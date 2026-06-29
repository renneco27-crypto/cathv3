import os, json, re, time, threading
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()
import study
import pytz
import telebot
from telebot import apihelper
from google import genai
from fastapi import FastAPI


# ==========================================
# INIT
# ==========================================
BOT_TOKEN          = os.environ.get("BOT_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY", "")

# Collect GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... in order until one's missing.
# Falls back to plain GEMINI_API_KEY (no suffix) if that's all that's set,
# so existing .env files with a single key still work with zero changes.
_gemini_keys = []
_i = 1
while True:
    _k = os.environ.get(f"GEMINI_API_KEY_{_i}", "")
    if not _k:
        break
    _gemini_keys.append(_k)
    _i += 1

if not _gemini_keys:
    _legacy = os.environ.get("GEMINI_API_KEY", "")
    if _legacy:
        _gemini_keys.append(_legacy)

GEMINI_API_KEY = _gemini_keys[0] if _gemini_keys else ""  # kept for any code that still reads this directly

bot = telebot.TeleBot(BOT_TOKEN, threaded=True, num_threads=8)

# One genai.Client per Gemini key. ai_client stays as the first one so any
# existing code calling ai_client.models.generate_content(...) directly
# keeps working unchanged. ai_clients (the full list) is what gets passed
# to study.py so it can rotate across keys on quota errors.
ai_clients = [genai.Client(api_key=k) for k in _gemini_keys]
ai_client  = ai_clients[0] if ai_clients else None

apihelper.READ_TIMEOUT    = 30
apihelper.CONNECT_TIMEOUT = 30

DATA_FILE = "bot_database.json"
PH_TZ     = pytz.timezone("Asia/Manila")

AI_PROVIDERS = (
    (["gemini"]     if GEMINI_API_KEY     else []) +
    (["openrouter"] if OPENROUTER_API_KEY else []) +
    (["nvidia"]     if NVIDIA_API_KEY     else [])
)
from study import (
    init_study, extract_knowledge,
    cmd_flashcard, cmd_mc, cmd_identify, cmd_tf,
    cmd_enum, cmd_search, cmd_answer,
    cmd_studystats, cmd_studydocs,
    QUIZ_STATE,
)
init_study(bot, ai_client, ai_clients_list=ai_clients)

# ==========================================
# DATABASE  (in-memory cache + file)
# ==========================================
_db_lock  = threading.Lock()
_db_cache: dict | None = None

_EMPTY_DB = lambda: {"notes": [], "schedules": [], "classes": {}, "recurring": [], "specials": []}

def load_db() -> dict:  # type: ignore[return]
    global _db_cache
    with _db_lock:
        if _db_cache is not None:
            return _db_cache
        if not os.path.exists(DATA_FILE):
            _db_cache = _EMPTY_DB()
            return _db_cache
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            _db_cache = _EMPTY_DB()
            return _db_cache
        if not isinstance(data.get("notes"), list):
            data["notes"] = []
        data.setdefault("schedules", [])
        data.setdefault("classes", {})
        data.setdefault("recurring", [])
        data.setdefault("specials", [])
        _db_cache = data
        return _db_cache

def save_db(data: dict) -> None:
    with _db_lock:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

def get_chat_schedules(db, chat_id): return [s for s in db["schedules"] if s.get("chat_id") == chat_id]
def get_chat_recurring(db, chat_id): return [r for r in db["recurring"]  if r.get("chat_id") == chat_id]
def get_chat_specials(db, chat_id):  return [s for s in db["specials"]   if s.get("chat_id") == chat_id]
def get_chat_notes(db, chat_id):     return [n for n in db["notes"]      if n.get("chat_id") == chat_id]

# ==========================================
# AI ENGINE
# ==========================================
def ask_ai(prompt: str) -> str:
    for provider in AI_PROVIDERS:
        try:
            if provider == "gemini":
                for model in ("gemini-2.0-flash", "gemini-1.5-flash"):
                    try:
                        resp = ai_client.models.generate_content(
                            model=model,
                            contents=f"Answer shortly and concisely: {prompt}"
                        )
                        return resp.text or "🤖 No response."
                    except Exception as me:
                        if "quota" in str(me).lower() or "429" in str(me):
                            continue
                        raise

            elif provider == "openrouter":
                import urllib.request, json as _j
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=_j.dumps({"model": "mistralai/mistral-7b-instruct",
                                   "messages": [{"role": "user", "content": f"Answer shortly: {prompt}"}],
                                   "max_tokens": 500}).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "HTTP-Referer": "https://github.com", "X-Title": "CathBot"}
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    return _j.loads(r.read())["choices"][0]["message"]["content"]

            elif provider == "nvidia":
                import urllib.request, json as _j
                req = urllib.request.Request(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    data=_j.dumps({"model": "meta/llama-3.1-8b-instruct",
                                   "messages": [{"role": "user", "content": f"Answer shortly: {prompt}"}],
                                   "max_tokens": 500}).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {NVIDIA_API_KEY}"}
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    return _j.loads(r.read())["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[AI] {provider} failed: {e}")
    return "🤖 All AI engines are offline. Try again later."

# ==========================================
# DATE / TIME PARSERS
# ==========================================
WEEKDAY_ABBREV = {
    "mon": "monday", "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "thu": "thursday", "thur": "thursday",
    "thurs": "thursday", "fri": "friday", "sat": "saturday", "sun": "sunday",
}
WEEKDAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
MONTH_NAME_MAP = {
    "january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,
    "april":4,"apr":4,"may":5,"june":6,"jun":6,"july":7,"jul":7,
    "august":8,"aug":8,"september":9,"sep":9,"sept":9,
    "october":10,"oct":10,"november":11,"nov":11,"december":12,"dec":12,
}

SPELLING_FIXES = [
    (r'\btomm?or+ow\b|\btomoro\b|\btmr[w]?\b|\btom\b', 'tomorrow'),
    (r'\btodat\b|\btodey\b', 'today'),
    (r'(\d)(am|pm)\b', r'\1 \2'),
]

def fuzzy_correct(text: str) -> str:
    for pat, rep in SPELLING_FIXES:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

def parse_time_only(text: str) -> str | None:
    for pat in [r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b',
                r'\b(\d{1,2})\s*(am|pm)\b',
                r'\b(\d{1,2}):(\d{2})\b']:
        m = re.search(pat, text, re.IGNORECASE)
        if not m: continue
        g = m.groups()
        if len(g) == 3:
            h, mn, mer = int(g[0]), int(g[1]), g[2].upper()
        elif len(g) == 2 and g[1].isalpha():
            h, mn, mer = int(g[0]), 0, g[1].upper()
        else:
            h, mn = int(g[0]), int(g[1])
            mer = "PM" if 1 <= h <= 6 else "AM"
        try:
            return datetime.strptime(f"{h}:{mn:02d} {mer}", "%I:%M %p").strftime("%I:%M %p")
        except ValueError:
            pass
    return None

def parse_date_only(raw_text: str, now_ph: datetime) -> str | None:
    text  = fuzzy_correct(raw_text)
    now_n = now_ph.replace(tzinfo=None)

    m = re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', text)
    if m:
        try: return datetime(int(m[1]),int(m[2]),int(m[3])).strftime("%Y-%m-%d")
        except ValueError: pass

    m = re.search(r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', text)
    if m:
        try: return datetime(int(m[3]),int(m[1]),int(m[2])).strftime("%Y-%m-%d")
        except ValueError: pass

    m = re.search(r'\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b', text)
    if m:
        mo, d = int(m[1]), int(m[2])
        y = int(m[3]) if m[3] else now_n.year
        if y < 100: y += 2000
        try: return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError: pass

    stripped = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b', '', text, flags=re.IGNORECASE)

    if re.search(r'\btomorrow\b', stripped, re.IGNORECASE):
        return (now_n + timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r'\btoday\b', stripped, re.IGNORECASE):
        return now_n.strftime("%Y-%m-%d")

    for name, wday in WEEKDAY_MAP.items():
        if re.search(rf'\b{name}\b', stripped, re.IGNORECASE):
            days = (wday - now_n.weekday() + 7) % 7 or 7
            return (now_n + timedelta(days=days)).strftime("%Y-%m-%d")
    for abbr, full in WEEKDAY_ABBREV.items():
        if re.search(rf'\b{abbr}\b', stripped, re.IGNORECASE):
            wday = WEEKDAY_MAP[full]
            days = (wday - now_n.weekday() + 7) % 7 or 7
            return (now_n + timedelta(days=days)).strftime("%Y-%m-%d")

    month_alt = '|'.join(MONTH_NAME_MAP)
    m = re.search(
        r'\b(' + month_alt + r')\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b'
        r'|\b(\d{1,2})(?:st|nd|rd|th)?\s+(' + month_alt + r')(?:,?\s*(\d{4}))?\b',
        stripped, re.IGNORECASE
    )
    if m:
        g = m.groups()
        if g[0]:
            mo, d, yr = MONTH_NAME_MAP[g[0].lower()], int(g[1]), g[2]
        else:
            mo, d, yr = MONTH_NAME_MAP[g[4].lower()], int(g[3]), g[5]
        y = int(yr) if yr else now_n.year
        try:
            c = datetime(y, mo, d)
            if not yr and c.date() < now_n.date():
                c = datetime(y+1, mo, d)
            return c.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def parse_schedule_time(raw_text: str, now_ph: datetime) -> tuple:
    now_n    = now_ph.replace(tzinfo=None)
    date_str = parse_date_only(raw_text, now_ph)
    time_str = parse_time_only(fuzzy_correct(raw_text))
    if date_str:
        base = datetime.strptime(date_str, "%Y-%m-%d")
        if time_str:
            t = datetime.strptime(time_str, "%I:%M %p")
            return datetime(base.year, base.month, base.day, t.hour, t.minute), ""
        return datetime(base.year, base.month, base.day, 8, 0), ""
    if time_str:
        t = datetime.strptime(time_str, "%I:%M %p")
        c = datetime(now_n.year, now_n.month, now_n.day, t.hour, t.minute)
        if c <= now_n: c += timedelta(days=1)
        return c, time_str
    return None, ""

# ==========================================
# TEXT CLEANER
# ==========================================
NOISE_WORDS = {
    'change','update','edit','modify','move','reschedule','reset','set',
    'to','at','the','it','a','an','sched','schedule','am','pm',
    'today','tomorrow','tmr','tmrw','tom','this','now','in','on',
    'date','time','day','from','for','my','please','remind','me',
    'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
    'mon','tue','wed','thu','fri','sat','sun',
    'january','february','march','april','may','june','july',
    'august','september','october','november','december',
}

_MONTH_ALT = '|'.join(MONTH_NAME_MAP)
_NOISE_PATTERNS = [
    r'\b\d{4}-\d{1,2}-\d{1,2}\b', r'\b\d{1,2}-\d{1,2}-\d{4}\b',
    r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
    r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b', r'\b\d{1,2}\s*(?:am|pm)\b', r'\b\d{1,2}:\d{2}\b',
    r'\b(' + _MONTH_ALT + r')\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(' + _MONTH_ALT + r')(?:,?\s+\d{4})?\b',
    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\b(mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b',
]

def clean_task_text(raw: str, matched_str: str) -> str:
    task = raw.replace(matched_str, "") if matched_str else raw
    for p in _NOISE_PATTERNS:
        task = re.sub(p, "", task, flags=re.IGNORECASE)
    words = re.split(r'\W+', task)
    words = [w for w in words if w.lower() not in NOISE_WORDS and len(w) > 1]
    result = " ".join(words).strip(" .,")
    return (result.capitalize() if result else raw.capitalize()) or ""

# ==========================================
# INTENT DETECTION
# ==========================================
_SCHED_SIGNALS = re.compile(
    r'\bsched\b|\bschedule\b|\bremind\b|\balert\b|\btell\b|\bnotify\b'
    r'|\bat\s+\d|\b\d+(am|pm)\b|\btoday\b|\btomorrow\b|\bnext\b|\btmr\b|\btmrw\b|\btom\b'
    r'|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{1,2}-\d{1,2}-\d{4}\b'
    r'|\b\d{1,2}:\d{2}\b|\b(mon|tue|wed|thu|fri|sat|sun)\b',
    re.IGNORECASE
)
_CONVO_SIGNALS = re.compile(
    r'^(hello|hi|hey|haha|hehe|lol|wow|omg|okay|ok|ay|uy|oo|yep|nope|sure|nice|aww|cute)\b'
    r'|^(creating|reading|watching|sending|checking)\b'
    r'|(haha|hehe|lmao|ang cute|charot|char)',
    re.IGNORECASE
)

def is_real_schedule_command(text: str) -> bool:
    t = text.lower().strip()
    return (bool(_SCHED_SIGNALS.search(t))
            and not bool(_CONVO_SIGNALS.search(t))
            and len(t.split()) >= 3)

# ==========================================
# RECURRING HELPERS
# ==========================================
_RECUR_PATTERNS = {
    'daily':   re.compile(r'\beveryday\b|\bdaily\b|\bevery\s+day\b|\beach\b', re.IGNORECASE),
    'weekly':  re.compile(r'\bevery\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b|\bweekly\b', re.IGNORECASE),
    'monthly': re.compile(r'\bevery\s+month\b|\bmonthly\b|\bevery\s+\d{1,2}(?:st|nd|rd|th)\b', re.IGNORECASE),
    'yearly':  re.compile(r'\bevery\s+year\b|\bannually\b|\byearly\b', re.IGNORECASE),
}

def detect_recur_type(text: str) -> str | None:
    t = re.sub(r'\b\d{1,2}:?\d{0,2}\s*(?:am|pm)?\b', '', text, flags=re.IGNORECASE)
    for rtype, pat in _RECUR_PATTERNS.items():
        if pat.search(t): return rtype
    return None

def detect_recur_weekday(text: str) -> str | None:
    for name in WEEKDAY_MAP:
        if re.search(rf'\b{name}\b', text, re.IGNORECASE): return name.capitalize()
    for abbr, full in WEEKDAY_ABBREV.items():
        if re.search(rf'\b{abbr}\b', text, re.IGNORECASE): return full.capitalize()
    return None

def detect_recur_monthday(text: str) -> int | None:
    m = re.search(r'\bevery\s+(\d{1,2})(?:st|nd|rd|th)?\b', text, re.IGNORECASE)
    return int(m[1]) if m else None

def next_recur_date(item: dict, now_ph: datetime) -> datetime:
    now_n = now_ph.replace(tzinfo=None)
    try:    t_obj = datetime.strptime(item.get("time","08:00 AM"), "%I:%M %p").time()
    except: t_obj = datetime.strptime("08:00 AM", "%I:%M %p").time()
    rt = item["recur_type"]
    if rt == "daily":
        c = datetime.combine(now_n.date(), t_obj)
        return c if c > now_n else c + timedelta(days=1)
    if rt == "weekly":
        wday = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(item.get("recur_day","Monday"))
        days = (wday - now_n.weekday() + 7) % 7
        if days == 0:
            c = datetime.combine(now_n.date(), t_obj)
            return c if c > now_n else datetime.combine((now_n + timedelta(days=7)).date(), t_obj)
        return datetime.combine((now_n + timedelta(days=days)).date(), t_obj)
    if rt == "monthly":
        dom = item.get("recur_monthday", 1)
        try:
            c = datetime(now_n.year, now_n.month, dom, t_obj.hour, t_obj.minute)
            if c > now_n: return c
        except ValueError: pass
        nm = now_n.month % 12 + 1
        ny = now_n.year + (1 if nm == 1 else 0)
        return datetime(ny, nm, dom, t_obj.hour, t_obj.minute)
    if rt == "yearly":
        mo, d = item.get("recur_month", now_n.month), item.get("recur_day_num", now_n.day)
        try:
            c = datetime(now_n.year, mo, d, t_obj.hour, t_obj.minute)
            if c > now_n: return c
        except ValueError: pass
        return datetime(now_n.year+1, mo, d, t_obj.hour, t_obj.minute)
    return now_n + timedelta(days=1)

# ==========================================
# SPECIAL EVENTS
# ==========================================
_SPECIAL_PATTERNS = {
    'birthday':    re.compile(r'\bbirthday\b|\bbday\b', re.IGNORECASE),
    'anniversary': re.compile(r'\banniversary\b|\banniv\b', re.IGNORECASE),
    'monthsary':   re.compile(r'\bmonthsary\b|\bmonthiversary\b', re.IGNORECASE),
}

def detect_special_type(text: str) -> str | None:
    for stype, pat in _SPECIAL_PATTERNS.items():
        if pat.search(text): return stype
    return None

def next_special_date(item: dict, now_ph: datetime) -> datetime:
    now_n = now_ph.replace(tzinfo=None)
    try:    t_obj = datetime.strptime(item.get("time","08:00 AM"), "%I:%M %p").time()
    except: t_obj = datetime.strptime("08:00 AM", "%I:%M %p").time()
    if item["special_type"] == "monthsary":
        dom = item.get("day_of_month", 1)
        try:
            c = datetime(now_n.year, now_n.month, dom, t_obj.hour, t_obj.minute)
            if c > now_n: return c
        except ValueError: pass
        nm = now_n.month % 12 + 1
        ny = now_n.year + (1 if nm == 1 else 0)
        return datetime(ny, nm, dom, t_obj.hour, t_obj.minute)
    mo, d = item.get("month", now_n.month), item.get("day", now_n.day)
    try:
        c = datetime(now_n.year, mo, d, t_obj.hour, t_obj.minute)
        if c > now_n: return c
    except ValueError: pass
    return datetime(now_n.year+1, mo, d, t_obj.hour, t_obj.minute)

# ==========================================
# EDIT HELPERS
# ==========================================
_EDIT_SIGNALS = re.compile(r'\bchange\b|\bupdate\b|\bedit\b|\bmodify\b|\bmove\b|\breset\b|\breschedule\b|\bset\b', re.IGNORECASE)
PENDING_EDITS: dict = {}

def detect_edit_intent(text: str) -> bool:
    return bool(_EDIT_SIGNALS.search(text))

def extract_task_keywords(text: str) -> list:
    t = re.sub(r'\b\d{1,2}:?\d{0,2}\s*(?:am|pm)?\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{1,2}-\d{1,2}-\d{4}\b', '', text, flags=re.IGNORECASE)
    quoted = re.search(r'["\'](.+?)["\']', t)
    if quoted: return quoted[1].lower().split()
    return [w for w in re.split(r'\W+', t.lower()) if w and w not in NOISE_WORDS and len(w) > 2]

def find_best_match(schedules: list, keywords: list):
    if not schedules: return None
    if not keywords:  return schedules[-1]
    best, best_score = None, 0
    for item in schedules:
        score = sum(1 for w in keywords if w in item["task"].lower())
        if score > best_score:
            best_score, best = score, item
    return best or schedules[-1]

def find_matching_tasks(schedules: list, keywords: list) -> list:
    if not keywords: return list(schedules)
    return [s for s in schedules if any(w in s["task"].lower() for w in keywords)]

def try_patch_edit(db: dict, raw_text: str, now_ph: datetime, chat_id) -> tuple:
    chat_schedules = get_chat_schedules(db, chat_id)
    if not chat_schedules: return False, ""
    keywords = extract_task_keywords(raw_text)
    target   = find_best_match(chat_schedules, keywords)
    if not target: return False, ""
    old_date, old_time = target["date"], target.get("time","08:00 AM")
    new_date = parse_date_only(raw_text, now_ph) or old_date
    new_time = parse_time_only(raw_text)          or old_time
    if new_date == old_date and new_time == old_time: return False, ""
    target["date"], target["time"] = new_date, new_time
    parts = []
    if new_date != old_date: parts.append(f"Date: {old_date} → {new_date}")
    if new_time != old_time: parts.append(f"Time: {old_time} → {new_time}")
    return True, f"✏️ Updated: {target['task']}\n" + "\n".join(parts)

def is_duplicate(db, task, date, time_str, chat_id) -> bool:
    return any(
        s["task"].lower() == task.lower() and s["date"] == date and s.get("time","") == time_str
        for s in get_chat_schedules(db, chat_id)
    )

# ==========================================
# UPLOAD STATE
# ==========================================
_pending_uploads: dict = {}
_pending_lock = threading.Lock()
PENDING_TIMEOUT = 300  # 5 minutes

def set_pending(chat_id, title):
    with _pending_lock:
        _pending_uploads[chat_id] = {"title": title, "collected": [], "expires": time.time()+PENDING_TIMEOUT}

def get_pending(chat_id):
    with _pending_lock:
        p = _pending_uploads.get(chat_id)
        if p and time.time() < p["expires"]: return p
        if p: del _pending_uploads[chat_id]
        return None

def clear_pending(chat_id):
    with _pending_lock:
        _pending_uploads.pop(chat_id, None)

def is_pending(chat_id) -> bool:
    return get_pending(chat_id) is not None

# ==========================================
# PROCESS SCHEDULE LINE
# ==========================================
def process_schedule_line(message, raw_text: str):
    recur_type = detect_recur_type(raw_text)
    if recur_type:
        _handle_recurring_create(message, raw_text, recur_type); return
    special_type = detect_special_type(raw_text)
    if special_type:
        _handle_special_create(message, raw_text, special_type); return
    if not is_real_schedule_command(raw_text): return

    db      = load_db()
    chat_id = message.chat.id
    now_ph  = datetime.now(PH_TZ)

    if detect_edit_intent(raw_text):
        ok, msg = try_patch_edit(db, raw_text, now_ph, chat_id)
        if ok:
            save_db(db); bot.reply_to(message, msg)
        else:
            bot.reply_to(message,
                "⚠️ Couldn't find a matching task.\n"
                "Tip: /edit \"task\" new_date_or_time\nOr /active to see task names.")
        return

    parsed_dt, matched = parse_schedule_time(raw_text, now_ph)
    if not parsed_dt:
        bot.reply_to(message, "📅 No date/time detected.\nTry: -sched biking tomorrow 7pm"); return
    if parsed_dt < now_ph.replace(tzinfo=None):
        bot.reply_to(message, f"⚠️ {parsed_dt.strftime('%Y-%m-%d %I:%M %p')} is in the past."); return

    ev_date = parsed_dt.strftime("%Y-%m-%d")
    ev_time = parsed_dt.strftime("%I:%M %p")
    task    = clean_task_text(raw_text, matched)

    if is_duplicate(db, task, ev_date, ev_time, chat_id):
        bot.reply_to(message, f"⚠️ Already saved: {task} on {ev_date} at {ev_time}."); return

    db["schedules"].append({"id": int(time.time()*1000), "date": ev_date, "time": ev_time,
                             "task": task, "chat_id": chat_id, "notified": False})
    save_db(db)
    bot.reply_to(message, f"📅 Saved!\n• {task}\n• {ev_date} at {ev_time}")

def _handle_recurring_create(message, raw_text: str, recur_type: str):
    db     = load_db()
    now_ph = datetime.now(PH_TZ)
    t_str  = parse_time_only(raw_text) or "08:00 AM"
    task   = raw_text
    for p in [r'\beveryday\b',r'\bdaily\b',r'\bevery\s+day\b',r'\beach\b',
              r'\bevery\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b',
              r'\bevery\s+month\b',r'\bmonthly\b',r'\bevery\s+\d{1,2}(?:st|nd|rd|th)\b',
              r'\bevery\s+year\b',r'\bannually\b',r'\byearly\b',r'\bweekly\b',
              r'\bremind\s+me\s+to\b',r'\bremind\s+to\b',
              r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',r'\b\d{1,2}\s*(?:am|pm)\b']:
        task = re.sub(p, '', task, flags=re.IGNORECASE)
    task = re.sub(r'\s+', ' ', task).strip(" .,").capitalize() or "Recurring Task"
    item: dict = {"id": int(time.time()*1000), "task": task, "time": t_str,
                  "recur_type": recur_type, "chat_id": message.chat.id, "last_notified": None}
    if recur_type == "weekly":  item["recur_day"]      = detect_recur_weekday(raw_text) or "Monday"
    if recur_type == "monthly": item["recur_monthday"] = detect_recur_monthday(raw_text) or now_ph.day
    if recur_type == "yearly":  item["recur_month"] = now_ph.month; item["recur_day_num"] = now_ph.day
    db["recurring"].append(item); save_db(db)
    if recur_type == "daily":     desc = f"Every day at {t_str}"
    elif recur_type == "weekly":  desc = f"Every {item.get('recur_day')} at {t_str}"
    elif recur_type == "monthly": desc = f"Every month, day {item.get('recur_monthday')} at {t_str}"
    else:                         desc = f"Every year at {t_str}"
    bot.reply_to(message, f"🔁 Recurring Saved!\n• {task}\n• {desc}\n• ID:{item['id']}\n/delrecurring id:{item['id']} to remove.")

def _handle_special_create(message, raw_text: str, special_type: str):
    db     = load_db()
    now_ph = datetime.now(PH_TZ)
    t_str  = parse_time_only(raw_text) or "08:00 AM"
    date_str = parse_date_only(raw_text, now_ph)
    label  = raw_text
    for pat in _SPECIAL_PATTERNS.values():
        label = pat.sub('', label)
    label = re.sub(r'\b\d{1,2}:?\d{0,2}\s*(?:am|pm)?\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}-\d{1,2}-\d{4}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', '', label, flags=re.IGNORECASE)
    label = re.sub(r'\b(' + _MONTH_ALT + r')\b\s*\d{0,2}', '', label, flags=re.IGNORECASE)
    label = re.sub(r'\b\d{1,2}\b|\s+', ' ', label).strip(" .,").capitalize() or special_type.capitalize()
    item: dict = {"id": int(time.time()*1000), "label": label, "special_type": special_type,
                  "time": t_str, "chat_id": message.chat.id, "last_notified": None}
    if date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        item.update({"month": dt.month, "day": dt.day, "day_of_month": dt.day})
        date_display = dt.strftime("%B %d")
    else:
        item.update({"month": now_ph.month, "day": now_ph.day, "day_of_month": now_ph.day})
        date_display = now_ph.strftime("%B %d")
    db["specials"].append(item); save_db(db)
    recur_note = "every month" if special_type == "monthsary" else "every year"
    bot.reply_to(message, f"🎉 {special_type.capitalize()} Saved!\n• {label}\n• {date_display} ({recur_note}) at {t_str}\n• ID:{item['id']}")

# ==========================================
# HANDLERS
# ==========================================

# --- Dash-line schedule (MUST be first non-command handler) ---
@bot.message_handler(func=lambda m: m.text and any(l.strip().startswith("-") for l in m.text.split("\n")))
def handle_dash_schedule(message):
    # Don't intercept during upload sessions
    if is_pending(message.chat.id): return
    for line in message.text.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            raw = line[1:].replace("@","").strip()
            if raw: process_schedule_line(message, raw)

# --- Photo upload handler ---
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    p = get_pending(chat_id)
    if not p: return
    db  = load_db()
    rec = {"id": int(time.time()*1000), "title": p["title"], "chat_id": chat_id,
           "type": "photo", "file_id": message.photo[-1].file_id}
    db["notes"].append(rec)
    save_db(db)
    with _pending_lock:
        if chat_id in _pending_uploads:
            _pending_uploads[chat_id]["collected"].append(rec["id"])
            _pending_uploads[chat_id]["expires"] = time.time() + PENDING_TIMEOUT
    count = len(_pending_uploads.get(chat_id, {}).get("collected", []))
    bot.reply_to(message, f"✅ Photo {count} saved under '{p['title']}'. Send more or /donesaving.")

# --- Text note handler during upload session ---
# This MUST come before smart_question and handle_show
@bot.message_handler(
    func=lambda m: (
        m.text is not None
        and not m.text.startswith('/')
        and is_pending(m.chat.id)
    ),
    content_types=['text']
)
def handle_text_note(message):
    chat_id = message.chat.id
    p = get_pending(chat_id)
    if not p: return
    db  = load_db()
    rec = {"id": int(time.time()*1000), "title": p["title"], "chat_id": chat_id,
           "type": "text", "content": message.text}
    db["notes"].append(rec)
    save_db(db)
    with _pending_lock:
        if chat_id in _pending_uploads:
            _pending_uploads[chat_id]["collected"].append(rec["id"])
            _pending_uploads[chat_id]["expires"] = time.time() + PENDING_TIMEOUT
    count = len(_pending_uploads.get(chat_id, {}).get("collected", []))
    bot.reply_to(message, f"✅ Note {count} saved under '{p['title']}'. Send more or /donesaving.")

# --- /upload ---
@bot.message_handler(commands=['upload'])
def handle_upload(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2: bot.reply_to(message, "Usage: /upload [title]"); return
    title = args[1].lower().strip()
    set_pending(message.chat.id, title)
    bot.reply_to(message, f"📎 Ready — send photos or text for *'{title}'*, then /donesaving (AI extraction) or /note (save only).\nSession expires in 5 minutes.", parse_mode="Markdown")

# --- /donesaving ---
@bot.message_handler(commands=['donesaving'])
def handle_done_saving(message):
    chat_id = message.chat.id
    p = get_pending(chat_id)
    if not p: bot.reply_to(message, "⚠️ No active upload session. /upload [title] first."); return
    count, title = len(p["collected"]), p["title"]
    clear_pending(chat_id)
    if count == 0:
        bot.reply_to(message, f"⚠️ No photos or notes were received for '{title}'.\nStart again with /upload {title}")
        return
    bot.reply_to(message, f"✅ Saved {count} item(s) under *'{title}'*.\n⏳ Extracting knowledge… I'll notify you when done.", parse_mode="Markdown")
    threading.Thread(target=extract_knowledge, args=(title, chat_id), daemon=True).start()

# --- /note ---
@bot.message_handler(commands=['note'])
def handle_note(message):
    chat_id = message.chat.id
    p = get_pending(chat_id)
    if not p:
        bot.reply_to(message, "⚠️ No active upload session. /upload [title] first.")
        return
    count, title = len(p["collected"]), p["title"]
    clear_pending(chat_id)
    if count == 0:
        bot.reply_to(message, f"⚠️ No photos or notes were received for '{title}'.\nStart again with /upload {title}")
        return
    bot.reply_to(message, f"✅ Saved {count} item(s) under *'{title}'*\\. Use /studydocs to see all docs\\.", parse_mode="MarkdownV2")

# --- /deleteupload ---
@bot.message_handler(commands=['deleteupload'])
def delete_upload(message):
    q = message.text.replace("/deleteupload","",1).strip()
    if not q: bot.reply_to(message, "Usage: /deleteupload [keyword | id:xxx | all]"); return
    db, chat_id = load_db(), message.chat.id
    if q.lower()=="all":
        n = len(get_chat_notes(db, chat_id))
        db["notes"] = [n for n in db["notes"] if n.get("chat_id") != chat_id]
        save_db(db); bot.reply_to(message, f"🗑️ Cleared {n} upload(s)."); return
    m = re.match(r'id:(\d+)', q, re.IGNORECASE)
    if m:
        tid = int(m[1]); before = len(db["notes"])
        db["notes"] = [n for n in db["notes"] if not (n.get("id")==tid and n.get("chat_id")==chat_id)]
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted ID {tid}." if before>len(db["notes"]) else f"🚫 ID {tid} not found."); return
    before = len(db["notes"])
    db["notes"] = [n for n in db["notes"] if not (n.get("chat_id")==chat_id and q.lower() in n.get("title","").lower())]
    d = before - len(db["notes"])
    if d: save_db(db); bot.reply_to(message, f"🗑️ Deleted {d} upload(s).")
    else: bot.reply_to(message, f"🚫 None found matching '{q}'.")

# --- /uploads ---
@bot.message_handler(commands=['uploads'])
def list_uploads(message):
    items = get_chat_notes(load_db(), message.chat.id)
    if not items: bot.reply_to(message, "📎 No uploads yet."); return
    e = {"photo":"🖼️","file":"📄","text":"📝"}
    lines = [f"{e.get(n['type'],'📎')} [ID:{n['id']}] {n['title']}" for n in items]
    bot.reply_to(message, "📎 Uploads:\n\n" + "\n".join(lines))

# --- /active ---
@bot.message_handler(commands=['active'])
def show_active(message):
    db      = load_db()
    chat_id = message.chat.id
    now_n   = datetime.now(PH_TZ).replace(tzinfo=None)
    cutoff  = now_n + timedelta(days=90)
    lines   = []
    for s in get_chat_schedules(db, chat_id):
        try:
            dt = datetime.strptime(f"{s['date']} {s.get('time','12:00 AM')}", "%Y-%m-%d %I:%M %p")
            if now_n <= dt <= cutoff:
                lines.append(f"• [ID:{s.get('id','?')}] {s['date']} {s.get('time','')} — {s['task']}")
        except ValueError: pass
    bot.reply_to(message, ("⏳ Upcoming (90 days)\n\n" + "\n".join(lines)) if lines else "✅ Nothing scheduled. You're free!")

# --- /delete ---
@bot.message_handler(commands=['delete'])
def delete_task(message):
    q = message.text.replace("/delete","",1).strip()
    if not q: bot.reply_to(message, "Usage: /delete [keyword | id:xxx | all]"); return
    db, chat_id = load_db(), message.chat.id
    if q.lower() == "all":
        n = len(get_chat_schedules(db, chat_id))
        db["schedules"] = [s for s in db["schedules"] if s.get("chat_id") != chat_id]
        save_db(db); bot.reply_to(message, f"🗑️ Cleared {n} task(s)."); return
    m = re.match(r'id:(\d+)', q, re.IGNORECASE)
    if m:
        tid = int(m[1]); before = len(db["schedules"])
        db["schedules"] = [s for s in db["schedules"] if not (s.get("id")==tid and s.get("chat_id")==chat_id)]
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted ID {tid}." if before>len(db["schedules"]) else f"🚫 ID {tid} not found."); return
    before = len(db["schedules"])
    db["schedules"] = [s for s in db["schedules"] if not (s.get("chat_id")==chat_id and q.lower() in s["task"].lower())]
    d = before - len(db["schedules"])
    if d: save_db(db); bot.reply_to(message, f"🗑️ Deleted {d} task(s) matching '{q}'.")
    else: bot.reply_to(message, f"🚫 No tasks matching '{q}'.")

# --- /edit ---
@bot.message_handler(commands=['edit'])
def edit_task(message):
    q = message.text.replace("/edit","",1).strip()
    if not q: bot.reply_to(message, "Usage: /edit [task keyword] [new date/time]\nExample: /edit run tom 9am"); return
    db, chat_id = load_db(), message.chat.id
    chat_scheds = get_chat_schedules(db, chat_id)
    if not chat_scheds: bot.reply_to(message, "🚫 No schedules."); return
    now_ph   = datetime.now(PH_TZ)
    new_date = parse_date_only(q, now_ph)
    new_time = parse_time_only(q)
    if not new_date and not new_time: bot.reply_to(message, "⚠️ Couldn't detect new date or time."); return
    keywords   = extract_task_keywords(q)
    candidates = find_matching_tasks(chat_scheds, keywords)
    if not candidates: bot.reply_to(message, f"🚫 No tasks matching '{q}'."); return
    if len(candidates) == 1:
        if new_date: candidates[0]["date"] = new_date
        if new_time: candidates[0]["time"] = new_time
        save_db(db)
        bot.reply_to(message, f"✏️ Updated: {candidates[0]['task']}\n• {candidates[0]['date']} at {candidates[0].get('time','')}"); return
    lines = [f"{i+1}. {c['task']} {c.get('time','')}" for i, c in enumerate(candidates)]
    PENDING_EDITS[chat_id] = {"candidates": [c["id"] for c in candidates], "new_date": new_date, "new_time": new_time}
    bot.reply_to(message, "Multiple matches — reply with number:\n\n" + "\n".join(lines))

@bot.message_handler(func=lambda m: m.text and m.text.strip().isdigit() and m.chat.id in PENDING_EDITS)
def handle_edit_selection(message):
    chat_id = message.chat.id
    pending = PENDING_EDITS.pop(chat_id, None)
    if not pending: return
    choice = int(message.text.strip()) - 1
    if choice < 0 or choice >= len(pending["candidates"]):
        bot.reply_to(message, "🚫 Invalid number."); return
    db = load_db()
    t = next((s for s in db["schedules"] if s.get("id")==pending["candidates"][choice] and s.get("chat_id")==chat_id), None)
    if not t: bot.reply_to(message, "🚫 Task not found."); return
    if pending["new_date"]: t["date"] = pending["new_date"]
    if pending["new_time"]: t["time"] = pending["new_time"]
    save_db(db)
    bot.reply_to(message, f"✏️ Updated: {t['task']}\n• {t['date']} at {t.get('time','')}")

# --- /parse ---
@bot.message_handler(commands=['parse'])
def parse_test(message):
    raw = message.text.replace("/parse","",1).strip()
    if not raw: bot.reply_to(message, "Usage: /parse [text]"); return
    dt, matched = parse_schedule_time(raw, datetime.now(PH_TZ))
    if not dt: bot.reply_to(message, "🔍 No date/time detected."); return
    task = clean_task_text(raw, matched)
    bot.reply_to(message, f"Date: {dt.strftime('%Y-%m-%d')}\nTime: {dt.strftime('%I:%M %p')}\nTask: {task}")

# --- /recurring ---
@bot.message_handler(commands=['recurring'])
def show_recurring(message):
    items = get_chat_recurring(load_db(), message.chat.id)
    if not items: bot.reply_to(message, "🔁 No recurring tasks."); return
    def desc(r):
        rt = r["recur_type"]
        if rt=="daily":   return f"Every day at {r.get('time','08:00 AM')}"
        if rt=="weekly":  return f"Every {r.get('recur_day','?')} at {r.get('time','08:00 AM')}"
        if rt=="monthly": return f"Every month, day {r.get('recur_monthday','?')} at {r.get('time','08:00 AM')}"
        return f"Every year at {r.get('time','08:00 AM')}"
    lines = [f"• [ID:{r.get('id','?')}] {r['task']} — {desc(r)}" for r in items]
    bot.reply_to(message, "🔁 Recurring Tasks\n\n" + "\n".join(lines))

@bot.message_handler(commands=['editrecurring'])
def edit_recurring(message):
    q = message.text.replace("/editrecurring","",1).strip()
    m = re.search(r'id:(\d+)', q, re.IGNORECASE)
    if not m: bot.reply_to(message, "Usage: /editrecurring id:xxx time:9pm  OR  id:xxx task:New name"); return
    tid = int(m[1]); db = load_db(); chat_id = message.chat.id
    t = next((r for r in db["recurring"] if r.get("id")==tid and r.get("chat_id")==chat_id), None)
    if not t: bot.reply_to(message, f"🚫 ID {tid} not found."); return
    nt = parse_time_only(q)
    if nt: t["time"] = nt
    tm = re.search(r'task:\s*(.+)', q, re.IGNORECASE)
    if tm: t["task"] = tm[1].strip().capitalize()
    save_db(db)
    bot.reply_to(message, f"✏️ Updated: {t['task']} — {t['recur_type']} at {t.get('time','08:00 AM')}")

@bot.message_handler(commands=['delrecurring'])
def del_recurring(message):
    q = message.text.replace("/delrecurring","",1).strip()
    db, chat_id = load_db(), message.chat.id
    if q.lower() == "all":
        n = len(get_chat_recurring(db, chat_id))
        db["recurring"] = [r for r in db["recurring"] if r.get("chat_id") != chat_id]
        save_db(db); bot.reply_to(message, f"🗑️ Cleared {n} recurring task(s)."); return
    m = re.search(r'id:(\d+)', q, re.IGNORECASE)
    if m:
        tid = int(m[1]); before = len(db["recurring"])
        db["recurring"] = [r for r in db["recurring"] if not (r.get("id")==tid and r.get("chat_id")==chat_id)]
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted ID {tid}." if before>len(db["recurring"]) else f"🚫 ID {tid} not found."); return
    before = len(db["recurring"])
    db["recurring"] = [r for r in db["recurring"] if not (r.get("chat_id")==chat_id and q.lower() in r["task"].lower())]
    d = before - len(db["recurring"])
    if d: save_db(db); bot.reply_to(message, f"🗑️ Deleted {d} matching.")
    else: bot.reply_to(message, f"🚫 None matching '{q}'.")

# --- /specials ---
@bot.message_handler(commands=['specials'])
def show_specials(message):
    items = get_chat_specials(load_db(), message.chat.id)
    if not items: bot.reply_to(message, "🎉 No special events."); return
    emoji_map = {'birthday':'🎂','anniversary':'💍','monthsary':'💕'}
    lines = []
    for s in items:
        e = emoji_map.get(s["special_type"],"🎉")
        if s["special_type"]=="monthsary": dd = f"Every month, day {s.get('day_of_month','?')}"
        else: dd = f"Every year on {datetime(2000,s.get('month',1),1).strftime('%B')} {s.get('day','?')}"
        lines.append(f"{e} [ID:{s.get('id','?')}] {s['label']} — {dd} at {s.get('time','08:00 AM')}")
    bot.reply_to(message, "🎉 Special Events\n\n" + "\n".join(lines))

@bot.message_handler(commands=['editspecial'])
def edit_special(message):
    q = message.text.replace("/editspecial","",1).strip()
    m = re.search(r'id:(\d+)', q, re.IGNORECASE)
    if not m: bot.reply_to(message, "Usage: /editspecial id:xxx time:9am  OR  id:xxx label:New Name"); return
    tid = int(m[1]); db = load_db(); chat_id = message.chat.id
    t = next((s for s in db["specials"] if s.get("id")==tid and s.get("chat_id")==chat_id), None)
    if not t: bot.reply_to(message, f"🚫 ID {tid} not found."); return
    nt = parse_time_only(q)
    if nt: t["time"] = nt
    lm = re.search(r'label:\s*(.+)', q, re.IGNORECASE)
    if lm: t["label"] = lm[1].strip().capitalize()
    save_db(db)
    bot.reply_to(message, f"✏️ Updated: {t['label']} ({t['special_type']}) at {t.get('time','08:00 AM')}")

@bot.message_handler(commands=['delspecial'])
def del_special(message):
    q = message.text.replace("/delspecial","",1).strip()
    db, chat_id = load_db(), message.chat.id
    if q.lower()=="all":
        n = len(get_chat_specials(db, chat_id))
        db["specials"] = [s for s in db["specials"] if s.get("chat_id") != chat_id]
        save_db(db); bot.reply_to(message, f"🗑️ Cleared {n} special event(s)."); return
    m = re.search(r'id:(\d+)', q, re.IGNORECASE)
    if m:
        tid = int(m[1]); before = len(db["specials"])
        db["specials"] = [s for s in db["specials"] if not (s.get("id")==tid and s.get("chat_id")==chat_id)]
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted ID {tid}." if before>len(db["specials"]) else f"🚫 ID {tid} not found."); return
    before = len(db["specials"])
    db["specials"] = [s for s in db["specials"] if not (s.get("chat_id")==chat_id and q.lower() in s["label"].lower())]
    d = before - len(db["specials"])
    if d: save_db(db); bot.reply_to(message, f"🗑️ Deleted {d} matching.")
    else: bot.reply_to(message, f"🚫 None matching '{q}'.")

# --- Study commands ---
@bot.message_handler(commands=['flashcard'])
def handle_flashcard(message):
    args = message.text.split(maxsplit=2); rest = args[1:] if len(args) > 1 else []
    reverse = bool(rest and rest[0].lower() == "reverse")
    title = " ".join(rest[1:] if reverse else rest).strip()
    if not title: bot.reply_to(message, "Usage: /flashcard [title]\n       /flashcard reverse [title]"); return
    cmd_flashcard(message, title, reverse)

@bot.message_handler(commands=['mc'])
def handle_mc(message):
    args = message.text.split(maxsplit=1)
    if len(args)<2: bot.reply_to(message,"Usage: /mc [title]"); return
    cmd_mc(message, args[1].strip())

@bot.message_handler(commands=['identify'])
def handle_identify(message):
    args = message.text.split(maxsplit=1)
    if len(args)<2: bot.reply_to(message,"Usage: /identify [title]"); return
    cmd_identify(message, args[1].strip())

@bot.message_handler(commands=['tf'])
def handle_tf(message):
    args = message.text.split(maxsplit=1)
    if len(args)<2: bot.reply_to(message,"Usage: /tf [title]"); return
    cmd_tf(message, args[1].strip())

@bot.message_handler(commands=['enum'])
def handle_enum(message):
    args = message.text.split(maxsplit=1)
    if len(args)<2: bot.reply_to(message,"Usage: /enum [title]"); return
    cmd_enum(message, args[1].strip())

@bot.message_handler(commands=['search'])
def handle_search(message):
    args = message.text.split(maxsplit=2)
    if len(args)<3: bot.reply_to(message,"Usage: /search [title] [keyword]"); return
    cmd_search(message, args[1].strip(), args[2].strip())

@bot.message_handler(commands=['answer'])
def handle_answer(message):
    args = message.text.split(maxsplit=1)
    if len(args)<2: bot.reply_to(message,"Usage: /answer [your answer]"); return
    cmd_answer(message, args[1].strip())

@bot.message_handler(commands=['studydocs'])
def handle_studydocs(message): cmd_studydocs(message)

@bot.message_handler(commands=['studystats'])
def handle_studystats(message):
    args = message.text.split(maxsplit=1)
    if len(args)<2: bot.reply_to(message,"Usage: /studystats [title]"); return
    cmd_studystats(message, args[1].strip())

# --- /help ---
@bot.message_handler(commands=['help','start'])
def show_help(message):
    bot.reply_to(message,
        "📖 COMMANDS\n\n"
        "SCHEDULES\n"
        "• -[task] [date] [time]  — save (start line with -)\n"
        "• /active — upcoming 90 days\n"
        "• /delete [keyword | id:xxx | all]\n"
        "• /edit [keyword] [new date/time]\n"
        "• /parse [text] — test date parsing\n\n"
        "RECURRING\n"
        "• -remind me to [task] everyday 8pm\n"
        "• /recurring • /editrecurring id:xxx • /delrecurring id:xxx\n\n"
        "SPECIAL EVENTS\n"
        "• -birthday [name] [date]\n"
        "• /specials • /editspecial id:xxx • /delspecial id:xxx\n\n"
        "UPLOADS\n"
        "• /upload [title] → send photos/text → /donesaving or /note\n"
        "• /uploads • /deleteupload [keyword | id:xxx | all]\n"
        "• show me [title] — retrieve\n\n"
        "STUDY\n"
        "• /studydocs • /studystats [title]\n"
        "• /flashcard [title] • /flashcard reverse [title]\n"
        "• /mc • /identify • /tf • /enum [title]\n"
        "• /search [title] [keyword]\n"
        "• /answer [reply]\n\n"
        "AI\n"
        "• Ask any question ending with ?"
    )

# --- "show me X" free-text (skip during upload session) ---
_SHOW_RE = re.compile(r'(?:show\s+me|show|get|find|give\s+me)\s+(.+)', re.IGNORECASE)

@bot.message_handler(func=lambda m: bool(m.text and _SHOW_RE.search(m.text.strip()) and not is_pending(m.chat.id)))
def handle_show(message):
    if is_pending(message.chat.id): return
    m = _SHOW_RE.search(message.text.strip())
    if not m: return
    kw = m[1].lower().strip()
    db = load_db(); chat_id = message.chat.id
    hits = [n for n in get_chat_notes(db, chat_id) if kw in n.get("title","").lower()]
    if not hits: bot.reply_to(message, f"🔍 Nothing found for *'{kw}'*.", parse_mode="Markdown"); return
    bot.reply_to(message, f"📎 {len(hits)} item(s) for *'{kw}'*:", parse_mode="Markdown")
    for note in hits:
        try:
            if note["type"]=="photo":   bot.send_photo(chat_id, note["file_id"], caption=f"🖼️ {note['title']}")
            elif note["type"]=="file":  bot.send_document(chat_id, note["file_id"], caption=f"📄 {note['title']}")
            elif note["type"]=="text":  bot.send_message(chat_id, f"📝 *{note['title']}*\n\n{note['content']}", parse_mode="Markdown")
        except Exception as e:
            bot.send_message(chat_id, f"⚠️ Couldn't retrieve ID {note['id']}: {e}")

# --- Question fallback / AI (skip during upload session) ---
_TOPIC_KW = {"sched","event","task","remind","tomorrow","today","tmr","pic","pics","photo","photos","note","notes"}
_STRIP_RE  = re.compile(r'\b(tomorrow|today|tmr|tmrw|sched|task|event|any|is|do|i|have|a|an|the|anything|about|my|for|are|there|pic|pics|photo|photos|note|notes)\b', re.IGNORECASE)

@bot.message_handler(func=lambda m: m.text and m.text.strip().endswith('?') and not is_pending(m.chat.id))
def smart_question(message):
    if is_pending(message.chat.id): return
    text = message.text.strip()
    tl   = text.lower()
    db   = load_db(); chat_id = message.chat.id

    if any(k in tl for k in _TOPIC_KW):
        now_ph = datetime.now(PH_TZ)
        date_filters = []
        if re.search(r'\btomorrow\b|\btmr\b|\btmrw\b', tl): date_filters.append((now_ph+timedelta(days=1)).strftime("%Y-%m-%d"))
        if re.search(r'\btoday\b', tl): date_filters.append(now_ph.strftime("%Y-%m-%d"))
        kw_str  = _STRIP_RE.sub('', tl).replace('?','')
        kws     = [w for w in kw_str.split() if len(w)>2]
        def matches(t): return (not kws) or any(w in t.lower() for w in kws)
        scheds  = get_chat_schedules(db, chat_id)
        if date_filters: scheds = [s for s in scheds if s["date"] in date_filters]
        lines  = [f"📅 [{s['date']} {s.get('time','')}] {s['task']}" for s in scheds if matches(s["task"])]
        lines += [f"🔁 {r['task']} (every {r['recur_type']} at {r.get('time','')})" for r in get_chat_recurring(db,chat_id) if matches(r["task"])]
        lines += [f"🎉 {s['label']} ({s['special_type']})" for s in get_chat_specials(db,chat_id) if matches(s["label"])]
        if lines: bot.reply_to(message, "Found:\n" + "\n".join(lines)); return
        notes = [n for n in get_chat_notes(db,chat_id) if matches(n.get("title",""))]
        if notes:
            for note in notes:
                try:
                    if note["type"]=="photo":   bot.send_photo(chat_id, note["file_id"], caption=f"📎 {note['title']}")
                    elif note["type"]=="file":  bot.send_document(chat_id, note["file_id"], caption=f"📎 {note['title']}")
                    else: bot.send_message(chat_id, f"📝 {note['title']}\n{note.get('content','')}")
                except: pass
            return
        bot.reply_to(message, "🔍 Nothing found. /active to see all."); return

    threading.Thread(target=lambda: bot.reply_to(message, ask_ai(text)), daemon=True).start()

# ==========================================
# REMINDER LOOP
# ==========================================
def reminder_loop():
    print("⏰ Reminder engine started.")
    while True:
        try:
            now_ph  = datetime.now(PH_TZ)
            db      = load_db()
            changed = False

            for item in db.get("schedules", []):
                if item.get("notified"): continue
                chat_id = item.get("chat_id")
                if not chat_id: continue
                try:
                    item_dt = PH_TZ.localize(datetime.strptime(f"{item['date']} {item.get('time','12:00 AM')}", "%Y-%m-%d %I:%M %p"))
                except ValueError: continue
                diff = (item_dt - now_ph).total_seconds()
                if -30 <= diff <= 60:
                    bot.send_message(chat_id, f"🔔 REMINDER\n\n{item['task']}\n{item['date']} at {item.get('time')}")
                    item["notified"] = True; changed = True
                elif 23*3600+55*60 <= diff <= 24*3600+5*60 and not item.get("warned_24h"):
                    bot.send_message(chat_id, f"⚠️ TOMORROW: {item['task']}\n📅 {item['date']} at {item.get('time')}")
                    item["warned_24h"] = True; changed = True

            for item in db.get("recurring", []):
                chat_id = item.get("chat_id")
                if not chat_id: continue
                try:
                    fire_dt = PH_TZ.localize(next_recur_date(item, now_ph))
                except: continue
                diff = (fire_dt - now_ph).total_seconds()
                last  = item.get("last_notified")
                fired = last and (now_ph.replace(tzinfo=None) - datetime.strptime(last, "%Y-%m-%d %H:%M")).total_seconds() < 23*3600
                if -30 <= diff <= 60 and not fired:
                    label = {'daily':'Daily','weekly':'Weekly','monthly':'Monthly','yearly':'Yearly'}.get(item["recur_type"],"Recurring")
                    bot.send_message(chat_id, f"🔁 {label.upper()} REMINDER\n\n{item['task']}\nEvery {item['recur_type']} at {item.get('time')}")
                    item["last_notified"] = now_ph.strftime("%Y-%m-%d %H:%M"); changed = True

            for item in db.get("specials", []):
                chat_id = item.get("chat_id")
                if not chat_id: continue
                try:
                    fire_dt = PH_TZ.localize(next_special_date(item, now_ph))
                except: continue
                diff  = (fire_dt - now_ph).total_seconds()
                last  = item.get("last_notified")
                fired = last and (now_ph.replace(tzinfo=None) - datetime.strptime(last, "%Y-%m-%d %H:%M")).total_seconds() < 23*3600
                e = {"birthday":"🎂","anniversary":"💍","monthsary":"💕"}.get(item["special_type"],"🎉")
                if 23*3600+55*60 <= diff <= 24*3600+5*60:
                    lw = item.get("last_warned_24h")
                    if not lw or lw != fire_dt.strftime("%Y-%m-%d"):
                        bot.send_message(chat_id, f"⚠️ {e} {item['special_type'].upper()} TOMORROW!\n{item['label']}\n📅 {fire_dt.strftime('%B %d')} at {item.get('time')}")
                        item["last_warned_24h"] = fire_dt.strftime("%Y-%m-%d"); changed = True
                if -30 <= diff <= 60 and not fired:
                    bot.send_message(chat_id, f"{e} {item['special_type'].upper()}!\n{item['label']}\n📅 {fire_dt.strftime('%B %d')}")
                    item["last_notified"] = now_ph.strftime("%Y-%m-%d %H:%M"); changed = True

            if changed:
                save_db(db)

        except Exception as e:
            print(f"[reminder] error: {e}")

        time.sleep(30)

# ==========================================
# STARTUP
# ==========================================
app = FastAPI()

@app.get("/health")
def health(): return {"status": "ok"}

threading.Thread(target=reminder_loop, daemon=True).start()
threading.Thread(target=lambda: __import__('uvicorn').run(app, host="0.0.0.0", port=8000), daemon=True).start()

print("✅ Bot armed. Polling…")
print(f"🤖 AI: {', '.join(AI_PROVIDERS) or 'none'}")

while True:
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        print(f"❌ Polling error, retry in 10s: {e}")
        time.sleep(10)