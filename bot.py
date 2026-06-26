import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
import re
import telebot
from telebot import apihelper
import threading
from datetime import datetime, timedelta
import dateparser
import pytz
from dateparser.search import search_dates
from google import genai


# ==========================================
# 1. INITIALIZATION & CREDENTIALS
# ==========================================
BOT_TOKEN          = os.environ.get("BOT_TOKEN", "")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY     = os.environ.get("NVIDIA_API_KEY", "")

bot       = telebot.TeleBot(BOT_TOKEN)
ai_client = genai.Client(api_key=GEMINI_API_KEY)
# Increase Telegram API timeouts to reduce ReadTimeout errors
from telebot import apihelper

telebot.apihelper.READ_TIMEOUT = 30 # type: ignore
telebot.apihelper.CONNECT_TIMEOUT = 30  # type: ignore
DATA_FILE = "bot_database.json"
PH_TZ     = pytz.timezone('Asia/Manila')

AI_PROVIDERS = []
if GEMINI_API_KEY:
    AI_PROVIDERS.append("gemini")
if OPENROUTER_API_KEY:
    AI_PROVIDERS.append("openrouter")
if NVIDIA_API_KEY:
    AI_PROVIDERS.append("nvidia")

PENDING_EDITS = {}

# ── Wire in study system ──────────────────────────────────────────────────────
from study import (
    init_study, extract_knowledge,
    cmd_flashcard, cmd_mc, cmd_identify, cmd_tf,
    cmd_enum, cmd_search, cmd_answer,
    cmd_studystats, cmd_studydocs,
    QUIZ_STATE,
)
init_study(bot, ai_client)
# ─────────────────────────────────────────────────────────────────────────────

# ==========================================
# 2. DATABASE HELPERS
# ==========================================
def load_db() -> dict:
    if not os.path.exists(DATA_FILE):
        return {"notes": [], "schedules": [], "classes": {}, "recurring": [], "specials": []}
    with open(DATA_FILE, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {"notes": [], "schedules": [], "classes": {}, "recurring": [], "specials": []}
    notes = data.get("notes")
    if not isinstance(notes, list):
        notes = []
    data["notes"] = notes
    data.setdefault("schedules", [])
    data.setdefault("classes", {})
    data.setdefault("recurring", [])
    data.setdefault("specials", [])
    return data

def save_db(data: dict) -> None:
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_chat_schedules(db: dict, chat_id) -> list:
    return [s for s in db.get("schedules", []) if s.get("chat_id") == chat_id]

def get_chat_recurring(db: dict, chat_id) -> list:
    return [r for r in db.get("recurring", []) if r.get("chat_id") == chat_id]

def get_chat_specials(db: dict, chat_id) -> list:
    return [s for s in db.get("specials", []) if s.get("chat_id") == chat_id]

def get_chat_notes(db: dict, chat_id) -> list:
    return [n for n in db.get("notes", []) if n.get("chat_id") == chat_id]

# ==========================================
# 3. AI FALLBACK ENGINE
# ==========================================
def ask_ai(prompt: str) -> str:
    for provider in AI_PROVIDERS:
        try:
            if provider == "gemini":
                resp = ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Answer this shortly and concisely: {prompt}"
                )
                return resp.text or "🤖 No response from Gemini."

            elif provider == "openrouter":
                import urllib.request
                import json as _json
                payload = _json.dumps({
                    "model": "mistralai/mistral-7b-instruct:free",
                    "messages": [{"role": "user", "content": f"Answer this shortly and concisely: {prompt}"}],
                    "max_tokens": 500
                }).encode()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "https://github.com",
                        "X-Title": "CathBot"
                    }
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = _json.loads(r.read())
                return data["choices"][0]["message"]["content"]

            elif provider == "nvidia":
                import urllib.request
                import json as _json
                payload = _json.dumps({
                    "model": "meta/llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": f"Answer this shortly and concisely: {prompt}"}],
                    "max_tokens": 500
                }).encode()
                req = urllib.request.Request(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {NVIDIA_API_KEY}"
                    }
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = _json.loads(r.read())
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[AI] {provider} failed: {e}")
            continue

    return "🤖 All AI engines are offline. Try again later."

# ==========================================
# 4. INTENT DETECTION
# ==========================================
SCHEDULE_SIGNALS = [
    r'\bsched\b', r'\bschedule\b', r'\bremind\b', r'\balert\b',
    r'\btell\b', r'\bnotify\b', r'\bat\s+\d', r'\b\d+(am|pm)\b',
    r'\btoday\b', r'\btomorrow\b', r'\bnext\b', r'\btmr\b', r'\btmrw\b', r'\btom\b',
    r'\b\d{4}-\d{2}-\d{2}\b',
    r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',
    r'\b\d{1,2}:\d{2}\b',
    r'\b(mon|tue|wed|thu|fri|sat|sun)\b',
]

CONVERSATIONAL_PATTERNS = [
    r'^(hello|hi|hey|haha|hehe|lol|wow|omg|okay|ok|ay|uy|oo|yep|nope|sure|nice|aww|cute)\b',
    r'^(creating|reading|watching|sending|checking)\b',
    r'^(haha|hehe|lmao|lmfao|😂|😭|🤣)',
    r'(ang cute|so cute|charot|char|haha+)',
]

def is_real_schedule_command(text: str) -> bool:
    text_lower = text.lower().strip()
    has_signal = any(re.search(p, text_lower) for p in SCHEDULE_SIGNALS)
    if not has_signal:
        return False
    is_chat = any(re.search(p, text_lower) for p in CONVERSATIONAL_PATTERNS)
    if is_chat:
        return False
    if len(text_lower.split()) < 3:
        return False
    return True

# ==========================================
# 5. DUPLICATE DETECTION
# ==========================================
def is_duplicate(db: dict, task: str, date: str, time_str: str, chat_id) -> bool:
    for item in get_chat_schedules(db, chat_id):
        if (
            item["task"].lower() == task.lower()
            and item["date"] == date
            and item.get("time", "") == time_str
        ):
            return True
    return False

# ==========================================
# 6. EDIT DETECTION SIGNALS
# ==========================================
EDIT_SIGNALS = [
    r'\bchange\b', r'\bupdate\b', r'\bedit\b', r'\bmodify\b',
    r'\bmove\b', r'\breset\b', r'\breschedule\b', r'\bset\b',
]

def detect_edit_intent(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in EDIT_SIGNALS)

# ==========================================
# 7. WHAT IS BEING CHANGED?
# ==========================================
DATE_SIGNALS = [
    r'\bdate\b', r'\bday\b',
    r'\b\d{4}-\d{2}-\d{2}\b',
    r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',
    r'\bmonday\b', r'\btuesday\b', r'\bwednesday\b', r'\bthursday\b',
    r'\bfriday\b', r'\bsaturday\b', r'\bsunday\b',
    r'\b(mon|tue|wed|thu|fri|sat|sun)\b',
    r'\btomorrow\b', r'\btoday\b', r'\bnext\b',
    r'\btmr\b', r'\btmrw\b', r'\btom\b',
    r'\bjanuary\b', r'\bfebruary\b', r'\bmarch\b', r'\bapril\b',
    r'\bmay\b', r'\bjune\b', r'\bjuly\b', r'\baugust\b',
    r'\bseptember\b', r'\boctober\b', r'\bnovember\b', r'\bdecember\b',
]

TIME_SIGNALS = [
    r'\btime\b',
    r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',
    r'\b\d{1,2}\s*(?:am|pm)\b',
    r'\b\d{1,2}:\d{2}\b',
]

def detect_what_to_change(text: str) -> tuple:
    text_lower = text.lower()
    has_date_signal = any(re.search(p, text_lower) for p in DATE_SIGNALS)
    has_time_signal = any(re.search(p, text_lower) for p in TIME_SIGNALS)
    explicit_date = bool(re.search(r'\bdate\b|\bday\b', text_lower))
    explicit_time = bool(re.search(r'\btime\b', text_lower))
    if explicit_date and not explicit_time:
        return True, False
    if explicit_time and not explicit_date:
        return False, True
    if has_date_signal and not has_time_signal:
        return True, False
    if has_time_signal and not has_date_signal:
        return False, True
    return True, True

# ==========================================
# 8. EXTRACT TASK NAME FROM UPDATE COMMAND
# ==========================================
TASK_EXTRACTION_NOISE = {
    'change', 'update', 'edit', 'modify', 'move', 'reschedule', 'reset', 'set',
    'to', 'at', 'the', 'it', 'a', 'an', 'sched', 'schedule',
    'am', 'pm', 'today', 'tomorrow', 'tmr', 'tmrw', 'tom', 'this', 'now', 'in', 'on',
    'date', 'time', 'day', 'from', 'for', 'my', 'please',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
}

def extract_task_keywords(raw_text: str) -> list:
    text = raw_text.lower()
    text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}\s*(?:am|pm)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', '', text)
    text = re.sub(r'\b\d{1,2}-\d{1,2}-\d{4}\b', '', text)
    quoted = re.search(r'["\'](.+?)["\']', text)
    if quoted:
        return quoted.group(1).lower().split()
    words = re.split(r'\W+', text)
    return [w for w in words if w and w not in TASK_EXTRACTION_NOISE and len(w) > 2]

def find_best_matching_task(schedules: list, keywords: list):
    if not schedules:
        return None
    if not keywords:
        return schedules[-1]
    target = None
    best_score = 0
    for item in schedules:
        task_lower = item["task"].lower()
        score = sum(1 for w in keywords if w in task_lower)
        if score > best_score:
            best_score = score
            target = item
    return target if target else schedules[-1]

def find_matching_tasks(schedules: list, keywords: list) -> list:
    if not keywords:
        return list(schedules)
    matches = [s for s in schedules if any(w in s["task"].lower() for w in keywords)]
    return matches

# ==========================================
# 9. PATCH EDIT
# ==========================================
def try_patch_edit(db: dict, raw_text: str, now_ph: datetime, chat_id) -> tuple:
    chat_schedules = get_chat_schedules(db, chat_id)
    if not chat_schedules:
        return False, ""
    change_date, change_time = detect_what_to_change(raw_text)
    keywords = extract_task_keywords(raw_text)
    target = find_best_matching_task(chat_schedules, keywords)
    if target is None:
        return False, ""
    old_date = target["date"]
    old_time = target.get("time", "08:00 AM")
    old_task = target["task"]
    new_date = old_date
    new_time = old_time
    if change_date:
        parsed_date = parse_date_only(raw_text, now_ph)
        if parsed_date:
            new_date = parsed_date
    if change_time:
        parsed_time_val = parse_time_only(raw_text)
        if parsed_time_val:
            new_time = parsed_time_val
    if new_date == old_date and new_time == old_time:
        return False, ""
    target["date"] = new_date
    target["time"] = new_time
    changed_parts = []
    if new_date != old_date:
        changed_parts.append(f"Date: {old_date} -> {new_date}")
    if new_time != old_time:
        changed_parts.append(f"Time: {old_time} -> {new_time}")
    msg = (
        f"✏️ Schedule Updated! (PATCH)\n"
        f"• Task: {old_task}\n"
        f"• {chr(10).join(['• ' + c for c in changed_parts]).lstrip('• ')}\n"
        f"\n✅ Final State:\n"
        f"• Date: {new_date}\n"
        f"• Time: {new_time}"
    )
    return True, msg

# ==========================================
# 10. ISOLATED DATE PARSER
# ==========================================
WEEKDAY_ABBREV = {
    'mon': 'monday', 'tue': 'tuesday', 'tues': 'tuesday', 'wed': 'wednesday',
    'thu': 'thursday', 'thur': 'thursday', 'thurs': 'thursday', 'fri': 'friday',
    'sat': 'saturday', 'sun': 'sunday',
}

MONTH_NAME_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12,
}

def parse_date_only(raw_text: str, now_ph: datetime) -> str | None:
    text = fuzzy_correct(raw_text)
    now_naive = now_ph.replace(tzinfo=None)

    iso_match = re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', text)
    if iso_match:
        try:
            return datetime(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))).strftime("%Y-%m-%d")
        except ValueError:
            pass

    dash_match = re.search(r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', text)
    if dash_match:
        m, d, y = int(dash_match.group(1)), int(dash_match.group(2)), int(dash_match.group(3))
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    slash_match = re.search(r'\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b', text)
    if slash_match:
        m, d = int(slash_match.group(1)), int(slash_match.group(2))
        y = int(slash_match.group(3)) if slash_match.group(3) else now_naive.year
        if y < 100:
            y += 2000
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    stripped = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', '', text, flags=re.IGNORECASE)
    stripped = re.sub(r'\b\d{1,2}\s*(?:am|pm)\b', '', stripped, flags=re.IGNORECASE)

    for pat, offset in [(r'\btomorrow\b', 1), (r'\btoday\b', 0)]:
        if re.search(pat, stripped, re.IGNORECASE):
            return (now_naive + timedelta(days=offset)).strftime("%Y-%m-%d")

    weekday_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    weekday_terms = list(weekday_map.items())
    for abbr, full in WEEKDAY_ABBREV.items():
        weekday_terms.append((abbr, weekday_map[full]))
    for name, wday in weekday_terms:
        if re.search(rf'\b{name}\b', stripped, re.IGNORECASE):
            days_ahead = (wday - now_naive.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (now_naive + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    month_alt = '|'.join(MONTH_NAME_MAP.keys())
    month_day = re.search(
        r'\b(' + month_alt + r')\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b'
        r'|\b(\d{1,2})(?:st|nd|rd|th)?\s+(' + month_alt + r')(?:,?\s+(\d{4}))?\b',
        stripped, re.IGNORECASE
    )
    if month_day:
        groups = month_day.groups()
        if groups[0]:
            month, day, year_group = MONTH_NAME_MAP[groups[0].lower()], int(groups[1]), groups[2]
        else:
            month, day, year_group = MONTH_NAME_MAP[groups[4].lower()], int(groups[3]), groups[5]
        year = int(year_group) if year_group else now_naive.year
        try:
            candidate = datetime(year, month, day)
            if not year_group and candidate.date() < now_naive.date():
                candidate = datetime(year + 1, month, day)
            return candidate.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None

# ==========================================
# 11. ISOLATED TIME PARSER
# ==========================================
def parse_time_only(raw_text: str) -> str | None:
    text = raw_text.strip()
    m = re.search(r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b', text, re.IGNORECASE)
    if m:
        h, mn, meridiem = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        try:
            return datetime.strptime(f"{h}:{mn:02d} {meridiem}", "%I:%M %p").strftime("%I:%M %p")
        except ValueError:
            pass
    m = re.search(r'\b(\d{1,2})\s*(am|pm)\b', text, re.IGNORECASE)
    if m:
        h, meridiem = int(m.group(1)), m.group(2).upper()
        try:
            return datetime.strptime(f"{h}:00 {meridiem}", "%I:%M %p").strftime("%I:%M %p")
        except ValueError:
            pass
    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        meridiem = "PM" if 1 <= h <= 6 else "AM"
        try:
            return datetime.strptime(f"{h}:{mn:02d} {meridiem}", "%I:%M %p").strftime("%I:%M %p")
        except ValueError:
            pass
    return None

# ==========================================
# 12A. FUZZY SPELLING CORRECTOR
# ==========================================
SPELLING_FIXES = {
    r'\btommorow\b': 'tomorrow',
    r'\btommorrow\b': 'tomorrow',
    r'\btomorow\b': 'tomorrow',
    r'\btomoro\b': 'tomorrow',
    r'\btmr\b': 'tomorrow',
    r'\btmrw\b': 'tomorrow',
    r'\btom\b': 'tomorrow',
    r'\btodat\b': 'today',
    r'\btodey\b': 'today',
    r'(\d)(am|pm)\b': r'\1 \2',
    r'(\d\s*(?:am|pm))\s*,': r'\1',
    r'\bto\s+(\d+\s*(?:am|pm))\b': r'\1',
}

def fuzzy_correct(text: str) -> str:
    for pattern, replacement in SPELLING_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ==========================================
# 12B. FULL DATETIME PARSER
# ==========================================
def parse_schedule_time(raw_text: str, now_ph: datetime) -> tuple:
    corrected = fuzzy_correct(raw_text)
    now_naive = now_ph.replace(tzinfo=None)

    explicit_date_str = parse_date_only(raw_text, now_ph)
    explicit_time_str = parse_time_only(corrected)
    if explicit_date_str:
        try:
            base_date = datetime.strptime(explicit_date_str, "%Y-%m-%d")
        except ValueError:
            base_date = None
        if base_date:
            if explicit_time_str:
                t_obj = datetime.strptime(explicit_time_str, "%I:%M %p")
                final_dt = datetime(base_date.year, base_date.month, base_date.day, t_obj.hour, t_obj.minute)
            else:
                final_dt = datetime(base_date.year, base_date.month, base_date.day, 8, 0)
            return final_dt, ""

    time_re = re.compile(
        r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b'
        r'|\b(\d{1,2})\s*(am|pm)\b'
        r'|\b(\d{1,2}):(\d{2})\b',
        re.IGNORECASE
    )
    relative_patterns = [
        (r'\btomorrow\b', 1), (r'\btoday\b', 0), (r'\bthis\b', 0),
    ]
    day_offset = None
    date_word_matched = ""
    for pat, offset in relative_patterns:
        m = re.search(pat, corrected, re.IGNORECASE)
        if m:
            day_offset = offset
            date_word_matched = m.group(0)
            break
    time_match = time_re.search(corrected)
    if day_offset is not None and time_match:
        target_date = (now_naive + timedelta(days=day_offset)).date()
        parsed_t = dateparser.parse(
            time_match.group(0).strip(),
            settings={'PREFER_DATES_FROM': 'current_period',
                      'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': False}
        )
        if parsed_t:
            final_dt = datetime(target_date.year, target_date.month, target_date.day,
                                parsed_t.hour, parsed_t.minute)
            return final_dt, f"{date_word_matched} {time_match.group(0)}"
    if time_match and day_offset is None:
        parsed_t = dateparser.parse(
            time_match.group(0).strip(),
            settings={'PREFER_DATES_FROM': 'current_period',
                      'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': False}
        )
        if parsed_t:
            candidate = datetime(now_naive.year, now_naive.month, now_naive.day,
                                 parsed_t.hour, parsed_t.minute)
            if candidate <= now_naive:
                candidate += timedelta(days=1)
            return candidate, time_match.group(0)
    prefer_today = bool(re.search(
        r'\btoday\b|\bthis\s+\d|\bthis\s+(morning|afternoon|evening|night|'
        r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d)',
        corrected, re.IGNORECASE
    ))
    found_dates = search_dates(
        corrected,
        settings={
            'RELATIVE_BASE': now_naive,
            'PREFER_DATES_FROM': 'current_period' if prefer_today else 'future',
            'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': False,
        }
    )
    if not found_dates:
        return None, ""
    best = found_dates[-1]
    for ms, md in found_dates:
        if md.hour != 0 or md.minute != 0:
            best = (ms, md)
            break
    matched_str, parsed_time_val = best
    return parsed_time_val, matched_str

# ==========================================
# 13. TEXT CLEANER
# ==========================================
def clean_task_text(raw_text: str, matched_date_str: str) -> str:
    task = raw_text.replace(matched_date_str, "") if matched_date_str else raw_text
    month_alt = '|'.join(MONTH_NAME_MAP.keys())
    noise_patterns = [
        r'\btoday\b', r'\btomorrow\b', r'\bthis\b', r'\bnow\b',
        r'\btommorow\b', r'\btommorrow\b', r'\btomorow\b', r'\btomoro\b',
        r'\btmr\b', r'\btmrw\b', r'\btom\b',
        r'\bat\b', r'\bin\b', r'\bon\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',
        r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
        r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b',
        r'\b\d{1,2}\s*(?:am|pm)\b',
        r'\b\d{1,2}:\d{2}\b',
        r'\b(' + month_alt + r')\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(' + month_alt + r')(?:,?\s+\d{4})?\b',
        r'\bmonday\b', r'\btuesday\b', r'\bwednesday\b', r'\bthursday\b',
        r'\bfriday\b', r'\bsaturday\b', r'\bsunday\b',
        r'\bmon\b', r'\btue\b', r'\btues\b', r'\bwed\b', r'\bthu\b', r'\bthur\b', r'\bthurs\b', r'\bfri\b', r'\bsat\b', r'\bsun\b',
        r'\bpm\b', r'\bam\b', r'\bsched\b', r'\bschedule\b',
        r'\bcreate sched to\b', r'\bchange it to\b', r'\bremind me to\b',
        r'\btell\b', r'\bchange\b', r'\bupdate\b', r',',
    ]
    for pattern in noise_patterns:
        task = re.sub(pattern, "", task, flags=re.IGNORECASE)
    task = re.sub(r'\s+', ' ', task).strip(" .,")
    return (task.capitalize() if task else raw_text.capitalize()) or ""

# ==========================================
# 14. RECURRING TASK HELPERS
# ==========================================
RECUR_TYPES = {
    'daily':   r'\beveryday\b|\bdaily\b|\bevery\s+day\b|\beach\b',
    'weekly':  r'\bevery\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b|\bweekly\b',
    'monthly': r'\bevery\s+month\b|\bmonthly\b|\bevery\s+\d{1,2}(?:st|nd|rd|th)\b',
    'yearly':  r'\bevery\s+year\b|\bannually\b|\byearly\b',
}

def detect_recur_type(text: str) -> str | None:
    t = text.lower()
    t = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\b\d{1,2}\s*(?:am|pm)\b', '', t, flags=re.IGNORECASE)
    for rtype, pat in RECUR_TYPES.items():
        if re.search(pat, t):
            return rtype
    return None

def detect_recur_weekday(text: str) -> str | None:
    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    t = text.lower()
    for d in days:
        if re.search(rf'\b{d}\b', t):
            return d.capitalize()
    for abbr, full in WEEKDAY_ABBREV.items():
        if re.search(rf'\b{abbr}\b', t):
            return full.capitalize()
    return None

def detect_recur_monthday(text: str) -> int | None:
    m = re.search(r'\bevery\s+(\d{1,2})(?:st|nd|rd|th)?\b', text.lower())
    if m:
        return int(m.group(1))
    return None

def next_recur_date(item: dict, now_ph: datetime) -> datetime:
    now_naive = now_ph.replace(tzinfo=None)
    t_str = item.get("time", "08:00 AM")
    try:
        t_obj = datetime.strptime(t_str, "%I:%M %p").time()
    except ValueError:
        t_obj = datetime.strptime("08:00 AM", "%I:%M %p").time()
    rtype = item["recur_type"]
    if rtype == "daily":
        candidate = datetime.combine(now_naive.date(), t_obj)
        if candidate <= now_naive:
            candidate += timedelta(days=1)
        return candidate
    if rtype == "weekly":
        day_name = item.get("recur_day", "Monday")
        target_wday = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(day_name)
        days_ahead = (target_wday - now_naive.weekday() + 7) % 7
        if days_ahead == 0:
            candidate = datetime.combine(now_naive.date(), t_obj)
            if candidate <= now_naive:
                days_ahead = 7
            else:
                return candidate
        return datetime.combine((now_naive + timedelta(days=days_ahead)).date(), t_obj)
    if rtype == "monthly":
        day_of_month = item.get("recur_monthday", 1)
        try:
            candidate = datetime(now_naive.year, now_naive.month, day_of_month, t_obj.hour, t_obj.minute)
        except ValueError:
            candidate = None
        if candidate and candidate > now_naive:
            return candidate
        if now_naive.month == 12:
            return datetime(now_naive.year + 1, 1, day_of_month, t_obj.hour, t_obj.minute)
        return datetime(now_naive.year, now_naive.month + 1, day_of_month, t_obj.hour, t_obj.minute)
    if rtype == "yearly":
        month = item.get("recur_month", now_naive.month)
        day   = item.get("recur_day_num", now_naive.day)
        try:
            candidate = datetime(now_naive.year, month, day, t_obj.hour, t_obj.minute)
        except ValueError:
            candidate = None
        if candidate and candidate > now_naive:
            return candidate
        return datetime(now_naive.year + 1, month, day, t_obj.hour, t_obj.minute)
    return now_naive + timedelta(days=1)

# ==========================================
# 15. SPECIAL EVENT HELPERS
# ==========================================
SPECIAL_TYPES = {
    'birthday':    r'\bbirthday\b|\bbday\b',
    'anniversary': r'\banniversary\b|\banniv\b',
    'monthsary':   r'\bmonthsary\b|\bmonthiversary\b',
}

def detect_special_type(text: str) -> str | None:
    t = text.lower()
    for stype, pat in SPECIAL_TYPES.items():
        if re.search(pat, t):
            return stype
    return None

def next_special_date(item: dict, now_ph: datetime) -> datetime:
    now_naive = now_ph.replace(tzinfo=None)
    t_str = item.get("time", "08:00 AM")
    try:
        t_obj = datetime.strptime(t_str, "%I:%M %p").time()
    except ValueError:
        t_obj = datetime.strptime("08:00 AM", "%I:%M %p").time()
    if item["special_type"] == "monthsary":
        day_of_month = item.get("day_of_month", 1)
        try:
            candidate = datetime(now_naive.year, now_naive.month, day_of_month, t_obj.hour, t_obj.minute)
        except ValueError:
            candidate = None
        if candidate and candidate > now_naive:
            return candidate
        if now_naive.month == 12:
            return datetime(now_naive.year + 1, 1, day_of_month, t_obj.hour, t_obj.minute)
        return datetime(now_naive.year, now_naive.month + 1, day_of_month, t_obj.hour, t_obj.minute)
    month = item.get("month", now_naive.month)
    day   = item.get("day", now_naive.day)
    try:
        candidate = datetime(now_naive.year, month, day, t_obj.hour, t_obj.minute)
    except ValueError:
        candidate = None
    if candidate and candidate > now_naive:
        return candidate
    return datetime(now_naive.year + 1, month, day, t_obj.hour, t_obj.minute)

# ==========================================
# 16. MAIN SCHEDULE HANDLER
# ==========================================
@bot.message_handler(func=lambda m: m.text and any(l.strip().startswith("-") for l in m.text.split("\n")))
def handle_cath_schedule(message):
    lines = [l.strip() for l in message.text.split("\n") if l.strip().startswith("-")]
    if not lines:
        return
    for line in lines:
        raw_text = line[1:].replace("@", "").strip()
        if raw_text:
            process_schedule_line(message, raw_text)

def process_schedule_line(message, raw_text: str):
    recur_type = detect_recur_type(raw_text)
    if recur_type:
        handle_recurring_create(message, raw_text, recur_type)
        return
    special_type = detect_special_type(raw_text)
    if special_type:
        handle_special_create(message, raw_text, special_type)
        return
    if not is_real_schedule_command(raw_text):
        return
    db = load_db()
    chat_id = message.chat.id
    now_ph = datetime.now(PH_TZ)
    is_edit = detect_edit_intent(raw_text)
    if is_edit:
        success, edit_msg = try_patch_edit(db, raw_text, now_ph, chat_id)
        if success:
            save_db(db)
            bot.reply_to(message, edit_msg)
            return
        else:
            bot.reply_to(
                message,
                "⚠️ Couldn't find a matching task to update.\n"
                "Tip: be more specific, e.g.:\n"
                "-update \"biking\" -> date: 2026-06-26\n"
                "-update \"biking\" -> time: 7:00 PM\n"
                "Or use /active to see your task names."
            )
            return
    parsed_time_val, matched_str = parse_schedule_time(raw_text, now_ph)
    if parsed_time_val:
        if parsed_time_val < now_ph.replace(tzinfo=None):
            bot.reply_to(
                message,
                f"⚠️ That time ({parsed_time_val.strftime('%Y-%m-%d %I:%M %p')}) is already in the past.\n"
                f"Did you mean a future date? Use /active to see current schedules."
            )
            return
        event_date = parsed_time_val.strftime("%Y-%m-%d")
        event_time = parsed_time_val.strftime("%I:%M %p")
        clean_task = clean_task_text(raw_text, matched_str)
        if is_duplicate(db, clean_task, event_date, event_time, chat_id):
            bot.reply_to(message, f"⚠️ Already saved: {clean_task} on {event_date} at {event_time}.")
            return
        db["schedules"].append({
            "id": int(time.time() * 1000),
            "date": event_date,
            "time": event_time,
            "task": clean_task,
            "chat_id": chat_id,
            "notified": False,
        })
        save_db(db)
        bot.reply_to(
            message,
            f"📅 Schedule Saved!\n"
            f"• Date: {event_date}\n"
            f"• Time: {event_time}\n"
            f"• Task: {clean_task}"
        )
    else:
        bot.reply_to(
            message,
            "📅 I couldn't detect a date or time.\n"
            "Try: -sched tomorrow 8pm tell mico about bot"
        )

# ==========================================
# 17. CREATE RECURRING TASK
# ==========================================
def handle_recurring_create(message, raw_text: str, recur_type: str):
    db = load_db()
    now_ph = datetime.now(PH_TZ)
    t_str = parse_time_only(raw_text) or "08:00 AM"
    task_text = raw_text
    task_text = re.sub(r'\beveryday\b|\bdaily\b|\bevery\s+day\b|\beach\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\bevery\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\bevery\s+month\b|\bmonthly\b|\bevery\s+\d{1,2}(?:st|nd|rd|th)\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\bevery\s+year\b|\bannually\b|\byearly\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\bweekly\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\bremind\s+me\s+to\b|\bremind\s+to\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\b\d{1,2}\s*(?:am|pm)\b', '', task_text, flags=re.IGNORECASE)
    task_text = re.sub(r'\s+', ' ', task_text).strip(" .,")
    task_label = task_text.capitalize() or "Recurring Task"
    item: dict = {
        "id": int(time.time() * 1000),
        "task": task_label,
        "time": t_str,
        "recur_type": recur_type,
        "chat_id": message.chat.id,
        "last_notified": None,
    }
    if recur_type == "weekly":
        day = detect_recur_weekday(raw_text)
        item["recur_day"] = day or "Monday"
    if recur_type == "monthly":
        md = detect_recur_monthday(raw_text)
        item["recur_monthday"] = md if md is not None else now_ph.day
    if recur_type == "yearly":
        item["recur_month"] = now_ph.month
        item["recur_day_num"] = now_ph.day
    db["recurring"].append(item)
    save_db(db)
    if recur_type == "daily":
        sched_desc = f"Every day at {t_str}"
    elif recur_type == "weekly":
        sched_desc = f"Every {item.get('recur_day')} at {t_str}"
    elif recur_type == "monthly":
        sched_desc = f"Every month on the {item.get('recur_monthday')} at {t_str}"
    else:
        sched_desc = f"Every year at {t_str}"
    bot.reply_to(
        message,
        f"🔁 Recurring Task Saved!\n"
        f"• Task: {task_label}\n"
        f"• Schedule: {sched_desc}\n"
        f"• ID: {item['id']}\n\n"
        f"Use /recurring to see all. /delrecurring id:{item['id']} to remove."
    )

# ==========================================
# 18. CREATE SPECIAL EVENT
# ==========================================
def handle_special_create(message, raw_text: str, special_type: str):
    db = load_db()
    now_ph = datetime.now(PH_TZ)
    t_str = parse_time_only(raw_text) or "08:00 AM"
    parsed_date_str = parse_date_only(raw_text, now_ph)
    label_text = raw_text
    for pat in SPECIAL_TYPES.values():
        label_text = re.sub(pat, '', label_text, flags=re.IGNORECASE)
    label_text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)\b', '', label_text, flags=re.IGNORECASE)
    label_text = re.sub(r'\b\d{1,2}\s*(?:am|pm)\b', '', label_text, flags=re.IGNORECASE)
    label_text = re.sub(r'\b\d{4}-\d{1,2}-\d{1,2}\b', '', label_text)
    label_text = re.sub(r'\b\d{1,2}-\d{1,2}-\d{4}\b', '', label_text)
    label_text = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', '', label_text)
    month_names = r'\b(' + '|'.join(MONTH_NAME_MAP.keys()) + r')\b'
    label_text = re.sub(month_names + r'\s*\d{0,2}', '', label_text, flags=re.IGNORECASE)
    label_text = re.sub(r'\b\d{1,2}\b', '', label_text)
    label_text = re.sub(r'\s+', ' ', label_text).strip(" .,")
    label = label_text.capitalize() or special_type.capitalize()
    item: dict = {
        "id": int(time.time() * 1000),
        "label": label,
        "special_type": special_type,
        "time": t_str,
        "chat_id": message.chat.id,
        "last_notified": None,
    }
    if parsed_date_str:
        dt = datetime.strptime(parsed_date_str, "%Y-%m-%d")
        item["month"] = dt.month
        item["day"] = dt.day
        item["day_of_month"] = dt.day
        date_display = dt.strftime("%B %d")
    else:
        item["month"] = now_ph.month
        item["day"] = now_ph.day
        item["day_of_month"] = now_ph.day
        date_display = now_ph.strftime("%B %d")
    db["specials"].append(item)
    save_db(db)
    type_labels = {'birthday': 'Birthday', 'anniversary': 'Anniversary', 'monthsary': 'Monthsary'}
    recur_note = "every month" if special_type == "monthsary" else "every year"
    bot.reply_to(
        message,
        f"🎉 {type_labels[special_type]} Saved!\n"
        f"• For: {label}\n"
        f"• Date: {date_display} ({recur_note})\n"
        f"• Reminder at: {t_str}\n"
        f"• ID: {item['id']}\n\n"
        f"Use /specials to see all. /delspecial id:{item['id']} to remove."
    )

# ==========================================
# 19. /active
# ==========================================
@bot.message_handler(commands=['active'])
def show_active_tasks(message):
    db = load_db()
    chat_id = message.chat.id
    now_ph = datetime.now(PH_TZ).replace(tzinfo=None)
    cutoff = now_ph + timedelta(days=90)
    active_list = []
    for item in get_chat_schedules(db, chat_id):
        try:
            item_dt = datetime.strptime(f"{item['date']} {item.get('time', '12:00 AM')}", "%Y-%m-%d %I:%M %p")
            if now_ph <= item_dt <= cutoff:
                active_list.append(
                    f"• [ID:{item.get('id','?')}] [{item['date']} at {item.get('time','Anytime')}] {item['task']}"
                )
        except ValueError:
            continue
    if active_list:
        bot.reply_to(message, "⏳ Active Schedule (Next 3 Months)\n\n" + "\n".join(active_list))
    else:
        bot.reply_to(message, "✅ No upcoming schedules. You're free!")

# ==========================================
# 20. /recurring
# ==========================================
@bot.message_handler(commands=['recurring'])
def show_recurring(message):
    db = load_db()
    items = get_chat_recurring(db, message.chat.id)
    if not items:
        bot.reply_to(message, "🔁 No recurring tasks saved.\nTry: -remind me to take meds 8pm everyday")
        return
    lines = []
    for r in items:
        if r["recur_type"] == "daily":
            sched = f"Every day at {r.get('time','08:00 AM')}"
        elif r["recur_type"] == "weekly":
            sched = f"Every {r.get('recur_day','?')} at {r.get('time','08:00 AM')}"
        elif r["recur_type"] == "monthly":
            sched = f"Every month, day {r.get('recur_monthday','?')} at {r.get('time','08:00 AM')}"
        else:
            sched = f"Every year at {r.get('time','08:00 AM')}"
        lines.append(f"• [ID:{r.get('id','?')}] {r['task']} — {sched}")
    bot.reply_to(message, "🔁 Recurring Tasks\n\n" + "\n".join(lines))

# ==========================================
# 21. /editrecurring
# ==========================================
@bot.message_handler(commands=['editrecurring'])
def edit_recurring(message):
    query = message.text.replace("/editrecurring", "").strip()
    id_match = re.search(r'id:(\d+)', query, re.IGNORECASE)
    if not id_match:
        bot.reply_to(message, "⚠️ Usage: /editrecurring id:123456 time: 9:00 PM\nor /editrecurring id:123456 task: New name")
        return
    target_id = int(id_match.group(1))
    db = load_db()
    chat_id = message.chat.id
    target = next((r for r in db["recurring"] if r.get("id") == target_id and r.get("chat_id") == chat_id), None)
    if not target:
        bot.reply_to(message, f"🚫 No recurring task with ID {target_id} in this chat. Check /recurring.")
        return
    new_time = parse_time_only(query)
    if new_time:
        target["time"] = new_time
    task_match = re.search(r'task:\s*(.+)', query, re.IGNORECASE)
    if task_match:
        target["task"] = task_match.group(1).strip().capitalize()
    save_db(db)
    bot.reply_to(
        message,
        f"✏️ Recurring Task Updated!\n"
        f"• ID: {target_id}\n"
        f"• Task: {target['task']}\n"
        f"• Time: {target.get('time','08:00 AM')}\n"
        f"• Schedule: {target['recur_type']}"
    )

# ==========================================
# 22. /delrecurring
# ==========================================
@bot.message_handler(commands=['delrecurring'])
def delete_recurring(message):
    query = message.text.replace("/delrecurring", "").strip()
    db = load_db()
    chat_id = message.chat.id
    chat_recurring_ids = {r.get("id") for r in get_chat_recurring(db, chat_id)}
    original = len(chat_recurring_ids)
    if query.lower() == "all":
        db["recurring"] = [r for r in db.get("recurring", []) if r.get("chat_id") != chat_id]
        save_db(db)
        bot.reply_to(message, f"🗑️ Cleared all {original} recurring task(s) for this chat.")
        return
    id_match = re.search(r'id:(\d+)', query, re.IGNORECASE)
    if id_match:
        target_id = int(id_match.group(1))
        before = len(db["recurring"])
        db["recurring"] = [r for r in db["recurring"] if not (r.get("id") == target_id and r.get("chat_id") == chat_id)]
        deleted = before - len(db["recurring"])
        save_db(db)
        if deleted:
            bot.reply_to(message, f"🗑️ Deleted recurring task ID {target_id}.")
        else:
            bot.reply_to(message, f"🚫 No recurring task with ID {target_id} in this chat.")
        return
    before = len(db.get("recurring", []))
    db["recurring"] = [r for r in db.get("recurring", []) if not (r.get("chat_id") == chat_id and query.lower() in r["task"].lower())]
    deleted = before - len(db["recurring"])
    if deleted:
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted {deleted} recurring task(s) matching '{query}'.")
    else:
        bot.reply_to(message, f"🚫 No recurring tasks matching '{query}'.")

# ==========================================
# 23. /specials
# ==========================================
@bot.message_handler(commands=['specials'])
def show_specials(message):
    db = load_db()
    items = get_chat_specials(db, message.chat.id)
    if not items:
        bot.reply_to(message, "🎉 No special events saved.\nTry: -birthday mico june 26 8am")
        return
    type_emoji = {'birthday': '🎂', 'anniversary': '💍', 'monthsary': '💕'}
    lines = []
    for s in items:
        emoji = type_emoji.get(s["special_type"], "🎉")
        if s["special_type"] == "monthsary":
            date_desc = f"Every month, day {s.get('day_of_month','?')}"
        else:
            month_name = datetime(2000, s.get("month", 1), 1).strftime("%B")
            date_desc = f"Every year on {month_name} {s.get('day','?')}"
        lines.append(f"{emoji} [ID:{s.get('id','?')}] {s['label']} ({s['special_type']}) — {date_desc} at {s.get('time','08:00 AM')}")
    bot.reply_to(message, "🎉 Special Events\n\n" + "\n".join(lines))

# ==========================================
# 24. /editspecial
# ==========================================
@bot.message_handler(commands=['editspecial'])
def edit_special(message):
    query = message.text.replace("/editspecial", "").strip()
    id_match = re.search(r'id:(\d+)', query, re.IGNORECASE)
    if not id_match:
        bot.reply_to(message, "⚠️ Usage: /editspecial id:123456 time: 9:00 AM\nor /editspecial id:123456 label: New Name")
        return
    target_id = int(id_match.group(1))
    db = load_db()
    chat_id = message.chat.id
    target = next((s for s in db["specials"] if s.get("id") == target_id and s.get("chat_id") == chat_id), None)
    if not target:
        bot.reply_to(message, f"🚫 No special event with ID {target_id} in this chat. Check /specials.")
        return
    new_time = parse_time_only(query)
    if new_time:
        target["time"] = new_time
    label_match = re.search(r'label:\s*(.+)', query, re.IGNORECASE)
    if label_match:
        target["label"] = label_match.group(1).strip().capitalize()
    save_db(db)
    bot.reply_to(
        message,
        f"✏️ Special Event Updated!\n"
        f"• ID: {target_id}\n"
        f"• Label: {target['label']}\n"
        f"• Type: {target['special_type']}\n"
        f"• Time: {target.get('time','08:00 AM')}"
    )

# ==========================================
# 25. /delspecial
# ==========================================
@bot.message_handler(commands=['delspecial'])
def delete_special(message):
    query = message.text.replace("/delspecial", "").strip()
    db = load_db()
    chat_id = message.chat.id
    original = len(get_chat_specials(db, chat_id))
    if query.lower() == "all":
        db["specials"] = [s for s in db.get("specials", []) if s.get("chat_id") != chat_id]
        save_db(db)
        bot.reply_to(message, f"🗑️ Cleared all {original} special event(s) for this chat.")
        return
    id_match = re.search(r'id:(\d+)', query, re.IGNORECASE)
    if id_match:
        target_id = int(id_match.group(1))
        before = len(db["specials"])
        db["specials"] = [s for s in db["specials"] if not (s.get("id") == target_id and s.get("chat_id") == chat_id)]
        deleted = before - len(db["specials"])
        save_db(db)
        if deleted:
            bot.reply_to(message, f"🗑️ Deleted special event ID {target_id}.")
        else:
            bot.reply_to(message, f"🚫 No special event with ID {target_id} in this chat.")
        return
    before = len(db.get("specials", []))
    db["specials"] = [s for s in db.get("specials", []) if not (s.get("chat_id") == chat_id and query.lower() in s["label"].lower())]
    deleted = before - len(db["specials"])
    if deleted:
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted {deleted} special event(s) matching '{query}'.")
    else:
        bot.reply_to(message, f"🚫 No special events matching '{query}'.")

# ==========================================
# 26. /delete
# ==========================================
@bot.message_handler(commands=['delete'])
def delete_task(message):
    query = message.text.replace("/delete", "").strip()
    if not query:
        bot.reply_to(
            message,
            "⚠️ Provide a keyword or ID.\n"
            "Examples:\n"
            "• /delete tell mico — by keyword\n"
            "• /delete id:1234567890 — by ID\n"
            "• /delete all — wipe everything"
        )
        return
    db = load_db()
    chat_id = message.chat.id
    original_count = len(get_chat_schedules(db, chat_id))
    if query.lower() == "all":
        db["schedules"] = [s for s in db.get("schedules", []) if s.get("chat_id") != chat_id]
        save_db(db)
        bot.reply_to(message, f"🗑️ Cleared all {original_count} scheduled task(s) for this chat.")
        return
    id_match = re.match(r'id:(\d+)', query, re.IGNORECASE)
    if id_match:
        target_id = int(id_match.group(1))
        before = len(db["schedules"])
        db["schedules"] = [s for s in db["schedules"] if not (s.get("id") == target_id and s.get("chat_id") == chat_id)]
        deleted = before - len(db["schedules"])
        save_db(db)
        if deleted:
            bot.reply_to(message, f"🗑️ Deleted task with ID {target_id}.")
        else:
            bot.reply_to(message, f"🚫 No task found with ID {target_id} in this chat. Check /active.")
        return
    before = len(db.get("schedules", []))
    db["schedules"] = [s for s in db.get("schedules", []) if not (s.get("chat_id") == chat_id and query.lower() in s["task"].lower())]
    deleted = before - len(db["schedules"])
    if deleted:
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted {deleted} task(s) matching '{query}'.")
    else:
        bot.reply_to(message, f"🚫 No tasks found matching '{query}'. Check /active.")

# ==========================================
# 26B. /edit
# ==========================================
def apply_schedule_edit(item: dict, new_date: str | None, new_time: str | None) -> None:
    if new_date:
        item["date"] = new_date
    if new_time:
        item["time"] = new_time

def format_edit_confirmation(item: dict) -> str:
    return (
        f"✏️ Schedule Updated!\n"
        f"• Task: {item['task']}\n"
        f"• Date: {item['date']}\n"
        f"• Time: {item.get('time','08:00 AM')}"
    )

@bot.message_handler(commands=['edit'])
def edit_by_task(message):
    query = message.text.replace("/edit", "", 1).strip()
    if not query:
        bot.reply_to(
            message,
            "⚠️ Usage: /edit [task keyword] [new date/time]\n"
            "Examples:\n"
            "/edit run tom 9am\n"
            "/edit enrollment 06-30-2026 10am"
        )
        return
    db = load_db()
    chat_id = message.chat.id
    chat_schedules = get_chat_schedules(db, chat_id)
    if not chat_schedules:
        bot.reply_to(message, "🚫 No schedules found for this chat. Use /active to check.")
        return
    now_ph = datetime.now(PH_TZ)
    new_date = parse_date_only(query, now_ph)
    new_time = parse_time_only(query)
    keywords = extract_task_keywords(query)
    candidates = find_matching_tasks(chat_schedules, keywords)
    if not candidates:
        bot.reply_to(message, f"🚫 No tasks matching '{query}'. Use /active to see task names.")
        return
    if not new_date and not new_time:
        bot.reply_to(message, "⚠️ I couldn't detect a new date or time in your request. Include one, e.g. '/edit run tom 9am'.")
        return
    if len(candidates) == 1:
        apply_schedule_edit(candidates[0], new_date, new_time)
        save_db(db)
        bot.reply_to(message, format_edit_confirmation(candidates[0]))
        return
    lines = [f"{i+1}. {c['task']} {c.get('time','')}".strip() for i, c in enumerate(candidates)]
    PENDING_EDITS[chat_id] = {
        "candidates": [c["id"] for c in candidates],
        "new_date": new_date,
        "new_time": new_time,
    }
    bot.reply_to(message, "⚠️ Multiple matches found. Reply with the number:\n\n" + "\n".join(lines))

@bot.message_handler(func=lambda m: m.text and m.text.strip().isdigit() and m.chat.id in PENDING_EDITS)
def handle_edit_selection(message):
    chat_id = message.chat.id
    pending = PENDING_EDITS.pop(chat_id, None)
    if not pending:
        return
    choice = int(message.text.strip()) - 1
    if choice < 0 or choice >= len(pending["candidates"]):
        bot.reply_to(message, "🚫 Invalid selection.")
        return
    target_id = pending["candidates"][choice]
    db = load_db()
    target = next((s for s in db["schedules"] if s.get("id") == target_id and s.get("chat_id") == chat_id), None)
    if not target:
        bot.reply_to(message, "🚫 That task no longer exists.")
        return
    apply_schedule_edit(target, pending["new_date"], pending["new_time"])
    save_db(db)
    bot.reply_to(message, format_edit_confirmation(target))

# ==========================================
# 26C. /parse
# ==========================================
@bot.message_handler(commands=['parse'])
def parse_test(message):
    raw_text = message.text.replace("/parse", "", 1).strip()
    if not raw_text:
        bot.reply_to(message, "⚠️ Usage: /parse [text]\nExample: /parse enrollment 06-30-2026 8am")
        return
    now_ph = datetime.now(PH_TZ)
    parsed_dt, matched_str = parse_schedule_time(raw_text, now_ph)
    if not parsed_dt:
        bot.reply_to(message, "🔍 No date/time detected.")
        return
    task = clean_task_text(raw_text, matched_str)
    bot.reply_to(
        message,
        f"Date: {parsed_dt.strftime('%Y-%m-%d')}\n"
        f"Time: {parsed_dt.strftime('%I:%M %p')}\n"
        f"Task: {task}"
    )

# ==========================================
# UPLOAD STATE
# ==========================================
pending_uploads = {}
pending_lock = threading.Lock()
PENDING_TIMEOUT = 60

def set_pending(chat_id, title):
    with pending_lock:
        pending_uploads[chat_id] = {
            "title": title,
            "collected": [],
            "expires": time.time() + PENDING_TIMEOUT
        }

def get_pending(chat_id):
    with pending_lock:
        p = pending_uploads.get(chat_id)
        if p and time.time() < p["expires"]:
            return p
        if p:
            del pending_uploads[chat_id]
        return None

def clear_pending(chat_id):
    with pending_lock:
        pending_uploads.pop(chat_id, None)

# ==========================================
# 27. /upload
# ==========================================
@bot.message_handler(commands=['upload'])
def handle_upload_command(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "⚠️ Use: /upload [title]\nExample: /upload math")
        return

    title = args[1].lower().strip()
    set_pending(message.chat.id, title)
    bot.reply_to(
        message,
        f"📎 Ready to save under *'{title}'*.\n"
        f"Send your photo(s) now, then /donesaving when finished.",
        parse_mode="Markdown"
    )

# ==========================================
# 27A. Photo handler
# ==========================================
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    pending = get_pending(chat_id)
    if not pending:
        return

    title = pending["title"]
    db = load_db()
    record = {
        "id": int(time.time() * 1000),
        "title": title,
        "chat_id": chat_id,
        "type": "photo",
        "file_id": message.photo[-1].file_id,
    }
    db["notes"].append(record)
    save_db(db)

    with pending_lock:
        if chat_id in pending_uploads:
            pending_uploads[chat_id]["collected"].append(record["id"])
            pending_uploads[chat_id]["expires"] = time.time() + PENDING_TIMEOUT

    count = len(pending_uploads.get(chat_id, {}).get("collected", []))
    bot.reply_to(
        message,
        f"✅ Photo {count} saved under *'{title}'*. Send more or /donesaving to finish.",
        parse_mode="Markdown"
    )

# ==========================================
# 27C. Text handler
# ==========================================
@bot.message_handler(content_types=['text'])
def handle_text(message):
    chat_id = message.chat.id

    # Don't treat commands as text notes — let their own handlers process them
    if message.text.startswith('/'):
        return

    pending = get_pending(chat_id)
    if not pending:
        return

    title = pending["title"]
    db = load_db()
    record = {
        "id": int(time.time() * 1000),
        "title": title,
        "chat_id": chat_id,
        "type": "text",
        "content": message.text,
    }
    db["notes"].append(record)
    save_db(db)

    with pending_lock:
        if chat_id in pending_uploads:
            pending_uploads[chat_id]["collected"].append(record["id"])
            pending_uploads[chat_id]["expires"] = time.time() + PENDING_TIMEOUT

    count = len(pending_uploads.get(chat_id, {}).get("collected", []))
    bot.reply_to(
        message,
        f"✅ Text note {count} saved under *'{title}'*. Send more or /donesaving to finish.",
        parse_mode="Markdown"
    )
# ==========================================
# 27B. /donesaving — now also triggers knowledge extraction
# ==========================================
@bot.message_handler(commands=['donesaving'])
def handle_done_saving(message):
    chat_id = message.chat.id
    pending = get_pending(chat_id)
    if not pending:
        bot.reply_to(message, "⚠️ No active upload session. Start one with /upload [title]")
        return

    count = len(pending["collected"])
    title = pending["title"]
    clear_pending(chat_id)
    bot.reply_to(
        message,
        f"✅ Done! Saved {count} photo(s) under *'{title}'*.\n"
        f"Type *show me {title}* anytime to view them.\n"
        f"⏳ Extracting knowledge in the background…",
        parse_mode="Markdown"
    )
    # Run OCR + extraction without blocking the bot
    threading.Thread(target=extract_knowledge, args=(title, chat_id), daemon=True).start()

# ==========================================
# 27C. /deleteupload
# ==========================================
@bot.message_handler(commands=['deleteupload'])
def delete_upload(message):
    query = message.text.replace("/deleteupload", "", 1).strip()
    if not query:
        bot.reply_to(
            message,
            "⚠️ Provide a keyword or ID.\n"
            "• /deleteupload math — by title\n"
            "• /deleteupload id:1234567890 — by ID\n"
            "• /deleteupload all — wipe all in this chat"
        )
        return

    db = load_db()
    chat_id = message.chat.id

    if query.lower() == "all":
        count = sum(1 for n in db.get("notes", []) if n.get("chat_id") == chat_id)
        db["notes"] = [n for n in db.get("notes", []) if n.get("chat_id") != chat_id]
        save_db(db)
        bot.reply_to(message, f"🗑️ Cleared {count} upload(s) for this chat.")
        return

    id_match = re.match(r'id:(\d+)', query, re.IGNORECASE)
    if id_match:
        target_id = int(id_match.group(1))
        before = len(db["notes"])
        db["notes"] = [n for n in db["notes"]
                       if not (n.get("id") == target_id and n.get("chat_id") == chat_id)]
        deleted = before - len(db["notes"])
        save_db(db)
        msg = f"🗑️ Deleted upload ID {target_id}." if deleted else f"🚫 No upload with ID {target_id} in this chat."
        bot.reply_to(message, msg)
        return

    before = len(db.get("notes", []))
    db["notes"] = [n for n in db.get("notes", [])
                   if not (n.get("chat_id") == chat_id and query.lower() in n.get("title", "").lower())]
    deleted = before - len(db["notes"])
    if deleted:
        save_db(db)
        bot.reply_to(message, f"🗑️ Deleted {deleted} upload(s) matching '{query}'.")
    else:
        bot.reply_to(message, f"🚫 No uploads found matching '{query}'.")

# ==========================================
# 27D. /uploads
# ==========================================
@bot.message_handler(commands=['uploads'])
def list_uploads(message):
    db = load_db()
    items = get_chat_notes(db, message.chat.id)
    if not items:
        bot.reply_to(message, "📎 No uploads yet. Use /upload [title] to start.")
        return
    type_emoji = {"photo": "🖼️", "file": "📄", "text": "📝"}
    lines = [f"{type_emoji.get(n['type'], '📎')} [ID:{n['id']}] {n['title']}" for n in items]
    bot.reply_to(message, "📎 Your uploads:\n\n" + "\n".join(lines))

# ==========================================
# 27E. Free-text retrieval
# ==========================================
@bot.message_handler(func=lambda m: bool(
    m.text and re.search(r'(?:show me|show|get|find|give me)\s+(.+)', m.text.strip(), re.IGNORECASE)
))
def handle_show_request(message):
    match = re.search(
        r'(?:show me|show|get|find|give me)\s+(.+)', message.text.strip(), re.IGNORECASE
    )
    if not match:
        return

    keyword = match.group(1).lower().strip()
    db = load_db()
    chat_id = message.chat.id
    matches = [n for n in get_chat_notes(db, chat_id) if keyword in n.get("title", "").lower()]

    if not matches:
        bot.reply_to(message, f"🔍 No uploads found matching *'{keyword}'*.", parse_mode="Markdown")
        return

    bot.reply_to(message, f"📎 Found {len(matches)} item(s) for *'{keyword}'*:", parse_mode="Markdown")
    for note in matches:
        try:
            if note["type"] == "photo":
                bot.send_photo(chat_id, note["file_id"], caption=f"🖼️ {note['title']}")
            elif note["type"] == "file":
                bot.send_document(chat_id, note["file_id"], caption=f"📄 {note['title']}")
            elif note["type"] == "text":
                bot.send_message(chat_id, f"📝 *{note['title']}*\n\n{note['content']}", parse_mode="Markdown")
        except Exception as e:
            bot.send_message(chat_id, f"⚠️ Couldn't retrieve item ID {note['id']}: {e}")

# ==========================================
# 28. /help
# ==========================================
@bot.message_handler(commands=['help', 'start'])
def show_help(message):
    bot.reply_to(message,
        "📖 COMMAND REFERENCE\n\n"
        "ONE-TIME SCHEDULES\n"
        "• -sched [task] [date] [time]\n"
        "• Multiple lines in one message are all saved (bulk paste)\n"
        "• /active — view upcoming\n"
        "• /delete [keyword or id:xxx or all]\n"
        "• /edit [task keyword] [new date/time]\n"
        "• /parse [text] — test date parsing\n\n"
        "UPLOAD & PROCESS\n"
        "• /upload [subject] — start uploading study notes\n"
    "• Send one or more photos of your notes\n"
    "• /donesaving — process images with Gemini OCR and save to study_data/[subject].json\n\n"
        "STUDY TOOLS\n"
        "• /studydocs — list all available study documents\n"
        "• /search [subject] [keyword] — search notes without using Gemini\n"
        "• /studystats [subject] — view quiz score, progress, and accuracy\n\n"
        "QUIZZES\n"
        "• /flashcard [subject] — term → definition flashcards\n"
        "• /flashcard reverse [subject] — definition → term flashcards\n"
        "• /mc [subject] — multiple choice quiz\n"
        "• /identify [subject] — fill-in-the-blank quiz\n"
        "• /tf [subject] — true or false quiz\n"
        "• /enum [subject] — enumeration quiz (5 lists at a time)\n"
        "• /answer [choice] — submit your answer (e.g. /answer B)\n\n"
        "RECURRING TASKS\n"
        "• -remind me to [task] [time] everyday\n"
        "• /recurring, /editrecurring, /delrecurring\n\n"
        "SPECIAL EVENTS\n"
        "• -birthday / -anniversary / -monthsary\n"
        "• /specials, /editspecial, /delspecial\n\n"
        "UPLOADS\n"
        "• /upload [title] → send photos → /donesaving\n"
        "• /uploads, /deleteupload\n\n"
        "📚 STUDY SYSTEM\n"
        "• /studydocs — list study documents\n"
        "• /flashcard [title] — review cards (spoiler answers)\n"
        "• /flashcard reverse [title] — reversed cards\n"
        "• /mc [title] — multiple choice\n"
        "• /identify [title] — fill in the blank\n"
        "• /tf [title] — true or false\n"
        "• /enum [title] — enumerations\n"
        "• /search [title] [keyword] — search your notes\n"
        "• /answer [reply] — submit your answer\n"
        "• /studystats [title] — your score and progress\n\n"
        "AI CHAT\n"
        "• Ask any question ending with ?"
    )

# ==========================================
# 29. QUESTION FALLBACK
# ==========================================
TOPIC_QUESTION_SIGNALS = [
    "sched", "event", "task", "remind", "cath", "mico", "tomorrow", "today", "tmr",
    "pic", "pics", "picture", "pictures", "photo", "photos", "image", "images",
    "note", "notes", "anything about", "have anything", "do i have",
]

def search_chat_topic(db: dict, chat_id, keywords: list) -> dict:
    def matches(text):
        return (not keywords) or any(w in text.lower() for w in keywords)
    return {
        "schedules": [s for s in get_chat_schedules(db, chat_id) if matches(s["task"])],
        "recurring": [r for r in get_chat_recurring(db, chat_id) if matches(r["task"])],
        "specials": [s for s in get_chat_specials(db, chat_id) if matches(s["label"])],
        "notes": [n for n in get_chat_notes(db, chat_id) if matches(n.get("title", ""))],
    }

@bot.message_handler(func=lambda m: m.text is not None)
def smart_question_fallback(message):
    text = message.text.strip()
    if not text.endswith('?'):
        return
    text_lower = text.lower()
    db = load_db()
    chat_id = message.chat.id
    if any(k in text_lower for k in TOPIC_QUESTION_SIGNALS):
        now_ph = datetime.now(PH_TZ)
        date_filters = []
        if re.search(r'\btomorrow\b|\btmr\b|\btmrw\b|\btom\b', text_lower):
            date_filters.append((now_ph + timedelta(days=1)).strftime("%Y-%m-%d"))
        if re.search(r'\btoday\b', text_lower):
            date_filters.append(now_ph.strftime("%Y-%m-%d"))
        keywords_str = re.sub(
            r'\b(tomorrow|today|tmr|tmrw|tom|sched|task|event|any|tasks|for|are|there|'
            r'is|do|i|have|a|an|the|anything|about|pic|pics|picture|pictures|photo|photos|'
            r'image|images|note|notes|my)\b',
            '', text_lower
        )
        keywords = [w for w in keywords_str.replace("?", "").split() if len(w) > 2]
        results = search_chat_topic(db, chat_id, keywords)
        if date_filters:
            results["schedules"] = [s for s in results["schedules"] if s["date"] in date_filters]
        reply_lines = []
        for item in results["schedules"]:
            reply_lines.append(f"📅 [{item['date']} at {item.get('time','Anytime')}] {item['task']}")
        for item in results["recurring"]:
            reply_lines.append(f"🔁 {item['task']} (every {item['recur_type']} at {item.get('time','08:00 AM')})")
        for item in results["specials"]:
            reply_lines.append(f"🎉 {item['label']} ({item['special_type']})")
        if reply_lines:
            bot.reply_to(message, "Found in your schedule:\n" + "\n".join(reply_lines))
        elif not results["notes"]:
            bot.reply_to(message, "🔍 No matching schedules, tasks, or uploads found. Try /active to see everything.")
        for note in results["notes"]:
            caption = f"📎 {note.get('title','').capitalize()}"
            try:
                if note["type"] == "photo":
                    bot.send_photo(chat_id, note["file_id"], caption=caption)
                elif note["type"] == "file":
                    bot.send_document(chat_id, note["file_id"], caption=caption)
                else:
                    bot.send_message(chat_id, f"{caption}\n{note.get('content','')}")
            except Exception as e:
                print(f"[upload-send] failed for note {note.get('id')}: {e}")
        return
    reply = ask_ai(text)
    bot.reply_to(message, reply)

# ==========================================
# 30. STUDY COMMANDS
# ==========================================

@bot.message_handler(commands=['flashcard'])
def handle_flashcard(message):
    args = message.text.split(maxsplit=2)
    if len(args) < 2:
        bot.reply_to(message, "Usage: /flashcard [title]\n       /flashcard reverse [title]")
        return
    rest    = args[1:]
    reverse = rest[0].lower() == "reverse"
    title   = " ".join(rest[1:]) if reverse else " ".join(rest)
    title   = title.strip()
    if not title:
        bot.reply_to(message, "⚠️ Provide a document title. See /studydocs")
        return
    cmd_flashcard(message, title, reverse)

@bot.message_handler(commands=['mc'])
def handle_mc(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "Usage: /mc [title]")
        return
    cmd_mc(message, args[1].strip())

@bot.message_handler(commands=['identify'])
def handle_identify(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "Usage: /identify [title]")
        return
    cmd_identify(message, args[1].strip())

@bot.message_handler(commands=['tf'])
def handle_tf(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "Usage: /tf [title]")
        return
    cmd_tf(message, args[1].strip())

@bot.message_handler(commands=['enum'])
def handle_enum(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "Usage: /enum [title]")
        return
    cmd_enum(message, args[1].strip())

@bot.message_handler(commands=['search'])
def handle_search(message):
    args = message.text.split(maxsplit=2)
    if len(args) < 3:
        bot.reply_to(message, "Usage: /search [title] [keyword]\nExample: /search physics radiation")
        return
    cmd_search(message, args[1].strip(), args[2].strip())

@bot.message_handler(commands=['answer'])
def handle_answer(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "Usage: /answer [your answer]")
        return
    cmd_answer(message, args[1].strip())

@bot.message_handler(commands=['studydocs'])
def handle_studydocs(message):
    cmd_studydocs(message)

@bot.message_handler(commands=['studystats'])
def handle_studystats(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        bot.reply_to(message, "Usage: /studystats [title]")
        return
    cmd_studystats(message, args[1].strip())

# ==========================================
# 31. REMINDER ENGINE
# ==========================================
def reminder_loop():
    print("⏰ Reminder engine started.")
    while True:
        try:
            now_ph = datetime.now(PH_TZ)
            db = load_db()
            changed = False

            for item in db.get("schedules", []):
                if item.get("notified"):
                    continue
                chat_id = item.get("chat_id")
                if not chat_id:
                    continue
                try:
                    item_dt = datetime.strptime(
                        f"{item['date']} {item.get('time','12:00 AM')}",
                        "%Y-%m-%d %I:%M %p"
                    )
                    item_dt_aware = PH_TZ.localize(item_dt)
                except ValueError:
                    continue
                diff = (item_dt_aware - now_ph).total_seconds()
                if -30 <= diff <= 60:
                    bot.send_message(chat_id,
                        f"🔔 REMINDER\n\n{item['task']}\n\n"
                        f"(Scheduled for {item['date']} at {item.get('time')})")
                    item["notified"] = True
                    changed = True
                elif 23 * 3600 + 55 * 60 <= diff <= 24 * 3600 + 5 * 60:
                    if not item.get("warned_24h"):
                        bot.send_message(chat_id,
                            f"⚠️ 24-HOUR NOTICE\n\nTomorrow: {item['task']}\n"
                            f"📅 {item['date']} at {item.get('time')}")
                        item["warned_24h"] = True
                        changed = True

            for item in db.get("recurring", []):
                chat_id = item.get("chat_id")
                if not chat_id:
                    continue
                try:
                    fire_dt = next_recur_date(item, now_ph)
                    fire_dt_aware = PH_TZ.localize(fire_dt)
                except Exception:
                    continue
                diff = (fire_dt_aware - now_ph).total_seconds()
                last = item.get("last_notified")
                already_fired_today = False
                if last:
                    try:
                        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
                        if (now_ph.replace(tzinfo=None) - last_dt).total_seconds() < 23 * 3600:
                            already_fired_today = True
                    except ValueError:
                        pass
                if -30 <= diff <= 60 and not already_fired_today:
                    recur_label = {
                        'daily': 'Daily', 'weekly': 'Weekly',
                        'monthly': 'Monthly', 'yearly': 'Yearly'
                    }.get(item["recur_type"], "Recurring")
                    bot.send_message(chat_id,
                        f"🔁 {recur_label.upper()} REMINDER\n\n{item['task']}\n\n"
                        f"(Every {item['recur_type']} at {item.get('time')})")
                    item["last_notified"] = now_ph.strftime("%Y-%m-%d %H:%M")
                    changed = True

            for item in db.get("specials", []):
                chat_id = item.get("chat_id")
                if not chat_id:
                    continue
                try:
                    fire_dt = next_special_date(item, now_ph)
                    fire_dt_aware = PH_TZ.localize(fire_dt)
                except Exception:
                    continue
                diff = (fire_dt_aware - now_ph).total_seconds()
                last = item.get("last_notified")
                already_fired = False
                if last:
                    try:
                        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
                        if (now_ph.replace(tzinfo=None) - last_dt).total_seconds() < 23 * 3600:
                            already_fired = True
                    except ValueError:
                        pass
                type_emoji = {"birthday": "🎂", "anniversary": "💍", "monthsary": "💕"}
                emoji = type_emoji.get(item["special_type"], "🎉")
                if 23 * 3600 + 55 * 60 <= diff <= 24 * 3600 + 5 * 60:
                    last_warn = item.get("last_warned_24h")
                    if not last_warn or last_warn != fire_dt_aware.strftime("%Y-%m-%d"):
                        type_label = item["special_type"].capitalize()
                        bot.send_message(chat_id,
                            f"⚠️ {emoji} {type_label} TOMORROW!\n\n"
                            f"{item['label']}\n"
                            f"📅 {fire_dt_aware.strftime('%B %d')} at {item.get('time')}")
                        item["last_warned_24h"] = fire_dt_aware.strftime("%Y-%m-%d")
                        changed = True
                if -30 <= diff <= 60 and not already_fired:
                    type_label = item["special_type"].capitalize()
                    bot.send_message(chat_id,
                        f"{emoji} {type_label.upper()}!\n\n"
                        f"Today is {item['label']}'s {type_label}!\n"
                        f"📅 {fire_dt_aware.strftime('%B %d')}")
                    item["last_notified"] = now_ph.strftime("%Y-%m-%d %H:%M")
                    changed = True

            if now_ph.weekday() == 5 and now_ph.hour == 9 and now_ph.minute == 0:
                chat_ids_seen: set = set()
                for item in db.get("schedules", []):
                    cid = item.get("chat_id")
                    if cid:
                        chat_ids_seen.add(cid)
                for cid in chat_ids_seen:
                    chat_items = [s for s in db.get("schedules", []) if s.get("chat_id") == cid]
                    audit_lines = [
                        f"• [{s['date']} at {s.get('time','Anytime')}] {s['task']}" for s in chat_items
                    ]
                    audit_msg = "📋 SATURDAY WEEKLY AUDIT\n\n"
                    audit_msg += "\n".join(audit_lines) if audit_lines else "No events logged."
                    bot.send_message(cid, audit_msg)
                import time as _time
                _time.sleep(70)

            if changed:
                save_db(db)

        except Exception as e:
            print(f"Reminder loop error: {e}")

        time.sleep(30)
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
# ==========================================
# 32. STARTUP
# ==========================================
threading.Thread(target=reminder_loop, daemon=True).start()

# ------------------------------------------
# FastAPI server startup for health check (apply.build)
# ------------------------------------------
def _run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=_run_fastapi, daemon=True).start()
print("✅ Bot armed and operational. Polling...")
active_providers = ", ".join(AI_PROVIDERS) if AI_PROVIDERS else "none configured"
print(f"🤖 AI providers: {active_providers}")

while True:
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        print(f"❌ Polling error. Retrying in 10s... ({e})")
        time.sleep(10)