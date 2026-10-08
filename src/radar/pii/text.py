"""Personal data in free text: find it and replace it with typed placeholders.

The same patterns drive both `clean_text` and `scan_text`, so a scan after cleaning must come back
empty; the structured checks in `radar.pii.records` are what make the scanner more than a mirror.
Names are caught where the shape is reliable: a patronymic, initials, a contact label, or a first
name from `first_names.csv` next to a surname-shaped word, alone on its line or in a signature line.
"""

import csv
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PLACEHOLDER = {
    "email": "[email]",
    "profile_link": "[link]",
    "nick": "[nick]",
    "inn": "[inn]",
    "ogrn": "[ogrn]",
    "phone": "[phone]",
    "person_name": "[name]",
}

# Public channels we collect from and service accounts that are not people.
DEFAULT_KEEP_HANDLES = frozenset(
    {
        "datasciencejobs",
        "geekjobs",
        "habr_career",
        "foranalysts",
        "datajobschannel",
        "llm_jobs",
        "nlp_jobs",
        "remoteit",
        "jobforjunior",
        "job_python",
    }
)

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[a-zа-яё]{2,}(?![\w-])", re.IGNORECASE)

_PROFILE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?"
    r"(?P<host>t\.me|telegram\.me|telegram\.dog|vk\.com|vk\.ru|wa\.me|instagram\.com|facebook\.com|fb\.com"
    r"|ok\.ru/profile|linkedin\.com/in|hh\.ru/resume)"
    r"/(?P<path>[\w.+\-/]*[\w+\-])",
    re.IGNORECASE,
)
_TG_HOSTS = {"t.me", "telegram.me", "telegram.dog"}
_TG_SERVICE = {"joinchat", "addlist", "share", "proxy", "iv", "c"}
_VK_COMMUNITY = re.compile(r"(?:club|public|event|wall|topic|board|video|album|market|app)-?\d", re.IGNORECASE)

_NICK = re.compile(r"(?<![\w@.])@(?P<name>[A-Za-z][A-Za-z0-9_]{3,31})(?![\w@])")

_DIGIT = re.compile(r"\d")
_INN = re.compile(r"ИНН\s*:?\s*№?\s*\d{12}(?!\d)")
_OGRNIP = re.compile(r"ОГРНИП\s*:?\s*№?\s*\d{15}(?!\d)")

_SEP = r"[ \t\-‐‑–.]*"
_PHONE = re.compile(
    rf"(?<![\w+])(?:(?:\+[ \t]*7|8|7){_SEP})?"  # country or trunk prefix
    rf"(?:\([ \t]*\d{{3,5}}[ \t]*\)|\d{{3,5}}){_SEP}"  # area or mobile operator code
    rf"\d{{1,3}}{_SEP}\d{{2}}{_SEP}\d{{2}}(?!\d)"
)
_LOCAL_PHONE = re.compile(r"(?<![\w+\-.])\d{2,3}-\d{2}-\d{2}(?![\w\-.])")

_CAP = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
_PATRONYMIC = r"[А-ЯЁ][а-яё]+(?:ич|вн|чн)[а-яё]{0,3}"
_INITIALS = r"[А-ЯЁ]\.[ \t]?[А-ЯЁ]\."
_NAME = re.compile(
    rf"\b{_CAP}[ \t]+{_CAP}[ \t]+{_PATRONYMIC}\b"
    rf"|\b{_CAP}[ \t]+{_PATRONYMIC}(?:[ \t]+{_CAP})?\b"
    rf"|\b{_CAP}[ \t]+{_INITIALS}(?!\w)"
    rf"|(?<!\w){_INITIALS}[ \t]?{_CAP}\b"
)
_LABELLED_NAME = re.compile(
    r"(?i:контактное\s+лицо|контакт|менеджер\s+по\s+(?:персоналу|подбору)|рекрутер|hr(?:-|\s)?менеджер"
    r"|обращаться\s+к|спросить|звонить|звоните|пишите|писать|меня\s+зовут)"
    rf"[ \t]*[:\-–]?[ \t]*(?P<name>{_CAP}(?:[ \t]+{_CAP}){{0,2}})"
)
_NOT_NAMES = frozenset(
    {"Отдел", "Служба", "Компания", "Москва", "Санкт-Петербург", "Телефон", "Почта", "Резюме", "Телеграм", "Вконтакте"}
)


def _load_first_names() -> dict[str, bool]:
    with Path(__file__).with_name("first_names.csv").open(encoding="utf-8", newline="") as f:
        return {row["name"]: row["ambiguous"] == "1" for row in csv.DictReader(f)}


# First name -> it is also a common word, a place or a product ("Вера", "Владимир", "Алиса", "Julia").
_FIRST_NAMES = _load_first_names()
_WORD = re.compile(r"(?<![\w-])(?:[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?|[A-Z][a-z]+(?:-[A-Z][a-z]+)?)(?![\w-])")
_SURNAME = re.compile(
    r"(?:ов|ев|ёв|ин|ын|ова|ева|ёва|ина|ына|ский|цкий|ская|цкая|енко|ук|юк|ич|ян|янц|дзе|швили|их|ых"
    r"|ov|ova|ev|eva|in|ina|yn|yna|sky|skiy|skii|skaya|enko|uk|yuk|ich|vich|yan|dze|shvili|ikh|ykh)$"
)
# A line that signs a post or gives a contact; placeholders count, so cleaned text reads the same.
_SIGNATURE = re.compile(
    r"(?<!\w)(?:hr|hrbp|recruit\w*|talent|contacts?|manager|telegram|tg)(?!\w)"
    r"|(?<!\w)(?:рекрут|контакт|пишите|напишите|писать|менеджер|телеграм)"
    r"|@\w|\[(?:nick|phone|email|link)\]",
    re.IGNORECASE,
)
# Capitalized words that stand next to a name in a signature without being a surname.
_NOT_SURNAMES = _NOT_NAMES | {
    "Контакт",
    "Контакты",
    "Пишите",
    "Напишите",
    "Менеджер",
    "Рекрутер",
    "Рекрутёр",
    "Директор",
    "Руководитель",
    "Специалист",
    "Разработчик",
    "Аналитик",
    "Инженер",
    "Ведущий",
    "Старший",
    "Главный",
    "Команда",
    "Связь",
    "Вакансия",
    "Привет",
    "Здравствуйте",
    "Ищет",
    "Ищем",
    "Ещё",
    "Еще",
    "Hey",
    "Hi",
    "Hello",
    "Спасибо",
    "Head",
    "Lead",
    "Senior",
    "Middle",
    "Junior",
    "Manager",
    "Recruiter",
    "Developer",
    "Engineer",
    "Analyst",
    "Team",
    "Talent",
    "Partner",
    "Founder",
    "Contact",
    "Contacts",
    "Telegram",
}
_EDGES = re.compile(r"^[\W\d_]+|[\W\d_]+$")


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    rule: str = ""  # for person_name found by first name: full_name, name_line or signature


def _same_script(a: str, b: str) -> bool:
    return a[0].isascii() == b[0].isascii()


def _is_surname(word: str, name: str) -> bool:
    return len(word) >= 4 and word not in _FIRST_NAMES and _same_script(word, name) and bool(_SURNAME.search(word))


def _is_other_word(word: str, name: str) -> bool:
    return _same_script(word, name) and word not in _NOT_SURNAMES


def _pair(name: str, other: str, ambiguous: bool, alone: bool, signature: bool, after: bool) -> bool:
    if _is_surname(other, name):
        # Latin names that are also tools (Julia, Victoria) need a signature line to count.
        return signature or not (ambiguous and name.isascii())
    # In a signature the word before a name is mostly a verb or a title ("Ищет", "Рекрутер").
    return (alone or (signature and after)) and not ambiguous and _is_other_word(other, name)


def _runs(words: list[re.Match[str]], text: str) -> Iterator[list[re.Match[str]]]:
    """Capitalized words separated by nothing but spaces."""
    run: list[re.Match[str]] = []
    for w in words:
        if run and text[run[-1].end() : w.start()].strip(" \t"):
            yield run
            run = []
        run.append(w)
    if run:
        yield run


def _flag_run(run: list[re.Match[str]], alone: bool, signature: bool) -> set[int]:
    flagged: set[int] = set()
    for i, w in enumerate(run):
        name = w.group()
        if name not in _FIRST_NAMES:
            continue
        ambiguous = _FIRST_NAMES[name]
        for j in (i - 1, i + 1):
            if not 0 <= j < len(run):
                continue
            if _pair(name, run[j].group(), ambiguous, alone, signature, after=j > i):
                flagged |= {i, j}
        if signature and not ambiguous:
            flagged.add(i)
    if alone and flagged:
        flagged = set(range(len(run)))
    return flagged


def _person_names(text: str, context: list[tuple[int, int]]) -> Iterator[tuple[str, int, int]]:
    """First name plus surname, a name alone on its line, or a name in a signature line."""
    hits = [m for m in _WORD.finditer(text) if m.group() in _FIRST_NAMES]
    for start in sorted({text.rfind("\n", 0, m.start()) + 1 for m in hits}):
        end = text.find("\n", start)
        end = len(text) if end < 0 else end
        body = text[start:end]
        core = _EDGES.sub("", body)
        signature = bool(_SIGNATURE.search(body)) or any(s < end and start < e for s, e in context)
        for run in _runs(list(_WORD.finditer(text, start, end)), text):
            alone = len(run) in (2, 3) and core == text[run[0].start() : run[-1].end()]
            flagged = _flag_run(run, alone, signature)
            rule = "name_line" if alone else "signature" if signature else "full_name"
            i = 0
            while i < len(run):
                if i not in flagged:
                    i += 1
                    continue
                j = i
                while j + 1 in flagged:
                    j += 1
                yield rule, run[i].start(), run[j].end()
                i = j + 1


def _keep_profile(match: re.Match[str], keep: frozenset[str]) -> bool:
    host = match.group("host").lower()
    parts = [p for p in match.group("path").split("/") if p]
    if not parts:
        return True
    if host in _TG_HOSTS:
        name = parts[1] if parts[0].lower() == "s" and len(parts) > 1 else parts[0]
        lowered = name.lower()
        return lowered in _TG_SERVICE or name.startswith("+") or lowered.endswith("bot") or lowered in keep
    if host in {"vk.com", "vk.ru"}:
        return bool(_VK_COMMUNITY.match(parts[0]))
    return False


def _is_phone(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11:
        return digits[0] in "78"
    if len(digits) == 10:
        return digits[0] in "3489"
    return False


def _spans(text: str, keep: frozenset[str]) -> list[Finding]:
    found: list[Finding] = []
    taken: list[tuple[int, int]] = []

    def add(kind: str, start: int, end: int, rule: str = "") -> None:
        if any(start < e and s < end for s, e in taken):
            return
        taken.append((start, end))
        found.append(Finding(kind, start, end, rule))

    if "@" in text:
        for m in _EMAIL.finditer(text):
            add("email", m.start(), m.end())
    if "/" in text:
        for m in _PROFILE.finditer(text):
            if not _keep_profile(m, keep):
                add("profile_link", m.start(), m.end())
    if "@" in text:
        for m in _NICK.finditer(text):
            name = m.group("name").lower()
            if not name.endswith("bot") and name not in keep:
                add("nick", m.start(), m.end())
    if _DIGIT.search(text):
        for m in _INN.finditer(text):
            add("inn", m.start(), m.end())
        for m in _OGRNIP.finditer(text):
            add("ogrn", m.start(), m.end())
        for m in _PHONE.finditer(text):
            if _is_phone(m.group(0)):
                add("phone", m.start(), m.end())
        for m in _LOCAL_PHONE.finditer(text):
            add("phone", m.start(), m.end())
    for m in _NAME.finditer(text):
        add("person_name", m.start(), m.end())
    for m in _LABELLED_NAME.finditer(text):
        name = m.group("name")
        if name.split()[0] not in _NOT_NAMES:
            add("person_name", m.start("name"), m.end("name"))
    contacts = [(f.start, f.end) for f in found if f.kind != "person_name"]
    for rule, start, end in _person_names(text, contacts):
        add("person_name", start, end, rule)
    return sorted(found, key=lambda f: f.start)


def scan_text(text: str, keep: Iterable[str] = DEFAULT_KEEP_HANDLES) -> list[Finding]:
    """Where personal data sits in `text`; positions only, so callers never have to print values."""
    return _spans(text, frozenset(h.lower() for h in keep))


Counts = tuple[tuple[str, int], ...]


def _clean(text: str, keep: frozenset[str]) -> tuple[str, Counts]:
    findings = _spans(text, keep)
    if not findings:
        return text, ()
    counts: Counter[str] = Counter()
    out: list[str] = []
    pos = 0
    for f in findings:
        out.append(text[pos : f.start])
        out.append(PLACEHOLDER[f.kind])
        counts[f.kind] += 1
        pos = f.end
    out.append(text[pos:])
    return "".join(out), tuple(counts.items())


@lru_cache(maxsize=200_000)
def _clean_short(text: str) -> tuple[str, Counts]:
    return _clean(text, DEFAULT_KEEP_HANDLES)


def clean_text_counts(text: str) -> tuple[str, Counts]:
    """`clean_text` with the default handles and counts as `(kind, n)` pairs, for the hot path."""
    if len(text) <= 200:
        # Short categorical values repeat across hundreds of thousands of records.
        return _clean_short(text)
    return _clean(text, DEFAULT_KEEP_HANDLES)


def clean_text(text: str, keep: Iterable[str] | None = None) -> tuple[str, Counter[str]]:
    """Replace personal data with placeholders; returns the text and counts by kind."""
    if keep is None:
        cleaned, items = clean_text_counts(text)
    else:
        cleaned, items = _clean(text, frozenset(h.lower() for h in keep))
    return cleaned, Counter(dict(items))
