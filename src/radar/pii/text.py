"""Personal data in free text: find it and replace it with typed placeholders.

The same patterns drive both `clean_text` and `scan_text`, so a scan after cleaning must come back
empty; the structured checks in `radar.pii.records` are what make the scanner more than a mirror.
Names are only caught where the shape is reliable: a patronymic, initials, or a contact label.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

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
    r"|обращаться\s+к|спросить|звонить|звоните|пишите|писать)"
    rf"[ \t]*[:\-–]?[ \t]*(?P<name>{_CAP}(?:[ \t]+{_CAP}){{0,2}})"
)
_NOT_NAMES = frozenset(
    {"Отдел", "Служба", "Компания", "Москва", "Санкт-Петербург", "Телефон", "Почта", "Резюме", "Телеграм", "Вконтакте"}
)


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int


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

    def add(kind: str, start: int, end: int) -> None:
        if any(start < e and s < end for s, e in taken):
            return
        taken.append((start, end))
        found.append(Finding(kind, start, end))

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
