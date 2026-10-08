"""Text and employer normalization for near-duplicate search.

Numbers become `0`: a repost that only moved the salary or the date should still look the same,
and stage B measures what this costs on pairs that differ only in amounts. `+` and `#` stay inside
tokens so that `c++` and `c#` are not the same word as `c`.
"""

import re

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_DIGITS = re.compile(r"\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?")
_NON_WORD = re.compile(r"[^0-9a-zа-я+#]+")

# Legal forms in short and full spelling; the order puts longer phrases first so that
# "публичное акционерное общество" is removed whole, not leaving "публичное".
_LEGAL_FORMS = (
    "общество с ограниченной ответственностью",
    "публичное акционерное общество",
    "непубличное акционерное общество",
    "закрытое акционерное общество",
    "открытое акционерное общество",
    "акционерное общество",
    "индивидуальный предприниматель",
    "федеральное государственное бюджетное учреждение",
    "государственное бюджетное учреждение",
    "государственное казенное учреждение",
    "государственное автономное учреждение",
    "муниципальное бюджетное учреждение",
    "муниципальное казенное учреждение",
    "муниципальное автономное учреждение",
    "автономная некоммерческая организация",
    "ооо",
    "пао",
    "нао",
    "зао",
    "оао",
    "ао",
    "ип",
    "фгбу",
    "гбу",
    "гку",
    "гау",
    "мбу",
    "мку",
    "мау",
    "ано",
    "llc",
    "ltd",
    "inc",
)
_LEGAL_RE = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in _LEGAL_FORMS) + r")\b")


def normalize_text(text: str) -> str:
    """Lowercase words for shingling: urls and e-mails dropped, numbers as 0, punctuation as space."""
    out = text.lower().replace("ё", "е")
    out = _URL.sub(" ", out)
    out = _EMAIL.sub(" ", out)
    out = _DIGITS.sub("0", out)
    return " ".join(_NON_WORD.sub(" ", out).split())


def shingles(text: str, k: int) -> set[str]:
    """Word k-grams of normalized text; a text shorter than k words is one shingle of all its words."""
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize_employer(name: str | None) -> str | None:
    """Key that groups spellings of one employer: «ПАО "МТС"» and «МТС» give `мтс`.

    A sole proprietor comes from the sources as just «ИП» (the name is personal data), which says
    nothing about which employer it is, so it gives None, as does an empty name.
    """
    if not name:
        return None
    out = name.lower().replace("ё", "е")
    out = _LEGAL_RE.sub(" ", out)
    out = " ".join(_NON_WORD.sub(" ", out).split())
    return out or None


def normalize_title(title: str | None) -> str | None:
    """Title key for blocking: normalized words without bracketed notes like «(удалённо)»."""
    if not title:
        return None
    out = re.sub(r"[(\[].*?[)\]]", " ", title)
    return normalize_text(out) or None
