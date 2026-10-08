"""Salary phrase -> canonical `Salary`: amounts in whole units of the stated currency, plus
currency, gross or net and period only when the phrase states them.

Handles one range per phrase: "300-400к net", "до $5000", "от 150 000 руб. до вычета НДФЛ",
"от 2,4 млн руб. в год", "80-100 т.р.", "1500 ₽/час". A phrase without an amount ("по итогам
собеседования", "конкурентная") gives None, never zero. Conversion to a monthly gross RUB amount is a
modelling choice and lives with the salary model, not here.
"""

import re
from collections.abc import Sequence
from typing import TypeVar

from radar.extract.schema import Currency, GrossNet, Period, Salary

T = TypeVar("T")

_SPACES = re.compile(r"[\s   ]+")
_DASHES = re.compile(r"[‐-―−]")
# "5000р." and "300руб" glue the unit to the digits, so a word boundary is not enough.
_NOT_LETTER = r"(?<![a-zа-я])"

# Order matters: Belarusian roubles before roubles.
_CURRENCY: list[tuple[Currency, re.Pattern[str]]] = [
    ("BYN", re.compile(rf"{_NOT_LETTER}byn\b|бел(?:орусск\w*)?\.?\s*руб")),
    (
        "RUB",
        re.compile(rf"₽|{_NOT_LETTER}rub\b|{_NOT_LETTER}руб|{_NOT_LETTER}р\.|{_NOT_LETTER}т\.\s?р\b|{_NOT_LETTER}тр\b"),
    ),
    ("USD", re.compile(rf"\$|{_NOT_LETTER}usd\b|{_NOT_LETTER}долл")),
    ("EUR", re.compile(rf"€|{_NOT_LETTER}eur\b|{_NOT_LETTER}евро\b")),
    ("KZT", re.compile(rf"₸|{_NOT_LETTER}kzt\b|{_NOT_LETTER}тенге\b")),
    ("UZS", re.compile(rf"{_NOT_LETTER}uzs\b|{_NOT_LETTER}сум(?:ов)?\b")),
    ("CNY", re.compile(rf"¥|{_NOT_LETTER}cny\b|{_NOT_LETTER}юан")),
    ("GBP", re.compile(rf"£|{_NOT_LETTER}gbp\b|{_NOT_LETTER}фунт")),
]
_NET = re.compile(r"на\s+руки|чистыми|после\s+(?:вычета|уплаты|удержания)|\bnet\b|нетто")
_GROSS = re.compile(r"до\s+(?:вычета|уплаты|удержания)|\bgross\b|гросс|брутто|включая\s+ндфл|с\s+учетом\s+ндфл")
_PERIOD: list[tuple[Period, re.Pattern[str]]] = [
    ("hour", re.compile(r"в\s+час|за\s+час|/\s*ч(?:ас)?\b|per\s+hour|/\s*hour|\bhourly\b")),
    ("day", re.compile(r"в\s+(?:день|смену)|за\s+(?:день|смену)|/\s*(?:день|смену|day)\b|per\s+day")),
    (
        "year",
        re.compile(
            r"в\s+год|за\s+год|/\s*(?:год|year)\b|per\s+year|\bannual\w*|\byearly\b"
            r"|годов(?:ой|ая)\s+(?:доход|оклад|зарплата|компенсаци)"
        ),
    ),
    ("month", re.compile(r"в\s+месяц|за\s+месяц|/\s*мес\w*|ежемесячн\w*|per\s+month|/\s*month|\bmonthly")),
]
_NEGOTIABLE = re.compile(r"собеседовани|договорн|договоренност|обсужда|конкурентн|не\s+указан")

# An amount: 150 000 | 150000 | 150,000 | 2,4 | 60 000,00, then an optional multiplier. A group of
# exactly three digits after a separator is thousands, one or two digits are a fraction. Digits glued
# to a percent sign are not money ("13% НДФЛ").
_AMOUNT = re.compile(
    r"(?<![\d.,])(?P<int>\d{1,3}(?:[ ,.]\d{3})+|\d+)(?!\d)(?:[.,](?P<frac>\d{1,2})(?!\d))?(?!\s*%)"
    r"\s*(?P<mult>тыс\w*\.?|т\.\s?р\.?|тр\b|k\b|к\b|млн\.?|миллион\w*)?"
)
_LOWER_WORD = re.compile(r"(?:\bот|\bfrom|не\s+менее|\bmin)\s*[$€₽£¥₸]?\s*$")
_UPPER_WORD = re.compile(r"(?:\bдо|\bto|up\s+to|не\s+более|\bmax)\s*[$€₽£¥₸]?\s*$")
_RANGE_GAP = re.compile(r"^\s*(?:[$€₽£¥₸]|руб\.?|rub|usd|eur)?\s*(?:-|\.\.\.|…|до|to)\s*[$€₽£¥₸]?\s*$")


def _clean(phrase: str) -> str:
    text = _DASHES.sub("-", phrase.lower().replace("ё", "е"))
    return _SPACES.sub(" ", text).strip()


class _Amount:
    def __init__(self, m: re.Match[str]) -> None:
        self.start, self.end = m.start(), m.end()
        self.base = float(re.sub(r"[ ,.]", "", m["int"]) + ("." + m["frac"] if m["frac"] else ""))
        mult = m["mult"] or ""
        self.mult = 1_000_000 if mult.startswith(("млн", "миллион")) else 1_000 if mult else 1

    @property
    def value(self) -> int:
        return round(self.base * self.mult)


def _first(patterns: Sequence[tuple[T, re.Pattern[str]]], text: str) -> T | None:
    return next((name for name, p in patterns if p.search(text)), None)


def parse_salary(phrase: str | None) -> Salary | None:
    if not phrase:
        return None
    text = _clean(phrase)
    amounts = [a for a in (_Amount(m) for m in _AMOUNT.finditer(text)) if a.value > 0]
    if not amounts:
        return None

    first = amounts[0]
    lower: int | None
    upper: int | None
    if len(amounts) >= 2 and _RANGE_GAP.match(text[first.end : amounts[1].start]):
        second = amounts[1]
        if first.mult == 1 and second.mult > 1 and first.base < 1000:
            first.mult = second.mult  # "300-400к", "от 2,4 до 3 млн": the multiplier covers both ends
        lower, upper = first.value, second.value
    elif _UPPER_WORD.search(text[: first.start]):
        lower, upper = None, first.value
    elif _LOWER_WORD.search(text[: first.start]):
        lower, upper = first.value, None
    else:
        lower = upper = first.value

    if lower is not None and upper is not None and lower > upper:
        return None
    gross_net: GrossNet | None = "net" if _NET.search(text) else "gross" if _GROSS.search(text) else None
    return Salary(
        from_=lower,
        to=upper,
        currency=_first(_CURRENCY, text),
        gross_net=gross_net,
        period=_first(_PERIOD, text),
    )


def is_negotiable(phrase: str) -> bool:
    """A salary line that names no amount on purpose ("по итогам собеседования")."""
    text = _clean(phrase)
    return bool(_NEGOTIABLE.search(text)) and not _AMOUNT.search(text)
