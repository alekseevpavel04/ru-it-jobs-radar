"""Employers that are people: sole proprietors (ИП), farm heads (глава КФХ), private persons.

For them the name, the 12-digit INN, the 15-digit OGRNIP, any code or link built from it and the
street-level address are personal data. A legal entity keeps all of these.
"""

import re

from radar.pii.text import clean_text

_IP_NAME = re.compile(r"^\s*[«\"']?\s*(?:ип|и\.\s?п\.|индивидуальный\s+предприниматель)(?![\w-])", re.IGNORECASE)
_KFH_NAME = re.compile(
    r"^\s*(?:глава\s+)?(?:к\(?ф\)?х|крестьянского\s*(?:\(\s*фермерского\s*\))?\s*хозяйства)(?![\w-])",
    re.IGNORECASE,
)
_STREET_WORDS = (
    r"ул|улица|пр|пр-т|пр-кт|просп|проспект|пер|переулок|ш|шоссе|д|дом|б-р|бульвар|наб|набережная|пл|площадь"
    r"|мкр|мкрн|микрорайон|проезд|туп|тупик|кв|квартира|корп|корпус|стр|строение|лит|литер|тракт|аллея|линия"
    r"|тер|территория|снт|офис|оф|помещ|помещение|влд|владение|зд|здание|к|кв-л|квартал"
)
_POSTCODE = re.compile(r"\d{6}")
_STREET_PART = re.compile(rf"^(?:{_STREET_WORDS})(?![\w-])|(?<![\w-])(?:{_STREET_WORDS})\.?$", re.IGNORECASE)
_DIGIT = re.compile(r"\d")


def digits(value: object) -> str:
    return re.sub(r"\D", "", str(value)) if value is not None else ""


def is_ip_name(name: str | None) -> bool:
    return bool(name) and bool(_IP_NAME.match(name or "") or _KFH_NAME.match(name or ""))


def is_person_employer(name: str | None, inn: object = None, ogrn: object = None) -> bool:
    return is_ip_name(name) or len(digits(inn)) == 12 or len(digits(ogrn)) == 15


def clean_employer_name(name: str | None) -> tuple[str | None, int]:
    """Name of an employer who is a person: the legal form stays, the person goes. Returns (name, removed)."""
    if not name or name in ("ИП", "Глава КФХ"):
        return name, 0
    if _IP_NAME.match(name):
        return "ИП", 1
    if _KFH_NAME.match(name):
        return "Глава КФХ", 1
    cleaned, counts = clean_text(name)
    return cleaned, int(sum(counts.values()) > 0)


def coarse_address(address: str | None) -> str | None:
    """Keep region and settlement, drop the street and everything after it.

    A part with a digit other than the postcode is street level too: a house, a block written as
    "кв-л 12" or a Zelenograd building "к. 1001Б", which no street word announces. A settlement
    whose name has a digit is lost with it; for a person's address that errs on the safe side.
    """
    if not address:
        return address
    kept: list[str] = []
    for part in (p.strip() for p in address.split(",")):
        if _POSTCODE.fullmatch(part):
            continue
        if _STREET_PART.search(part) or _DIGIT.search(part):
            break
        if part:
            kept.append(part)
    return ", ".join(kept) or None
