"""Whether two cards can be in the same place, the rule the labeling guide puts before any text.

trudvsem gives a region and a free-form address; the API and the snapshot spell the same address
differently («г Москва» and «Город Москва», «Пискарёвский проспект, 2к2» and «..., 2, корп. 2»).
So the address is reduced to a settlement and a street name, and only a stated difference counts:
two different settlements or two different streets mean two places, a missing part means unknown.
Habr gives a list of cities, which is compared with the trudvsem region or settlement.
"""

import re
from dataclasses import dataclass

_SETTLEMENT_BEFORE = re.compile(r"^(?:г|город)\.?\s+(.+)$")
_SETTLEMENT_AFTER = re.compile(
    r"^(.+?)\s+(?:город|село|деревня|поселок городского типа|рабочий поселок|поселок|станица|хутор|аул)$"
)
_STREET_TYPES = (
    "улица|ул|проспект|пр-кт|переулок|пер|шоссе|площадь|пл|бульвар|б-р|проезд|набережная|наб|"
    "квартал|микрорайон|мкр|тракт|километр|км|линия|аллея|тупик"
)
_STREET = re.compile(rf"(?:^|\s)(?:{_STREET_TYPES})\.?(?:\s|$)")
_STREET_WORDS = re.compile(rf"\b(?:{_STREET_TYPES}|им)\b\.?")
_REGION_WORDS = re.compile(r"\b(?:город|г|область|обл|край|республика|респ|автономный округ|ао)\b\.?")
_NON_WORD = re.compile(r"[^0-9a-zа-я]+")


def _norm(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower().replace("ё", "е")).split())


@dataclass(frozen=True)
class Place:
    """What a card says about where the work is."""

    names: frozenset[str]
    settlement: str | None
    street: frozenset[str] | None
    remote: bool

    @property
    def known(self) -> bool:
        return bool(self.names or self.settlement)


def parse_address(address: str | None) -> tuple[str | None, frozenset[str] | None]:
    """Settlement and street words of a trudvsem address; None where the address does not say."""
    settlement: str | None = None
    street: frozenset[str] | None = None
    for raw in (address or "").split(","):
        part = raw.strip().lower().replace("ё", "е")
        if not part:
            continue
        if settlement is None:
            m = _SETTLEMENT_BEFORE.match(part) or _SETTLEMENT_AFTER.match(part)
            if m:
                settlement = _norm(m.group(1))
                continue
        if street is None and _STREET.search(part):
            words = _norm(_STREET_WORDS.sub(" ", part)).split()
            street = frozenset(words) or None
    return settlement, street


def place_of(location: str | None, address: str | None, remote: bool | None, source: str) -> Place:
    """`location` is the trudvsem region, the SuperJob town or the comma-joined Habr cities.

    SuperJob goes the trudvsem way: its town stands where the region does, and its address, cut to
    region and settlement by the raw cleaner, gives the settlement when it says «г. ...».
    """
    if source == "habr":
        names = frozenset(_norm(c) for c in (location or "").split(",") if _norm(c))
        return Place(names, None, None, bool(remote))
    settlement, street = parse_address(address)
    region = _norm(_REGION_WORDS.sub(" ", (location or "").lower()))
    names = frozenset(n for n in (region, settlement) if n)
    return Place(names, settlement, street, bool(remote))


def same_place(a: Place, b: Place) -> bool:
    if a.remote and b.remote:
        return True
    if not a.known or not b.known:
        return True
    if a.settlement and b.settlement:
        if a.settlement != b.settlement:
            return False
        return a.street is None or b.street is None or a.street == b.street
    return bool(a.names & b.names)
