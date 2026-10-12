"""What an extractor reads: the vacancy as a job page shows it.

A page shows the title and the structured facts the employer filled in (salary, address, experience,
employment) above the free text. trudvsem keeps those facts only in the payload, so its `text` alone
has no salary or city. The input therefore renders the facts as a short header in the page's own
words; the same string goes to teachers, the gold set, the regex baseline and the students, and a
text pasted into the Space has the same shape. The header renders what the source states and nothing
it does not: trudvsem's experience "0" is not shown, because senior vacancies carry it too and it
cannot be told from "not filled in".
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Facts = list[tuple[str, str]]


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _years(n: int) -> str:
    word = "года" if n % 10 == 1 and n % 100 != 11 else "лет"
    return f"от {n} {word}"


def _int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except ValueError:
        return 0


def amount_range(low: int, high: int, unit: str) -> str | None:
    """A salary range as a job page shows it; zero means not filled in."""
    if low <= 0 and high <= 0:
        return None
    if high <= 0 or high < low:
        return f"от {_thousands(low)} {unit}"
    if low <= 0:
        return f"до {_thousands(high)} {unit}"
    if low == high:
        return f"{_thousands(low)} {unit}"
    return f"от {_thousands(low)} до {_thousands(high)} {unit}"


def rub_range(low: int, high: int) -> str | None:
    """trudvsem salaryMin/salaryMax as the portal shows them."""
    return amount_range(low, high, "руб.")


def _trudvsem_page(salary_min: Any, salary_max: Any, address: Any, experience: Any, busy: Any, schedule: Any) -> Facts:
    """The trudvsem vacancy page; the snapshot and the API name its fields differently."""
    facts: Facts = []
    salary = rub_range(_int(salary_min), _int(salary_max))
    if salary:
        facts.append(("Зарплата", salary))
    if address:
        facts.append(("Адрес", str(address).rstrip("; ")))
    years = _int(experience)
    if years > 0:
        facts.append(("Требуемый опыт", _years(years)))
    for value, label in ((busy, "Тип занятости"), (schedule, "График")):
        if value:
            facts.append((label, str(value)))
    return facts


def facts_trudvsem_snapshot(payload: Mapping[str, Any]) -> Facts:
    return _trudvsem_page(
        payload.get("salaryMin", 0),
        payload.get("salaryMax", 0),
        payload.get("vacancyAddress") or payload.get("regionName"),
        payload.get("experienceRequirements", 0),
        payload.get("busyType"),
        payload.get("scheduleType"),
    )


def facts_trudvsem(payload: Mapping[str, Any]) -> Facts:
    """The API record of the same page as the snapshot, so the same header. The API's own `salary`
    string keeps only the lower bound ("от 33867" for 33 867 to 50 000), so the range is rebuilt
    from `salary_min` and `salary_max`."""
    addresses = (payload.get("addresses") or {}).get("address") or []
    address = "; ".join(str(a["location"]).rstrip("; ") for a in addresses if a.get("location"))
    return _trudvsem_page(
        payload.get("salary_min", 0),
        payload.get("salary_max", 0),
        address or (payload.get("region") or {}).get("name"),
        (payload.get("requirement") or {}).get("experience", 0),
        payload.get("employment"),
        payload.get("schedule"),
    )


HABR_EMPLOYMENT = {"full_time": "Полный рабочий день", "part_time": "Неполный рабочий день"}


def _titles(items: Any) -> str:
    return ", ".join(str(item["title"]) for item in items or () if item.get("title"))


def _habr_card_localities(card: Mapping[str, Any]) -> str:
    places = card.get("jobLocation") or []
    if isinstance(places, Mapping):
        places = [places]
    localities = []
    for place in places:
        address = place.get("address")
        if isinstance(address, Mapping) and address.get("addressLocality"):
            localities.append(str(address["addressLocality"]))
    return ", ".join(localities)


def facts_habr(payload: Mapping[str, Any]) -> Facts:
    """A Habr Career vacancy as its list entry shows it: salary as published, specializations,
    qualification, skill tags, cities, employment and the "можно удаленно" flag.

    Habr's own salary estimate (`predictedSalary`) is the site's guess, not the employer's, and is
    never rendered. The card's JSON-LD repeats the list (salary, remote, employment) and says
    FULL_TIME also where the list states no employment, so it is used for one thing only: a vacancy
    with an office address instead of a city list gets the address's locality as its city.

    A vacancy whose card was not read yet has no description: the extractor sees the title and this
    header only, so nothing is known about experience, gross or net, or the language of the text.
    """
    facts: Facts = []
    salary = (payload.get("salary") or {}).get("formatted")
    if salary:
        facts.append(("Зарплата", str(salary)))
    divisions = _titles(payload.get("divisions"))
    if divisions:
        facts.append(("Специализация", divisions))
    qualification = (payload.get("salaryQualification") or {}).get("title") or payload.get("qualification")
    if qualification:
        facts.append(("Квалификация", str(qualification)))
    skills = _titles(payload.get("skills"))
    if skills:
        facts.append(("Навыки", skills))
    places = _titles(payload.get("locations")) or _habr_card_localities(payload.get("card") or {})
    if places:
        facts.append(("Местоположение", places))
    employment = payload.get("employment")
    if employment:
        facts.append(("Тип занятости", HABR_EMPLOYMENT.get(employment, str(employment))))
    if payload.get("remoteWork") is True:
        facts.append(("Формат работы", "Можно удаленно"))
    return facts


SUPERJOB_CURRENCY = {"rub": "руб.", "usd": "$", "eur": "€", "kzt": "тенге", "uzs": "сум", "byn": "бел. руб."}
# SuperJob's "no matter" answers: the employer chose not to state the fact.
SUPERJOB_UNSTATED = "Не имеет значения"


def _superjob_title(payload: Mapping[str, Any], key: str) -> str | None:
    title = (payload.get(key) or {}).get("title")
    return str(title) if title and title != SUPERJOB_UNSTATED else None


def facts_superjob(payload: Mapping[str, Any]) -> Facts:
    """A SuperJob vacancy page: salary as published (zero is not filled in, "по договоренности" is no
    amount), the town, required experience, type of work and place of work when it is stated. The
    address is left out: the collector cuts it to the town level, which the town already says."""
    facts: Facts = []
    currency = str(payload.get("currency") or "rub")
    salary = amount_range(
        _int(payload.get("payment_from") or 0),
        _int(payload.get("payment_to") or 0),
        SUPERJOB_CURRENCY.get(currency, currency.upper()),
    )
    if salary:
        facts.append(("Зарплата", salary))
    town = (payload.get("town") or {}).get("title")
    if town:
        facts.append(("Местоположение", str(town)))
    for key, label in (
        ("experience", "Требуемый опыт"),
        ("type_of_work", "Тип занятости"),
        ("place_of_work", "Формат работы"),
    ):
        value = _superjob_title(payload, key)
        if value:
            facts.append((label, value))
    return facts


def _no_facts(payload: Mapping[str, Any]) -> Facts:
    return []


FACTS: dict[str, Callable[[Mapping[str, Any]], Facts]] = {
    "trudvsem": facts_trudvsem,
    "trudvsem_snapshot": facts_trudvsem_snapshot,
    "habr": facts_habr,
    "superjob": facts_superjob,
    "telegram": _no_facts,
}


def build_input(title: str | None, text: str | None, facts: Sequence[tuple[str, str]] = ()) -> str:
    head = [f"Должность: {title.strip()}"] if title and title.strip() else []
    head += [f"{label}: {value}" for label, value in facts]
    parts = ["\n".join(head)] if head else []
    if text and text.strip():
        parts.append(text.strip())
    return "\n\n".join(parts)


def input_for_record(record: Mapping[str, Any]) -> str:
    """Extractor input for a raw-layer row. A source without a facts adapter is an error,
    not a silently thinner input: the adapter is written together with the source's collector."""
    source = record["source"]
    if source not in FACTS:
        raise KeyError(f"no facts adapter for source {source!r}; add one to radar.extract.inputs.FACTS")
    payload = record.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    return build_input(record.get("title"), record.get("text"), FACTS[source](payload))
