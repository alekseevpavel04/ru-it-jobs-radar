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


def rub_range(low: int, high: int) -> str | None:
    """trudvsem salaryMin/salaryMax as the portal shows them; zero means not filled in."""
    if low <= 0 and high <= 0:
        return None
    if high <= 0 or high < low:
        return f"от {_thousands(low)} руб."
    if low <= 0:
        return f"до {_thousands(high)} руб."
    if low == high:
        return f"{_thousands(low)} руб."
    return f"от {_thousands(low)} до {_thousands(high)} руб."


def facts_trudvsem_snapshot(payload: Mapping[str, Any]) -> Facts:
    facts: Facts = []
    salary = rub_range(_int(payload.get("salaryMin", 0)), _int(payload.get("salaryMax", 0)))
    if salary:
        facts.append(("Зарплата", salary))
    address = payload.get("vacancyAddress") or payload.get("regionName")
    if address:
        facts.append(("Адрес", str(address).rstrip("; ")))
    experience = _int(payload.get("experienceRequirements", 0))
    if experience > 0:
        facts.append(("Требуемый опыт", _years(experience)))
    for key, label in (("busyType", "Тип занятости"), ("scheduleType", "График")):
        if payload.get(key):
            facts.append((label, str(payload[key])))
    return facts


def _no_facts(payload: Mapping[str, Any]) -> Facts:
    return []


FACTS: dict[str, Callable[[Mapping[str, Any]], Facts]] = {
    "trudvsem_snapshot": facts_trudvsem_snapshot,
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
