import json

import pytest

from radar.extract.inputs import build_input, facts_trudvsem_snapshot, input_for_record, rub_range
from radar.extract.salary import parse_salary

PAYLOAD = {
    "vacancyName": "Инженер-программист",
    "salary": "от 81000",
    "salaryMin": "81000",
    "salaryMax": "100000",
    "vacancyAddress": "Нижегородская область, Город Дзержинск, Зеленая улица, 10;",
    "regionName": "Нижегородская область",
    "experienceRequirements": "1",
    "busyType": "Полная занятость",
    "scheduleType": "Полный рабочий день",
}


@pytest.mark.parametrize(
    ("low", "high", "text"),
    [
        (81000, 100000, "от 81 000 до 100 000 руб."),
        (50000, 50000, "50 000 руб."),
        (43500, 0, "от 43 500 руб."),
        (60000, 50000, "от 60 000 руб."),
        (0, 0, None),
    ],
)
def test_rub_range(low: int, high: int, text: str | None) -> None:
    assert rub_range(low, high) == text


@pytest.mark.parametrize(("low", "high"), [(81000, 100000), (50000, 50000), (43500, 0)])
def test_rendered_salary_parses_back(low: int, high: int) -> None:
    rendered = rub_range(low, high)
    assert rendered is not None
    s = parse_salary(rendered)
    assert s is not None and s.currency == "RUB"
    assert (s.from_, s.to) == (low, high if high else None)


def test_trudvsem_snapshot_facts() -> None:
    assert facts_trudvsem_snapshot(PAYLOAD) == [
        ("Зарплата", "от 81 000 до 100 000 руб."),
        ("Адрес", "Нижегородская область, Город Дзержинск, Зеленая улица, 10"),
        ("Требуемый опыт", "от 1 года"),
        ("Тип занятости", "Полная занятость"),
        ("График", "Полный рабочий день"),
    ]


def test_unfilled_facts_are_not_rendered() -> None:
    payload = {"salaryMin": "0", "salaryMax": "0", "experienceRequirements": "0", "regionName": "Город Москва"}
    assert facts_trudvsem_snapshot(payload) == [("Адрес", "Город Москва")]


@pytest.mark.parametrize(("years", "text"), [("2", "от 2 лет"), ("5", "от 5 лет"), ("11", "от 11 лет")])
def test_experience_wording(years: str, text: str) -> None:
    assert dict(facts_trudvsem_snapshot({"experienceRequirements": years}))["Требуемый опыт"] == text


def test_build_input_layout() -> None:
    assert build_input("Аналитик", "Обязанности: SQL", [("Зарплата", "50 000 руб.")]) == (
        "Должность: Аналитик\nЗарплата: 50 000 руб.\n\nОбязанности: SQL"
    )
    assert build_input(None, " текст ") == "текст"
    assert build_input(" ", None) == ""


def test_input_for_record_reads_payload_json() -> None:
    record = {
        "source": "trudvsem_snapshot",
        "title": "Инженер-программист",
        "text": "Обязанности: код",
        "payload": json.dumps(PAYLOAD),
    }
    text = input_for_record(record)
    assert text.startswith("Должность: Инженер-программист\nЗарплата: от 81 000 до 100 000 руб.\n")
    assert text.endswith("\n\nОбязанности: код")


def test_source_without_adapter_is_an_error() -> None:
    with pytest.raises(KeyError, match="habr"):
        input_for_record({"source": "habr", "title": "x", "text": "y", "payload": {}})
