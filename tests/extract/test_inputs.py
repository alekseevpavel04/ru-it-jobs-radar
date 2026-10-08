import json

import pytest

from radar.extract.inputs import (
    build_input,
    facts_habr,
    facts_trudvsem,
    facts_trudvsem_snapshot,
    input_for_record,
    rub_range,
)
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


API_PAYLOAD = {
    "job-name": "Инженер-программист",
    "salary": "от 81000",
    "salary_min": 81000,
    "salary_max": 100000,
    "currency": "«руб.»",
    "addresses": {"address": [{"location": "Нижегородская область, Город Дзержинск, Зеленая улица, 10;"}]},
    "region": {"region_code": "5200000000000", "name": "Нижегородская область"},
    "requirement": {"education": "Высшее образование", "experience": 1},
    "employment": "Полная занятость",
    "schedule": "Полный рабочий день",
    "qualification": "Специалист",
    "_queries": ["industry_it"],
}


def test_trudvsem_api_header_matches_snapshot() -> None:
    assert facts_trudvsem(API_PAYLOAD) == facts_trudvsem_snapshot(PAYLOAD)


def test_trudvsem_api_unfilled_facts_are_not_rendered() -> None:
    payload = {
        "salary": "от 0",
        "salary_min": 0,
        "salary_max": 0,
        "addresses": {"address": []},
        "region": {"name": "Город Москва"},
        "requirement": {"experience": 0},
        "employment": None,
    }
    assert facts_trudvsem(payload) == [("Адрес", "Город Москва")]


def test_trudvsem_api_several_addresses() -> None:
    payload = {
        "addresses": {"address": [{"location": "г Казань, ул. Баумана, 1;"}, {"location": "г Казань, ул. Пушкина, 2"}]}
    }
    assert facts_trudvsem(payload) == [("Адрес", "г Казань, ул. Баумана, 1; г Казань, ул. Пушкина, 2")]


HABR_PAYLOAD = {
    "id": 1,
    "title": "Python-разработчик",
    "salary": {"from": 200000, "to": 300000, "currency": "rur", "formatted": "от 200 000 до 300 000 ₽"},
    "predictedSalary": {"from": 150000, "to": 250000, "currency": "rur", "formatted": "от 150 000 до 250 000 ₽"},
    "qualification": "Middle",
    "salaryQualification": {"title": "Средний (Middle)", "href": "/vacancies?qid=4"},
    "divisions": [{"title": "Бэкенд разработчик", "href": "/vacancies/spec/development/backend"}],
    "skills": [{"title": "Python", "href": "/x"}, {"title": "PostgreSQL", "href": "/y"}],
    "locations": [{"title": "Москва", "href": "/vacancies?city_id=678"}, {"title": "Казань", "href": "/z"}],
    "employment": "full_time",
    "remoteWork": True,
    "company": {"title": "Компания"},
}


def test_habr_facts() -> None:
    assert facts_habr(HABR_PAYLOAD) == [
        ("Зарплата", "от 200 000 до 300 000 ₽"),
        ("Специализация", "Бэкенд разработчик"),
        ("Квалификация", "Средний (Middle)"),
        ("Навыки", "Python, PostgreSQL"),
        ("Местоположение", "Москва, Казань"),
        ("Тип занятости", "Полный рабочий день"),
        ("Формат работы", "Можно удаленно"),
    ]


def test_habr_estimate_and_empty_fields_are_not_rendered() -> None:
    payload = {
        "salary": {"from": None, "to": None, "currency": None, "formatted": ""},
        "predictedSalary": {"from": 100000, "to": 200000, "formatted": "от 100 000 до 200 000 ₽"},
        "qualification": None,
        "divisions": [],
        "skills": [],
        "locations": None,
        "employment": None,
        "remoteWork": False,
        "card": {"employmentType": "FULL_TIME", "jobLocation": {"@type": "Place", "address": "Russia"}},
    }
    assert facts_habr(payload) == []


@pytest.mark.parametrize(
    ("formatted", "currency"), [("до 188 000 ₽", "RUB"), ("от 2000 до 3000 $", "USD"), ("до 8000 €", "EUR")]
)
def test_habr_salary_parses_back(formatted: str, currency: str) -> None:
    rendered = dict(facts_habr({"salary": {"formatted": formatted}}))["Зарплата"]
    s = parse_salary(rendered)
    assert s is not None and s.currency == currency


def test_habr_card_locality_when_list_has_no_cities() -> None:
    card = {
        "jobLocation": {
            "@type": "Place",
            "address": {"@type": "PostalAddress", "streetAddress": "ул. Лесная, д. 7", "addressLocality": "Москва"},
        }
    }
    assert facts_habr({"locations": None, "card": card}) == [("Местоположение", "Москва")]
    listed = {"locations": [{"title": "Пенза"}], "card": card}
    assert facts_habr(listed) == [("Местоположение", "Пенза")]


def test_habr_without_card_text_is_header_only() -> None:
    record = {"source": "habr", "title": "Python-разработчик", "text": None, "payload": json.dumps(HABR_PAYLOAD)}
    assert input_for_record(record) == (
        "Должность: Python-разработчик\n"
        "Зарплата: от 200 000 до 300 000 ₽\n"
        "Специализация: Бэкенд разработчик\n"
        "Квалификация: Средний (Middle)\n"
        "Навыки: Python, PostgreSQL\n"
        "Местоположение: Москва, Казань\n"
        "Тип занятости: Полный рабочий день\n"
        "Формат работы: Можно удаленно"
    )


def test_trudvsem_api_record() -> None:
    record = {"source": "trudvsem", "title": "Инженер-программист", "text": "Обязанности: код", "payload": API_PAYLOAD}
    assert input_for_record(record) == input_for_record({**record, "source": "trudvsem_snapshot", "payload": PAYLOAD})


def test_source_without_adapter_is_an_error() -> None:
    with pytest.raises(KeyError, match="superjob"):
        input_for_record({"source": "superjob", "title": "x", "text": "y", "payload": {}})
