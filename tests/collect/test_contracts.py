"""Contract tests on recorded responses: a changed fixture (field gone, type changed) must fail."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from radar.collect.contracts import (
    HabrListItem,
    HabrListResponse,
    TrudvsemApiResponse,
    TrudvsemApiVacancy,
    TrudvsemSnapshotVacancy,
    parse_habr_card,
)
from radar.collect.trudvsem_snapshot import iter_vacancies

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _api_vacancies() -> list[dict[str, Any]]:
    return [item["vacancy"] for item in _json("trudvsem/vacancies_it_page.json")["results"]["vacancies"]]


def _snapshot_vacancies() -> list[dict[str, Any]]:
    with (FIXTURES / "trudvsem/snapshot_sample.xml").open("rb") as f:
        return list(iter_vacancies(f))


def test_trudvsem_api_fixture_passes() -> None:
    page = TrudvsemApiResponse.model_validate(_json("trudvsem/vacancies_it_page.json"))
    assert page.meta.total == 6567
    vacancies = [TrudvsemApiVacancy.model_validate(v) for v in _api_vacancies()]
    assert [v.job_name for v in vacancies][:2] == ["Программист 1С", "Разработчик Python"]
    assert vacancies[0].date_modify.utcoffset() is not None


def test_trudvsem_api_error_page_fails() -> None:
    with pytest.raises(ValidationError):
        TrudvsemApiResponse.model_validate(_json("trudvsem/vacancies_500_offset_over_limit.json"))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("id",), None),
        (("job-name",), None),
        (("date_modify",), None),
        (("company",), None),
        (("date_modify",), "07.10.2026"),
        (("salary_min",), "53340"),
        (("region", "region_code"), 2400000000000),
    ],
)
def test_trudvsem_api_changed_fixture_fails(path: tuple[str, ...], value: Any) -> None:
    vacancy = copy.deepcopy(_api_vacancies()[1])
    node = vacancy
    for key in path[:-1]:
        node = node[key]
    if value is None:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    with pytest.raises(ValidationError):
        TrudvsemApiVacancy.model_validate(vacancy)


def test_trudvsem_api_new_field_is_fine() -> None:
    vacancy = {**_api_vacancies()[1], "brand_new_field": [1, 2]}
    extra = TrudvsemApiVacancy.model_validate(vacancy).model_extra
    assert extra is not None
    assert extra["brand_new_field"] == [1, 2]


def test_trudvsem_snapshot_fixture() -> None:
    vacancies = _snapshot_vacancies()
    assert len(vacancies) == 3
    ok = [TrudvsemSnapshotVacancy.model_validate(v) for v in vacancies[:2]]
    assert ok[0].dateModify.isoformat() == "2026-06-28T10:00:00+03:00"
    with pytest.raises(ValidationError):
        TrudvsemSnapshotVacancy.model_validate(vacancies[2])


@pytest.mark.parametrize(
    ("key", "value"),
    [("id", None), ("vacancyName", None), ("dateModify", None), ("deleted", "maybe"), ("salaryMin", "сорок")],
)
def test_trudvsem_snapshot_changed_fixture_fails(key: str, value: Any) -> None:
    vacancy = dict(_snapshot_vacancies()[0])
    if value is None:
        del vacancy[key]
    else:
        vacancy[key] = value
    with pytest.raises(ValidationError):
        TrudvsemSnapshotVacancy.model_validate(vacancy)


def test_habr_list_fixture() -> None:
    page = HabrListResponse.model_validate(_json("habr/vacancies_page.json"))
    assert page.meta.totalResults == 1160
    items = [HabrListItem.model_validate(i) for i in page.list_]
    assert len(items) == 3
    assert items[0].publishedDate.date.utcoffset() is not None


@pytest.mark.parametrize(
    ("key", "value"),
    [("id", None), ("title", None), ("publishedDate", None), ("id", "1000000101"), ("archived", "no")],
)
def test_habr_list_changed_fixture_fails(key: str, value: Any) -> None:
    item = dict(_json("habr/vacancies_page.json")["list"][0])
    if value is None:
        del item[key]
    else:
        item[key] = value
    with pytest.raises(ValidationError):
        HabrListItem.model_validate(item)


def test_habr_card_fixture() -> None:
    card = parse_habr_card((FIXTURES / "habr/vacancy_card_1000150058.html").read_text(encoding="utf-8"))
    assert card.vacancy.id == 1000150058
    assert card.vacancy.description.startswith("<h3>")
    assert card.job_posting.datePosted == "2026-09-17"


def test_habr_card_without_state_fails() -> None:
    html = (FIXTURES / "habr/vacancy_card_1000150058.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="SSR"):
        parse_habr_card(html.replace('data-ssr-state="true"', 'data-state="true"'))


def test_habr_card_without_description_fails() -> None:
    html = (FIXTURES / "habr/vacancy_card_1000150058.html").read_text(encoding="utf-8")
    head, rest = html.split('data-ssr-state="true">', 1)
    state_json, tail = rest.split("</script>", 1)
    state = json.loads(state_json)
    del state["vacancy"]["description"]
    with pytest.raises(ValidationError):
        parse_habr_card(head + 'data-ssr-state="true">' + json.dumps(state, ensure_ascii=False) + "</script>" + tail)
