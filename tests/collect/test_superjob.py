import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

import radar.results
from radar.collect import superjob
from radar.collect.contracts import SuperjobResponse, SuperjobVacancy
from radar.collect.http import Fetcher
from radar.pii import clean_superjob_vacancy, scan_record

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "superjob" / "vacancies_page.json").read_text(encoding="utf-8")
)
OBJECTS = FIXTURE["objects"]
IT_COMPANY, SOLE_PROPRIETOR, PERSON = OBJECTS
SERVED_PER_PAGE = 40


class Api:
    """The SuperJob vacancy search over a list of objects, with the limits seen on 2026-10-12:
    at most 40 per page whatever `count` asks and nothing past the 500th record of a query."""

    def __init__(self, objects: list[dict[str, Any]]) -> None:
        self.objects = copy.deepcopy(objects)
        self.requests: list[httpx.Request] = []

    def matches(self, obj: dict[str, Any], params: httpx.QueryParams) -> bool:
        if "catalogues" in params and int(params["catalogues"]) not in {c["id"] for c in obj["catalogues"]}:
            return False
        phrase = params.get("keywords[0][keys]")
        if phrase is not None and phrase.lower() not in obj.get("profession", "").lower():
            return False
        lo = int(params.get("date_published_from", 0))
        hi = int(params.get("date_published_to", 2**40))
        return lo <= obj["date_published"] <= hi

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        params = request.url.params
        found = sorted(
            (o for o in self.objects if self.matches(o, params)),
            key=lambda o: o["date_published"],
            reverse=params.get("order_direction", "desc") == "desc",
        )
        size = min(int(params["count"]), SERVED_PER_PAGE)
        start = int(params["page"]) * size
        end = min(start + size, superjob.MAX_RECORDS, len(found))
        body = {"objects": found[start:end], "total": len(found), "more": end < min(len(found), superjob.MAX_RECORDS)}
        return httpx.Response(200, json=body)

    def fetcher(self) -> Fetcher:
        return Fetcher(httpx.Client(transport=httpx.MockTransport(self.handle)), pause_s=0)


def _rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in sorted(root.glob("superjob/dt=*/part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return rows


def _many(n: int, start: int = 1791000000, step: int = 600) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        obj = copy.deepcopy(IT_COMPANY)
        obj["id"] = 60000000 + i
        obj["date_published"] = start + i * step
        obj["profession"] = f"Инженер техподдержки {i}"
        out.append(obj)
    return out


def test_contract_on_the_fixture() -> None:
    page = SuperjobResponse.model_validate(FIXTURE)
    assert (page.total, page.more) == (3, False)
    vacancies = [SuperjobVacancy.model_validate(o) for o in page.objects]
    assert [v.id for v in vacancies] == [50000001, 50000002, 50000003]
    broken = copy.deepcopy(IT_COMPANY)
    broken["date_published"] = "вчера"
    with pytest.raises(ValidationError):
        SuperjobVacancy.model_validate(broken)


def test_cleaner_removes_contacts_place_and_person_employers() -> None:
    company, counts = clean_superjob_vacancy(IT_COMPANY)
    assert counts == {"phone": 2, "address": 1, "email": 1, "person_name": 1}
    assert company["address"] == "Москва"
    for key in ("phone", "phones", "fax", "faxes", "latitude", "longitude", "metro"):
        assert key not in company
    assert company["firm_name"] == "ООО Пример Софт"
    assert company["client"]["link"] == IT_COMPANY["client"]["link"]
    assert company["id_client"] == 1000001

    sole, counts = clean_superjob_vacancy(SOLE_PROPRIETOR)
    assert sole["firm_name"] == "ИП"
    assert sole["client"]["title"] == "ИП"
    assert "id_client" not in sole and "firm_activity" not in sole
    assert set(sole["client"]) & {"id", "link", "description", "address", "addresses"} == set()
    assert sole["address"] == "Казань"
    assert counts == {"phone": 2, "address": 3, "employer_name": 1, "company_code": 1}

    person, counts = clean_superjob_vacancy(PERSON)
    assert person["firm_name"] == "[name]"
    assert person["client"]["title"] == "[name]"
    assert "link" not in person["client"]
    assert counts["employer_name"] == 1
    for cleaned in (company, sole, person):
        assert scan_record({"payload": cleaned}) == {}


@pytest.mark.parametrize(
    ("firm_name", "client_title", "expected"),
    [
        (
            "Альфа Тех (ИП Сидоров Олег Петрович)",
            "Альфа Тех (ИП Сидоров Олег Петрович)",
            ("Альфа Тех (ИП [name])",) * 2,
        ),
        ("Сидоров", "Сидоров Олег Петрович", ("[name]", "[name]")),
        ("ИП Сидоров Олег Петрович", "Альфа Тех", ("ИП", "Альфа Тех")),
    ],
)
def test_person_employer_variants(firm_name: str, client_title: str, expected: tuple[str, str]) -> None:
    vacancy = copy.deepcopy(IT_COMPANY)
    vacancy["firm_name"] = firm_name
    vacancy["client"]["title"] = client_title
    cleaned, counts = clean_superjob_vacancy(vacancy)
    assert (cleaned["firm_name"], cleaned["client"]["title"]) == expected
    assert "link" not in cleaned["client"] and "id_client" not in cleaned
    assert counts["employer_name"] == 1
    assert scan_record({"payload": cleaned}) == {}


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("Российская Федерация, г. Москва, г. Зеленоград, к. 1001Б", "Российская Федерация, г. Москва, г. Зеленоград"),
        ("Москва, Примерный р-н, кв-л 12", "Москва, Примерный р-н"),
        ("Самара, Примерная улица, 5", "Самара"),
        ("ул. Примерная, 5", None),
    ],
)
def test_address_keeps_the_town_only(address: str, expected: str | None) -> None:
    vacancy = copy.deepcopy(IT_COMPANY)
    vacancy["address"] = address
    cleaned, counts = clean_superjob_vacancy(vacancy)
    assert cleaned.get("address") == expected
    assert counts["address"] == 1


def test_first_run_writes_clean_records(tmp_path: Path) -> None:
    run = superjob.collect(Api(OBJECTS).fetcher(), root=tmp_path)
    assert run["distinct_vacancies"] == 3
    assert run["catalogue_coverage"] == {"n": 2, "denominator": 2}
    assert run["only_by_title"] == 1
    rows = {r["source_id"]: r for r in _rows(tmp_path)}
    assert set(rows) == {"50000001", "50000002", "50000003"}
    for row in rows.values():
        assert scan_record({**row, "payload": json.loads(row["payload"])}) == {}
    first = rows["50000001"]
    assert first["title"] == "Python-разработчик"
    assert first["url"] == IT_COMPANY["link"]
    assert first["text"].startswith("Обязанности:\n- Разработка сервисов на Python")
    assert "[email]" in first["text"] and "[name]" in first["text"]
    assert dict(first["pii_removed"]) == {"phone": 2, "address": 1, "email": 1, "person_name": 1}
    payload = json.loads(first["payload"])
    assert "vacancyRichText" not in payload and "candidat" not in payload
    assert payload["_queries"] == ["catalogue_33", "title_python", "title_разработчик"]
    assert rows["50000002"]["text"].endswith("Звоните [phone].")
    assert rows["50000003"]["text"] == (
        "Обязанности: Доработка конфигураций 1С.\n\nТребования: Опыт с 1С от года.\n\nУсловия: Удалённая работа."
    )
    assert json.loads(rows["50000003"]["payload"])["_queries"] == ["title_1С", "title_программист"]


def test_query_above_the_cap_is_split_into_date_windows(tmp_path: Path) -> None:
    api = Api(_many(1100))
    run = superjob.collect(api.fetcher(), root=tmp_path)
    catalogue = run["queries"]["catalogue_33"]
    assert catalogue["total"] == 1100
    assert catalogue["windows"] >= 3
    assert catalogue["truncated"] is False
    assert run["catalogue_coverage"] == {"n": 1100, "denominator": 1100}
    assert run["written"] == 1100
    windowed = [r.url.params for r in api.requests if "date_published_from" in r.url.params]
    assert windowed and all(int(p["date_published_to"]) >= int(p["date_published_from"]) for p in windowed)


def test_window_that_cannot_be_split_is_marked_truncated(tmp_path: Path) -> None:
    run = superjob.collect(Api(_many(600, step=0)).fetcher(), root=tmp_path)
    catalogue = run["queries"]["catalogue_33"]
    assert catalogue["truncated"] is True
    assert run["catalogue_coverage"] == {"n": 500, "denominator": 600}


def test_rerun_and_raise_write_nothing_new_but_an_edit_does(tmp_path: Path) -> None:
    superjob.collect(Api(OBJECTS).fetcher(), root=tmp_path)
    raised = copy.deepcopy(OBJECTS)
    raised[0]["date_published"] += 3600
    raised[0]["date_pub_to"] += 3600
    raised[0]["client"]["vacancy_count"] = 9
    raised[0]["highlight"] = False
    run = superjob.collect(Api(raised).fetcher(), root=tmp_path)
    assert (run["written"], run["duplicates_skipped"]) == (0, 3)
    raised[1]["payment_from"] = 90000
    run = superjob.collect(Api(raised).fetcher(), root=tmp_path)
    assert (run["written"], run["duplicates_skipped"]) == (1, 2)


def test_broken_object_is_rejected(tmp_path: Path) -> None:
    objects = copy.deepcopy(OBJECTS)
    del objects[0]["profession"]
    objects[1]["id"] = "not a number"
    run = superjob.collect(Api(objects).fetcher(), root=tmp_path)
    assert run["written"] == 1
    assert (run["rejected"], run["objects_without_id"]) == (1, 2)
    reasons = [
        json.loads(line)["reason"]
        for line in next(tmp_path.glob("superjob/_rejects/*.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    assert reasons == ["profession: missing"]


def test_summarize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    superjob.collect(Api(OBJECTS).fetcher(), root=tmp_path)
    doc = json.loads(superjob.summarize(root=tmp_path).read_text(encoding="utf-8"))
    assert doc["metrics"]["last_run_catalogue_coverage"] == {"value": 1.0, "n": 2, "denominator": 2}
    assert doc["metrics"]["last_run_only_by_title"] == {"value": 1, "n": 1, "denominator": 3}
    assert doc["days"][0]["rows"] == 3
    assert doc["days"][0]["pii_removed"]["employer_name"] == 2


def test_run_without_key_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(superjob, "setup_logging", lambda: None)
    monkeypatch.delenv(superjob.KEY_VAR, raising=False)

    def no_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no request without a key")

    monkeypatch.setattr(superjob, "make_client", no_network)
    monkeypatch.setattr(superjob, "summarize", no_network)
    assert superjob.main([]) == 0
