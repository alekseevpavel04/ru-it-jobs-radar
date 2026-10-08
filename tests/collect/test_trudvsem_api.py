import json
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
import pytest

import radar.results
from radar.collect import trudvsem as tv
from radar.collect.http import Fetcher
from radar.pii import scan_record

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trudvsem"
PAGE = json.loads((FIXTURES / "vacancies_it_page.json").read_text(encoding="utf-8"))


def _fetcher(pages: list[Any] | None = None) -> tuple[Fetcher, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=PAGE if pages is None else pages[len(seen) - 1])

    return Fetcher(httpx.Client(transport=httpx.MockTransport(handle)), pause_s=0), seen


def _rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in sorted(root.glob("trudvsem/dt=*/part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return rows


def test_full_run_writes_cleaned_records_once(tmp_path: Path) -> None:
    fetcher, seen = _fetcher()
    run = tv.collect(fetcher, full=True, root=tmp_path)
    assert run["mode"] == "full"
    assert len(seen) == len(tv.QUERIES)
    assert all("modifiedFrom" not in r.url.params for r in seen)
    assert (run["distinct_vacancies"], run["written"], run["records_fetched"]) == (3, 3, 3 * len(tv.QUERIES))

    rows = _rows(tmp_path)
    assert len(rows) == 3
    for row in rows:
        payload = json.loads(row["payload"])
        assert scan_record({**row, "payload": payload}) == {}
        assert payload["_queries"] == sorted(tv.QUERIES)
    sole = next(r for r in rows if r["source_id"].endswith("0001"))
    assert sole["url"] is None
    assert sole["text"].startswith("Обязанности: Реализация технических задач")
    assert dict(sole["pii_removed"])["contact_person"] == 1


def test_same_day_rerun_does_not_duplicate(tmp_path: Path) -> None:
    tv.collect(_fetcher()[0], full=True, root=tmp_path)
    again = tv.collect(_fetcher()[0], full=True, root=tmp_path)
    assert (again["written"], again["duplicates_skipped"]) == (0, 3)
    assert len(_rows(tmp_path)) == 3


def test_second_run_is_incremental(tmp_path: Path) -> None:
    first = tv.collect(_fetcher()[0], full=True, root=tmp_path)
    fetcher, seen = _fetcher()
    run = tv.collect(fetcher, full=False, root=tmp_path)
    assert run["mode"] == "incremental"
    assert run["modified_from"] < first["started_at"]
    assert all(r.url.params["modifiedFrom"].endswith("Z") for r in seen)


def test_query_above_the_page_cap_is_marked_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tv, "PAGE_SIZE", 1)
    monkeypatch.setattr(tv, "MAX_PAGES", 2)
    monkeypatch.setattr(tv, "QUERIES", {"industry_it": {"industry": "InformationTechnology"}})
    fetcher, seen = _fetcher()
    run = tv.collect(fetcher, full=True, root=tmp_path)
    assert [r.url.params["offset"] for r in seen] == ["0", "1"]
    assert run["queries"]["industry_it"] == {"total": 6567, "pages": 2, "records": 6, "truncated": True}


def test_empty_page_and_broken_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tv, "QUERIES", {"a": {"text": "a"}, "b": {"text": "b"}})
    broken = json.loads(json.dumps(PAGE))
    del broken["results"]["vacancies"][1]["vacancy"]["job-name"]
    empty = {"status": "200", "request": {"api": "v1"}, "meta": {"total": 0, "limit": 100}, "results": {}}
    run = tv.collect(_fetcher([broken, empty])[0], full=True, root=tmp_path)
    assert (run["written"], run["rejected"]) == (2, 1)
    rejects = next(tmp_path.glob("trudvsem/_rejects/*.jsonl")).read_text(encoding="utf-8")
    assert json.loads(rejects)["reason"] == "job-name: missing"


def test_changed_api_format_stops_the_run(tmp_path: Path) -> None:
    changed = json.loads(json.dumps(PAGE))
    changed["meta"]["total"] = "6567"
    with pytest.raises(ValueError):
        tv.collect(_fetcher([changed])[0], full=True, root=tmp_path)
    assert not list(tmp_path.glob("trudvsem/dt=*/part-*.parquet"))


def test_summarize_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    tv.collect(_fetcher()[0], full=True, root=tmp_path)
    doc = json.loads(tv.summarize(root=tmp_path).read_text(encoding="utf-8"))
    assert doc["metrics"]["rows_total"]["value"] == 3
    assert doc["metrics"]["last_run_truncated_queries"] == {"value": 0, "n": 0, "denominator": len(tv.QUERIES)}
    assert doc["days"][0]["pii_removed"]["contact_person"] == 3
