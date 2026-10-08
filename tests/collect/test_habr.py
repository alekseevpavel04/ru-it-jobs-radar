import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
import pytest

import radar.results
from radar.collect import habr
from radar.collect.http import Fetcher
from radar.pii import scan_record

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "habr"
LIST = json.loads((FIXTURES / "vacancies_page.json").read_text(encoding="utf-8"))
FILTERS = (FIXTURES / "vacancies_filters_page.html").read_text(encoding="utf-8")
CARD = (FIXTURES / "vacancy_card_1000150058.html").read_text(encoding="utf-8")
IDS = [item["id"] for item in LIST["list"]]


class Site:
    """Recorded Habr responses behind a mock transport; tests tweak what it serves."""

    def __init__(self) -> None:
        self.list = copy.deepcopy(LIST)
        self.capped_group: str | None = None
        self.card_status: dict[int, int] = {}
        self.card_error: Exception | None = None
        self.card_for: Callable[[int], int] = lambda vid: vid
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/frontend/vacancies":
            body = copy.deepcopy(self.list)
            specs = request.url.params.get_list("s[]")
            capped = self.capped_group is not None and specs and specs[0] == self.capped_group and len(specs) > 1
            body["meta"]["totalPages"] = 21 if capped else 1
            if specs:
                body["meta"]["totalResults"] = 25
            return httpx.Response(200, json=body)
        if path == "/vacancies":
            return httpx.Response(200, text=FILTERS)
        vid = int(path.rsplit("/", 1)[1])
        if self.card_error is not None:
            raise self.card_error
        status = self.card_status.get(vid, 200)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, text=CARD.replace("1000150058", str(self.card_for(vid))))

    def fetcher(self) -> Fetcher:
        return Fetcher(httpx.Client(transport=httpx.MockTransport(self.handle)), pause_s=0)

    def card_requests(self) -> list[str]:
        return [r.url.path for r in self.requests if r.url.path.startswith("/vacancies/")]


def _rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in sorted(root.glob("habr/dt=*/part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return rows


def test_first_run(tmp_path: Path) -> None:
    site = Site()
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert run["site_counter"] == 1160
    assert run["coverage"] == {"n": 3, "denominator": 1160}
    assert len(run["passes"]) == 3 + 17
    assert run["cards"] == {"fetched": 3, "carried": 0, "gone": 0, "failed": 0, "deferred": 0, "blocked": False}
    rows = _rows(tmp_path)
    assert sorted(r["source_id"] for r in rows) == sorted(str(i) for i in IDS)
    for row in rows:
        payload = json.loads(row["payload"])
        assert scan_record({**row, "payload": payload}) == {}
        assert row["text"].startswith("О компании и команде")
        assert row["published_at"] == datetime(2026, 9, 16, 21, 0, tzinfo=UTC)
        assert payload["card"]["validThrough"]
        assert "all_by_date" in payload["_passes"]
    sole = next(json.loads(r["payload"]) for r in rows if r["source_id"] == "1000000002")
    assert sole["company"] == {"title": "ИП", "accredited": False}


def test_same_day_rerun_reads_no_cards_and_writes_nothing(tmp_path: Path) -> None:
    habr.collect(Site().fetcher(), root=tmp_path)
    site = Site()
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert run["cards"] == {"fetched": 0, "carried": 3, "gone": 0, "failed": 0, "deferred": 0, "blocked": False}
    assert site.card_requests() == []
    assert (run["written"], run["duplicates_skipped"]) == (0, 3)


def test_changed_list_entry_refetches_its_card(tmp_path: Path) -> None:
    habr.collect(Site().fetcher(), root=tmp_path)
    site = Site()
    site.list["list"][1]["title"] = "Python backend (senior)"
    site.list["list"][2]["reactions"]["items"][0]["count"] = 99
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert site.card_requests() == [f"/vacancies/{IDS[1]}"]
    assert (run["written"], run["duplicates_skipped"]) == (1, 2)


def test_missing_and_foreign_cards(tmp_path: Path) -> None:
    site = Site()
    site.card_status[IDS[0]] = 410
    site.card_for = lambda vid: 1000150058 if vid == IDS[1] else vid
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert run["cards"] == {"fetched": 1, "carried": 0, "gone": 1, "failed": 1, "deferred": 0, "blocked": False}
    texts = {r["source_id"]: r["text"] for r in _rows(tmp_path)}
    assert texts[str(IDS[0])] is None
    assert texts[str(IDS[1])] is None
    again = Site()
    habr.collect(again.fetcher(), root=tmp_path)
    assert sorted(again.card_requests()) == sorted(f"/vacancies/{i}" for i in IDS[:2])


def test_capped_group_is_split_by_specialization(tmp_path: Path) -> None:
    development = next(specs for title, specs in habr.parse_groups(FILTERS) if title == "Разработка")
    site = Site()
    site.capped_group = str(development[0])
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert run["capped_passes"] == ["group:Разработка"]
    assert len(run["passes"]) == 3 + 17 + len(development)


def test_list_format_change_stops_the_run(tmp_path: Path) -> None:
    site = Site()
    del site.list["meta"]["perPage"]
    with pytest.raises(ValueError):
        habr.collect(site.fetcher(), root=tmp_path)


def test_broken_list_item_is_rejected(tmp_path: Path) -> None:
    site = Site()
    site.list["list"][0]["id"] = "not a number"
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert run["written"] == 2
    assert run["rejected"] == 1
    reasons = {
        json.loads(line)["reason"] for line in next(tmp_path.glob("habr/_rejects/*.jsonl")).open(encoding="utf-8")
    }
    assert reasons == {"id: int_type"}


def test_html_to_text() -> None:
    html = "<h3>О нас</h3><p>Мы&nbsp;делаем <strong>ML</strong>.<br>Ещё строка</p><ul><li>Python</li><li>SQL</li></ul>"
    assert habr.html_to_text(html) == "О нас\nМы делаем ML.\nЕщё строка\n- Python\n- SQL"
    assert habr.html_to_text("") == ""


def test_summarize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    habr.collect(Site().fetcher(), root=tmp_path)
    doc = json.loads(habr.summarize(root=tmp_path).read_text(encoding="utf-8"))
    assert doc["metrics"]["last_run_coverage_of_site_counter"] == {"value": 0.0026, "n": 3, "denominator": 1160}
    assert doc["days"][0]["rows"] == 3


def test_card_cap_reads_newest_first_and_defers_the_rest(tmp_path: Path) -> None:
    site = Site()
    run = habr.collect(site.fetcher(), root=tmp_path, max_cards=1)
    newest = max(LIST["list"], key=lambda i: i["publishedDate"]["date"])["id"]
    assert site.card_requests() == [f"/vacancies/{newest}"]
    assert (run["cards"]["fetched"], run["cards"]["deferred"], run["written"]) == (1, 2, 3)
    later = Site()
    habr.collect(later.fetcher(), root=tmp_path)
    assert len(later.card_requests()) == 2


def test_blocked_site_still_writes_the_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(habr.time, "sleep", lambda _s: None)
    monkeypatch.setattr("radar.collect.http.time.sleep", lambda _s: None)
    site = Site()
    site.card_error = httpx.ConnectError("handshake failed")
    run = habr.collect(site.fetcher(), root=tmp_path)
    assert run["cards"]["blocked"] is True
    assert (run["cards"]["fetched"], run["cards"]["deferred"]) == (0, 3)
    assert len(site.card_requests()) == 4
    assert {r["text"] for r in _rows(tmp_path)} == {None}
    assert (tmp_path / "habr" / f"dt={run['dt']}" / "_SUCCESS").exists()


def test_changed_vacancy_waiting_for_its_card_keeps_the_old_text(tmp_path: Path) -> None:
    habr.collect(Site().fetcher(), root=tmp_path)
    site = Site()
    site.list["list"][1]["title"] = "Python backend (senior)"
    habr.collect(site.fetcher(), root=tmp_path, max_cards=0)
    changed = [r for r in _rows(tmp_path) if r["title"] == "Python backend (senior)"]
    assert len(changed) == 1
    assert changed[0]["text"].startswith("О компании и команде")
    later = Site()
    later.list["list"][1]["title"] = "Python backend (senior)"
    habr.collect(later.fetcher(), root=tmp_path)
    assert later.card_requests() == [f"/vacancies/{IDS[1]}"]
