import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

import radar.results
from radar.collect import trudvsem_snapshot as ts
from radar.collect.writer import is_complete, partition_dir
from radar.pii import scan_record

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trudvsem"
SNAPSHOT = (FIXTURES / "snapshot_sample.xml").read_bytes()
PASSPORT = (FIXTURES / "passport_sample.xml").read_bytes()
VERSION = ts.Version("20260630T135247", "https://opendata.trudvsem.ru/x/data-20260630T135247.xml")
FETCHED = datetime(2026, 10, 8, 10, 56, 5, tzinfo=UTC)


def _with_control_char(xml: bytes) -> bytes:
    return xml.replace("Консультирование покупателей.".encode(), "Консультирование\x01 покупателей.".encode())


def test_passport_versions_oldest_first() -> None:
    versions = ts.parse_passport(PASSPORT)
    assert [v.created for v in versions] == ["20260630T135247", "20260701T135325", "20261007T135128"]
    assert versions[0].dt == date(2026, 6, 30)
    assert versions[0].url.endswith("data-20260630T135247-structure-20161130T143000.xml")


def test_process_file(tmp_path: Path) -> None:
    xml = tmp_path / "snapshot.xml"
    xml.write_bytes(_with_control_char(SNAPSHOT))
    summary = ts.process_file(xml, VERSION, FETCHED, root=tmp_path)
    assert (summary["vacancies_in_file"], summary["written"], summary["rejected"]) == (3, 2, 1)
    assert summary["control_bytes_dropped"] == 1
    assert summary["pii_removed"]["contact_person"] == 2
    assert is_complete("trudvsem_snapshot", VERSION.dt, tmp_path)

    rows = pq.read_table(next(partition_dir("trudvsem_snapshot", VERSION.dt, tmp_path).glob("part-*.parquet")))
    first, sole = rows.to_pylist()
    assert first["source"] == "trudvsem_snapshot"
    assert first["fetched_at"] == FETCHED
    assert first["published_at"] == datetime(2026, 6, 20, 6, 15, tzinfo=UTC)
    assert first["text"].count("Опыт разработки на Python") == 1
    assert first["text"].startswith("Обязанности: Разработка сервисов на Python.")
    payload = json.loads(first["payload"])
    assert payload["snapshot_version"] == "20260630T135247"
    assert not {"requirements", "qualifications", "responsibilities", "organization", "region"} & set(payload)
    assert sole["url"] is None
    assert "Консультирование покупателей." in sole["text"]
    for row in (first, sole):
        assert scan_record({**row, "payload": json.loads(row["payload"])}) == {}

    again = ts.process_file(xml, VERSION, FETCHED, root=tmp_path)
    assert (again["written"], again["duplicates_skipped"]) == (0, 2)
    assert again["pii_removed"] == summary["pii_removed"]


class _Breaks(httpx.SyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self) -> Iterator[bytes]:
        yield self.data
        raise httpx.ReadError("connection reset")


def _server(body: bytes, break_at: int | None = None, gone: bool = False) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if gone:
            return httpx.Response(404)
        if request.url.path.endswith("/"):
            return httpx.Response(200, content=PASSPORT)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(body))})
        rng = request.headers.get("range")
        if rng:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=body[start:])
        if break_at is not None:
            return httpx.Response(200, stream=_Breaks(body[:break_at]))
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handle)), seen


def test_download_resumes_after_a_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ts, "CHUNK", 100)
    client, seen = _server(SNAPSHOT, break_at=1000)
    result = ts.download(client, VERSION, tmp_path)
    assert result.path.read_bytes() == SNAPSHOT
    assert result.bytes_transferred == len(SNAPSHOT)
    assert result.runs == 1
    assert result.seconds is not None
    assert [r.headers.get("range") for r in seen if r.method == "GET"] == [None, "bytes=1000-"]
    assert not list(tmp_path.glob("*.part"))

    again = ts.download(client, VERSION, tmp_path)
    assert again.reused
    assert again.fetched_at == result.fetched_at
    assert again.seconds == result.seconds


def test_download_of_a_foreign_partial_file_has_no_timing(tmp_path: Path) -> None:
    (tmp_path / "data-20260630T135247.xml.part").write_bytes(SNAPSHOT[:500])
    client, seen = _server(SNAPSHOT)
    result = ts.download(client, VERSION, tmp_path)
    assert result.path.read_bytes() == SNAPSHOT
    assert result.seconds is None
    assert [r.headers.get("range") for r in seen if r.method == "GET"] == ["bytes=500-"]


def test_gone_version(tmp_path: Path) -> None:
    client, _ = _server(SNAPSHOT, gone=True)
    with pytest.raises(ts.GoneError):
        ts.download(client, VERSION, tmp_path)


def test_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts.time, "sleep", lambda _s: None)
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(ts, "LOG_PATH", tmp_path / "absent.log")
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    client, _ = _server(SNAPSHOT)
    tmp_dir = tmp_path / "data" / "tmp"

    assert ts._run(client, limit=2, keep_xml=False, tmp_dir=tmp_dir) == 2
    assert is_complete("trudvsem_snapshot", date(2026, 7, 1), tmp_path / "data" / "raw")
    assert not list(tmp_dir.iterdir())
    result = json.loads((tmp_path / "results" / ts.RESULT_PATH).read_text(encoding="utf-8"))
    assert result["metrics"]["snapshots_complete"]["value"] == 2
    assert [s["version"] for s in result["snapshots"]] == ["20260630T135247", "20260701T135325"]
    first = result["snapshots"][0]
    assert (first["rows"], first["vacancies_in_file"], first["rejected_not_written"]) == (2, 3, 1)
    assert first["rejected_ids"] == ["00000000-0000-4000-8000-0000000000a3"]
    assert first["download"]["source"] == "collector"
    assert first["pii_removed"]["contact_person"] == 2

    assert ts._run(client, limit=None, keep_xml=False, tmp_dir=tmp_dir) == 1
    result = json.loads((tmp_path / "results" / ts.RESULT_PATH).read_text(encoding="utf-8"))
    assert result["metrics"]["snapshots_complete"]["value"] == 3
    assert result["metrics"]["rows_lost"] == {"value": 0.333333333, "n": 3, "denominator": 9}
    assert result["metrics"]["downloads_timed"] == {"value": 3, "n": 3, "denominator": 3}
