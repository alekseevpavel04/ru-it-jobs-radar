import json
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from radar.collect.envelope import RawRecord, content_hash, parse_source_time
from radar.collect.writer import PartitionWriter, is_complete, partition_dir, write_records

DT = date(2026, 6, 30)
FETCHED = datetime(2026, 10, 8, 10, 56, 5, tzinfo=UTC)


def _record(source_id: str, text: str = "Обязанности: код.") -> RawRecord:
    payload = {"id": source_id, "dateModify": "2026-06-28T10:00:00+0300"}
    return RawRecord(
        source="trudvsem_snapshot",
        source_id=source_id,
        fetched_at=FETCHED,
        published_at=parse_source_time("2026-06-20T09:15:00+0300"),
        modified_at=parse_source_time("2026-06-28"),
        url=None,
        title="Программист",
        text=text,
        payload=payload,
        content_hash=content_hash("Программист", text, payload, ignore=["dateModify"]),
        pii_removed={"phone": 2},
    )


def _rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for part in sorted(directory.glob("part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return rows


def test_write_and_read_back(tmp_path: Path) -> None:
    writer = write_records([_record("a"), _record("b")], "trudvsem_snapshot", DT, root=tmp_path)
    directory = partition_dir("trudvsem_snapshot", DT, tmp_path)
    assert writer.written == 2
    assert is_complete("trudvsem_snapshot", DT, tmp_path)
    rows = _rows(directory)
    assert [r["source_id"] for r in rows] == ["a", "b"]
    assert rows[0]["schema_version"] == 1
    assert rows[0]["fetched_at"] == FETCHED
    assert rows[0]["modified_at"] == datetime(2026, 6, 27, 21, 0, tzinfo=UTC)
    assert rows[0]["pii_removed"] == [("phone", 2)]
    assert json.loads(str(rows[0]["payload"]))["id"] == "a"
    assert not list(directory.glob(".*"))


def test_same_key_same_date_is_written_once(tmp_path: Path) -> None:
    write_records([_record("a"), _record("a")], "trudvsem_snapshot", DT, root=tmp_path)
    second = write_records([_record("a"), _record("b")], "trudvsem_snapshot", DT, root=tmp_path)
    directory = partition_dir("trudvsem_snapshot", DT, tmp_path)
    assert sorted(r["source_id"] for r in _rows(directory)) == ["a", "b"]
    assert second.duplicates == 1
    assert second.written == 1


def test_changed_content_is_a_new_key(tmp_path: Path) -> None:
    write_records([_record("a")], "trudvsem_snapshot", DT, root=tmp_path)
    write_records([_record("a", text="Обязанности: другой код.")], "trudvsem_snapshot", DT, root=tmp_path)
    assert len(_rows(partition_dir("trudvsem_snapshot", DT, tmp_path))) == 2


def test_stamp_only_change_keeps_the_hash() -> None:
    payload = {"id": "a", "dateModify": "2026-06-28T10:00:00+0300"}
    later = {**payload, "dateModify": "2026-07-01T10:00:00+0300"}
    assert content_hash("t", "x", payload, ignore=["dateModify"]) == content_hash(
        "t", "x", later, ignore=["dateModify"]
    )
    assert content_hash("t", "x", payload) != content_hash("t", "x", later)
    assert content_hash("T", " x\n", payload) == content_hash("t", "x", payload)


def test_existing_parts_are_never_touched(tmp_path: Path) -> None:
    write_records([_record("a")], "trudvsem_snapshot", DT, root=tmp_path)
    directory = partition_dir("trudvsem_snapshot", DT, tmp_path)
    first = next(directory.glob("part-*.parquet"))
    before = (first.stat().st_mtime_ns, first.read_bytes())
    write_records([_record("b")], "trudvsem_snapshot", DT, root=tmp_path)
    assert (first.stat().st_mtime_ns, first.read_bytes()) == before
    assert len(list(directory.glob("part-*.parquet"))) == 2


def test_failed_run_leaves_parts_without_marker(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), PartitionWriter("trudvsem_snapshot", DT, root=tmp_path, batch_rows=1) as w:
        w.add(_record("a"))
        raise RuntimeError("network down")
    assert not is_complete("trudvsem_snapshot", DT, tmp_path)
    resumed = write_records([_record("a"), _record("b")], "trudvsem_snapshot", DT, root=tmp_path)
    assert resumed.already_present == 1
    assert resumed.written == 1
    assert is_complete("trudvsem_snapshot", DT, tmp_path)


def test_rejects_log_has_id_and_reason_only(tmp_path: Path) -> None:
    with PartitionWriter("trudvsem_snapshot", DT, root=tmp_path) as writer:
        writer.reject("x1", "dateModify: datetime_from_date_parsing")
    lines = (
        (tmp_path / "trudvsem_snapshot" / "_rejects" / "dt=2026-06-30.jsonl").read_text(encoding="utf-8").splitlines()
    )
    entry = json.loads(lines[0])
    assert set(entry) == {"at", "source_id", "reason"}
    assert entry["source_id"] == "x1"
    marker = json.loads((partition_dir("trudvsem_snapshot", DT, tmp_path) / "_SUCCESS").read_text(encoding="utf-8"))
    assert marker["rejected"] == 1


def test_wrong_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="habr"), PartitionWriter("trudvsem", DT, root=tmp_path) as writer:
        writer.add(_record("a").model_copy(update={"source": "habr"}))


def test_envelope_rejects_naive_time_and_bad_hash() -> None:
    good = _record("a").model_dump()
    with pytest.raises(ValueError):
        RawRecord.model_validate({**good, "fetched_at": datetime(2026, 10, 8, 10, 0)})
    with pytest.raises(ValueError):
        RawRecord.model_validate({**good, "content_hash": "abc"})
    with pytest.raises(ValueError):
        RawRecord.model_validate({**good, "source": "hh"})
