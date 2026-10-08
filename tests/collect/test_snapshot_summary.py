import json
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import radar.results
from radar.collect import trudvsem_snapshot as ts
from radar.collect.summary import partition_stats

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trudvsem"
SNAPSHOT = (FIXTURES / "snapshot_sample.xml").read_bytes()
VERSION = ts.Version("20260630T135247", "x")
FETCHED = datetime(2026, 10, 8, 10, 56, 5, tzinfo=UTC)
DT = date(2026, 6, 30)

LOG = """\
2026-10-08 14:33:34,283 INFO version 20260701T135325: download
2026-10-08 14:35:21,589 INFO version 20260701T135325: 3818659995 bytes in 106 s, parse
2026-10-08 14:42:19,389 INFO version 20260702T135403: download
2026-10-08 14:44:53,830 INFO version 20260702T135403: 3822393515 bytes in 153 s, parse
2026-10-08 14:46:39,391 INFO version 20260702T135403: download
2026-10-08 14:46:40,410 INFO version 20260702T135403: 3822393515 bytes in 0 s (file from an earlier run), parse
2026-10-08 14:53:38,644 INFO version 20260703T135823: download
2026-10-08 14:54:25,745 INFO version 20260703T135823: download
2026-10-08 14:55:23,000 INFO version 20260703T135823: 3803079962 bytes in 57 s, parse
"""


def test_log_timings_only_for_uninterrupted_downloads(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_text(LOG, encoding="utf-8")
    timings = ts.downloads_from_log(log)
    assert timings == {
        "20260701T135325": {"file_bytes": 3818659995, "seconds": 106, "source": "log"},
        "20260702T135403": {"file_bytes": 3822393515, "seconds": 153, "source": "log"},
    }


def _partition_with_lost_record(tmp_path: Path) -> Path:
    """A partition as the first probe left it: one record rejected, the rest written."""
    xml = tmp_path / "tmp" / "data-20260630T135247.xml"
    xml.parent.mkdir()
    xml.write_bytes(SNAPSHOT)
    xml.with_suffix(".fetched_at").write_text(FETCHED.isoformat(), encoding="utf-8")
    ts.process_file(xml, VERSION, FETCHED, root=tmp_path / "raw")
    return xml


def test_repair_appends_only_missing_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xml = _partition_with_lost_record(tmp_path)
    root = tmp_path / "raw"
    directory = root / "trudvsem_snapshot" / "dt=2026-06-30"
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    # The bad record becomes valid, as the epoch-millisecond fix did for the real one.
    xml.write_bytes(
        SNAPSHOT.replace(
            b"<dateModify>\xd0\xbd\xd0\xb5 \xd0\xb4\xd0\xb0\xd1\x82\xd0\xb0</dateModify>",
            b"<dateModify>1771493955091</dateModify>",
        )
    )

    result = ts.repair(DT, xml.parent, root=root)

    assert (result["rows_appended"], result["present_with_other_hash"], result["still_rejected"]) == (1, 0, 0)
    assert result["appended_ids"] == ["00000000-0000-4000-8000-0000000000a3"]
    after = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert {k: after[k] for k in before} == before
    new_parts = sorted(set(after) - set(before))
    assert len(new_parts) == 1
    assert pq.read_table(directory / new_parts[0]).num_rows == 1

    monkeypatch.setattr(ts, "LOG_PATH", tmp_path / "absent.log")
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    doc = json.loads(ts.summarize(root=root).read_text(encoding="utf-8"))
    entry = doc["snapshots"][0]
    assert (entry["rows"], entry["rejected_not_written"]) == (3, 0)
    assert entry["pii_method"].startswith("recomputed over the local XML")
    assert entry["download"] is None
    assert entry["download_note"].startswith("not measured")


def test_repair_needs_the_local_file(tmp_path: Path) -> None:
    xml = _partition_with_lost_record(tmp_path)
    xml.unlink()
    with pytest.raises(RuntimeError, match="no local XML"):
        ts.repair(DT, xml.parent, root=tmp_path / "raw")


def test_partition_stats_cache_follows_the_parts(tmp_path: Path) -> None:
    _partition_with_lost_record(tmp_path)
    root = tmp_path / "raw"
    first = partition_stats("trudvsem_snapshot", DT, root)
    assert (first["rows"], first["rejected_not_written"]) == (2, 1)
    assert partition_stats("trudvsem_snapshot", DT, root)["computed_at"] == first["computed_at"]
    cache = root / "trudvsem_snapshot" / "_meta" / "stats-dt=2026-06-30.json"
    stale = json.loads(cache.read_text(encoding="utf-8"))
    stale["parts"] = []
    cache.write_text(json.dumps(stale), encoding="utf-8")
    assert partition_stats("trudvsem_snapshot", DT, root)["rows"] == 2
