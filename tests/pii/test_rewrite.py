import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from radar.collect.envelope import RawRecord, content_hash
from radar.collect.summary import read_meta
from radar.collect.telegram import MSK, VOLATILE, keep_handles
from radar.collect.writer import PartitionWriter, partition_dir
from radar.pii import rewrite

KEEP = keep_handles()
NOW = datetime.now(UTC)
TODAY = NOW.astimezone(MSK).date()
PAST = TODAY - timedelta(days=7)
# Stored by the cleaner before name lines were caught: the nick is gone, the name line is not.
WITH_NAME = "Иванова Мария\nДата-инженер, удалённо\nОтклик: [nick]"
CLEAN = "Python-разработчик, офис в Казани"


def record(post_id: int, text: str, pii: dict[str, int] | None = None) -> RawRecord:
    payload: dict[str, Any] = {"channel": "job_python", "views": post_id, "rule_kind": "vacancy"}
    return RawRecord(
        source="telegram",
        source_id=f"job_python/{post_id}",
        fetched_at=NOW,
        published_at=NOW - timedelta(days=30),
        modified_at=None,
        url=f"https://t.me/job_python/{post_id}",
        title=None,
        text=text,
        payload=payload,
        content_hash=content_hash(None, text, payload, ignore=VOLATILE),
        pii_removed=pii or {},
        is_vacancy=True,
        multi_vacancy=False,
    )


def write(root: Path, dt: Any, records: list[RawRecord], age: timedelta | None = None) -> list[Path]:
    with PartitionWriter("telegram", dt, root=root) as writer:
        writer.add_many(records)
    if age is not None:
        stamp = time.time() - age.total_seconds()
        for part in writer.parts:
            os.utime(part, (stamp, stamp))
    return writer.parts


def rows(part: Path) -> list[dict[str, Any]]:
    return pq.read_table(part).to_pylist()


def test_rewrites_old_parts_keeps_rows_and_skips_fresh_ones(tmp_path: Path) -> None:
    hour = timedelta(hours=1)
    (old,) = write(tmp_path, PAST, [record(1, WITH_NAME, {"nick": 1}), record(2, CLEAN)], age=hour)
    (fresh,) = write(tmp_path, PAST, [record(3, WITH_NAME)])
    marker = (partition_dir("telegram", PAST, tmp_path) / "_SUCCESS").read_bytes()
    (today,) = write(tmp_path, TODAY, [record(4, WITH_NAME)], age=hour)
    before = rows(old)
    untouched = {p: p.read_bytes() for p in (fresh, today)}

    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)

    after = rows(old)
    assert [r["source_id"] for r in after] == [r["source_id"] for r in before]
    changed, same = after
    assert changed["text"] == "[name]\nДата-инженер, удалённо\nОтклик: [nick]"
    assert dict(changed["pii_removed"]) == {"nick": 1, "person_name": 1}
    payload = json.loads(changed["payload"])
    assert changed["content_hash"] == content_hash(None, changed["text"], payload, ignore=VOLATILE)
    assert changed["content_hash"] != before[0]["content_hash"]
    for key in ("fetched_at", "published_at", "url", "payload", "is_vacancy"):
        assert changed[key] == before[0][key]
    assert same == before[1]
    assert {p: p.read_bytes() for p in (fresh, today)} == untouched
    assert (partition_dir("telegram", PAST, tmp_path) / "_SUCCESS").read_bytes() == marker
    assert not list(tmp_path.rglob("*.tmp"))

    assert summary["rows_checked"] == 2
    assert summary["rows_changed"] == 1
    assert summary["pii_added"] == {"person_name": 1}
    assert summary["hash_not_reproduced"] == 0
    assert summary["parts_rewritten"] == 1
    assert {(s["dt"], s.get("part"), s["reason"]) for s in summary["skipped"]} == {
        (PAST.isoformat(), fresh.name, "modified in the last 30 minutes"),
        (TODAY.isoformat(), None, "partition of today (Moscow)"),
    }
    (entry,) = read_meta("telegram", rewrite.META, tmp_path)
    assert entry["part"] == old.name and entry["rows_changed"] == 1 and entry["pii_added"] == {"person_name": 1}

    again = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)
    assert again["rows_changed"] == 0
    assert rows(old) == after


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    (part,) = write(tmp_path, PAST, [record(1, WITH_NAME)], age=timedelta(hours=1))
    content = part.read_bytes()
    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW, dry_run=True)
    assert summary["rows_changed"] == 1
    assert part.read_bytes() == content
    assert read_meta("telegram", rewrite.META, tmp_path) == []


def test_versions_that_differed_only_in_a_name_keep_both_rows(tmp_path: Path) -> None:
    other = WITH_NAME.replace("Иванова Мария", "Петрова Анна")
    (part,) = write(tmp_path, PAST, [record(1, WITH_NAME), record(1, other)], age=timedelta(hours=1))
    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)
    after = rows(part)
    assert len(after) == 2
    assert after[0]["content_hash"] == after[1]["content_hash"]
    assert summary["duplicate_keys_in_rewritten_partitions"] == 1


def test_scan_counts_what_is_left(tmp_path: Path) -> None:
    write(tmp_path, PAST, [record(1, WITH_NAME), record(2, CLEAN), record(3, "Пишите @someone_private")])
    found = rewrite.scan("telegram", KEEP, tmp_path)
    assert found["rows"] == 3
    assert found["rows_with_findings"] == 2
    assert found["findings"] == {"nick": 1, "person_name": 1}
    assert found["person_name_by_rule"] == {"name_line": 1}


def test_verify_refuses_a_lost_row(tmp_path: Path) -> None:
    (part,) = write(tmp_path, PAST, [record(1, WITH_NAME), record(2, CLEAN)])
    table = pq.read_table(part)
    with pytest.raises(rewrite.RewriteError, match="rows"):
        rewrite._verify(table, table.slice(0, 1))
    with pytest.raises(rewrite.RewriteError, match="column url"):
        rewrite._verify(table, table.set_column(6, "url", table.column("title")))
