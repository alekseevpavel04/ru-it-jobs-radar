"""Raw-layer writer: `data/raw/<source>/dt=YYYY-MM-DD/part-*.parquet`.

The raw layer cannot be rebuilt (past days are gone at the source), so the writer only ever adds
files: a part is written to a temporary name and renamed into place, existing parts are never
touched, and a key `(source, source_id, content_hash)` already present in the partition is skipped.
A partition is complete when its `_SUCCESS` marker exists; readers should glob `part-*.parquet`.
"""

import json
import os
import secrets
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.collect.envelope import RawRecord
from radar.config import data_dir

_FIELDS: "list[pa.Field[Any]]" = [
    pa.field("schema_version", pa.int16()),
    pa.field("source", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("fetched_at", pa.timestamp("us", tz="UTC")),
    pa.field("published_at", pa.timestamp("us", tz="UTC")),
    pa.field("modified_at", pa.timestamp("us", tz="UTC")),
    pa.field("url", pa.string()),
    pa.field("title", pa.string()),
    pa.field("text", pa.string()),
    pa.field("payload", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("pii_removed", pa.map_(pa.string(), pa.int32())),
    pa.field("is_vacancy", pa.bool_()),
    pa.field("multi_vacancy", pa.bool_()),
]
SCHEMA = pa.schema(_FIELDS)

SUCCESS = "_SUCCESS"


def raw_root() -> Path:
    return data_dir() / "raw"


def partition_dir(source: str, dt: date, root: Path | None = None) -> Path:
    return (root or raw_root()) / source / f"dt={dt.isoformat()}"


def is_complete(source: str, dt: date, root: Path | None = None) -> bool:
    return (partition_dir(source, dt, root) / SUCCESS).exists()


def read_marker(source: str, dt: date, root: Path | None = None) -> dict[str, Any] | None:
    path = partition_dir(source, dt, root) / SUCCESS
    if not path.exists():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def existing_keys(directory: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for part in sorted(directory.glob("part-*.parquet")):
        table = pq.read_table(part, columns=["source", "source_id", "content_hash"])
        keys.update(zip(*(table.column(c).to_pylist() for c in ("source", "source_id", "content_hash")), strict=True))
    return keys


def _row(record: RawRecord) -> dict[str, Any]:
    row = record.model_dump()
    row["payload"] = json.dumps(record.payload, ensure_ascii=False, separators=(",", ":"))
    row["pii_removed"] = list(record.pii_removed.items())
    return row


class PartitionWriter:
    """Append records to one partition in batches; use as a context manager.

    `close(complete=True)` (the normal exit of the `with` block) writes the `_SUCCESS` marker with
    the run's counters. An exception leaves the parts written so far and no marker: a rerun skips
    the keys already on disk and finishes the partition.
    """

    def __init__(self, source: str, dt: date, root: Path | None = None, batch_rows: int = 50_000) -> None:
        self.source = source
        self.dt = dt
        self.dir = partition_dir(source, dt, root)
        self.rejects_path = (root or raw_root()) / source / "_rejects" / f"dt={dt.isoformat()}.jsonl"
        self.batch_rows = batch_rows
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keys = existing_keys(self.dir)
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
        self.batch: list[dict[str, Any]] = []
        self.parts: list[Path] = []
        self.written = 0
        self.duplicates = 0
        self.rejected = 0
        self.already_present = len(self.keys)
        self.closed = False

    def add(self, record: RawRecord) -> bool:
        if record.source != self.source:
            raise ValueError(f"record of {record.source} in a {self.source} partition")
        if record.key in self.keys:
            self.duplicates += 1
            return False
        self.keys.add(record.key)
        self.batch.append(_row(record))
        if len(self.batch) >= self.batch_rows:
            self.flush()
        return True

    def add_many(self, records: Iterable[RawRecord]) -> int:
        return sum(self.add(r) for r in records)

    def reject(self, source_id: str | None, reason: str) -> None:
        """Log a row that failed validation. Only the id and the reason: the row may hold personal data."""
        self.rejected += 1
        self.rejects_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "source_id": source_id, "reason": reason}
        with self.rejects_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if not self.batch:
            return
        table = pa.Table.from_pylist(self.batch, schema=SCHEMA)
        final = self.dir / f"part-{self.run_id}-{len(self.parts):05d}.parquet"
        tmp = self.dir / f".{final.name}.tmp"
        pq.write_table(table, tmp, compression="zstd")
        if final.exists():
            raise FileExistsError(final)
        os.replace(tmp, final)
        self.parts.append(final)
        self.written += len(self.batch)
        self.batch = []

    def close(self, complete: bool, **summary: Any) -> None:
        if self.closed:
            return
        self.closed = True
        self.flush()
        if not complete:
            return
        marker = {
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows_in_partition": self.already_present + self.written,
            "written_this_run": self.written,
            "duplicates_skipped": self.duplicates,
            "rejected": self.rejected,
            **summary,
        }
        tmp = self.dir / f".{SUCCESS}.tmp"
        tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.dir / SUCCESS)

    def __enter__(self) -> "PartitionWriter":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close(complete=exc_type is None)


def write_records(records: Iterable[RawRecord], source: str, dt: date, root: Path | None = None) -> PartitionWriter:
    with PartitionWriter(source, dt, root) as writer:
        writer.add_many(records)
    return writer
