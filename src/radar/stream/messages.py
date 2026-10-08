"""Raw-layer rows as stream messages: the value is the raw-layer envelope as JSON, the key is `source_id`.

The raw partition date `dt` travels in a header: it is a property of the file the record was
written to (the collection day, or the snapshot date), not of the record, so the envelope stays as
the contract defines it.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from radar.collect.envelope import RawRecord

TOPIC_PREFIX = "raw."
DT_HEADER = "dt"


@dataclass(frozen=True)
class Message:
    topic: str
    key: bytes
    value: bytes
    headers: list[tuple[str, str | bytes | None]]


def topic_for(source: str) -> str:
    return TOPIC_PREFIX + source


def record_from_row(row: dict[str, Any]) -> RawRecord:
    """A Parquet row of `radar.collect.writer.SCHEMA` back into the envelope it was written from."""
    return RawRecord.model_validate(
        {**row, "payload": json.loads(row["payload"]), "pii_removed": dict(row["pii_removed"] or [])}
    )


def to_message(record: RawRecord, dt: date) -> Message:
    return Message(
        topic=topic_for(record.source),
        key=record.source_id.encode("utf-8"),
        value=record.model_dump_json().encode("utf-8"),
        headers=[(DT_HEADER, dt.isoformat().encode("ascii"))],
    )


def iter_records(directory: Path, batch_rows: int = 10_000) -> Iterator[RawRecord]:
    """Records of every part in a partition directory, read in batches to bound memory."""
    for part in sorted(directory.glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                yield record_from_row(row)
