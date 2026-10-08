"""Collection summaries computed from what is on disk, never from a collector's memory.

For each partition of a source: rows and distinct ids from the Parquet parts, `pii_removed` summed
from the column, rejected ids from the rejects log minus ids that a later run did write. Stats of a
complete partition are cached in `<source>/_meta/` keyed by the list of part files and their sizes,
so a summary after every collected day does not reread the whole history.
"""

import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from radar.collect.writer import SUCCESS, raw_root


def meta_dir(source: str, root: Path | None = None) -> Path:
    return (root or raw_root()) / source / "_meta"


def append_meta(source: str, name: str, entry: dict[str, Any], root: Path | None = None) -> None:
    """Append one JSON line to `<source>/_meta/<name>.jsonl` (download timings, gone versions)."""
    path = meta_dir(source, root) / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_meta(source: str, name: str, root: Path | None = None) -> list[dict[str, Any]]:
    path = meta_dir(source, root) / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pii_sums(table: pa.Table) -> dict[str, int]:
    column = cast("pa.MapArray[Any, Any]", table.column("pii_removed").combine_chunks())
    if len(column) == 0:
        return {}
    pairs = pa.table({"kind": column.keys, "n": column.items})
    grouped = pairs.group_by("kind").aggregate([("n", "sum")]).to_pylist()
    return {g["kind"]: int(g["n_sum"]) for g in sorted(grouped, key=lambda g: g["kind"])}


def _rejected_ids(source: str, dt: date, root: Path | None) -> set[str | None]:
    path = (root or raw_root()) / source / "_rejects" / f"dt={dt.isoformat()}.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line).get("source_id") for line in path.read_text(encoding="utf-8").splitlines() if line}


def partition_stats(source: str, dt: date, root: Path | None = None) -> dict[str, Any]:
    directory = (root or raw_root()) / source / f"dt={dt.isoformat()}"
    parts = sorted(directory.glob("part-*.parquet"))
    signature = [[p.name, p.stat().st_size] for p in parts]
    complete = (directory / SUCCESS).exists()
    cache = meta_dir(source, root) / f"stats-dt={dt.isoformat()}.json"
    if complete and cache.exists():
        cached: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("parts") == signature:
            return cached
    ids: set[str] = set()
    pii: Counter[str] = Counter()
    rows = 0
    for part in parts:
        table = pq.read_table(part, columns=["source_id", "pii_removed"])
        rows += table.num_rows
        ids.update(str(i) for i in table.column("source_id").to_pylist())
        pii.update(_pii_sums(table))
    rejected = _rejected_ids(source, dt, root)
    missing = sorted((i for i in rejected if i not in ids), key=lambda i: i or "")
    marker = json.loads((directory / SUCCESS).read_text(encoding="utf-8")) if complete else None
    stats = {
        "dt": dt.isoformat(),
        "complete": complete,
        "rows": rows,
        "distinct_ids": len(ids),
        "parquet_bytes": sum(p.stat().st_size for p in parts),
        "pii_removed": dict(sorted(pii.items())),
        "rejected_not_written": len(missing),
        "rejected_ids": missing,
        "marker": marker,
        "parts": signature,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if complete:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def partition_dates(source: str, root: Path | None = None) -> list[date]:
    base = (root or raw_root()) / source
    if not base.exists():
        return []
    return sorted(date.fromisoformat(p.name[3:]) for p in base.glob("dt=*") if p.is_dir())
