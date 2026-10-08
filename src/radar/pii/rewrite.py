"""Run the current text cleaner over stored Telegram partitions and rewrite the parts it changes.

The raw layer only grows (radar.collect.writer); this command is the one exception, and it keeps
every row. A changed part is written next to itself, read back and compared with the original (same
row count, same ids in the same order, every column except text, content_hash and pii_removed
equal), then renamed over it; `_SUCCESS` and part names stay. Counts of what changed go to
`telegram/_meta/pii_rewrites.jsonl`, one line per rewritten part and one per run (a dry run and a
scan record their counts there too).

Left alone: the partition of today's Moscow date (a collector may still be writing it) and any part
modified in the last 30 minutes. They are listed as skipped; run the command again later.

    python -m radar.pii.rewrite --source telegram [--dry-run | --scan-only]
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.collect.envelope import content_hash
from radar.collect.summary import append_meta, meta_dir, partition_dates
from radar.collect.telegram import MSK, VOLATILE, keep_handles
from radar.collect.writer import partition_dir
from radar.pii.records import scan_record
from radar.pii.text import clean_text, scan_text

MIN_AGE = timedelta(minutes=30)
CLEANED = ("text", "content_hash", "pii_removed")
META = "pii_rewrites"

Row = dict[str, Any]


class RewriteError(RuntimeError):
    pass


def clean_row(row: Row, keep: frozenset[str]) -> tuple[Row, Counter[str], bool]:
    """The row after cleaning its text again, what was found, and whether the stored hash reproduced."""
    text = row["text"]
    if not text:
        return row, Counter(), True
    cleaned, found = clean_text(text, keep)
    if not found:
        return row, found, True
    payload = json.loads(row["payload"])
    reproduced = content_hash(row["title"], text, payload, ignore=VOLATILE) == row["content_hash"]
    stored = cleaned.strip() or None
    removed = Counter(dict(row["pii_removed"] or []))
    removed.update(found)
    out = {
        **row,
        "text": stored,
        "content_hash": content_hash(row["title"], stored, payload, ignore=VOLATILE),
        "pii_removed": list(removed.items()),
    }
    return out, found, reproduced


def _verify(original: pa.Table, written: pa.Table) -> None:
    if written.num_rows != original.num_rows:
        raise RewriteError(f"{written.num_rows} rows written, {original.num_rows} read")
    if not written.schema.equals(original.schema):
        raise RewriteError("schema changed")
    if written.column("source_id").to_pylist() != original.column("source_id").to_pylist():
        raise RewriteError("ids or their order changed")
    for name in original.column_names:
        if name not in CLEANED and not written.column(name).equals(original.column(name)):
            raise RewriteError(f"column {name} changed")


def rewrite_part(part: Path, keep: frozenset[str], dry_run: bool = False) -> dict[str, Any]:
    original = pq.read_table(part)
    rows: list[Row] = []
    added: Counter[str] = Counter()
    changed = not_reproduced = 0
    for row in original.to_pylist():
        new, found, reproduced = clean_row(row, keep)
        if found:
            changed += 1
            added.update(found)
            not_reproduced += not reproduced
        rows.append(new)
    result = {
        "part": part.name,
        "rows": original.num_rows,
        "rows_changed": changed,
        "pii_added": dict(sorted(added.items())),
        # Rows whose stored hash this code does not reproduce from the stored text and payload.
        "hash_not_reproduced": not_reproduced,
    }
    if not changed or dry_run:
        return result
    tmp = part.with_name(f".{part.name}.rewrite.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=original.schema), tmp, compression="zstd")
    try:
        _verify(original, pq.read_table(tmp))
    except RewriteError:
        tmp.unlink()
        raise
    os.replace(tmp, part)
    return result


def _duplicate_keys(directory: Path) -> int:
    keys: Counter[tuple[str, str, str]] = Counter()
    for part in sorted(directory.glob("part-*.parquet")):
        table = pq.read_table(part, columns=["source", "source_id", "content_hash"])
        keys.update(zip(*(table.column(c).to_pylist() for c in ("source", "source_id", "content_hash")), strict=True))
    return sum(n - 1 for n in keys.values() if n > 1)


def _invalidate_stats(source: str, dt: str, root: Path | None) -> None:
    """The cached partition stats are keyed by part names and sizes; a rewrite may keep both."""
    cache = meta_dir(source, root) / f"stats-dt={dt}.json"
    if not cache.exists():
        return
    stats = json.loads(cache.read_text(encoding="utf-8"))
    stats["parts"] = None
    tmp = cache.with_name(f".{cache.name}.tmp")
    tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, cache)


def scan(source: str, keep: frozenset[str], root: Path | None = None) -> dict[str, Any]:
    """Personal data the scanner still finds in every stored part, today's partition included."""
    rows = with_findings = 0
    findings: Counter[str] = Counter()
    rows_by_kind: Counter[str] = Counter()
    name_rules: Counter[str] = Counter()
    for dt in partition_dates(source, root):
        for part in sorted(partition_dir(source, dt, root).glob("part-*.parquet")):
            table = pq.read_table(part, columns=["source_id", "url", "title", "text", "payload"])
            for row in table.to_pylist():
                rows += 1
                found = scan_record({**row, "payload": json.loads(row["payload"])}, keep)
                if not found:
                    continue
                with_findings += 1
                findings.update(found)
                rows_by_kind.update(found.keys())
                names = [f for f in scan_text(row["text"] or "", keep) if f.kind == "person_name"]
                name_rules.update(f.rule or "shape_or_label" for f in names)
    return {
        "rows": rows,
        "rows_with_findings": with_findings,
        "findings": dict(sorted(findings.items())),
        "rows_by_kind": dict(sorted(rows_by_kind.items())),
        "person_name_by_rule": dict(sorted(name_rules.items())),
    }


def rewrite(
    source: str, keep: frozenset[str], root: Path | None = None, now: datetime | None = None, dry_run: bool = False
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    today = now.astimezone(MSK).date()
    skipped: list[dict[str, Any]] = []
    rows = changed = not_reproduced = duplicates = rewritten = 0
    added: Counter[str] = Counter()
    for dt in partition_dates(source, root):
        directory = partition_dir(source, dt, root)
        parts = sorted(directory.glob("part-*.parquet"))
        if dt == today:
            skipped.append({"dt": dt.isoformat(), "parts": len(parts), "reason": "partition of today (Moscow)"})
            continue
        touched = False
        for part in parts:
            if datetime.fromtimestamp(part.stat().st_mtime, UTC) > now - MIN_AGE:
                skipped.append({"dt": dt.isoformat(), "part": part.name, "reason": "modified in the last 30 minutes"})
                continue
            result = rewrite_part(part, keep, dry_run)
            rows += result["rows"]
            changed += result["rows_changed"]
            not_reproduced += result["hash_not_reproduced"]
            added.update(result["pii_added"])
            if result["rows_changed"] and not dry_run:
                touched = True
                entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "dt": dt.isoformat(), **result}
                append_meta(source, META, entry, root)
                rewritten += 1
        if touched:
            _invalidate_stats(source, dt.isoformat(), root)
            duplicates += _duplicate_keys(directory)
    return {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": "dry_run" if dry_run else "run",
        "rows_checked": rows,
        "rows_changed": changed,
        "pii_added": dict(sorted(added.items())),
        "hash_not_reproduced": not_reproduced,
        "parts_rewritten": rewritten,
        # Two versions of a post that differed only in removed data now share a key; both rows stay.
        "duplicate_keys_in_rewritten_partitions": duplicates,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["telegram"], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="count what would change, rewrite no part")
    mode.add_argument("--scan-only", action="store_true", help="only run the scanner over all partitions")
    args = parser.parse_args(argv)
    keep = keep_handles()
    before = scan(args.source, keep)
    if args.scan_only:
        entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "kind": "scan", "scan": before}
        append_meta(args.source, META, entry)
        print(json.dumps(entry, ensure_ascii=False, indent=1))
        return 0
    summary = rewrite(args.source, keep, dry_run=args.dry_run)
    summary["scan_before"] = before
    summary["scan_after"] = before if args.dry_run else scan(args.source, keep)
    append_meta(args.source, META, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
