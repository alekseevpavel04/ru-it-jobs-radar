"""Run the current cleaner over stored partitions and rewrite the parts it changes.

The raw layer only grows (radar.collect.writer); this command is the one exception, and it keeps
every row. Each stored record goes through the same cleaning calls its collector makes, on the same
units (a Telegram post; a Habr card text and every list and company string; each labelled section
and payload string of trudvsem), and its content hash is recomputed the way the collector computes
it, so the next collection of an unchanged vacancy produces the same key. For Habr the list hash
that decides whether a card is read again is recomputed too (see `_clean_habr`).

A changed part is written next to itself, read back and compared with the original (same row count,
same ids in the same order, every column the source's cleaner does not touch equal), then renamed
over it; `_SUCCESS` and part names stay. Counts of what changed go to
`<source>/_meta/pii_rewrites.jsonl`, one line per rewritten part and one per run (a dry run and a
scan record their counts there too). Rewriting is per part and idempotent: an interrupted run is
continued by running it again.

Left alone: the partition of today's Moscow date and any partition without `_SUCCESS` (a collector
may still be writing them), and any part modified in the last 30 minutes. They are listed as
skipped; run the command again later.

    python -m radar.pii.rewrite --source {telegram,habr,trudvsem,trudvsem_snapshot}
        [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--dry-run | --scan-only] [--no-scan]
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.collect import habr, setup_logging, telegram, trudvsem, trudvsem_snapshot
from radar.collect.envelope import content_hash
from radar.collect.summary import append_meta, meta_dir, partition_dates
from radar.collect.writer import SUCCESS, partition_dir
from radar.pii.records import _clean_strings, scan_record
from radar.pii.text import DEFAULT_KEEP_HANDLES, clean_text, scan_text

log = logging.getLogger("radar.pii.rewrite")

MSK = telegram.MSK
MIN_AGE = timedelta(minutes=30)
META = "pii_rewrites"
# Rows the scanner holds as Python objects at a time; a snapshot part is 50 000 rows of about 4 KB.
SCAN_BATCH = 10_000
SOURCES = ("telegram", "habr", "trudvsem", "trudvsem_snapshot")

Row = dict[str, Any]
# The cleaned row, what the cleaner found, and notes about the stored row (hash not reproduced...).
Cleaned = tuple[Row, Counter[str], Counter[str]]


class RewriteError(RuntimeError):
    pass


def _dumps(payload: Row) -> str:
    # As radar.collect.writer stores it.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _clean_payload(payload: Row, found: Counter[str], as_is: frozenset[str]) -> Row:
    """Every payload value the collector passed through the cleaner, in the stored key order."""
    return {k: v if k in as_is else _clean_strings(v, found, k) for k, v in payload.items()}


def _clean_telegram(row: Row, keep: frozenset[str]) -> Cleaned:
    text = row["text"]
    if not text:
        return row, Counter(), Counter()
    cleaned, found = clean_text(text, keep)
    if not found:
        return row, found, Counter()
    payload = json.loads(row["payload"])
    notes: Counter[str] = Counter()
    notes["hash_not_reproduced"] += (
        content_hash(row["title"], text, payload, ignore=telegram.VOLATILE) != row["content_hash"]
    )
    stored = cleaned.strip() or None
    out = {
        **row,
        "text": stored,
        "content_hash": content_hash(row["title"], stored, payload, ignore=telegram.VOLATILE),
    }
    return out, found, notes


# Payload keys the Habr collector adds next to the cleaned list entry.
HABR_ADDED = frozenset({"list_hash", "_passes", "card", "company_card", "card_fetched_at", "card_list_hash"})
# Of those, `company_card` went through the cleaner (clean_habr_card_company); the rest did not.
_HABR_AS_IS = HABR_ADDED - {"company_card"}


def _list_hash(payload: Row) -> str:
    """`habr.list_entry_hash` of the list entry this payload was built from."""
    entry = {k: v for k, v in payload.items() if k not in HABR_ADDED}
    return content_hash(None, None, entry, ignore=habr.LIST_VOLATILE)


def _clean_habr(row: Row, keep: frozenset[str]) -> Cleaned:
    """Card text, list entry and card company strings; the `title` column is the list title as served.

    `list_hash` is the hash of the cleaned list entry and the next run compares it with the
    `card_list_hash` stored here to decide whether to read the card again. When cleaning changes
    the list entry, both are recomputed: `list_hash` always, `card_list_hash` when it equalled the
    old `list_hash` (the card was read for this very list entry). Otherwise the next run, hashing
    the same entry with the same cleaner, would see a change and spend a card request on it.
    """
    found: Counter[str] = Counter()
    text = row["text"]
    new_text = text
    if text:
        new_text, found = clean_text(text)
    payload = json.loads(row["payload"])
    new_payload = _clean_payload(payload, found, _HABR_AS_IS)
    if not found:
        return row, found, Counter()
    notes: Counter[str] = Counter()
    notes["hash_not_reproduced"] += (
        content_hash(row["title"], text, payload, ignore=habr.RECORD_VOLATILE) != row["content_hash"]
    )
    old_hash = payload.get("list_hash")
    notes["list_hash_not_reproduced"] += _list_hash(payload) != old_hash
    new_list = _list_hash(new_payload)
    if new_list != _list_hash(payload):
        notes["list_hash_changed"] += 1
        new_payload["list_hash"] = new_list
        if "card_list_hash" in new_payload and new_payload["card_list_hash"] == old_hash:
            new_payload["card_list_hash"] = new_list
    out = {
        **row,
        "text": new_text,
        "payload": _dumps(new_payload),
        "content_hash": content_hash(row["title"], new_text, new_payload, ignore=habr.RECORD_VOLATILE),
    }
    return out, found, notes


@dataclass(frozen=True)
class SectionedText:
    """How a trudvsem collector builds `text`: `Label: value` blocks in label order, a blank line apart."""

    labels: tuple[str, ...]
    # The snapshot collector leaves out a section equal to an earlier one (after cleaning).
    dedupe: bool

    @property
    def head(self) -> re.Pattern[str]:
        alternatives = "|".join(re.escape(label) for label in sorted(self.labels, key=len, reverse=True))
        return re.compile(rf"(?:^|\n\n)({alternatives}): ")

    def split(self, text: str, head: re.Pattern[str]) -> tuple[list[tuple[str, str]], bool]:
        """Sections and whether a section holds what looks like another heading (kept inside it)."""
        heads = list(head.finditer(text))
        if not heads or heads[0].start() != 0:
            raise RewriteError("text does not start with a section heading")
        kept: list[re.Match[str]] = []
        for m in heads:
            if not kept or self.labels.index(m.group(1)) > self.labels.index(kept[-1].group(1)):
                kept.append(m)
        bounds = [m.start() for m in kept[1:]] + [len(text)]
        sections = [(m.group(1), text[m.end() : end]) for m, end in zip(kept, bounds, strict=True)]
        return sections, len(kept) != len(heads)

    def clean(self, text: str | None, found: Counter[str], head: re.Pattern[str]) -> tuple[str | None, bool]:
        if not text:
            return text, False
        sections, ambiguous = self.split(text, head)
        values = [_clean_strings(value, found, "") for _, value in sections]
        # Equal stored sections mean the split is not the collector's one: nothing is dropped then.
        dedupe = self.dedupe and len({v for _, v in sections}) == len(sections)
        out: list[str] = []
        seen: set[str] = set()
        for (label, _), value in zip(sections, values, strict=True):
            if dedupe and value in seen:
                continue
            seen.add(value)
            out.append(f"{label}: {value}")
        return "\n\n".join(out) or None, ambiguous


def _trudvsem_cleaner(
    sections: SectionedText, title_key: str, volatile: tuple[str, ...], as_is: frozenset[str]
) -> Callable[[Row, frozenset[str]], Cleaned]:
    head = sections.head

    def clean(row: Row, keep: frozenset[str]) -> Cleaned:
        found: Counter[str] = Counter()
        payload = json.loads(row["payload"])
        new_payload = _clean_payload(payload, found, as_is)
        new_text, ambiguous = sections.clean(row["text"], found, head)
        if not found:
            return row, found, Counter()
        notes: Counter[str] = Counter()
        notes["hash_not_reproduced"] += (
            content_hash(row["title"], row["text"], payload, ignore=volatile) != row["content_hash"]
        )
        notes["text_split_ambiguous"] += ambiguous
        title = new_payload.get(title_key)
        out = {
            **row,
            "title": title,
            "text": new_text,
            "payload": _dumps(new_payload),
            "content_hash": content_hash(title, new_text, new_payload, ignore=volatile),
        }
        return out, found, notes

    return clean


@dataclass(frozen=True)
class Source:
    clean: Callable[[Row, frozenset[str]], Cleaned]
    # Columns the cleaner may change; `pii_removed` gains what it found.
    cleaned: tuple[str, ...]


SPECS = {
    "telegram": Source(_clean_telegram, ("text", "content_hash", "pii_removed")),
    "habr": Source(_clean_habr, ("text", "payload", "content_hash", "pii_removed")),
    "trudvsem": Source(
        _trudvsem_cleaner(
            SectionedText(tuple(label for _, label in trudvsem.TEXT_SECTIONS), dedupe=False),
            "job-name",
            trudvsem.VOLATILE,
            frozenset({"_queries"}),
        ),
        ("title", "text", "payload", "content_hash", "pii_removed"),
    ),
    "trudvsem_snapshot": Source(
        _trudvsem_cleaner(
            SectionedText(tuple(label for _, label in trudvsem_snapshot.TEXT_SECTIONS), dedupe=True),
            "vacancyName",
            trudvsem_snapshot.VOLATILE,
            frozenset({"snapshot_version"}),
        ),
        ("title", "text", "payload", "content_hash", "pii_removed"),
    ),
}
_READ = ("source_id", "url", "title", "text", "payload", "content_hash", "pii_removed")


def keep_for(source: str) -> frozenset[str]:
    """Handles that are not people: Telegram adds its channels, the other collectors use the default."""
    return telegram.keep_handles() if source == "telegram" else DEFAULT_KEEP_HANDLES


def clean_row(source: str, row: Row, keep: frozenset[str]) -> Cleaned:
    """`row` as stored (payload a JSON string, pii_removed as pairs) after cleaning it again."""
    new, found, notes = SPECS[source].clean(row, keep)
    if found:
        removed = Counter(dict(row["pii_removed"] or []))
        removed.update(found)
        new = {**new, "pii_removed": list(removed.items())}
    return new, found, notes


def _scan_row(row: Row, keep: frozenset[str]) -> Counter[str]:
    return scan_record({**row, "payload": json.loads(row["payload"])}, keep)


def _verify(original: pa.Table, written: pa.Table, cleaned: tuple[str, ...]) -> None:
    if written.num_rows != original.num_rows:
        raise RewriteError(f"{written.num_rows} rows written, {original.num_rows} read")
    if not written.schema.equals(original.schema):
        raise RewriteError("schema changed")
    if written.column("source_id").to_pylist() != original.column("source_id").to_pylist():
        raise RewriteError("ids or their order changed")
    for name in original.column_names:
        if name not in cleaned and not written.column(name).equals(original.column(name)):
            raise RewriteError(f"column {name} changed")


def _replace(tmp: Path, part: Path) -> None:
    # On Windows a reader in another process (a collector rebuilding its summary) blocks the rename.
    for pause in (1, 2, 4, 8, 16, None):
        try:
            os.replace(tmp, part)
            return
        except PermissionError:
            if pause is None:
                tmp.unlink()
                raise
            time.sleep(pause)


def rewrite_part(source: str, part: Path, keep: frozenset[str], dry_run: bool = False) -> dict[str, Any]:
    spec = SPECS[source]
    original = pq.read_table(part)
    columns = {c: original.column(c).to_pylist() for c in _READ}
    new_columns: dict[str, list[Any]] = {c: list(columns[c]) for c in spec.cleaned}
    added: Counter[str] = Counter()
    notes: Counter[str] = Counter()
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    rows_with_names = [0, 0]
    changed = 0
    for i in range(original.num_rows):
        row = {c: columns[c][i] for c in _READ}
        new, found, row_notes = clean_row(source, row, keep)
        if not found:
            continue
        changed += 1
        added.update(found)
        notes.update(row_notes)
        for c in spec.cleaned:
            new_columns[c][i] = new[c]
        # Unchanged rows scan the same before and after; changed ones are scanned here both ways.
        for counter, n, scanned in ((before, 0, row), (after, 1, new)):
            found_now = _scan_row(scanned, keep)
            counter.update(found_now)
            rows_with_names[n] += bool(found_now.get("person_name"))
    result: dict[str, Any] = {
        "part": part.name,
        "rows": original.num_rows,
        "rows_changed": changed,
        "pii_added": dict(sorted(added.items())),
        # Rows whose stored hash this code does not reproduce from the stored title, text and payload.
        "hash_not_reproduced": notes["hash_not_reproduced"],
        "notes": dict(sorted((k, v) for k, v in notes.items() if v and k != "hash_not_reproduced")),
        "changed_rows_scan": {
            "before": dict(sorted(before.items())),
            "after": dict(sorted(after.items())),
            "rows_with_person_name_before": rows_with_names[0],
            "rows_with_person_name_after": rows_with_names[1],
        },
    }
    if not changed or dry_run:
        return result
    table = original
    for c in spec.cleaned:
        index = table.schema.get_field_index(c)
        table = table.set_column(index, table.schema.field(index), pa.array(new_columns[c], table.schema.field(c).type))
    tmp = part.with_name(f".{part.name}.rewrite.tmp")
    pq.write_table(table, tmp, compression="zstd")
    try:
        _verify(original, pq.read_table(tmp), spec.cleaned)
    except RewriteError:
        tmp.unlink()
        raise
    _replace(tmp, part)
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


def _dates(source: str, root: Path | None, since: date | None, until: date | None) -> Iterator[date]:
    for dt in partition_dates(source, root):
        if (since is None or dt >= since) and (until is None or dt <= until):
            yield dt


def scan(
    source: str, keep: frozenset[str], root: Path | None = None, since: date | None = None, until: date | None = None
) -> dict[str, Any]:
    """Personal data the scanner still finds in every stored part of the range, today's partition included."""
    rows = with_findings = with_names = 0
    findings: Counter[str] = Counter()
    rows_by_kind: Counter[str] = Counter()
    name_rules: Counter[str] = Counter()
    by_dt: dict[str, dict[str, int]] = {}
    for dt in _dates(source, root, since, until):
        day_rows = day_names = 0
        for part in sorted(partition_dir(source, dt, root).glob("part-*.parquet")):
            batches = pq.ParquetFile(part).iter_batches(
                SCAN_BATCH, columns=["source_id", "url", "title", "text", "payload"]
            )
            for row in (r for batch in batches for r in batch.to_pylist()):
                day_rows += 1
                found = _scan_row(row, keep)
                if not found:
                    continue
                with_findings += 1
                findings.update(found)
                rows_by_kind.update(found.keys())
                if found.get("person_name"):
                    day_names += 1
                names = [f for f in scan_text(row["text"] or "", keep) if f.kind == "person_name"]
                name_rules.update(f.rule or "shape_or_label" for f in names)
        rows += day_rows
        with_names += day_names
        by_dt[dt.isoformat()] = {"rows": day_rows, "rows_with_person_name": day_names}
        log.info("scan %s dt=%s: %d of %d rows with a person name", source, dt, day_names, day_rows)
    return {
        "rows": rows,
        "rows_with_findings": with_findings,
        "rows_with_person_name": with_names,
        "findings": dict(sorted(findings.items())),
        "rows_by_kind": dict(sorted(rows_by_kind.items())),
        "person_name_by_rule": dict(sorted(name_rules.items())),
        "by_dt": by_dt,
    }


def rewrite(
    source: str,
    keep: frozenset[str],
    root: Path | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    since: date | None = None,
    until: date | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    today = now.astimezone(MSK).date()
    skipped: list[dict[str, Any]] = []
    rows = changed = not_reproduced = duplicates = rewritten = 0
    added: Counter[str] = Counter()
    notes: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    rewritten_dts: list[str] = []
    for dt in _dates(source, root, since, until):
        directory = partition_dir(source, dt, root)
        parts = sorted(directory.glob("part-*.parquet"))
        if dt == today:
            skipped.append({"dt": dt.isoformat(), "parts": len(parts), "reason": "partition of today (Moscow)"})
            continue
        if not (directory / SUCCESS).exists():
            skipped.append({"dt": dt.isoformat(), "parts": len(parts), "reason": "partition without _SUCCESS"})
            continue
        started = time.monotonic()
        touched = False
        day_rows = day_changed = 0
        for part in parts:
            if datetime.fromtimestamp(part.stat().st_mtime, UTC) > now - MIN_AGE:
                skipped.append({"dt": dt.isoformat(), "part": part.name, "reason": "modified in the last 30 minutes"})
                continue
            result = rewrite_part(source, part, keep, dry_run)
            day_rows += result["rows"]
            day_changed += result["rows_changed"]
            not_reproduced += result["hash_not_reproduced"]
            added.update(result["pii_added"])
            notes.update(result["notes"])
            scan_result = result["changed_rows_scan"]
            scanned["rows_with_person_name_before"] += scan_result["rows_with_person_name_before"]
            scanned["rows_with_person_name_after"] += scan_result["rows_with_person_name_after"]
            if result["rows_changed"] and not dry_run:
                touched = True
                entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "dt": dt.isoformat(), **result}
                append_meta(source, META, entry, root)
                rewritten += 1
        rows += day_rows
        changed += day_changed
        if touched:
            _invalidate_stats(source, dt.isoformat(), root)
            duplicates += _duplicate_keys(directory)
            rewritten_dts.append(dt.isoformat())
        log.info(
            "%s dt=%s: %d of %d rows changed, %.0f s", source, dt, day_changed, day_rows, time.monotonic() - started
        )
    return {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": "dry_run" if dry_run else "run",
        "source": source,
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "rows_checked": rows,
        "rows_changed": changed,
        "pii_added": dict(sorted(added.items())),
        "hash_not_reproduced": not_reproduced,
        "notes": dict(sorted(notes.items())),
        "changed_rows_scan": dict(scanned),
        "parts_rewritten": rewritten,
        "dts_rewritten": rewritten_dts,
        # Two versions that differed only in removed data now share a key; both rows stay.
        "duplicate_keys_in_rewritten_partitions": duplicates,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--since", type=date.fromisoformat, metavar="YYYY-MM-DD", help="first partition date")
    parser.add_argument("--until", type=date.fromisoformat, metavar="YYYY-MM-DD", help="last partition date")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="count what would change, rewrite no part")
    mode.add_argument("--scan-only", action="store_true", help="only run the scanner over the partitions")
    parser.add_argument(
        "--no-scan", action="store_true", help="skip the full scans before and after (changed rows are still scanned)"
    )
    args = parser.parse_args(argv)
    setup_logging()
    keep = keep_for(args.source)
    window = {"since": args.since, "until": args.until}
    if args.scan_only:
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": "scan",
            "since": args.since.isoformat() if args.since else None,
            "until": args.until.isoformat() if args.until else None,
            "scan": scan(args.source, keep, **window),
        }
        append_meta(args.source, META, entry)
        print(json.dumps(entry, ensure_ascii=False, indent=1))
        return 0
    before = None if args.no_scan else scan(args.source, keep, **window)
    summary = rewrite(args.source, keep, dry_run=args.dry_run, **window)
    if before is not None:
        summary["scan_before"] = before
        summary["scan_after"] = before if args.dry_run else scan(args.source, keep, **window)
    append_meta(args.source, META, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
