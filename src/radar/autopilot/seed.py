"""One-time start of aggregates/telegram/ from the local stack: the history the daily run adds to.

    python -m radar.autopilot.seed

- Counts: the IT units of staging.int_extracted_weekly_units (source telegram, weeks from FIRST_YEAR),
  each with its channel from stg_telegram, the same rows radar.site.skill_trends reads.
- Cursors: the daily cursor of each enabled channel as the collector keeps it, rebuilt from the local raw
  parquet: the highest stored post id and the content hashes of the newest posts. The partitions come
  from the daily server (scripts/sync_from_vps.sh), whose cursor files are not synced back, so the local
  ones are behind the data; first_seen_id and subscribers are taken from them.
- Week hashes: radar.autopilot.aggregates.text_hash of the stg_telegram posts of the current and the
  previous week, from the local raw parquet.

The corpus run lags the collector by a day or two, and a post collected but not yet extracted would stay
out of the counts for good once the cursor is past it. So in a channel with such a post in the two open
weeks the cursor goes back to just below its first one, and every post of that channel above the new
cursor is left out of the counts and the hashes: the first daily run collects and counts them. A unit
without fields in an older week is left as dbt has it (units_without_fields_older in the summary).
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.autopilot import aggregates
from radar.autopilot.aggregates import AGGREGATES, MSK, Counts, moscow_week, text_hash
from radar.autopilot.daily import channel_of, counted, held_back, latest_versions, post_id
from radar.collect import setup_logging, telegram
from radar.collect.writer import raw_root
from radar.extract.telegram_corpus import channel_langs
from radar.results import write_result
from radar.site.market_facts import fetch
from radar.stream.clickhouse import ClickHouse

FIRST_YEAR = 2019
UNITS_SQL = f"""
SELECT toString(u.week) AS week, t.channel AS channel, u.record_source_id AS source_id,
       coalesce(u.role, '') AS role, coalesce(u.grade, '') AS grade, u.skills AS skills
FROM staging.int_extracted_weekly_units AS u
INNER JOIN (SELECT source_id, any(channel) AS channel FROM staging.stg_telegram GROUP BY source_id) AS t
    ON u.record_source_id = t.source_id
WHERE u.source = 'telegram' AND u.is_it AND toYear(u.week) >= {FIRST_YEAR}
"""
MISSING_SQL = """
SELECT record_source_id AS source_id, toString(week) AS week
FROM staging.int_extracted_weekly_units
WHERE source = 'telegram' AND NOT has_fields AND NOT not_extractable
"""


def cuts(missing: Sequence[Mapping[str, Any]], weeks: Sequence[str]) -> dict[str, int]:
    """channel -> the highest post id the counts keep: just below its first unit without fields."""
    out: dict[str, int] = {}
    for r in missing:
        if r["week"] in weeks:
            channel, pid = channel_of(r["source_id"]), post_id(r["source_id"])
            out[channel] = min(out.get(channel, pid), pid)
    return {c: pid - 1 for c, pid in out.items()}


def kept(source_id: str, cut: Mapping[str, int]) -> bool:
    channel = channel_of(source_id)
    return channel not in cut or post_id(source_id) <= cut[channel]


def counts_of(units: Sequence[Mapping[str, Any]], cut: Mapping[str, int]) -> Counts:
    counts = Counts()
    for u in units:
        if kept(u["source_id"], cut):
            counts.add(date.fromisoformat(u["week"]), u["channel"], u["role"], u["grade"], u["skills"])
    return counts


def week_hashes(cursors: Mapping[str, Mapping[str, Any]], weeks: Sequence[str]) -> dict[str, set[str]]:
    """Hashes of the stg_telegram posts of these weeks at or below their channel's cursor."""
    # a post is first stored on or after the Moscow day it was published
    first_day = date.fromisoformat(min(weeks)) - timedelta(days=1)
    days = [d.name.removeprefix("dt=") for d in (raw_root() / telegram.SOURCE).glob("dt=*")]
    langs = channel_langs()
    out: dict[str, set[str]] = {w: set() for w in weeks}
    for row in latest_versions(raw_root(), [d for d in days if date.fromisoformat(d) >= first_day]):
        last = cursors.get(channel_of(row["source_id"]), {}).get("last_id")
        if last is None or post_id(row["source_id"]) > last or not counted(row, langs):
            continue
        week = moscow_week(row["published_at"]).isoformat()
        if week in out:
            out[week].add(text_hash(row["text"]))
    return out


def raw_cursors(local: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """channel -> the daily cursor the collector would hold after storing the local raw partitions."""
    parts = sorted((raw_root() / telegram.SOURCE).glob("dt=*/part-*.parquet"))
    columns = ["source_id", "fetched_at", "content_hash"]
    table = pa.concat_tables(pq.read_table(p, columns=columns) for p in parts).sort_by("fetched_at")
    digests: dict[str, dict[int, str]] = {}
    for row in table.to_pylist():
        digests.setdefault(channel_of(row["source_id"]), {})[post_id(row["source_id"])] = row["content_hash"][:16]
    out = {}
    for channel, state in local.items():
        stored = digests.get(channel, {})
        newest = sorted(stored)[-telegram.RECENT_IDS :]
        out[channel] = {
            **state,
            "last_id": max([*stored, state["last_id"]]),
            "recent": {str(i): stored[i] for i in newest},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregates", type=Path, default=AGGREGATES)
    args = parser.parse_args(argv)
    setup_logging()
    today = datetime.now(MSK).date()
    weeks = aggregates.kept_weeks(today)
    with ClickHouse() as ch:
        missing = fetch(ch, MISSING_SQL)
        units = fetch(ch, UNITS_SQL)
    cut = cuts(missing, weeks)
    counts = counts_of(units, cut)
    written = aggregates.save(counts, args.aggregates)
    local = {c: s for c in telegram.enabled_channels() if (s := telegram.read_state("daily", c))}
    cursors = {}
    for channel, state in raw_cursors(local).items():
        if channel in cut and cut[channel] < state["last_id"]:
            state = held_back(state, cut[channel] + 1)
        aggregates.write_cursor(channel, state, args.aggregates)
        cursors[channel] = state
    hashes = week_hashes(cursors, weeks)
    aggregates.write_week_hashes(hashes, today, args.aggregates)
    held = [u for u in units if not kept(u["source_id"], cut)]
    metrics = {
        "it_units": {"value": sum(counts.posts.values())},
        "it_units_held_back": {"value": len(held)},
        "channels_held_back": {"value": len(cut)},
        "units_without_fields_open_weeks": {"value": sum(1 for r in missing if r["week"] in weeks)},
        "units_without_fields_older": {"value": sum(1 for r in missing if r["week"] not in weeks)},
        "cursors": {"value": len(cursors)},
        "post_rows": {"value": len(counts.posts)},
        "skill_rows": {"value": len(counts.skills)},
        **{f"week_hashes.{w}": {"value": len(hashes[w])} for w in weeks},
    }
    path = write_result(
        "autopilot/seed.json",
        metrics,
        source_script="src/radar/autopilot/seed.py",
        first_year=FIRST_YEAR,
        open_weeks=list(weeks),
        files=sorted(p.relative_to(args.aggregates).as_posix() for p in written),
    )
    print(json.dumps({k: m["value"] for k, m in metrics.items()}, ensure_ascii=False))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
