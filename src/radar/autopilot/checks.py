"""Checks of the daily run's Python rules against the dbt tables they stand in for (local stack).

    python -m radar.autopilot.checks parity    skill_trends from aggregates/ against ClickHouse
    python -m radar.autopilot.checks skills    radar.autopilot.skillmap against int_extracted_fields skills
    python -m radar.autopilot.checks dedup     the weekly text hash dedup against stg_telegram

parity: the IT units of radar.autopilot.seed, all of them, written to a temporary directory and read back
must give the metrics of the ClickHouse run exactly; the committed aggregates differ from it only by the
units the seed held back for the first daily run. skills: a sample of extracted Telegram records, the
spellings of the version int_extracted_fields took its fields from, mapped in Python and compared with
its skills as sets. dedup: is_week_duplicate of every stg_telegram post of the two open weeks against the
first-copy rule of radar.autopilot.daily over the local raw parquet. Results under results/autopilot/.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from radar.autopilot import aggregates
from radar.autopilot.aggregates import AGGREGATES, MSK, moscow_week, text_hash
from radar.autopilot.daily import counted, latest_versions
from radar.autopilot.seed import UNITS_SQL, counts_of
from radar.autopilot.skillmap import SkillMap
from radar.collect import setup_logging, telegram
from radar.collect.writer import raw_root
from radar.extract.telegram_corpus import channel_langs
from radar.results import write_result
from radar.site import skill_trends
from radar.site.market_facts import fetch, share
from radar.stream.clickhouse import ClickHouse

# Records whose latest version is extracted: that version's spellings and text, as int_extracted_fields
# took its skills from them (latest version: the order of radar.extract.corpus_run _TELEGRAM_SQL).
SKILLS_SQL = """
WITH sample AS (
    SELECT source_id, skills AS mapped
    FROM staging.int_extracted_fields
    WHERE source = 'telegram' AND valid AND is_extracted AND cityHash64(source_id) % {every:UInt32} = 0
),
latest AS (
    SELECT source_id, content_hash, text
    FROM raw.vacancies FINAL
    WHERE source = 'telegram' AND source_id IN (SELECT source_id FROM sample)
    ORDER BY source_id, fetched_at DESC, dt DESC, content_hash DESC
    LIMIT 1 BY source_id
),
answers AS (
    SELECT source_id, content_hash, skills
    FROM extract.vacancy_fields FINAL
    WHERE source = 'telegram' AND valid AND model_id IN {models:Array(String)}
        AND source_id IN (SELECT source_id FROM sample)
    ORDER BY source_id, content_hash, indexOf({models:Array(String)}, model_id), created_at DESC
    LIMIT 1 BY source_id, content_hash
)
SELECT s.source_id AS source_id, s.mapped AS mapped, a.skills AS written, l.text AS text
FROM sample AS s
INNER JOIN latest AS l ON l.source_id = s.source_id
INNER JOIN answers AS a ON a.source_id = l.source_id AND a.content_hash = l.content_hash
FORMAT JSONEachRow
"""
DEDUP_SQL = """
SELECT source_id, is_week_duplicate FROM staging.stg_telegram WHERE toString(published_week) IN {weeks:Array(String)}
FORMAT JSONEachRow
"""


def differing(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def parity() -> dict[str, Any]:
    reference = skill_trends.metrics_of(*skill_trends.clickhouse_rows())
    with ClickHouse() as ch:
        units = fetch(ch, UNITS_SQL)
    with tempfile.TemporaryDirectory() as tmp:
        aggregates.save(counts_of(units, {}), Path(tmp))
        roundtrip = skill_trends.metrics_of(*skill_trends.aggregate_rows(Path(tmp)))
    committed = skill_trends.metrics_of(*skill_trends.aggregate_rows(AGGREGATES))
    full, seeded = differing(reference, roundtrip), differing(reference, committed)
    metrics = {
        "metrics_clickhouse": {"value": len(reference)},
        "it_units": {"value": len(units)},
        "all_units_metrics_differing": {"value": len(full)},
        "committed_metrics_differing": {"value": len(seeded)},
    }
    write_result(
        "autopilot/parity.json",
        metrics,
        source_script="src/radar/autopilot/checks.py",
        all_units_differing=full[:50],
        committed_differing=seeded[:50],
    )
    return {k: m["value"] for k, m in metrics.items()}


def skills_check(every: int) -> dict[str, Any]:
    models = ["qwen3_1p7b_r16_clean-awq", "qwen3_1p7b_r16_clean-q4_k_m"]
    with ClickHouse() as ch:
        body = ch.query(SKILLS_SQL, models=f"{models}", every=str(every))
    rows = [json.loads(line) for line in body.splitlines() if line]
    mapper = SkillMap()
    agree = with_link_skill = link_agree = 0
    for r in rows:
        same = mapper.skills(r["written"], "\n" + (r["text"] or "")) == sorted(r["mapped"])
        agree += same
        if any(set(mapper.names(s)) & mapper.dictionary.outside_links for s in r["written"]):
            with_link_skill += 1
            link_agree += same
    metrics = {
        "records_same_skills": share(agree, len(rows)),
        "records_with_http_skill_same": share(link_agree, with_link_skill),
        "spellings_unseen_by_seed": {"value": mapper.unseen},
    }
    write_result(
        "autopilot/skill_map_check.json",
        metrics,
        source_script="src/radar/autopilot/checks.py",
        sample=f"cityHash64(source_id) % {every} = 0 of the valid Telegram records of int_extracted_fields",
        models=models,
    )
    return {k: m["value"] for k, m in metrics.items()} | {"records": len(rows)}


def dedup_check(today: date) -> dict[str, Any]:
    weeks = aggregates.kept_weeks(today)
    first_day = date.fromisoformat(weeks[0]) - timedelta(days=1)
    days = [d.name.removeprefix("dt=") for d in (raw_root() / telegram.SOURCE).glob("dt=*")]
    langs = channel_langs()
    rows = [
        r
        for r in latest_versions(raw_root(), [d for d in days if date.fromisoformat(d) >= first_day])
        if counted(r, langs) and moscow_week(r["published_at"]).isoformat() in weeks
    ]
    seen: dict[str, set[str]] = {}
    ours = {}
    for r in sorted(rows, key=lambda r: (r["published_at"], r["source_id"])):
        week_seen = seen.setdefault(moscow_week(r["published_at"]).isoformat(), set())
        digest = text_hash(r["text"])
        ours[r["source_id"]] = digest in week_seen
        week_seen.add(digest)
    with ClickHouse() as ch:
        body = ch.query(DEDUP_SQL, weeks=f"{list(weeks)}")
    theirs = {(x := json.loads(line))["source_id"]: bool(x["is_week_duplicate"]) for line in body.splitlines() if line}
    both = ours.keys() & theirs.keys()
    metrics = {
        "posts_same_flag": share(sum(ours[k] == theirs[k] for k in both), len(both)),
        "duplicates_python": {"value": sum(ours[k] for k in both)},
        "duplicates_clickhouse": {"value": sum(theirs[k] for k in both)},
        "posts_only_python": {"value": len(ours.keys() - theirs.keys())},
        "posts_only_clickhouse": {"value": len(theirs.keys() - ours.keys())},
    }
    write_result(
        "autopilot/dedup_check.json", metrics, source_script="src/radar/autopilot/checks.py", weeks=list(weeks)
    )
    return {k: m["value"] for k, m in metrics.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=("parity", "skills", "dedup"))
    parser.add_argument("--every", type=int, default=50, help="skills: one record in this many by id hash")
    args = parser.parse_args(argv)
    setup_logging()
    if args.check == "parity":
        out = parity()
    elif args.check == "skills":
        out = skills_check(args.every)
    else:
        out = dedup_check(datetime.now(MSK).date())
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
