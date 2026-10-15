"""The daily run of the site's Telegram data: new posts in, weekly counts of aggregates/telegram/ up.

    RADAR_USER_AGENT='...' PYTHONPATH=src:airflow/include python -m radar.autopilot.daily \
        --base-url http://127.0.0.1:8080 --time-budget 3000

1. The Telegram collector (radar.collect.telegram, daily mode) runs against a temporary data directory
   that holds only the committed cursors, so the posts it stores never reach the repository.
2. The posts above each channel's committed cursor go through the rules of stg_telegram: a vacancy post,
   in the channel's language where the channel has one. Within a Moscow week the same text counts once,
   whichever channel carried it (week_hashes.json holds the hashes of the current and the previous week;
   copies of a post counted on an earlier day are found there). Edits of posts below the cursor are not
   counted again: the weekly units of dbt count a post once, with its latest text.
3. A single-vacancy post that the corpus rules would extract (radar.extract.telegram_corpus.drop_reason:
   no digest, enough text of its own) goes to the student behind llama-server on the CPU
   (airflow/include/student_cpu.py, model qwen3_1p7b_r16_clean-q4_k_m, the fallback model of the dbt
   marts), with the input every extractor reads. A valid answer whose role is not non_it is an IT unit;
   its skills are mapped as int_extracted_fields maps them (radar.autopilot.skillmap) and counted.
4. Cursors, hashes and counts are written back. Posts are handled in order of publication; when the time
   budget runs out or the server fails a request, the rest stays above the cursor of its channel and the
   next run reads it again (a post read twice is a duplicate of itself by its hash, never counted twice).

The summary (counts and seconds only) goes to stdout and to results/autopilot/daily.json. A channel the
collector could not read or a failed request is reported there and does not stop the run; the workflow
fails the job on them after the commit.
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.autopilot import aggregates
from radar.autopilot.aggregates import AGGREGATES, MSK, Counts, moscow_week, text_hash
from radar.autopilot.skillmap import SkillMap
from radar.collect import setup_logging, telegram
from radar.collect.http import Fetcher, make_client
from radar.config import ROOT
from radar.extract.corpus_run import Answer, Backend, checked_output, load_backend, parse_options
from radar.extract.inputs import input_for_record
from radar.extract.telegram_corpus import channel_langs, drop_reason, post_lang
from radar.extract.telegram_corpus import load_config as corpus_config
from radar.results import write_result

log = logging.getLogger("radar.autopilot.daily")

BACKEND = "student_cpu:backend"
MODEL_ID = "qwen3_1p7b_r16_clean-q4_k_m"
RESULT = "autopilot/daily.json"
COLUMNS = ["source_id", "fetched_at", "published_at", "text", "payload", "is_vacancy", "multi_vacancy"]


def post_id(source_id: str) -> int:
    return int(source_id.rsplit("/", 1)[1])


def channel_of(source_id: str) -> str:
    return source_id.rsplit("/", 1)[0]


def latest_versions(raw_root: Path, days: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """The latest stored version of every post in the Telegram partitions (all, or of these days)."""
    wanted = None if days is None else set(days)
    parts = [
        part
        for d in sorted((raw_root / telegram.SOURCE).glob("dt=*"))
        if wanted is None or d.name.removeprefix("dt=") in wanted
        for part in sorted(d.glob("part-*.parquet"))
    ]
    if not parts:
        return []
    table = pa.concat_tables(pq.read_table(p, columns=COLUMNS) for p in parts).sort_by("fetched_at")
    return list({row["source_id"]: row for row in table.to_pylist()}.values())


def counted(row: Mapping[str, Any], langs: Mapping[str, str]) -> bool:
    """A row of stg_telegram: a vacancy post with a date, in its channel's language if it has one."""
    lang = langs.get(channel_of(row["source_id"]))
    return bool(row["is_vacancy"]) and row["published_at"] is not None and (not lang or post_lang(row["text"]) == lang)


@dataclass
class Outcome:
    counts: Counts = field(default_factory=Counts)
    stats: Counter[str] = field(default_factory=Counter)
    # channel -> the lowest post id left for the next run
    left: dict[str, int] = field(default_factory=dict)
    seconds_extracting: float = 0.0


def process(
    rows: Sequence[Mapping[str, Any]],
    hashes: dict[str, set[str]],
    extract: Callable[[str], Answer],
    mapper: SkillMap,
    langs: Mapping[str, str],
    min_content_chars: int,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Count new posts into weekly units; `hashes` (week -> text hashes) is updated in place."""
    out = Outcome()
    eligible = sorted((r for r in rows if counted(r, langs)), key=lambda r: (r["published_at"], r["source_id"]))
    out.stats["vacancy_posts"] = len(eligible)
    for i, row in enumerate(eligible):
        week = moscow_week(row["published_at"])
        seen = hashes.setdefault(week.isoformat(), set())
        digest = text_hash(row["text"])
        why = drop_reason(row, min_content_chars)
        if row["multi_vacancy"]:
            out.stats["multi_vacancy"] += 1
        elif digest in seen:
            out.stats["duplicates"] += 1
        elif why is not None:
            out.stats["units"] += 1
            out.stats[f"not_extracted_{why}"] += 1
        else:
            if clock() >= deadline:
                out.left = first_left(eligible[i:])
                out.stats["left_over"] = len(eligible) - i
                break
            text = str(row["text"] or "")
            started = clock()
            answer = extract(
                input_for_record({"source": "telegram", "title": None, "text": text, "payload": row["payload"]})
            )
            out.seconds_extracting += clock() - started
            if answer.error is not None:
                log.error("%s: %s", row["source_id"], answer.error)
                out.stats["extraction_errors"] += 1
                out.left = first_left(eligible[i:])
                out.stats["left_over"] = len(eligible) - i
                break
            out.stats["units"] += 1
            out.stats["extracted"] += 1
            output = checked_output(answer)
            if output is not None:
                fields = json.loads(output)
                out.stats["valid"] += 1
                if fields.get("role") != "non_it":
                    out.stats["it"] += 1
                    # the text the extractor read, as int_extracted_fields checks HTTP and HTTPS in it
                    found = mapper.skills(fields.get("skills") or [], "\n" + text)
                    out.counts.add(week, channel_of(row["source_id"]), fields.get("role"), fields.get("grade"), found)
        seen.add(digest)
    return out


def first_left(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """channel -> the lowest post id among rows not handled."""
    left: dict[str, int] = {}
    for r in rows:
        channel, pid = channel_of(r["source_id"]), post_id(r["source_id"])
        left[channel] = min(left.get(channel, pid), pid)
    return left


def held_back(state: Mapping[str, Any], first: int) -> dict[str, Any]:
    """A cursor moved below post `first`, which was not handled, so that the next daily run reads it again."""
    last_id = first - 1
    recent = {k: v for k, v in state.get("recent", {}).items() if int(k) <= last_id}
    return {**state, "last_id": last_id, "recent": recent}


def extractor(backend: Backend) -> Callable[[str], Answer]:
    def extract(text: str) -> Answer:
        (answer,) = backend.extract([text])
        return answer

    return extract


def run(
    backend: Backend,
    budget_s: float,
    channels: Sequence[str],
    root: Path = AGGREGATES,
    today: date | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + budget_s
    today = today or datetime.now(MSK).date()
    cursors = aggregates.read_cursors(root)
    hashes = aggregates.read_week_hashes(root)
    with tempfile.TemporaryDirectory(prefix="radar-autopilot-") as tmp:
        raw = Path(tmp) / "raw"
        for channel, state in cursors.items():
            telegram.write_state("daily", channel, state, raw)
        with make_client(telegram.TIMEOUT_S) as client:
            entry = telegram.collect(
                Fetcher(client, telegram.PAUSE_S, backoff_s=telegram.BACKOFF_S), list(channels), root=raw
            )
        seconds_collecting = time.monotonic() - started
        new = [
            r
            for r in latest_versions(raw)
            if (last := cursors.get(channel_of(r["source_id"]), {}).get("last_id")) is None
            or post_id(r["source_id"]) > last
        ]
        mapper = SkillMap()
        outcome = process(
            new,
            hashes,
            extractor(backend),
            mapper,
            channel_langs(),
            corpus_config().min_content_chars,
            deadline,
        )
        after = {c: telegram.read_state("daily", c, raw) for c in channels}
    for channel, state in after.items():
        if not state:
            continue
        if channel in outcome.left:
            state = held_back(state, outcome.left[channel])
        aggregates.write_cursor(channel, state, root)
    years = outcome.counts.years()
    if years:
        totals = aggregates.load(root, years)
        totals.merge(outcome.counts)
        aggregates.save(totals, root)
    aggregates.write_week_hashes(hashes, today, root)
    stats = outcome.stats
    return {
        "channels": len(channels),
        "channels_failed": len(entry["failed"]),
        "failed_channels": sorted(entry["failed"]),
        # three channels failed in a row: t.me refuses the runner, the channels after them were not read
        "collector_aborted": int(entry["aborted"]),
        "requests": entry["requests"],
        "posts_fetched": len(new),
        **{k: stats[k] for k in ("vacancy_posts", "multi_vacancy", "duplicates", "units", "extracted", "valid", "it")},
        "not_extracted": {
            k.removeprefix("not_extracted_"): n for k, n in stats.items() if k.startswith("not_extracted_")
        },
        "left_over": stats["left_over"],
        "extraction_errors": stats["extraction_errors"],
        "skill_spellings_unseen_by_seed": mapper.unseen,
        "weeks": sorted({k[0] for k in outcome.counts.posts}),
        "seconds_collecting": round(seconds_collecting, 1),
        "seconds_extracting": round(outcome.seconds_extracting, 1),
        "seconds_total": round(time.monotonic() - started, 1),
    }


def relative(path: Path) -> str:
    full = path.resolve()
    return full.relative_to(ROOT).as_posix() if full.is_relative_to(ROOT) else full.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server serving the student")
    parser.add_argument("--backend", default=BACKEND, help="module:factory of radar.extract.corpus_run")
    parser.add_argument("--time-budget", type=float, default=3000.0, help="seconds; the rest waits for the next run")
    parser.add_argument("--channel", action="append", help="only these channels (repeatable)")
    parser.add_argument("--aggregates", type=Path, default=AGGREGATES)
    parser.add_argument("--result", default=RESULT, help="result file under results/")
    args = parser.parse_args(argv)
    setup_logging()
    backend = load_backend(args.backend, parse_options([f"base_url={args.base_url}"]))
    if backend.model_id != MODEL_ID:
        raise SystemExit(f"the backend is {backend.model_id}, the counts are made with {MODEL_ID}")
    enabled = telegram.enabled_channels()
    channels = args.channel or enabled
    unknown = sorted(set(channels) - set(enabled))
    if unknown:
        parser.error(f"not enabled in {telegram.CONFIG.name}: {', '.join(unknown)}")
    summary = run(backend, args.time_budget, channels, args.aggregates)
    metrics = {k: {"value": v} for k, v in summary.items() if isinstance(v, int | float)}
    write_result(
        args.result,
        metrics,
        source_script="src/radar/autopilot/daily.py",
        model_id=MODEL_ID,
        aggregates=relative(args.aggregates),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        failed_channels=summary["failed_channels"],
        not_extracted=summary["not_extracted"],
        weeks=summary["weeks"],
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
