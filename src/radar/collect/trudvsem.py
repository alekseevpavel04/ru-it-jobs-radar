"""Daily collector of IT vacancies from the trudvsem open-data API.

The IT slice is wide on purpose (docs/sources.md): the InformationTechnology sphere misses most
programmers and the text search matches any word of a phrase, so the union of the sphere and role
queries is collected and the precise IT filter is left to staging. The API serves at most 100
records per page and 100 pages per query; a query above that is recorded as truncated.

The first run (or `--full`) walks every query completely; later runs ask only for records modified
since the last complete run started, with an hour of overlap. Records go to
`data/raw/trudvsem/dt=<run date>/`; the same record twice in a day is written once. The partition
is then published to the topic `raw.trudvsem`; a stream that is down costs a warning, not the run.

    RADAR_USER_AGENT='...' python -m radar.collect.trudvsem [--full] [--summarize]
"""

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radar.collect import setup_logging
from radar.collect.contracts import TrudvsemApiResponse, TrudvsemApiVacancy
from radar.collect.envelope import RawRecord, content_hash
from radar.collect.http import Fetcher, make_client
from radar.collect.summary import append_meta, partition_dates, partition_stats, read_meta
from radar.collect.writer import PartitionWriter
from radar.pii import clean_trudvsem_api
from radar.results import write_result
from radar.stream.producer import publish_after_collect

log = logging.getLogger("radar.collect.trudvsem")

SOURCE = "trudvsem"
API_URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
RESULT_PATH = "collect/trudvsem_api.json"
PAGE_SIZE = 100
MAX_PAGES = 100
PAUSE_S = 2.0
TIMEOUT_S = 60.0
OVERLAP = timedelta(hours=1)
# Partitions are dated by the Moscow calendar day, as the snapshots are.
MSK = timezone(timedelta(hours=3))

# Each query is a set of API parameters; its name goes into the payload as `_queries`.
QUERIES: dict[str, dict[str, str]] = {
    "industry_it": {"industry": "InformationTechnology"},
    "text_программист": {"text": "программист"},
    "text_разработчик": {"text": "разработчик"},
    "text_аналитик": {"text": "аналитик"},
    "text_аналитик данных": {"text": "аналитик данных"},
    "text_инженер данных": {"text": "инженер данных"},
    "text_машинное обучение": {"text": "машинное обучение"},
    "text_data scientist": {"text": "data scientist"},
    "text_data": {"text": "data"},
    "text_python": {"text": "python"},
    "text_devops": {"text": "devops"},
    "text_тестировщик": {"text": "тестировщик"},
    "text_системный администратор": {"text": "системный администратор"},
}
TEXT_SECTIONS = (("duty", "Обязанности"), ("requirements", "Требования"), ("conditions", "Условия"))
# Moves without the vacancy changing, or describes how we found it rather than what it says.
VOLATILE = ("date_modify", "_queries")


@dataclass
class QueryStats:
    total: int | None = None
    pages: int = 0
    records: int = 0
    truncated: bool = False


@dataclass
class RunStats:
    queries: dict[str, QueryStats] = field(default_factory=dict)
    rejected: int = 0


def to_record(vacancy: dict[str, Any], queries: list[str], fetched_at: datetime) -> RawRecord:
    contract = TrudvsemApiVacancy.model_validate(vacancy)
    cleaned, removed = clean_trudvsem_api(vacancy)
    sections: list[str] = []
    for key, label in TEXT_SECTIONS:
        value = cleaned.pop(key, None)
        if isinstance(value, str) and value.strip():
            sections.append(f"{label}: {value}")
    cleaned["_queries"] = sorted(queries)
    title = cleaned.get("job-name")
    text = "\n\n".join(sections) or None
    return RawRecord(
        source=SOURCE,
        source_id=contract.id,
        fetched_at=fetched_at,
        published_at=contract.creation_date,
        modified_at=contract.date_modify,
        url=cleaned.get("vac_url"),
        title=title,
        text=text,
        payload=cleaned,
        content_hash=content_hash(title, text, cleaned, ignore=VOLATILE),
        pii_removed=dict(removed),
    )


def iter_query(
    fetcher: Fetcher, name: str, params: dict[str, str], modified_from: datetime | None, stats: QueryStats
) -> Iterator[tuple[dict[str, Any], datetime]]:
    """Pages of one query; yields each vacancy with the moment its page was requested."""
    for page in range(MAX_PAGES):
        query = {**params, "offset": str(page), "limit": str(PAGE_SIZE)}
        if modified_from is not None:
            query["modifiedFrom"] = modified_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        requested_at = datetime.now(UTC)
        response = fetcher.get(API_URL, params=query)
        response.raise_for_status()
        body = TrudvsemApiResponse.model_validate(response.json())
        stats.total = body.meta.total
        stats.pages += 1
        items = [item.vacancy for item in body.results.vacancies]
        stats.records += len(items)
        if stats.pages % 10 == 0:
            log.info("query %s: %d pages, %d of %s records", name, stats.pages, stats.records, stats.total)
        for item in items:
            yield item, requested_at
        if len(items) < PAGE_SIZE or (page + 1) * PAGE_SIZE >= body.meta.total:
            return
    stats.truncated = stats.total is not None and stats.total > MAX_PAGES * PAGE_SIZE
    log.warning("query %s: %s records, only %d reachable", name, stats.total, MAX_PAGES * PAGE_SIZE)


def last_complete_run(root: Path | None = None) -> dict[str, Any] | None:
    runs = [r for r in read_meta(SOURCE, "runs", root) if r.get("complete")]
    return runs[-1] if runs else None


def collect(fetcher: Fetcher, full: bool, root: Path | None = None) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    previous = None if full else last_complete_run(root)
    modified_from = datetime.fromisoformat(previous["started_at"]) - OVERLAP if previous else None
    mode = "incremental" if modified_from else "full"
    log.info("mode %s%s", mode, f", modifiedFrom {modified_from:%Y-%m-%dT%H:%M:%SZ}" if modified_from else "")

    # A vacancy found by several queries is written once, with all query names in `_queries`.
    found: dict[str, tuple[dict[str, Any], datetime]] = {}
    matched: dict[str, list[str]] = {}
    stats = RunStats()
    for name, params in QUERIES.items():
        stats.queries[name] = QueryStats()
        for vacancy, requested_at in iter_query(fetcher, name, params, modified_from, stats.queries[name]):
            vid = vacancy.get("id")
            if not isinstance(vid, str) or not vid:
                stats.rejected += 1
                continue
            found.setdefault(vid, (vacancy, requested_at))
            matched.setdefault(vid, []).append(name)
        q = stats.queries[name]
        log.info("query %s: total %s, %d pages, %d records", name, q.total, q.pages, q.records)

    dt = started_at.astimezone(MSK).date()
    with PartitionWriter(SOURCE, dt, root=root) as writer:
        for vid, (vacancy, requested_at) in found.items():
            try:
                record = to_record(vacancy, matched[vid], requested_at)
            except ValidationError as exc:
                first = exc.errors()[0]
                writer.reject(vid, ".".join(str(p) for p in first["loc"]) + ": " + first["type"])
                continue
            writer.add(record)
    run = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dt": dt.isoformat(),
        "mode": mode,
        "modified_from": modified_from.isoformat() if modified_from else None,
        "complete": True,
        "queries": {k: vars(v) for k, v in stats.queries.items()},
        "records_fetched": sum(q.records for q in stats.queries.values()),
        "distinct_vacancies": len(found),
        "written": writer.written,
        "duplicates_skipped": writer.duplicates,
        "rejected": writer.rejected + stats.rejected,
        "requests": fetcher.requests,
        "retries": fetcher.retries,
        "seconds_in_requests": round(fetcher.seconds_in_requests, 1),
    }
    append_meta(SOURCE, "runs", run, root)
    return run


def summarize(root: Path | None = None) -> Path:
    """Rebuild `results/collect/trudvsem_api.json` from the partitions and the run log on disk."""
    runs = read_meta(SOURCE, "runs", root)
    days = []
    for dt in partition_dates(SOURCE, root):
        stats = partition_stats(SOURCE, dt, root)
        day_runs = [r for r in runs if r["dt"] == stats["dt"]]
        days.append(
            {
                "dt": stats["dt"],
                "rows": stats["rows"],
                "distinct_ids": stats["distinct_ids"],
                "rejected_not_written": stats["rejected_not_written"],
                "parquet_bytes": stats["parquet_bytes"],
                "pii_removed": stats["pii_removed"],
                "runs": day_runs,
            }
        )
    last = runs[-1] if runs else None
    metrics: dict[str, dict[str, Any]] = {
        "days": {"value": len(days)},
        "rows_total": {"value": sum(d["rows"] for d in days)},
    }
    if last is not None:
        it_total = last["queries"]["industry_it"]["total"]
        metrics["last_run_distinct_vacancies"] = {"value": last["distinct_vacancies"]}
        metrics["last_run_written"] = {"value": last["written"]}
        metrics["last_run_industry_it_total"] = {"value": it_total}
        metrics["last_run_requests"] = {"value": last["requests"]}
        truncated = [k for k, q in last["queries"].items() if q["truncated"]]
        metrics["last_run_truncated_queries"] = {
            "value": len(truncated),
            "n": len(truncated),
            "denominator": len(last["queries"]),
        }
    return write_result(
        RESULT_PATH,
        metrics,
        source_script="src/radar/collect/trudvsem.py",
        method="recomputed from data/raw/trudvsem partitions and _meta/runs.jsonl",
        days=days,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--full", action="store_true", help="walk every query completely")
    parser.add_argument("--summarize", action="store_true", help="only rebuild the results file from disk")
    args = parser.parse_args(argv)
    setup_logging()
    if not args.summarize:
        started = time.monotonic()
        with make_client(TIMEOUT_S) as client:
            run = collect(Fetcher(client, PAUSE_S), full=args.full)
        log.info(
            "done in %.0f s: %d distinct vacancies, %d written, %d rejected",
            time.monotonic() - started,
            run["distinct_vacancies"],
            run["written"],
            run["rejected"],
        )
        publish_after_collect(SOURCE, date.fromisoformat(run["dt"]))
    log.info("results: %s", summarize())
    return 0


if __name__ == "__main__":
    sys.exit(main())
