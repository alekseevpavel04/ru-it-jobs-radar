"""Daily collector of IT vacancies from the SuperJob API.

Every run is a full snapshot of the active IT slice: the catalogue "IT, Интернет, связь, телеком"
(33) plus title queries for IT roles that employers often file under other catalogues: most
programmers and system administrators are outside 33 (docs/sources.md). As for trudvsem, the slice
is wide on purpose and the precise IT filter is left to staging.

The API gives at most 500 records per query and about 40 per page whatever `count` asks (plus a few
promoted vacancies on the first pages). A query above that is split into windows of
`date_published`, halved until each fits, and walked oldest first: a vacancy raised during the run
moves into a window still ahead. The share of the query's `total` reached is stored with every run.

Search results carry the full text (`vacancyRichText`), so no card is requested. Contacts, the
street address, coordinates and metro are removed before writing (`clean_superjob_vacancy`). The
partition is then published to the topic `raw.superjob`; a stream that is down costs a warning, not
the run. Without `SUPERJOB_APP_KEY` the run is skipped.

    RADAR_USER_AGENT='...' SUPERJOB_APP_KEY='...' python -m radar.collect.superjob [--summarize]
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radar.collect import setup_logging
from radar.collect.contracts import SuperjobResponse, SuperjobVacancy
from radar.collect.envelope import RawRecord, content_hash
from radar.collect.habr import html_to_text
from radar.collect.http import Fetcher, make_client
from radar.collect.summary import append_meta, partition_dates, partition_stats, read_meta
from radar.collect.writer import PartitionWriter
from radar.config import optional
from radar.pii import clean_superjob_vacancy, clean_text
from radar.results import write_result
from radar.stream.producer import publish_after_collect

log = logging.getLogger("radar.collect.superjob")

SOURCE = "superjob"
API_URL = "https://api.superjob.ru/2.0/vacancies/"
RESULT_PATH = "collect/superjob.json"
KEY_VAR = "SUPERJOB_APP_KEY"
PAGE_SIZE = 40
MAX_RECORDS = 500
# A window above this many records is split: its last page would cross MAX_RECORDS.
SPLIT_ABOVE = MAX_RECORDS // PAGE_SIZE * PAGE_SIZE
MIN_WINDOW_S = 60
# The documented limit is 120 requests a minute from one address; a whole run is about 60 requests.
PAUSE_S = 2.0
TIMEOUT_S = 30.0
MSK = timezone(timedelta(hours=3))

CATALOGUE_IT = "catalogue_33"
TITLE_PHRASES = (
    "программист",
    "разработчик",
    "developer",
    "аналитик",
    "data",
    "python",
    "java",
    "frontend",
    "backend",
    "devops",
    "тестировщик",
    "qa",
    "1С",
    "системный администратор",
    "machine learning",
    "инженер данных",
)
# Each query is a set of API parameters; its name goes into the payload as `_queries`.
QUERIES: dict[str, dict[str, str]] = {
    CATALOGUE_IT: {"catalogues": "33"},
    **{
        f"title_{phrase}": {"keywords[0][srws]": "1", "keywords[0][skwc]": "particular", "keywords[0][keys]": phrase}
        for phrase in TITLE_PHRASES
    },
}
TEXT_SECTIONS = (("work", "Обязанности"), ("candidat", "Требования"), ("compensation", "Условия"))
TEXT_FIELDS = ("vacancyRichText", "work", "candidat", "compensation")
# Moves without the vacancy changing: the raise date and its expiry, viewer state, the employer's
# profile (its vacancy count changes daily), and how we found the vacancy.
VOLATILE = (
    "date_published",
    "date_pub_to",
    "date_archived",
    "highlight",
    "canEdit",
    "favorite",
    "already_sent_on_vacancy",
    "rejected",
    "response_info",
    "client",
    "_queries",
)

Sink = Callable[[dict[str, Any], datetime], None]


@dataclass
class QueryStats:
    total: int | None = None
    windows: int = 0
    pages: int = 0
    records: int = 0
    distinct: int = 0
    truncated: bool = False


def fetch_page(fetcher: Fetcher, params: dict[str, str], page: int) -> tuple[SuperjobResponse, datetime]:
    requested_at = datetime.now(UTC)
    response = fetcher.get(API_URL, params={**params, "count": str(PAGE_SIZE), "page": str(page)})
    response.raise_for_status()
    return SuperjobResponse.model_validate(response.json()), requested_at


def consume(
    fetcher: Fetcher,
    params: dict[str, str],
    first: tuple[SuperjobResponse, datetime],
    stats: QueryStats,
    sink: Sink,
) -> None:
    body, requested_at = first
    page = 0
    while True:
        stats.pages += 1
        stats.records += len(body.objects)
        for raw in body.objects:
            sink(raw, requested_at)
        page += 1
        if not body.more:
            return
        if page * PAGE_SIZE >= MAX_RECORDS:
            stats.truncated = True
            return
        body, requested_at = fetch_page(fetcher, params, page)


def walk_window(fetcher: Fetcher, params: dict[str, str], lo: int, hi: int, stats: QueryStats, sink: Sink) -> None:
    # Bounds overlap by one second between halves: whether the API counts them inclusively is not
    # documented, and the sink folds a vacancy seen twice.
    window = {**params, "date_published_from": str(lo), "date_published_to": str(hi)}
    first = fetch_page(fetcher, window, 0)
    if first[0].total > SPLIT_ABOVE and hi - lo >= MIN_WINDOW_S:
        mid = (lo + hi) // 2
        walk_window(fetcher, params, lo, mid, stats, sink)
        walk_window(fetcher, params, mid, hi, stats, sink)
        return
    stats.windows += 1
    if first[0].total > SPLIT_ABOVE:
        stats.truncated = True
        log.warning("window %d-%d: %d records, only %d reachable", lo, hi, first[0].total, MAX_RECORDS)
    consume(fetcher, window, first, stats, sink)


def walk(fetcher: Fetcher, name: str, params: dict[str, str], stats: QueryStats, sink: Sink) -> None:
    first = fetch_page(fetcher, params, 0)
    stats.total = first[0].total
    if stats.total <= SPLIT_ABOVE:
        stats.windows = 1
        consume(fetcher, params, first, stats, sink)
    else:
        # From the epoch: sorting by date ascending does not start at the oldest date_published
        # (on 2026-10-12 it began 3.2 days back while 261 of 660 were older), and the empty halves
        # on the way down cost one request each. A day ahead: vacancies raised while the run goes on
        # stay inside the last window.
        walk_window(fetcher, params, 0, int(time.time()) + 86_400, stats, sink)
    log.info(
        "query %s: total %s, %d windows, %d pages, %d records",
        name,
        stats.total,
        stats.windows,
        stats.pages,
        stats.records,
    )


def vacancy_text(sections: dict[str, Any]) -> str | None:
    rich = sections.get("vacancyRichText")
    if isinstance(rich, str) and rich.strip():
        return html_to_text(rich) or None
    parts = [
        f"{label}: {html_to_text(value)}"
        for key, label in TEXT_SECTIONS
        if isinstance(value := sections.get(key), str) and value.strip()
    ]
    return "\n\n".join(parts) or None


def to_record(vacancy: dict[str, Any], queries: list[str], fetched_at: datetime) -> RawRecord:
    contract = SuperjobVacancy.model_validate(vacancy)
    body = dict(vacancy)
    sections = {key: body.pop(key, None) for key in TEXT_FIELDS}
    cleaned, counts = clean_superjob_vacancy(body)
    text = vacancy_text(sections)
    if text is not None:
        text, removed = clean_text(text)
        counts.update(removed)
    cleaned["_queries"] = sorted(queries)
    title = cleaned["profession"]
    return RawRecord(
        source=SOURCE,
        source_id=str(contract.id),
        fetched_at=fetched_at,
        published_at=datetime.fromtimestamp(contract.date_published, UTC),
        modified_at=None,
        url=contract.link,
        title=title,
        text=text,
        payload=cleaned,
        content_hash=content_hash(title, text, cleaned, ignore=VOLATILE),
        pii_removed=dict(+counts),
    )


def collect(fetcher: Fetcher, root: Path | None = None) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    # A vacancy found by several queries is written once, with all query names in `_queries`.
    found: dict[str, tuple[dict[str, Any], datetime]] = {}
    matched: dict[str, list[str]] = {}
    stats: dict[str, QueryStats] = {}
    unidentified = 0

    for name, params in QUERIES.items():
        seen: set[str] = set()

        def sink(raw: dict[str, Any], requested_at: datetime, name: str = name, seen: set[str] = seen) -> None:
            nonlocal unidentified
            vid = raw.get("id")
            if not isinstance(vid, int):
                unidentified += 1
                return
            key = str(vid)
            found.setdefault(key, (raw, requested_at))
            if key not in seen:
                seen.add(key)
                matched.setdefault(key, []).append(name)

        stats[name] = QueryStats()
        walk(fetcher, name, params, stats[name], sink)
        stats[name].distinct = len(seen)

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
    catalogue = stats[CATALOGUE_IT]
    run = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dt": dt.isoformat(),
        "complete": True,
        "queries": {k: vars(v) for k, v in stats.items()},
        "records_fetched": sum(q.records for q in stats.values()),
        "distinct_vacancies": len(found),
        "only_by_title": sum(1 for names in matched.values() if CATALOGUE_IT not in names),
        "written": writer.written,
        "duplicates_skipped": writer.duplicates,
        "rejected": writer.rejected,
        # Occurrences, not vacancies: an object without an id cannot be told apart across queries.
        "objects_without_id": unidentified,
        "requests": fetcher.requests,
        "retries": fetcher.retries,
        "seconds_in_requests": round(fetcher.seconds_in_requests, 1),
        "catalogue_coverage": {"n": catalogue.distinct, "denominator": catalogue.total or 0},
    }
    append_meta(SOURCE, "runs", run, root)
    return run


def summarize(root: Path | None = None) -> Path:
    """Rebuild `results/collect/superjob.json` from the partitions and the run log on disk."""
    runs = read_meta(SOURCE, "runs", root)
    days = []
    for dt in partition_dates(SOURCE, root):
        stats = partition_stats(SOURCE, dt, root)
        days.append(
            {
                "dt": stats["dt"],
                "rows": stats["rows"],
                "distinct_ids": stats["distinct_ids"],
                "rejected_not_written": stats["rejected_not_written"],
                "parquet_bytes": stats["parquet_bytes"],
                "pii_removed": stats["pii_removed"],
                "runs": [r for r in runs if r["dt"] == stats["dt"]],
            }
        )
    metrics: dict[str, dict[str, Any]] = {
        "days": {"value": len(days)},
        "rows_total": {"value": sum(d["rows"] for d in days)},
    }
    if runs:
        last = runs[-1]
        n, denominator = last["catalogue_coverage"]["n"], last["catalogue_coverage"]["denominator"]
        metrics["last_run_distinct_vacancies"] = {"value": last["distinct_vacancies"]}
        metrics["last_run_only_by_title"] = {
            "value": last["only_by_title"],
            "n": last["only_by_title"],
            "denominator": last["distinct_vacancies"],
        }
        metrics["last_run_catalogue_total"] = {"value": denominator}
        metrics["last_run_catalogue_coverage"] = {
            "value": round(n / denominator, 4) if denominator else 0,
            "n": n,
            "denominator": denominator,
        }
        metrics["last_run_written"] = {"value": last["written"]}
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
        source_script="src/radar/collect/superjob.py",
        method="recomputed from data/raw/superjob partitions and _meta/runs.jsonl",
        days=days,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--summarize", action="store_true", help="only rebuild the results file from disk")
    args = parser.parse_args(argv)
    setup_logging()
    if not args.summarize:
        key = optional(KEY_VAR)
        if key is None:
            log.warning("%s is not set, SuperJob is skipped", KEY_VAR)
            return 0
        started = time.monotonic()
        with make_client(TIMEOUT_S) as client:
            client.headers["X-Api-App-Id"] = key
            run = collect(Fetcher(client, PAUSE_S))
        log.info(
            "done in %.0f s: %d distinct vacancies, %d of %d in catalogue 33, %d written, %d rejected",
            time.monotonic() - started,
            run["distinct_vacancies"],
            run["catalogue_coverage"]["n"],
            run["catalogue_coverage"]["denominator"],
            run["written"],
            run["rejected"],
        )
        publish_after_collect(SOURCE, date.fromisoformat(run["dt"]))
    log.info("results: %s", summarize())
    return 0


if __name__ == "__main__":
    sys.exit(main())
