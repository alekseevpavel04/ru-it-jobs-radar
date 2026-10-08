"""Daily collector of Habr Career: a full snapshot of the active list plus cards of new and changed vacancies.

The list API stops at 20 pages of 50, below the site's own counter, so the list is walked several
ways and united by id: by date and by salary both ways, then each specialization group of the
site's filter (a group that is itself above the cap is split into its specializations). The share
of the counter reached is measured and stored with every run.

The list has no description; it comes from the card page. A card is fetched when the vacancy is new
or its list entry changed since the latest stored day; otherwise the description is carried from
that day, with `card_fetched_at` telling when it was read and `card_list_hash` what the list said
then. Cards are read slowly and at most `MAX_CARDS` per run (new vacancies first, newest first); a
vacancy whose card waits keeps the text of its last card, or none, and is read on a later run. If the
site stops answering, the remaining cards wait and the list snapshot is still written whole. The
partition is then published to the topic `raw.habr`; a stream that is down costs a warning, not the run.

    RADAR_USER_AGENT='...' python -m radar.collect.habr [--summarize]
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import lxml.html
import pyarrow.parquet as pq
from pydantic import ValidationError

from radar.collect import setup_logging
from radar.collect.contracts import HabrCard, HabrListItem, HabrListResponse, parse_habr_card
from radar.collect.envelope import RawRecord, content_hash, parse_source_time
from radar.collect.http import Fetcher, make_client
from radar.collect.summary import append_meta, partition_dates, partition_stats, read_meta
from radar.collect.writer import PartitionWriter, partition_dir
from radar.pii import clean_habr_card_company, clean_habr_vacancy, clean_text
from radar.results import write_result
from radar.stream.producer import publish_after_collect

log = logging.getLogger("radar.collect.habr")

SOURCE = "habr"
BASE = "https://career.habr.com"
LIST_URL = f"{BASE}/api/frontend/vacancies"
FILTERS_URL = f"{BASE}/vacancies?type=all"
RESULT_PATH = "collect/habr.json"
PER_PAGE = 50
MAX_PAGES = 20
PAUSE_S = 2.0
# Card pages are HTML behind QRATOR, which blocked the address after about 400 requests at 2-3 s
# on 2026-10-08; cards go slower and at most MAX_CARDS per run, the rest waits for the next run.
CARD_PAUSE_S = 6.0
MAX_CARDS = 150
TIMEOUT_S = 30.0
MSK = timezone(timedelta(hours=3))

# List fields that change without the vacancy changing: viewer state, reactions, the "raised" date.
LIST_VOLATILE = (
    "reactions",
    "favorite",
    "can_edit",
    "response",
    "publishedDate",
    "isMarked",
    "userVacancyBanHref",
    "quickResponseHref",
    "hidden",
)
RECORD_VOLATILE = (*LIST_VOLATILE, "list_hash", "card_list_hash", "card_fetched_at", "_passes")
CARD_FIELDS = ("datePosted", "validThrough", "employmentType", "jobLocationType", "jobLocation", "baseSalary")
_BLOCKS = ("p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote", "pre")
_SSR = re.compile(r'<script type="application/json" data-ssr-state="true">(.*?)</script>', re.DOTALL)


@dataclass
class Pass:
    name: str
    params: list[tuple[str, str]]
    counter: int | None = None
    pages: int = 0
    items: int = 0
    capped: bool = False


@dataclass
class Listing:
    items: dict[int, dict[str, Any]] = field(default_factory=dict)
    fetched_at: dict[int, datetime] = field(default_factory=dict)
    passes: dict[int, list[str]] = field(default_factory=dict)
    stats: list[Pass] = field(default_factory=list)
    site_counter: int | None = None
    # One entry per id: the same broken item shows up in several passes.
    rejected: dict[str | None, str] = field(default_factory=dict)


def html_to_text(fragment: str) -> str:
    if not fragment.strip():
        return ""
    root = lxml.html.fragment_fromstring(fragment, create_parent="div")
    for br in root.iter("br"):
        br.tail = "\n" + (br.tail or "")
    for li in root.iter("li"):
        li.text = "- " + (li.text or "")
    for block in root.iter(*_BLOCKS):
        block.tail = "\n" + (block.tail or "")
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in root.text_content().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_groups(html: str) -> list[tuple[str, list[int]]]:
    """Specialization groups of the site's filter: (title, specialization ids).

    Membership comes from each specialization's `parent_id`: the groups' own `children` lists miss
    the newest specializations.
    """
    match = _SSR.search(html)
    if match is None:
        raise ValueError("filters page without SSR state")
    state = json.loads(match.group(1))
    titles = {int(g["id"]): str(g["title"]) for g in state["search"]["groups"]}
    members: dict[int, list[int]] = {gid: [] for gid in titles}
    for option in state["options"]["s"]:
        members.setdefault(int(option["parent_id"]), []).append(int(option["value"]))
    return [(titles.get(gid, f"group {gid}"), specs) for gid, specs in members.items() if specs]


def walk(fetcher: Fetcher, p: Pass, listing: Listing) -> None:
    for page in range(1, MAX_PAGES + 1):
        requested_at = datetime.now(UTC)
        response = fetcher.get(LIST_URL, params=[*p.params, ("per_page", str(PER_PAGE)), ("page", str(page))])
        response.raise_for_status()
        body = HabrListResponse.model_validate(response.json())
        p.counter = body.meta.totalResults
        p.pages += 1
        p.capped = body.meta.totalPages > MAX_PAGES
        for raw in body.list_:
            p.items += 1
            try:
                item = HabrListItem.model_validate(raw)
            except ValidationError as exc:
                first = exc.errors()[0]
                vid = raw.get("id")
                listing.rejected[str(vid) if vid is not None else None] = f"{first['loc'][0]}: {first['type']}"
                continue
            listing.items.setdefault(item.id, raw)
            listing.fetched_at.setdefault(item.id, requested_at)
            listing.passes.setdefault(item.id, []).append(p.name)
        if page >= body.meta.totalPages:
            break
    listing.stats.append(p)
    log.info(
        "pass %s: counter %s, %d pages, %d items, %d distinct so far",
        p.name,
        p.counter,
        p.pages,
        p.items,
        len(listing.items),
    )


def collect_list(fetcher: Fetcher) -> Listing:
    listing = Listing()
    base = [("type", "all"), ("sort", "date")]
    walk(fetcher, Pass("all_by_date", base), listing)
    listing.site_counter = listing.stats[0].counter
    # Other orders show other windows of the same capped list.
    for order in ("salary_desc", "salary_asc"):
        walk(fetcher, Pass(f"all_by_{order}", [("type", "all"), ("sort", order)]), listing)
    response = fetcher.get(FILTERS_URL)
    response.raise_for_status()
    for title, specs in parse_groups(response.text):
        group = Pass(f"group:{title}", base + [("s[]", str(s)) for s in specs])
        walk(fetcher, group, listing)
        if group.capped:
            for spec in specs:
                walk(fetcher, Pass(f"spec:{spec}", [*base, ("s[]", str(spec))]), listing)
    return listing


def latest_previous(dt: date, root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Per vacancy id, what the latest stored day up to `dt` knew: list hash, text and card fields."""
    days = [d for d in partition_dates(SOURCE, root) if d <= dt]
    if not days:
        return {}
    known: dict[str, dict[str, Any]] = {}
    for part in sorted(partition_dir(SOURCE, days[-1], root).glob("part-*.parquet")):
        table = pq.read_table(part, columns=["source_id", "text", "payload", "published_at"])
        for row in table.to_pylist():
            payload = json.loads(row["payload"])
            if payload.get("card_fetched_at"):
                known[row["source_id"]] = {
                    "card_list_hash": payload.get("card_list_hash") or payload.get("list_hash"),
                    "text": row["text"],
                    "card": payload.get("card"),
                    "company_card": payload.get("company_card"),
                    "card_fetched_at": payload["card_fetched_at"],
                    "published_at": row["published_at"],
                }
    return known


def card_fields(card: HabrCard) -> tuple[str, dict[str, Any], dict[str, Any], Counter[str]]:
    text, removed = clean_text(html_to_text(card.vacancy.description))
    posting = card.job_posting.model_dump(by_alias=True)
    fields = {k: posting[k] for k in CARD_FIELDS if posting.get(k) is not None}
    company, company_removed = clean_habr_card_company(
        {k: v for k, v in card.company.items() if k in ("id", "name", "href", "url", "description")}
    )
    return text, fields, company, removed + company_removed


def list_entry_hash(raw: dict[str, Any]) -> str:
    """What the list says about a vacancy, cleaned; a change here means the card is read again."""
    cleaned, _ = clean_habr_vacancy(raw)
    return content_hash(None, None, cleaned, ignore=LIST_VOLATILE)


def build_record(
    raw: dict[str, Any],
    fetched_at: datetime,
    passes: list[str],
    card: dict[str, Any] | None,
) -> RawRecord:
    item = HabrListItem.model_validate(raw)
    cleaned, removed = clean_habr_vacancy(raw)
    payload: dict[str, Any] = {**cleaned, "list_hash": list_entry_hash(raw), "_passes": sorted(passes)}
    text = None
    published_at = None
    if card is not None:
        text = card["text"]
        payload["card"] = card["card"]
        payload["company_card"] = card["company_card"]
        payload["card_fetched_at"] = card["card_fetched_at"]
        payload["card_list_hash"] = card["card_list_hash"]
        removed.update(card.get("pii_removed", {}))
        published_at = card["published_at"]
    return RawRecord(
        source=SOURCE,
        source_id=str(item.id),
        fetched_at=fetched_at,
        published_at=published_at,
        modified_at=None,
        url=BASE + item.href,
        title=item.title,
        text=text,
        payload=payload,
        content_hash=content_hash(item.title, text, payload, ignore=RECORD_VOLATILE),
        pii_removed=dict(+removed),
    )


@dataclass
class CardStats:
    fetched: int = 0
    carried: int = 0
    gone: int = 0
    failed: int = 0
    deferred: int = 0
    blocked: bool = False


def fetch_card(fetcher: Fetcher, vid: int, stats: CardStats) -> dict[str, Any] | None:
    requested_at = datetime.now(UTC)
    response = fetcher.get(f"{BASE}/vacancies/{vid}")
    if response.status_code in (404, 410):
        stats.gone += 1
        return None
    response.raise_for_status()
    try:
        card = parse_habr_card(response.text)
    except (ValueError, ValidationError) as exc:
        stats.failed += 1
        log.warning("card %s: %s", vid, type(exc).__name__)
        return None
    if card.vacancy.id != vid:
        stats.failed += 1
        log.warning("card %s: page is about vacancy %s", vid, card.vacancy.id)
        return None
    stats.fetched += 1
    text, fields, company, removed = card_fields(card)
    posted = card.job_posting.datePosted
    return {
        "text": text,
        "card": fields,
        "company_card": company,
        "card_fetched_at": requested_at.isoformat(),
        "published_at": parse_source_time(posted),
        "pii_removed": dict(removed),
    }


def collect(
    fetcher: Fetcher,
    root: Path | None = None,
    card_fetcher: Fetcher | None = None,
    max_cards: int = MAX_CARDS,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    dt = started_at.astimezone(MSK).date()
    listing = collect_list(fetcher)
    previous = latest_previous(dt, root)
    hashes = {vid: list_entry_hash(raw) for vid, raw in listing.items.items()}
    stale = [
        vid for vid in listing.items if (k := previous.get(str(vid))) is None or k["card_list_hash"] != hashes[vid]
    ]
    # New vacancies before changed ones, the most recently raised first.
    stale.sort(key=lambda vid: str(listing.items[vid]["publishedDate"]["date"]), reverse=True)
    stale.sort(key=lambda vid: str(vid) in previous)
    to_fetch = set(stale[:max_cards])
    cards = CardStats()
    with PartitionWriter(SOURCE, dt, root=root) as writer:
        for source_id, reason in listing.rejected.items():
            writer.reject(source_id, reason)
        for vid, raw in listing.items.items():
            known = previous.get(str(vid))
            card: dict[str, Any] | None = known
            if vid not in stale:
                cards.carried += 1
            elif vid in to_fetch and not cards.blocked:
                try:
                    fresh = fetch_card(card_fetcher or fetcher, vid, cards)
                except RuntimeError as exc:
                    cards.blocked = True
                    log.warning("cards stopped, the rest waits for the next run: %s", exc)
                    fresh = None
                    cards.deferred += 1
                if fresh is not None:
                    card = {**fresh, "card_list_hash": hashes[vid]}
            else:
                cards.deferred += 1
            record = build_record(raw, listing.fetched_at[vid], listing.passes[vid], card)
            writer.add(record)
    counter = listing.site_counter or 0
    run = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dt": dt.isoformat(),
        "complete": True,
        "site_counter": listing.site_counter,
        "distinct_listed": len(listing.items),
        "passes": [{k: v for k, v in vars(p).items() if k != "params"} for p in listing.stats],
        "capped_passes": [p.name for p in listing.stats if p.capped],
        "cards": vars(cards),
        "written": writer.written,
        "duplicates_skipped": writer.duplicates,
        "rejected": writer.rejected,
        "requests": fetcher.requests + (card_fetcher.requests if card_fetcher else 0),
        "retries": fetcher.retries + (card_fetcher.retries if card_fetcher else 0),
        "seconds_in_requests": round(
            fetcher.seconds_in_requests + (card_fetcher.seconds_in_requests if card_fetcher else 0), 1
        ),
        "coverage": {"n": len(listing.items), "denominator": counter},
    }
    append_meta(SOURCE, "runs", run, root)
    return run


def summarize(root: Path | None = None) -> Path:
    """Rebuild `results/collect/habr.json` from the partitions and the run log on disk."""
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
        n, denominator = last["coverage"]["n"], last["coverage"]["denominator"]
        metrics["last_run_coverage_of_site_counter"] = {
            "value": round(n / denominator, 4) if denominator else 0,
            "n": n,
            "denominator": denominator,
        }
        metrics["last_run_cards_fetched"] = {"value": last["cards"]["fetched"]}
        metrics["last_run_cards_carried"] = {"value": last["cards"]["carried"]}
        metrics["last_run_cards_failed"] = {"value": last["cards"]["failed"] + last["cards"]["gone"]}
        metrics["last_run_cards_deferred"] = {"value": last["cards"]["deferred"]}
        metrics["last_run_requests"] = {"value": last["requests"]}
    return write_result(
        RESULT_PATH,
        metrics,
        source_script="src/radar/collect/habr.py",
        method="recomputed from data/raw/habr partitions and _meta/runs.jsonl",
        days=days,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--summarize", action="store_true", help="only rebuild the results file from disk")
    parser.add_argument("--max-cards", type=int, default=MAX_CARDS, help="card pages to read in this run")
    args = parser.parse_args(argv)
    setup_logging()
    if not args.summarize:
        started = time.monotonic()
        with make_client(TIMEOUT_S) as client:
            run = collect(
                Fetcher(client, PAUSE_S), card_fetcher=Fetcher(client, CARD_PAUSE_S), max_cards=args.max_cards
            )
        log.info(
            "done in %.0f s: %d of %s listed, cards %s, %d written",
            time.monotonic() - started,
            run["distinct_listed"],
            run["site_counter"],
            run["cards"],
            run["written"],
        )
        publish_after_collect(SOURCE, date.fromisoformat(run["dt"]))
    log.info("results: %s", summarize())
    return 0


if __name__ == "__main__":
    sys.exit(main())
