"""Telegram collector: public channels through `t.me/s/<channel>`, for weekly site aggregates and the
extraction corpus (DECISIONS 10).

Posts stay in the local raw layer and are never published (DECISIONS 2). Personal data is cut before writing (DECISIONS 3): phones, emails, people's
@names and t.me links to people; the channels of configs/telegram_channels.toml keep their names.
Each post is classified by rules v2 (radar.collect.telegram_rules) on its raw text and links; only
the verdict, the cleaned text and the hosts of job-board links are stored.

Partitions are dated by the Moscow day the run started, as for the other API sources, not
by the post date: the raw layer says what was seen when, an edit lands as a new content_hash on the
day it is seen, and staging takes the week of a post from `published_at`.

Daily mode reads each enabled channel from the newest page back to the last stored post id; a stored
post on those pages is written again only if its text changed (edits of older posts go unseen).
Deleted posts are not recorded. The
first daily run of a channel reads its newest page only, except DAILY_ONLY channels, which are read
to their first live post. Backfill walks `?before=` from the oldest post the daily mode stored to
the channel's first post; its cursor is saved per channel in `_meta/state/` after every flush, so a
broken run resumes where it stopped (pages read since the last flush are read again, their keys are
skipped). Both modes hold one lock, so t.me sees a single stream of requests, PAUSE_S apart.

A run killed hard leaves its partition without `_SUCCESS`; `--close-partition` marks it complete
once the saved cursors show that no run will store its posts again (see `close_partition`).

    RADAR_USER_AGENT='...' python -m radar.collect.telegram [--backfill] [--channel NAME] [--summarize]
    python -m radar.collect.telegram --close-partition YYYY-MM-DD
"""

import argparse
import json
import logging
import os
import sys
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import ValidationError

from radar.collect import setup_logging
from radar.collect.envelope import RawRecord, content_hash
from radar.collect.http import Fetcher, make_client
from radar.collect.lock import AlreadyRunning, single_instance
from radar.collect.summary import append_meta, meta_dir, partition_dates, partition_stats, read_meta
from radar.collect.telegram_page import Page, Post, job_links, page_url, parse, post_url
from radar.collect.telegram_rules import RULES_VERSION, classify
from radar.collect.writer import SUCCESS, PartitionWriter, partition_dir, raw_root
from radar.config import ROOT
from radar.pii import DEFAULT_KEEP_HANDLES, clean_text
from radar.results import write_result
from radar.stream.producer import publish_after_collect

log = logging.getLogger("radar.collect.telegram")

SOURCE = "telegram"
CONFIG = ROOT / "configs" / "telegram_channels.toml"
RESULT_PATH = "collect/telegram.json"
BACKFILL_RESULT_PATH = "collect/backfill_summary.json"
# The 1.5 survey made 1 012 requests at this pace without a single 429.
PAUSE_S = 3.0
BACKOFF_S = (60.0, 120.0, 300.0)
TIMEOUT_S = 30.0
MSK = timezone(timedelta(hours=3))
FLUSH_PAGES = 50
# Stored hashes of the newest posts per channel: an edit is noticed while the post is this fresh.
RECENT_IDS = 200
MAX_FAILED_IN_ROW = 3
# A part this fresh may belong to a run that is still going.
CLOSE_MIN_AGE = timedelta(minutes=30)
# Payload fields that change while the post does not, or that the rules derive from it.
VOLATILE = ("views", "edited", "rule_kind", "vacancies_in_post", "rules_version")


class ChannelError(RuntimeError):
    pass


def load_channels(path: Path = CONFIG) -> list[dict[str, Any]]:
    return list(tomllib.loads(path.read_text(encoding="utf-8"))["channel"])


def enabled_channels(path: Path = CONFIG) -> list[str]:
    return [c["name"] for c in load_channels(path) if c["enabled"]]


# Channels with `backfill = false` keep only recent posts and delete older ids: nothing to backfill.
DAILY_ONLY = frozenset(c["name"] for c in load_channels() if c.get("backfill") is False)


def keep_handles(path: Path = CONFIG) -> frozenset[str]:
    """Public channels whose @names and t.me links are not personal data."""
    return DEFAULT_KEEP_HANDLES | {c["name"].lower() for c in load_channels(path)}


def build_record(channel: str, post: Post, fetched_at: datetime, keep: frozenset[str]) -> RawRecord:
    verdict = classify(post.text, post.links)
    text, removed = clean_text(post.text, keep)
    stored = text.strip() or None
    payload: dict[str, Any] = {
        "channel": channel,
        "views": post.views,
        "edited": post.edited,
        "forwarded": post.forwarded,
        "forwarded_from": post.forwarded_from,
        "media": post.media,
        "job_links": job_links(post.links),
        "rule_kind": verdict.kind,
        "vacancies_in_post": verdict.n_units,
        "rules_version": RULES_VERSION,
    }
    return RawRecord(
        source=SOURCE,
        source_id=f"{channel}/{post.id}",
        fetched_at=fetched_at,
        published_at=datetime.fromisoformat(post.date),
        modified_at=None,
        url=post_url(channel, post.id),
        title=None,
        text=stored,
        payload=payload,
        content_hash=content_hash(None, stored, payload, ignore=VOLATILE),
        pii_removed=dict(+removed),
        is_vacancy=verdict.vacancy,
        multi_vacancy=verdict.vacancy and verdict.multi,
    )


def state_path(kind: str, channel: str, root: Path | None = None) -> Path:
    return meta_dir(SOURCE, root) / "state" / f"{channel}.{kind}.json"


def read_state(kind: str, channel: str, root: Path | None = None) -> dict[str, Any]:
    path = state_path(kind, channel, root)
    if not path.exists():
        return {}
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def write_state(kind: str, channel: str, state: dict[str, Any], root: Path | None = None) -> None:
    path = state_path(kind, channel, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Run:
    fetcher: Fetcher
    writer: PartitionWriter
    keep: frozenset[str]
    root: Path | None = None
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def state(self, kind: str, channel: str) -> dict[str, Any]:
        return dict(self.pending.get((kind, channel)) or read_state(kind, channel, self.root))

    def save(self, kind: str, channel: str, state: dict[str, Any]) -> None:
        self.pending[(kind, channel)] = dict(state)

    def commit(self) -> None:
        """Parts first, then the cursors that point past them: a crash in between rereads, never skips."""
        self.writer.flush()
        for (kind, channel), state in self.pending.items():
            write_state(kind, channel, state, self.root)
        self.pending.clear()

    def stats(self, channel: str) -> dict[str, Any]:
        return self.channels.setdefault(
            channel, {"pages": 0, "posts_seen": 0, "written": 0, "vacancy_written": 0, "edits": 0, "rejected": 0}
        )

    def page(self, channel: str, before: int | None = None) -> tuple[Page, datetime]:
        fetched_at = datetime.now(UTC)
        response = self.fetcher.get(page_url(channel, before))
        page = parse(response.status_code, fetched_at.isoformat(), response.text)
        self.stats(channel)["pages"] += 1
        if not page.exists:
            raise ChannelError(f"{channel}: no public preview (HTTP {response.status_code})")
        return page, fetched_at

    def record(self, channel: str, post: Post, fetched_at: datetime) -> RawRecord | None:
        self.stats(channel)["posts_seen"] += 1
        try:
            return build_record(channel, post, fetched_at, self.keep)
        except ValidationError as exc:
            first = exc.errors()[0]
            reason = f"{first['loc'][0] if first['loc'] else 'record'}: {first['type']}"
        except ValueError as exc:
            reason = f"{type(exc).__name__}: post date {post.date!r}"
        self.stats(channel)["rejected"] += 1
        self.writer.reject(f"{channel}/{post.id}", reason)
        return None

    def write(self, channel: str, record: RawRecord, edit: bool = False) -> bool:
        if not self.writer.add(record):
            return False
        stats = self.stats(channel)
        stats["written"] += 1
        stats["vacancy_written"] += bool(record.is_vacancy)
        stats["edits"] += edit
        return True


def daily_channel(run: Run, channel: str) -> None:
    state = run.state("daily", channel)
    last_id: int | None = state.get("last_id")
    recent: dict[str, str] = dict(state.get("recent", {}))
    page, at = run.page(channel)
    if not page.posts:
        raise ChannelError(f"{channel}: the newest page has no posts, has the page format changed?")
    subscribers = page.subscribers
    seen = [(p, at) for p in page.posts]
    to_start = last_id is None and channel in DAILY_ONLY
    while page.prev_before is not None and (to_start or (last_id is not None and page.prev_before > last_id)):
        before = page.prev_before
        page, at = run.page(channel, before)
        older = [p for p in page.posts if p.id < before]
        if not older:
            break
        seen += [(p, at) for p in older]
    for post, fetched_at in seen:
        if post.service:
            continue
        record = run.record(channel, post, fetched_at)
        if record is None:
            continue
        key, digest = str(post.id), record.content_hash[:16]
        if last_id is None or post.id > last_id:
            run.write(channel, record)
        elif key in recent and recent[key] != digest:
            run.write(channel, record, edit=True)
        recent[key] = digest
    ids = [p.id for p, _ in seen]
    newest = sorted(recent, key=int)[-RECENT_IDS:]
    run.save(
        "daily",
        channel,
        {
            "last_id": max(ids) if last_id is None else max(last_id, *ids),
            "first_seen_id": state.get("first_seen_id", min(ids)),
            "recent": {k: recent[k] for k in newest},
            "subscribers": subscribers,
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def backfill_channel(run: Run, channel: str) -> None:
    state = run.state("backfill", channel)
    if state.get("done"):
        return
    if "first_seen_id" not in run.state("daily", channel):
        daily_channel(run, channel)
    before: int = state.get("before") or run.state("daily", channel)["first_seen_id"]
    state.setdefault("started_at", datetime.now(UTC).isoformat(timespec="seconds"))
    state.setdefault("pages", 0)
    state.setdefault("posts_written", 0)
    started, since_flush, written = time.monotonic(), 0, 0
    log.info("backfill %s: from before=%d, %d pages done earlier", channel, before, state["pages"])
    while True:
        page, at = run.page(channel, before)
        older = [p for p in page.posts if p.id < before]
        for post in older:
            if not post.service and (record := run.record(channel, post, at)) is not None:
                written += run.write(channel, record)
        done = not older or page.prev_before is None
        if older:
            before = min(p.id for p in older)
        since_flush += 1
        if done or since_flush >= FLUSH_PAGES:
            state["pages"] += since_flush
            state["posts_written"] += written
            state.update(before=before, done=done, updated_at=datetime.now(UTC).isoformat(timespec="seconds"))
            if done:
                state["reached_start_at"] = state["updated_at"]
                state["first_id"] = before
            run.save("backfill", channel, state)
            run.commit()
            log.info(
                "backfill %s: %d pages, before=%d, %d posts written, %.1f s per page%s",
                channel,
                state["pages"],
                before,
                state["posts_written"],
                (time.monotonic() - started) / since_flush,
                ", reached the start" if done else "",
            )
            started, since_flush, written = time.monotonic(), 0, 0
        if done:
            return


def collect(fetcher: Fetcher, channels: list[str], backfill: bool = False, root: Path | None = None) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    dt = started_at.astimezone(MSK).date()
    mode = "backfill" if backfill else "daily"
    failed: dict[str, str] = {}
    aborted = False
    skipped = [c for c in channels if backfill and c in DAILY_ONLY]
    with PartitionWriter(SOURCE, dt, root=root) as writer:
        run = Run(fetcher, writer, keep_handles(), root)
        in_row = 0
        for channel in channels:
            if channel in skipped:
                continue
            try:
                (backfill_channel if backfill else daily_channel)(run, channel)
                in_row = 0
            except RuntimeError as exc:
                run.commit()
                failed[channel] = str(exc)
                log.error("%s %s failed: %s", mode, channel, exc)
                in_row += 1
                if in_row >= MAX_FAILED_IN_ROW:
                    aborted = True
                    log.error("%d channels failed in a row, stopping the run", in_row)
                    break
        run.commit()
        writer.close(complete=True, mode=mode, partial=bool(failed), failed_channels=sorted(failed))
    totals = Counter[str]()
    for stats in run.channels.values():
        totals.update(stats)
    entry = {
        "mode": mode,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "dt": dt.isoformat(),
        "channels_requested": len(channels),
        "channels_processed": len(run.channels) - len(failed),
        "skipped_daily_only": skipped,
        "failed": failed,
        "aborted": aborted,
        **{k: totals[k] for k in ("pages", "posts_seen", "written", "vacancy_written", "edits", "rejected")},
        "requests": fetcher.requests,
        "retries": fetcher.retries,
        "failed_attempts": dict(fetcher.failed_attempts),
        "http_429": fetcher.failed_attempts.get("HTTP 429", 0),
        "seconds_in_requests": round(fetcher.seconds_in_requests, 1),
        "seconds_total": round((datetime.now(UTC) - started_at).total_seconds(), 1),
        "per_channel": run.channels,
    }
    append_meta(SOURCE, "runs", entry, root)
    return entry


class PartitionNotClosable(RuntimeError):
    pass


def _behind_cursor(channel: str, post_id: int, root: Path | None) -> bool:
    """Whether a resumed run skips this post: daily reads above `last_id`, backfill below `before`."""
    daily = read_state("daily", channel, root)
    first_seen, last_id = daily.get("first_seen_id"), daily.get("last_id")
    if first_seen is not None and post_id >= first_seen:
        return last_id is not None and post_id <= last_id
    before = read_state("backfill", channel, root).get("before")
    return before is not None and post_id >= before


def close_partition(dt: date, root: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Write `_SUCCESS` for the partition of a run that was killed before it could.

    A hard kill (out of memory) leaves the flushed parts and no marker, so the partition is never
    published or rewritten. The parts are kept as they are; the marker goes in only when no run will
    store their posts again: each post sits behind its channel's saved cursor (cursors are written
    after the parts they point past, see `Run.commit`), the partition holds no key twice, and no
    part is younger than 30 minutes or the day is still today. The marker says the run was
    interrupted and gives only what the parts show; requests and timings of the run are not known.
    """
    now = now or datetime.now(UTC)
    directory = partition_dir(SOURCE, dt, root)
    parts = sorted(directory.glob("part-*.parquet"))
    if (directory / SUCCESS).exists():
        raise PartitionNotClosable(f"{directory} already has {SUCCESS}")
    if not parts:
        raise PartitionNotClosable(f"{directory} has no parts")
    if dt >= now.astimezone(MSK).date():
        raise PartitionNotClosable(f"dt={dt} is today in Moscow; a run may still be writing it")
    if list(directory.glob(".*.tmp")):
        raise PartitionNotClosable(f"{directory} holds a temporary file of an unfinished write")
    if any(datetime.fromtimestamp(p.stat().st_mtime, UTC) > now - CLOSE_MIN_AGE for p in parts):
        raise PartitionNotClosable(f"{directory} has a part modified in the last 30 minutes")
    table = pa.concat_tables(pq.read_table(p, columns=["source_id", "content_hash", "fetched_at"]) for p in parts)
    ids = [str(i) for i in table.column("source_id").to_pylist()]
    keys = set(zip(ids, table.column("content_hash").to_pylist(), strict=True))
    if len(keys) != len(ids):
        raise PartitionNotClosable(f"{directory} holds {len(ids) - len(keys)} repeated keys")
    by_channel: Counter[str] = Counter()
    ahead: Counter[str] = Counter()
    for source_id in ids:
        channel, post_id = source_id.rsplit("/", 1)
        by_channel[channel] += 1
        ahead[channel] += not _behind_cursor(channel, int(post_id), root)
    if +ahead:
        raise PartitionNotClosable(f"posts a resumed run would store again, by channel: {dict(+ahead)}")
    fetched = table.column("fetched_at")
    marker = {
        "closed_at": now.isoformat(timespec="seconds"),
        "interrupted": True,
        "rows_in_partition": len(ids),
        "distinct_ids": len(set(ids)),
        "first_fetched_at": pc.min(fetched).as_py().isoformat(),
        "last_fetched_at": pc.max(fetched).as_py().isoformat(),
        "run_ids": sorted({p.name[len("part-") :].rsplit("-", 1)[0] for p in parts}),
        "rows_by_channel": dict(sorted(by_channel.items())),
    }
    tmp = directory / f".{SUCCESS}.tmp"
    tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, directory / SUCCESS)
    append_meta(SOURCE, "closed_partitions", {"dt": dt.isoformat(), **marker}, root)
    return marker


def _latest_versions(root: Path | None) -> dict[str, dict[str, Any]]:
    """Per post, its latest stored version: an edited post counts once, as it reads now."""
    columns = ["source_id", "fetched_at", "published_at", "is_vacancy", "multi_vacancy"]
    tables = [
        pq.read_table(part, columns=columns)
        for dt in partition_dates(SOURCE, root)
        for part in sorted(((root or raw_root()) / SOURCE / f"dt={dt.isoformat()}").glob("part-*.parquet"))
    ]
    if not tables:
        return {}
    table = pa.concat_tables(tables).sort_by("fetched_at")
    return {row["source_id"]: row for row in table.to_pylist()}


def _share(n: int, denominator: int) -> dict[str, Any]:
    return {"value": round(n / denominator, 4) if denominator else 0, "n": n, "denominator": denominator}


def summarize(root: Path | None = None, config: Path = CONFIG) -> Path:
    """Rebuild `results/collect/telegram.json` from the partitions, run log and backfill state on disk;
    once every backfilled channel reached its first post, also `results/collect/backfill_summary.json`.
    """
    enabled = enabled_channels(config)
    channels: dict[str, dict[str, Any]] = {
        c: {"posts": 0, "vacancy_posts": 0, "multi_vacancy_posts": 0, "first_published": None, "last_published": None}
        for c in enabled
    }
    years: dict[str, Counter[str]] = defaultdict(Counter)
    channel_years: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for source_id, row in _latest_versions(root).items():
        channel = source_id.rsplit("/", 1)[0]
        stats = channels.setdefault(
            channel,
            {"posts": 0, "vacancy_posts": 0, "multi_vacancy_posts": 0, "first_published": None, "last_published": None},
        )
        published = row["published_at"].astimezone(MSK)
        year = str(published.year)
        day = published.date().isoformat()
        stats["posts"] += 1
        stats["vacancy_posts"] += bool(row["is_vacancy"])
        stats["multi_vacancy_posts"] += bool(row["multi_vacancy"])
        stats["first_published"] = min(filter(None, (stats["first_published"], day)))
        stats["last_published"] = max(filter(None, (stats["last_published"], day)))
        for counter in (years[year], channel_years[channel][year]):
            counter["posts"] += 1
            counter["vacancy_posts"] += bool(row["is_vacancy"])
    to_backfill = [c for c in enabled if c not in DAILY_ONLY]
    for name in enabled:
        state = read_state("backfill", name, root)
        channels[name]["backfill"] = (
            "daily only"
            if name in DAILY_ONLY
            else {k: state.get(k) for k in ("done", "pages", "posts_written", "started_at", "reached_start_at")}
        )
    reached = [c for c in to_backfill if read_state("backfill", c, root).get("done")]
    posts = sum(s["posts"] for s in channels.values())
    vacancies = sum(s["vacancy_posts"] for s in channels.values())
    multi = sum(s["multi_vacancy_posts"] for s in channels.values())
    runs = read_meta(SOURCE, "runs", root)
    metrics: dict[str, dict[str, Any]] = {
        "channels_enabled": {"value": len(enabled)},
        "posts": {"value": posts},
        "vacancy_posts": _share(vacancies, posts),
        "multi_vacancy_posts": _share(multi, vacancies),
        "backfill_channels_reached_start": _share(len(reached), len(to_backfill)),
    }
    for mode in ("daily", "backfill"):
        done = [r for r in runs if r["mode"] == mode]
        if done:
            last = done[-1]
            metrics[f"last_{mode}_run_requests"] = {"value": last["requests"]}
            metrics[f"last_{mode}_run_http_429"] = {"value": last["http_429"]}
            metrics[f"last_{mode}_run_seconds"] = {"value": last["seconds_total"]}
            metrics[f"last_{mode}_run_posts_written"] = {"value": last["written"]}
    by_year = {y: dict(c) for y, c in sorted(years.items())}
    by_channel_year = {ch: {y: dict(c) for y, c in sorted(ys.items())} for ch, ys in sorted(channel_years.items())}
    days = []
    for dt in partition_dates(SOURCE, root):
        stats = partition_stats(SOURCE, dt, root)
        days.append({k: stats[k] for k in ("dt", "complete", "rows", "distinct_ids", "parquet_bytes", "pii_removed")})
    method = (
        "recomputed from data/raw/telegram partitions (latest version of each post), _meta/runs.jsonl and "
        "_meta/state; years by published_at in Moscow time; vacancy posts by rules v2"
    )
    if to_backfill and len(reached) == len(to_backfill):
        backfill_runs = [r for r in runs if r["mode"] == "backfill"]
        write_result(
            BACKFILL_RESULT_PATH,
            {
                "posts": {"value": posts},
                "vacancy_posts": _share(vacancies, posts),
                "channels_backfilled": {"value": len(reached)},
                # Pages from the channel cursors, which survive a killed run; the rest from finished runs.
                "backfill_pages": {"value": sum(read_state("backfill", c, root)["pages"] for c in to_backfill)},
                "backfill_finished_runs": {"value": len(backfill_runs)},
                "backfill_requests_in_finished_runs": {"value": sum(r["requests"] for r in backfill_runs)},
                "backfill_http_429_in_finished_runs": {"value": sum(r["http_429"] for r in backfill_runs)},
                "backfill_seconds_in_finished_runs": {
                    "value": round(sum(r["seconds_total"] for r in backfill_runs), 1)
                },
            },
            source_script="src/radar/collect/telegram.py",
            method=method,
            years=by_year,
            channels=channels,
            channel_years=by_channel_year,
        )
    return write_result(
        RESULT_PATH,
        metrics,
        source_script="src/radar/collect/telegram.py",
        method=method,
        years=by_year,
        channels=channels,
        channel_years=by_channel_year,
        days=days,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--backfill", action="store_true", help="walk channels back to their first post")
    parser.add_argument("--channel", help="one channel from configs/telegram_channels.toml")
    parser.add_argument("--summarize", action="store_true", help="only rebuild the results files from disk")
    parser.add_argument(
        "--close-partition",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="write _SUCCESS for the partition of a killed run, if no run will store its posts again",
    )
    args = parser.parse_args(argv)
    setup_logging()
    if args.close_partition is not None:
        try:
            marker = close_partition(args.close_partition)
        except PartitionNotClosable as exc:
            log.error("dt=%s stays open: %s", args.close_partition, exc)
            return 1
        print(json.dumps(marker, ensure_ascii=False, indent=1))
        log.info("next: python -m radar.pii.rewrite --source telegram, then radar.stream.replay for this dt")
        return 0
    status = 0
    if not args.summarize:
        channels = enabled_channels()
        if args.channel is not None:
            if args.channel not in {c["name"] for c in load_channels()}:
                parser.error(f"{args.channel} is not in {CONFIG.name}")
            channels = [args.channel]
        try:
            with single_instance(meta_dir(SOURCE) / "collector.lock"), make_client(TIMEOUT_S) as client:
                run = collect(Fetcher(client, PAUSE_S, backoff_s=BACKOFF_S), channels, backfill=args.backfill)
        except AlreadyRunning as exc:
            log.error("another Telegram run is going on: %s", exc)
            return 1
        log.info(
            "%s done in %.0f s: %d channels, %d pages, %d posts written (%d vacancy), %d requests, %d HTTP 429, "
            "failed %s",
            run["mode"],
            run["seconds_total"],
            run["channels_processed"],
            run["pages"],
            run["written"],
            run["vacancy_written"],
            run["requests"],
            run["http_429"],
            sorted(run["failed"]) or "none",
        )
        publish_after_collect(SOURCE, date.fromisoformat(run["dt"]))
        status = 1 if run["failed"] else 0
    log.info("results: %s", summarize())
    return status


if __name__ == "__main__":
    sys.exit(main())
