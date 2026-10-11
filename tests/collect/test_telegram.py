import html
import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
import pytest

import radar.results
from radar.collect import telegram
from radar.collect.http import Fetcher
from radar.collect.telegram_page import Post
from radar.pii import scan_text

CHANNEL = "job_python"  # a configured channel: its own @name is public and stays in texts
KEEP = telegram.keep_handles()


def vacancy_text(i: int) -> str:
    return (
        f"Python-разработчик в ООО «Пример {i}»\nЗарплата: от 200 000 руб.\nУдалённо, полная занятость\n"
        f"Отклик: @hr_person{i:04d}"
    )


class Channel:
    """A t.me/s channel: pages of up to 20 posts, `?before=` paging, deleted ids leave gaps."""

    def __init__(self, name: str, ids: range) -> None:
        self.name = name
        self.texts = {i: vacancy_text(i) for i in ids if i % 7}
        self.fail_at: int | None = None  # index of the request that gets HTTP 500
        self.requests: list[httpx.URL] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url)
        assert request.url.path == f"/s/{self.name}"
        if len(self.requests) - 1 == self.fail_at:
            return httpx.Response(500)
        before = request.url.params.get("before")
        ids = sorted((i for i in self.texts if before is None or i < int(before)), reverse=True)[:20]
        return httpx.Response(200, text=self.render(sorted(ids)))

    def render(self, ids: list[int]) -> str:
        prev = f'<link rel="prev" href="/s/{self.name}?before={ids[0]}">' if ids and ids[0] > min(self.texts) else ""
        posts = "".join(
            f'<div class="tgme_widget_message_wrap"><div class="tgme_widget_message js-widget_message" '
            f'data-post="{self.name}/{i}"><div class="tgme_widget_message_text js-message_text">'
            f"{html.escape(self.texts[i]).replace(chr(10), '<br/>')}</div>"
            f'<span class="tgme_widget_message_views">{i}</span><span class="tgme_widget_message_meta">'
            f'<a class="tgme_widget_message_date" href="https://t.me/{self.name}/{i}">'
            f'<time datetime="{2020 + i // 100}-03-01T10:00:00+00:00"></time></a></span></div></div>'
            for i in ids
        )
        return (
            f'<html><head>{prev}</head><body><div class="tgme_channel_info"><div class="tgme_channel_info_counter">'
            f'<span class="counter_value">1K</span><span class="counter_type">subscribers</span></div></div>'
            f"{posts}</body></html>"
        )

    def fetcher(self) -> Fetcher:
        return Fetcher(httpx.Client(transport=httpx.MockTransport(self.handle)), pause_s=0, backoff_s=())

    def befores(self) -> list[int | None]:
        return [int(b) if (b := u.params.get("before")) else None for u in self.requests]


def rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in sorted(root.glob("telegram/dt=*/part-*.parquet")):
        out.extend(pq.read_table(part).to_pylist())
    return out


def test_record_envelope_and_personal_data() -> None:
    text = (
        "Python-разработчик в ООО «Пример»\nЗарплата: от 250 000 руб.\nУдалённо\n"
        "Пишите @ivan_petrov_hr, hr.person@example.com, +7 900 000-00-07, https://t.me/ivan_petrov_hr\n"
        "Больше вакансий в @datasciencejobs и https://t.me/job_python"
    )
    post = Post(
        id=42,
        date="2026-10-01T09:00:00+00:00",
        text=text,
        links=[
            "https://t.me/ivan_petrov_hr",
            "https://career.habr.com/vacancies/1000000001",
            "https://t.me/job_python",
        ],
        forwarded_from="other_jobs_channel",
        service=False,
        forwarded=True,
        views=1200,
        media=["photo"],
    )
    fetched_at = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
    record = telegram.build_record(CHANNEL, post, fetched_at, KEEP)
    assert record.source == "telegram"
    assert record.source_id == "job_python/42"
    assert record.url == "https://t.me/job_python/42"
    assert record.title is None
    assert record.published_at == datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    assert record.pii_removed == {"nick": 1, "email": 1, "phone": 1, "profile_link": 1}
    assert record.text is not None
    assert scan_text(record.text, KEEP) == []
    assert "@datasciencejobs" in record.text and "https://t.me/job_python" in record.text
    assert (record.is_vacancy, record.multi_vacancy) == (True, False)
    assert record.payload["job_links"] == [{"host": "career.habr.com", "habr_vacancy_id": 1000000001}]
    assert record.payload["forwarded_from"] == "other_jobs_channel"
    assert "ivan" not in json.dumps(record.payload)
    # Views and the edit mark change while the post does not.
    later = telegram.build_record(CHANNEL, Post(**{**post.__dict__, "views": 5000, "edited": True}), fetched_at, KEEP)
    assert later.content_hash == record.content_hash
    changed = telegram.build_record(CHANNEL, Post(**{**post.__dict__, "text": text + "!"}), fetched_at, KEEP)
    assert changed.content_hash != record.content_hash


def test_daily_first_run_reads_the_newest_page_and_rerun_writes_nothing(tmp_path: Path) -> None:
    site = Channel(CHANNEL, range(1, 101))
    run = telegram.collect(site.fetcher(), [CHANNEL], root=tmp_path)
    assert site.befores() == [None]
    assert run["written"] == 20 and run["requests"] == 1 and run["http_429"] == 0
    state = telegram.read_state("daily", CHANNEL, tmp_path)
    assert (state["last_id"], state["first_seen_id"]) == (100, 78)
    for row in rows(tmp_path):
        assert scan_text(row["text"], KEEP) == []
        assert row["is_vacancy"] is True
    again = Channel(CHANNEL, range(1, 101))
    run = telegram.collect(again.fetcher(), [CHANNEL], root=tmp_path)
    assert run["written"] == 0
    assert len(rows(tmp_path)) == 20


def test_daily_catches_up_to_the_last_stored_post_and_records_edits(tmp_path: Path) -> None:
    telegram.collect(Channel(CHANNEL, range(1, 101)).fetcher(), [CHANNEL], root=tmp_path)
    site = Channel(CHANNEL, range(1, 146))
    site.texts[100] += "\nОбновлено: вилка выше"
    run = telegram.collect(site.fetcher(), [CHANNEL], root=tmp_path)
    # 39 new posts: the newest page, then one older page that reaches post 100.
    assert site.befores()[0] is None and len(site.requests) == 2
    assert run["per_channel"][CHANNEL]["edits"] == 1
    new_ids = {i for i in range(101, 146) if i % 7}
    assert run["written"] == len(new_ids) + 1
    written = Counter(r["source_id"] for r in rows(tmp_path))
    assert written[f"{CHANNEL}/100"] == 2
    assert telegram.read_state("daily", CHANNEL, tmp_path)["last_id"] == 145


def test_backfill_resumes_after_a_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram, "FLUSH_PAGES", 2)
    ids = range(1, 301)
    telegram.collect(Channel(CHANNEL, ids).fetcher(), [CHANNEL], root=tmp_path)
    broken = Channel(CHANNEL, ids)
    broken.fail_at = 4  # four pages read and flushed, the fifth fails
    run = telegram.collect(broken.fetcher(), [CHANNEL], backfill=True, root=tmp_path)
    assert list(run["failed"]) == [CHANNEL]
    state = telegram.read_state("backfill", CHANNEL, tmp_path)
    assert not state["done"] and state["pages"] == 4
    cursor = state["before"]

    resumed = Channel(CHANNEL, ids)
    run = telegram.collect(resumed.fetcher(), [CHANNEL], backfill=True, root=tmp_path)
    assert not run["failed"]
    assert resumed.befores()[0] == cursor
    state = telegram.read_state("backfill", CHANNEL, tmp_path)
    assert state["done"] and state["first_id"] == 1
    written = Counter(r["source_id"] for r in rows(tmp_path))
    assert set(written) == {f"{CHANNEL}/{i}" for i in ids if i % 7}
    assert max(written.values()) == 1

    finished = Channel(CHANNEL, ids)
    run = telegram.collect(finished.fetcher(), [CHANNEL], backfill=True, root=tmp_path)
    assert finished.requests == [] and run["written"] == 0


def test_backfill_without_daily_state_reads_the_newest_page_first(tmp_path: Path) -> None:
    site = Channel(CHANNEL, range(1, 41))
    telegram.collect(site.fetcher(), [CHANNEL], backfill=True, root=tmp_path)
    assert site.befores()[0] is None
    assert telegram.read_state("daily", CHANNEL, tmp_path)["last_id"] == 40
    assert len(rows(tmp_path)) == len([i for i in range(1, 41) if i % 7])


def test_daily_only_channel_is_read_to_its_start_and_never_backfilled(tmp_path: Path) -> None:
    name = "runello_rus_frontend"
    site = Channel(name, range(500, 560))
    run = telegram.collect(site.fetcher(), [name], root=tmp_path)
    assert run["written"] == len([i for i in range(500, 560) if i % 7])
    again = Channel(name, range(500, 560))
    run = telegram.collect(again.fetcher(), [name], backfill=True, root=tmp_path)
    assert again.requests == [] and run["skipped_daily_only"] == [name]


def test_failing_channels_stop_the_run(tmp_path: Path) -> None:
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(down)), pause_s=0, backoff_s=())
    run = telegram.collect(fetcher, ["job_python", "geekjobs", "remoteit", "datajob"], root=tmp_path)
    assert run["aborted"] and sorted(run["failed"]) == ["geekjobs", "job_python", "remoteit"]
    assert run["failed_attempts"] == {"ConnectError": 3}
    marker = json.loads(next(tmp_path.glob("telegram/dt=*/_SUCCESS")).read_text(encoding="utf-8"))
    assert marker["partial"] is True


def test_summarize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    config = tmp_path / "channels.toml"
    config.write_text(f'[[channel]]\nname = "{CHANNEL}"\nenabled = true\n', encoding="utf-8")
    ids = range(1, 251)
    telegram.collect(Channel(CHANNEL, ids).fetcher(), [CHANNEL], root=tmp_path)
    doc = json.loads(telegram.summarize(root=tmp_path, config=config).read_text(encoding="utf-8"))
    assert doc["metrics"]["posts"]["value"] == 20
    assert doc["metrics"]["backfill_channels_reached_start"] == {"value": 0, "n": 0, "denominator": 1}
    assert not (tmp_path / "results" / telegram.BACKFILL_RESULT_PATH).exists()

    telegram.collect(Channel(CHANNEL, ids).fetcher(), [CHANNEL], backfill=True, root=tmp_path)
    doc = json.loads(telegram.summarize(root=tmp_path, config=config).read_text(encoding="utf-8"))
    total = len([i for i in ids if i % 7])
    assert doc["metrics"]["vacancy_posts"] == {"value": 1.0, "n": total, "denominator": total}
    assert doc["metrics"]["last_backfill_run_http_429"]["value"] == 0
    assert doc["channels"][CHANNEL]["backfill"]["done"] is True
    in_2020 = len([i for i in range(1, 100) if i % 7])
    assert doc["years"]["2020"] == {"posts": in_2020, "vacancy_posts": in_2020}
    assert sum(y["posts"] for y in doc["channel_years"][CHANNEL].values()) == total
    backfill = json.loads((tmp_path / "results" / telegram.BACKFILL_RESULT_PATH).read_text(encoding="utf-8"))
    assert backfill["metrics"]["posts"]["value"] == total
    assert backfill["metrics"]["channels_backfilled"]["value"] == 1


def test_killed_daily_run_rereads_without_duplicates(tmp_path: Path) -> None:
    first, second = Channel(CHANNEL, range(1, 61)), Channel("geekjobs", range(1, 61))

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/s/geekjobs":
            raise KeyboardInterrupt
        return first.handle(request)

    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(handle)), pause_s=0, backoff_s=())
    with pytest.raises(KeyboardInterrupt):
        telegram.collect(fetcher, [CHANNEL, "geekjobs"], root=tmp_path)
    assert telegram.read_state("daily", CHANNEL, tmp_path) == {}
    assert len(rows(tmp_path)) == 20

    def both(request: httpx.Request) -> httpx.Response:
        return (second if request.url.path == "/s/geekjobs" else first).handle(request)

    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(both)), pause_s=0, backoff_s=())
    run = telegram.collect(fetcher, [CHANNEL, "geekjobs"], root=tmp_path)
    assert run["written"] == 20 and run["per_channel"][CHANNEL]["written"] == 0
    assert len(rows(tmp_path)) == len({r["source_id"] for r in rows(tmp_path)}) == 40


def _killed_backfill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_at: int | None) -> Path:
    """A daily run and a backfill whose partition lost its marker, as after a hard kill."""
    monkeypatch.setattr(telegram, "FLUSH_PAGES", 2)
    ids = range(1, 301)
    telegram.collect(Channel(CHANNEL, ids).fetcher(), [CHANNEL], root=tmp_path)
    site = Channel(CHANNEL, ids)

    def handle(request: httpx.Request) -> httpx.Response:
        if len(site.requests) == interrupt_at:
            raise KeyboardInterrupt
        return site.handle(request)

    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(handle)), pause_s=0, backoff_s=())
    if interrupt_at is None:
        telegram.collect(fetcher, [CHANNEL], backfill=True, root=tmp_path)
    else:
        with pytest.raises(KeyboardInterrupt):
            telegram.collect(fetcher, [CHANNEL], backfill=True, root=tmp_path)
    directory = next(tmp_path.glob("telegram/dt=*"))
    (directory / telegram.SUCCESS).unlink(missing_ok=True)
    return directory


def test_close_partition_of_a_killed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = _killed_backfill(tmp_path, monkeypatch, interrupt_at=None)
    dt = date.fromisoformat(directory.name.removeprefix("dt="))
    with pytest.raises(telegram.PartitionNotClosable, match="today"):
        telegram.close_partition(dt, tmp_path, now=datetime.now(UTC))
    tomorrow = datetime.now(UTC) + timedelta(days=1)
    marker = telegram.close_partition(dt, tmp_path, now=tomorrow)
    stored = rows(tmp_path)
    assert marker["interrupted"] is True
    assert marker["rows_in_partition"] == marker["distinct_ids"] == len(stored)
    assert marker["rows_by_channel"] == {CHANNEL: len(stored)}
    assert json.loads((directory / telegram.SUCCESS).read_text(encoding="utf-8")) == marker
    with pytest.raises(telegram.PartitionNotClosable, match="already"):
        telegram.close_partition(dt, tmp_path, now=tomorrow)


def test_partition_with_posts_past_the_cursor_stays_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The daily page and two backfill pages are committed; the next page reaches the parts on the way
    # out but not the cursor, so a resumed run on another day would store it again.
    directory = _killed_backfill(tmp_path, monkeypatch, interrupt_at=3)
    dt = date.fromisoformat(directory.name.removeprefix("dt="))
    cursor = telegram.read_state("backfill", CHANNEL, tmp_path)["before"]
    assert min(int(r["source_id"].rsplit("/", 1)[1]) for r in rows(tmp_path)) < cursor
    with pytest.raises(telegram.PartitionNotClosable, match=CHANNEL):
        telegram.close_partition(dt, tmp_path, now=datetime.now(UTC) + timedelta(days=1))
    assert not (directory / telegram.SUCCESS).exists()
