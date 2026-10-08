"""t.me/s client with an on-disk cache, and the parser of a channel page.

Every page is fetched once: the cache file holds the status, the fetch time and the body, so a rerun
with `offline=True` repeats the survey without touching t.me.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from lxml import html as lhtml

UA = "ru-it-jobs-radar/0.1 (+https://github.com/alekseevpavel04/ru-it-jobs-radar)"
PAUSE = 3.0
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "tmp" / "telegram_survey" / "cache"
TME_LINK = re.compile(r"(?:https?://)?(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z][A-Za-z0-9_]{3,31})(?:/(\d+))?", re.I)


class CacheMiss(RuntimeError):
    pass


@dataclass
class Post:
    id: int
    date: str
    text: str
    links: list[str]
    forwarded_from: str | None  # channel username, only when forwarded from a public channel
    service: bool  # "channel created", "pinned" and the like, not a post

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Page:
    status: int
    exists: bool
    fetched_at: str
    title: str = ""
    subscribers: str | None = None
    posts: list[Post] = field(default_factory=list)
    prev_before: int | None = None


class Client:
    def __init__(self, cache: Path = CACHE, offline: bool = False) -> None:
        self.cache = cache
        self.offline = offline
        self.cache.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=False)
        self.last_request = 0.0
        self.requests = 0
        self.touched: list[str] = []

    def _path(self, url: str) -> Path:
        return self.cache / hashlib.sha1(url.encode()).hexdigest()

    def is_cached(self, url: str) -> bool:
        return self._path(url).exists()

    def fetch(self, url: str) -> tuple[int, str, str]:
        self.touched.append(url)
        path = self._path(url)
        if path.exists():
            head, _, body = path.read_text(encoding="utf-8").partition("\n")
            status, fetched_at = head.split(" ", 1)
            return int(status), fetched_at, body
        if self.offline:
            raise CacheMiss(url)
        for attempt in range(3):
            time.sleep(max(0.0, PAUSE - (time.monotonic() - self.last_request)))
            try:
                r = self.http.get(url)
            except httpx.TransportError:
                self.last_request = time.monotonic()
                time.sleep(10 * (attempt + 1))
                continue
            self.last_request = time.monotonic()
            self.requests += 1
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 60)))
                continue
            fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
            path.write_text(f"{r.status_code} {fetched_at}\n{r.text}", encoding="utf-8", newline="\n")
            return r.status_code, fetched_at, r.text
        raise RuntimeError(f"giving up on {url}")

    def page(self, channel: str, before: int | None = None, after: int | None = None) -> Page:
        return parse(*self.fetch(page_url(channel, before, after)))


def page_url(channel: str, before: int | None = None, after: int | None = None) -> str:
    url = f"https://t.me/s/{channel}"
    if before is not None:
        url += f"?before={before}"
    elif after is not None:
        url += f"?after={after}"
    return url


def parse(status: int, fetched_at: str, body: str) -> Page:
    # A missing channel answers with a redirect to t.me/<name>; a user or a group has no channel header.
    if status != 200:
        return Page(status, False, fetched_at)
    doc = lhtml.fromstring(body)
    if not doc.xpath('//div[contains(@class,"tgme_channel_info")]'):
        return Page(status, False, fetched_at)
    title = "".join(doc.xpath('//div[contains(@class,"tgme_channel_info_header_title")]//text()')).strip()
    subscribers = None
    for counter in doc.xpath('//div[contains(@class,"tgme_channel_info_counter")]'):
        if "".join(counter.xpath('.//span[@class="counter_type"]/text()')).startswith("subscriber"):
            subscribers = "".join(counter.xpath('.//span[@class="counter_value"]/text()'))
    posts = []
    for m in doc.xpath('//div[contains(@class,"tgme_widget_message ") and @data-post]'):
        dates = m.xpath('.//a[contains(@class,"tgme_widget_message_date")]/time/@datetime')
        text_nodes = m.xpath('.//div[contains(@class,"tgme_widget_message_text")]')
        text, links = "", []
        if text_nodes:
            for br in text_nodes[0].iter("br"):
                br.tail = "\n" + (br.tail or "")
            text = text_nodes[0].text_content()
            links = [a.get("href") for a in text_nodes[0].xpath(".//a[@href]")]
        forwarded = m.xpath('.//a[contains(@class,"tgme_widget_message_forwarded_from_name")]/@href')
        source = TME_LINK.search(forwarded[0]) if forwarded else None
        posts.append(
            Post(
                id=int(m.get("data-post").rsplit("/", 1)[1]),
                date=dates[0] if dates else "",
                text=text,
                links=links,
                forwarded_from=source.group(1).lower() if source else None,
                service="service_message" in (m.get("class") or ""),
            )
        )
    prev = doc.xpath('//link[@rel="prev"]/@href')
    before = re.search(r"before=(\d+)", prev[0]) if prev else None
    return Page(status, True, fetched_at, title, subscribers, posts, int(before.group(1)) if before else None)
