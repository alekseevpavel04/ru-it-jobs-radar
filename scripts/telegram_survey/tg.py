"""t.me/s client with an on-disk cache; the page parser is radar.collect.telegram_page.

Every page is fetched once: the cache file holds the status, the fetch time and the body, so a rerun
with `offline=True` repeats the survey without touching t.me.
"""

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from radar.collect.telegram_page import Page, page_url, parse

UA = "ru-it-jobs-radar/0.1 (+https://github.com/alekseevpavel04/ru-it-jobs-radar)"
PAUSE = 3.0
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "tmp" / "telegram_survey" / "cache"


class CacheMiss(RuntimeError):
    pass


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
