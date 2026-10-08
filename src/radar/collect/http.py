"""Polite HTTP for collectors: one request at a time, a pause between requests, retries with backoff."""

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from radar.config import user_agent

log = logging.getLogger("radar.collect.http")

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class Fetcher:
    """Sequential GETs with at least `pause_s` between the end of one request and the start of the next.

    Network errors, 429 and 5xx are retried after the pauses in `backoff_s` (429 honours Retry-After);
    any other error status is returned to the caller, which knows what it means for that source.
    """

    def __init__(
        self,
        client: httpx.Client,
        pause_s: float,
        backoff_s: Sequence[float] = (10, 30, 90),
        retry_status: frozenset[int] = RETRY_STATUS,
    ) -> None:
        self.client = client
        self.pause_s = pause_s
        self.backoff_s = tuple(backoff_s)
        self.retry_status = retry_status
        self.requests = 0
        self.retries = 0
        self.seconds_in_requests = 0.0
        self._last_end: float | None = None

    def _wait(self) -> None:
        if self._last_end is not None:
            remaining = self.pause_s - (time.monotonic() - self._last_end)
            if remaining > 0:
                time.sleep(remaining)

    def get(self, url: str, params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None) -> httpx.Response:
        for attempt, pause in enumerate((*self.backoff_s, None)):
            self._wait()
            started = time.monotonic()
            try:
                response = self.client.get(url, params=params)  # type: ignore[arg-type]
            except httpx.TransportError as exc:
                problem: str = f"{type(exc).__name__}: {exc}"
                retry_after = None
            else:
                if response.status_code not in self.retry_status:
                    self._done(started)
                    return response
                problem = f"HTTP {response.status_code}"
                retry_after = response.headers.get("retry-after")
            finally:
                self._last_end = time.monotonic()
            self._done(started)
            if pause is None:
                raise RuntimeError(f"GET {url} failed after {attempt + 1} attempts: {problem}")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else pause
            log.warning("GET %s: %s, retry %d in %.0f s", url, problem, attempt + 1, wait)
            self.retries += 1
            time.sleep(wait)
        raise AssertionError("unreachable")

    def _done(self, started: float) -> None:
        self.requests += 1
        self.seconds_in_requests += time.monotonic() - started


def make_client(timeout_s: float) -> httpx.Client:
    return httpx.Client(headers={"User-Agent": user_agent()}, timeout=timeout_s, follow_redirects=True)
