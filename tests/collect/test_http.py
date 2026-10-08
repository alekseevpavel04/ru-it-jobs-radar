import httpx
import pytest

from radar.collect import http
from radar.collect.http import Fetcher


def _fetcher(statuses: list[int | Exception], **kwargs: object) -> tuple[Fetcher, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = statuses[len(seen) - 1]
        if isinstance(status, Exception):
            raise status
        headers = {"retry-after": "7"} if status == 429 else {}
        return httpx.Response(status, headers=headers)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return Fetcher(client, pause_s=0, backoff_s=(1, 2), **kwargs), seen  # type: ignore[arg-type]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(http.time, "sleep", slept.append)
    return slept


def test_retries_then_succeeds(sleeps: list[float]) -> None:
    fetcher, seen = _fetcher([503, 429, 200])
    assert fetcher.get("https://example.org/x").status_code == 200
    assert len(seen) == 3
    assert sleeps == [1, 7]
    assert (fetcher.requests, fetcher.retries) == (3, 2)


def test_network_error_is_retried(sleeps: list[float]) -> None:
    fetcher, _ = _fetcher([httpx.ConnectError("refused"), 200])
    assert fetcher.get("https://example.org/x").status_code == 200
    assert sleeps == [1]


def test_gives_up_after_backoff(sleeps: list[float]) -> None:
    fetcher, seen = _fetcher([500, 500, 500])
    with pytest.raises(RuntimeError, match="3 attempts: HTTP 500"):
        fetcher.get("https://example.org/x")
    assert len(seen) == 3


def test_other_errors_go_back_to_the_caller(sleeps: list[float]) -> None:
    fetcher, _ = _fetcher([404])
    assert fetcher.get("https://example.org/x").status_code == 404
    assert sleeps == []


def test_pause_between_requests(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    # start, end of first request, its recorded end; then the wait check and the second request
    clock = iter([0.0, 0.3, 0.5, 1.0, 1.2, 1.3, 1.4])
    monkeypatch.setattr(http.time, "monotonic", lambda: next(clock))
    fetcher, _ = _fetcher([200, 200])
    fetcher.pause_s = 2.0
    fetcher.get("https://example.org/a")
    fetcher.get("https://example.org/b")
    assert sleeps == [pytest.approx(1.5)]
