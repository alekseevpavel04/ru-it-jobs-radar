import pytest

COLLECTORS = ("habr", "telegram", "trudvsem", "trudvsem_snapshot")


@pytest.fixture(autouse=True)
def no_publish_from_collectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collector tests run the collectors end to end, and a collector's last step publishes the new
    partition to the stream. With the local stack up that put fixture rows into the real
    raw.vacancies (dt 2026-06-30, 2026-07-01 and 2026-10-07 on 2026-10-11), so no test publishes
    from a collector; tests/stream covers publishing itself."""
    import importlib

    for name in COLLECTORS:
        module = importlib.import_module(f"radar.collect.{name}")
        monkeypatch.setattr(module, "publish_after_collect", lambda source, dt, root=None: False)
