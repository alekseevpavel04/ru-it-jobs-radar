"""The raw layer on disk as the DAGs see it: which partitions of a date are complete and how far
behind each source is.

Standard library only: Airflow's own interpreter imports it in tasks (PYTHONPATH of the scheduler
in infra/compose/docker-compose.yml), the tests import it under the project's. The layout is the
one of radar.collect.writer: `<raw>/<source>/dt=<YYYY-MM-DD>/`, complete once `_SUCCESS` exists.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

SUCCESS = "_SUCCESS"


@dataclass(frozen=True)
class Threshold:
    warn_after_hours: float
    error_after_hours: float


# The thresholds of dbt/models/staging/_sources.yml. dbt measures from fetched_at in ClickHouse;
# here the clock starts when the partition was due locally, so both alert after about one missed
# day (warn) and two (error). superjob is not collected (no key) and has no threshold.
THRESHOLDS = {
    "trudvsem": Threshold(26, 50),
    "habr": Threshold(26, 50),
    "telegram": Threshold(26, 50),
    # A snapshot is published about 13:50 Moscow time and takes hours to download and parse.
    "trudvsem_snapshot": Threshold(48, 96),
}


def complete_sources(raw: Path, sources: tuple[str, ...], dt: date) -> list[str]:
    return [s for s in sources if (raw / s / f"dt={dt.isoformat()}" / SUCCESS).is_file()]


def latest_complete(raw: Path, source: str) -> date | None:
    days = [date.fromisoformat(p.name[3:]) for p in (raw / source).glob("dt=*") if (p / SUCCESS).is_file()]
    return max(days, default=None)


def due_at(dt: date) -> datetime:
    """The radar_collect run for day D starts at 00:00 UTC of D+1, after the server's 21:30 Moscow run of D."""
    return datetime.combine(dt + timedelta(days=1), time(), tzinfo=UTC)


def freshness(raw: Path, now: datetime, thresholds: dict[str, Threshold] = THRESHOLDS) -> list[dict[str, Any]]:
    """Per source: the latest complete day, hours since it was due locally and the status.

    The status is `pass`, `warn` or `error` by the thresholds; a source without any complete
    partition is `error` with no lag.
    """
    rows = []
    for source, limit in thresholds.items():
        latest = latest_complete(raw, source)
        # A partition collected on its own day locally arrives before it is due: not late, lag 0.
        lag = None if latest is None else max(0.0, round((now - due_at(latest)).total_seconds() / 3600, 1))
        if lag is None or lag > limit.error_after_hours:
            status = "error"
        elif lag > limit.warn_after_hours:
            status = "warn"
        else:
            status = "pass"
        rows.append(
            {
                "source": source,
                "latest_dt": latest.isoformat() if latest else None,
                "lag_hours": lag,
                "warn_after_hours": limit.warn_after_hours,
                "error_after_hours": limit.error_after_hours,
                "status": status,
            }
        )
    return rows
