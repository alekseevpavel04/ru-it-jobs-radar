"""The standard-library helpers of airflow/include that the DAG tasks import under Airflow's interpreter."""

import importlib.util
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

INCLUDE = Path(__file__).resolve().parents[2] / "airflow" / "include"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"airflow_include_{name}", INCLUDE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


raw_layer = load("raw_layer")
dbt_handover = load("dbt_handover")


def partition(raw: Path, source: str, dt: str, complete: bool = True) -> None:
    directory = raw / source / f"dt={dt}"
    directory.mkdir(parents=True)
    (directory / "part-0.parquet").write_bytes(b"")
    if complete:
        (directory / raw_layer.SUCCESS).write_text("{}", encoding="utf-8")


def test_complete_sources_need_the_marker(tmp_path: Path) -> None:
    partition(tmp_path, "habr", "2026-07-22")
    partition(tmp_path, "trudvsem", "2026-07-22", complete=False)
    partition(tmp_path, "trudvsem_snapshot", "2026-07-23")
    found = raw_layer.complete_sources(tmp_path, ("trudvsem", "habr", "trudvsem_snapshot"), date(2026, 7, 22))
    assert found == ["habr"]


def test_latest_complete_ignores_an_unfinished_newer_day(tmp_path: Path) -> None:
    partition(tmp_path, "habr", "2026-10-07")
    partition(tmp_path, "habr", "2026-10-08")
    partition(tmp_path, "habr", "2026-10-09", complete=False)
    assert raw_layer.latest_complete(tmp_path, "habr") == date(2026, 10, 8)
    assert raw_layer.latest_complete(tmp_path, "telegram") is None


def test_partition_is_due_when_the_collect_run_starts() -> None:
    assert raw_layer.due_at(date(2026, 10, 8)) == datetime(2026, 10, 9, tzinfo=UTC)


@pytest.mark.parametrize(
    ("hours_after_due", "status"),
    [(0.5, "pass"), (26.0, "pass"), (26.5, "warn"), (50.0, "warn"), (50.5, "error")],
)
def test_freshness_status_by_threshold(tmp_path: Path, hours_after_due: float, status: str) -> None:
    partition(tmp_path, "habr", "2026-10-08")
    now = datetime(2026, 10, 9, tzinfo=UTC) + timedelta(hours=hours_after_due)
    (row,) = raw_layer.freshness(tmp_path, now, {"habr": raw_layer.Threshold(26, 50)})
    assert row == {
        "source": "habr",
        "latest_dt": "2026-10-08",
        "lag_hours": hours_after_due,
        "warn_after_hours": 26,
        "error_after_hours": 50,
        "status": status,
    }


def test_partition_ahead_of_its_due_time_is_not_late(tmp_path: Path) -> None:
    partition(tmp_path, "telegram", "2026-10-08")
    limits = {"telegram": raw_layer.Threshold(26, 50)}
    (row,) = raw_layer.freshness(tmp_path, datetime(2026, 10, 8, 17, tzinfo=UTC), limits)
    assert (row["lag_hours"], row["status"]) == (0.0, "pass")


def test_source_without_partitions_is_an_error(tmp_path: Path) -> None:
    (row,) = raw_layer.freshness(tmp_path, datetime(2026, 10, 9, tzinfo=UTC), {"telegram": raw_layer.Threshold(26, 50)})
    assert row["status"] == "error"
    assert row["latest_dt"] is None and row["lag_hours"] is None


def test_thresholds_cover_the_collected_sources() -> None:
    assert set(raw_layer.THRESHOLDS) == {"trudvsem", "habr", "telegram", "trudvsem_snapshot"}


def test_batch_hands_over_once_after_the_last_run() -> None:
    pending: list[str] = []
    handed = []
    queued_behind = [6, 5, 4, 3, 2, 1, 0]
    for day, waiting in zip(range(22, 29), queued_behind, strict=True):
        now, pending = dbt_handover.hand_over(pending, [f"trudvsem_snapshot/2026-07-{day}"], waiting)
        handed.append(now)
    assert handed[:-1] == [[]] * 6
    assert handed[-1] == [f"trudvsem_snapshot/2026-07-{day}" for day in range(22, 29)]
    assert pending == []


def test_last_run_hands_over_what_earlier_runs_replayed() -> None:
    now, later = dbt_handover.hand_over(["habr/2026-10-07"], [], 0)
    assert (now, later) == (["habr/2026-10-07"], [])


def test_nothing_replayed_hands_over_nothing() -> None:
    assert dbt_handover.hand_over([], [], 0) == ([], [])
