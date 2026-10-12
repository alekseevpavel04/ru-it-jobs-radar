from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from radar.monitor.drift import (
    BY_NAME,
    Source,
    analyse_source,
    category,
    choose_windows,
    complete_weeks,
    metrics_of,
    volume_anomalies,
)
from radar.results import Metric

WEEKS = [date(2026, 7, 6) + timedelta(weeks=i) for i in range(5)]
CURRENT = WEEKS[-1]
LAST_DAY = CURRENT + timedelta(days=6)
GONE_DAY = CURRENT + timedelta(days=2)
# Large enough that an unchanged feature stays well under the 0.1 threshold.
PER_WEEK = 2000


class FakeReader:
    """Five complete weeks; the last one moves the role mix and the salaries and loses a day."""

    def __init__(self, extent: tuple[date, date] | None = (WEEKS[0] - timedelta(days=1), LAST_DAY)) -> None:
        self._extent = extent
        self.rng = np.random.default_rng(7)

    def extent(self, source: Source) -> tuple[date, date] | None:
        return self._extent

    def daily(self, source: Source, first: date, last: date) -> pd.DataFrame:
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        days = [d for d in days if d != GONE_DAY]
        rows = self.rng.integers(950, 1050, len(days))
        rows[days.index(CURRENT)] = 400
        return pd.DataFrame(
            {
                "day": days,
                "rows": rows,
                "it_rows": rows // 50,
                "it_without_text": 0,
                "it_without_title": 0,
                "it_without_published_at": [1] + [0] * (len(days) - 1),
            }
        )

    def inflow(self, source: Source, first: date, last: date) -> pd.DataFrame:
        frames = []
        for week in WEEKS:
            n = PER_WEEK
            moved = week == CURRENT
            has_salary = self.rng.random(n) < 0.7
            frames.append(
                pd.DataFrame(
                    {
                        "week": week,
                        "is_duplicate": self.rng.random(n) < 0.05,
                        "role_bucket": self.rng.choice(
                            ["ml", "backend", "qa"], n, p=[0.6, 0.3, 0.1] if moved else [0.2, 0.5, 0.3]
                        ),
                        "grade": self.rng.choice(["junior", "middle", "senior", "unknown"], n),
                        "work_format": self.rng.choice(["office", "remote", "hybrid"], n),
                        "region_group": self.rng.choice(["moscow", "spb", "other"], n),
                        "region": None,
                        "has_salary_rub": has_salary,
                        "salary_mid_rub": np.where(
                            has_salary, self.rng.normal(200_000 if moved else 120_000, 30_000, n), np.nan
                        ),
                        "experience_min_years": self.rng.integers(0, 4, n).astype(float),
                        "no_experience": self.rng.random(n) < 0.2,
                        "n_skills": self.rng.poisson(4, n),
                        "text_len": self.rng.normal(1500, 300, n).round(),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def test_complete_weeks_need_monday_to_sunday() -> None:
    # Tuesday start: its week is partial; Saturday end: so is the last one.
    assert complete_weeks(date(2026, 6, 30), date(2026, 7, 25)) == [date(2026, 7, 6), date(2026, 7, 13)]
    assert complete_weeks(date(2026, 7, 6), date(2026, 7, 12)) == [date(2026, 7, 6)]
    assert complete_weeks(date(2026, 10, 8), date(2026, 10, 11)) == []


def test_reference_is_the_first_weeks_and_current_the_latest() -> None:
    assert choose_windows(WEEKS[:2]) is None
    assert choose_windows(WEEKS[:3]) == (WEEKS[:2], WEEKS[2])
    seven = [WEEKS[0] + timedelta(weeks=i) for i in range(7)]
    assert choose_windows(seven) == (seven[:4], seven[6])


def test_volume_anomalies_flag_far_and_missing_days() -> None:
    reference = pd.Series([100, 102, 98, 101, 99, 100, 103])
    days = [CURRENT + timedelta(days=i) for i in range(7)]
    current = pd.Series({days[0]: 100, days[1]: 40, days[2]: 300, days[4]: 97, days[5]: 104, days[6]: 99})
    out = volume_anomalies(reference, current, days)
    kinds = {f["day"]: f["kind"] for f in out["flagged"]}
    assert kinds == {days[1].isoformat(): "low", days[2].isoformat(): "high", days[3].isoformat(): "missing"}
    assert out["days"] == {"value": round(3 / 7, 4), "n": 3, "denominator": 7}


def test_volume_sigma_has_a_floor_when_the_reference_is_flat() -> None:
    days = [CURRENT]
    # MAD is 0; the floor (5% of 100) keeps 110 inside 3 sigma.
    assert volume_anomalies(pd.Series([100] * 7), pd.Series({CURRENT: 110}), days)["flagged"] == []


def test_categories_are_strings_with_flags_as_words() -> None:
    assert category(True) == "true"
    assert category(np.bool_(False)) == "false"
    assert category(None) is None
    assert category(float("nan")) is None
    assert category("Город Москва") == "Город Москва"


@pytest.fixture(scope="module")
def habr(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    html_dir = tmp_path_factory.mktemp("evidently")
    return analyse_source(BY_NAME["habr"], FakeReader(), html_dir=html_dir), html_dir


def test_source_report_finds_the_moved_features(habr: tuple[dict[str, Any], Path]) -> None:
    out, html_dir = habr

    assert out["status"] == "ok"
    assert out["reference_weeks"] == [w.isoformat() for w in WEEKS[:4]]
    assert out["current_week"] == CURRENT.isoformat()
    features = {f["feature"]: f for f in out["drift"]["features"]}
    assert {name for name, f in features.items() if f["drifted"]} == {"role_bucket", "salary_mid_rub"}
    assert features["role_bucket"]["beyond_spread"] and features["salary_mid_rub"]["beyond_spread"]
    assert features["salary_mid_rub"]["method"] == "wasserstein"
    assert features["role_bucket"]["method"] == "jensenshannon"
    assert out["drift"]["drifted_features"] == {"value": round(2 / 10, 4), "n": 2, "denominator": 10}
    # A field the source lacks is left out of the comparison and named.
    assert out["not_available"] == ["region"]
    assert (html_dir / CURRENT.isoformat() / "habr.html").stat().st_size > 0


def test_source_report_counts_quality_with_denominators() -> None:
    reader = FakeReader()
    out = analyse_source(BY_NAME["trudvsem_snapshot"], reader, html_dir=None)
    quality = out["quality"]

    assert quality["days_collected"]["current"] == {"value": round(6 / 7, 4), "n": 6, "denominator": 7}
    assert quality["days_collected"]["reference"]["n"] == 28
    flagged = {f["day"]: f["kind"] for f in quality["volume"]["flagged"]}
    assert flagged == {CURRENT.isoformat(): "low", GONE_DAY.isoformat(): "missing"}
    assert quality["it_without_published_at"]["reference"]["n"] == 1
    duplicates = quality["duplicates"]["current"]
    assert 0 < duplicates["n"] < duplicates["denominator"] == PER_WEEK
    # Duplicates are counted in quality and left out of the drift units.
    assert out["units"]["current"] == PER_WEEK - duplicates["n"]
    assert quality["new_vacancies"]["current"] == out["units"]["current"]
    assert "report_html" not in out


def test_telegram_has_no_it_share() -> None:
    out = analyse_source(BY_NAME["telegram"], FakeTelegram(), html_dir=None)
    assert out["status"] == "ok"
    assert "it_share" not in out["quality"]
    assert {f["feature"] for f in out["drift"]["features"]} == {"role_bucket", "n_skills"}
    assert set(out["not_available"]) == {"grade", "work_format", "no_experience", "is_multi_vacancy", "channel_topic"}


class FakeTelegram(FakeReader):
    def daily(self, source: Source, first: date, last: date) -> pd.DataFrame:
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        return pd.DataFrame({"day": days, "rows": 100})

    def inflow(self, source: Source, first: date, last: date) -> pd.DataFrame:
        frame = super().inflow(source, first, last)
        for column in ("grade", "work_format", "no_experience", "is_multi_vacancy", "channel_topic"):
            frame[column] = None
        return frame


@pytest.mark.parametrize(
    ("extent", "status"),
    [(None, "no_data"), ((date(2026, 10, 8), date(2026, 10, 11)), "insufficient_history")],
)
def test_short_or_empty_sources_are_reported_not_compared(extent: tuple[date, date] | None, status: str) -> None:
    out = analyse_source(BY_NAME["superjob"], FakeReader(extent), html_dir=None)
    assert out["status"] == status


def test_metrics_follow_the_result_convention(habr: tuple[dict[str, Any], Path]) -> None:
    results = {
        "habr": habr[0],
        "superjob": {"status": "no_data"},
    }
    metrics = metrics_of(results)
    assert set(metrics) == {
        "habr.drifted_features",
        "habr.volume_anomaly_days",
        "habr.duplicates_current",
        "habr.new_vacancies_current",
    }
    for value in metrics.values():
        Metric.model_validate(value)
