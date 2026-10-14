from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from radar.monitor.drift import BY_NAME, Source
from radar.monitor.drift_output import analyse_source, extraction_model, metrics_of, output_frame
from radar.results import Metric

WEEKS = [date(2026, 7, 6) + timedelta(weeks=i) for i in range(5)]
CURRENT = WEEKS[-1]
PER_WEEK = 2000
ROLES = ["ml_engineer", "backend", "qa"]


class FakeReader:
    """Five complete weeks of student outputs; the last one moves the role mix and the skill count.

    current_coverage: the share of the current week's units that have an extracted version."""

    def __init__(self, current_coverage: float = 1.0) -> None:
        self.current_coverage = current_coverage
        self.rng = np.random.default_rng(11)

    def extent(self, source: Source) -> tuple[date, date] | None:
        return WEEKS[0] - timedelta(days=1), CURRENT + timedelta(days=6)

    def outputs(self, source: Source, first: date, last: date) -> pd.DataFrame:
        frames = []
        for week in WEEKS:
            n, moved = PER_WEEK, week == CURRENT
            extracted = self.rng.random(n) < (self.current_coverage if moved else 1.0)
            valid = extracted & (self.rng.random(n) < 0.99)
            frames.append(
                pd.DataFrame(
                    {
                        "week": week,
                        "is_duplicate": self.rng.random(n) < 0.05,
                        "source_id": [f"{week}-{i}" for i in range(n)],
                        "content_hash": "h",
                        "is_extracted": extracted,
                        "valid": valid,
                        "role": self.rng.choice(ROLES, n, p=[0.6, 0.3, 0.1] if moved else [0.2, 0.5, 0.3]),
                        "grade": self.rng.choice(["junior", "middle", None], n),
                        "work_format": self.rng.choice(["office", "remote", "hybrid"], n),
                        "employment_type": None,
                        "lang": "ru",
                        "has_salary": self.rng.random(n) < 0.6,
                        "has_experience": self.rng.random(n) < 0.5,
                        "n_skills": self.rng.poisson(8 if moved else 4, n),
                        "n_cities": self.rng.poisson(1, n),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def test_output_frame_keeps_null_as_a_category_and_invalid_answers_only_as_invalid() -> None:
    units = pd.DataFrame(
        {
            "is_extracted": [True, True, False],
            "valid": [True, False, False],
            "role": ["qa", None, None],
            "grade": [None, None, None],
            "work_format": ["remote", None, None],
            "employment_type": [None, None, None],
            "lang": ["ru", None, None],
            "has_salary": [np.bool_(True), np.bool_(False), np.bool_(False)],
            "has_experience": [np.bool_(False), np.bool_(False), np.bool_(False)],
            "n_skills": [3, 0, 0],
            "n_cities": [1, 0, 0],
        }
    )
    frame = output_frame(units)

    assert len(frame) == 2  # the unit without an extracted version is not compared
    assert frame.loc[0, ["valid", "role", "grade", "has_salary", "has_experience"]].tolist() == [
        "true",
        "qa",
        "null",
        "true",
        "false",
    ]
    assert frame.loc[0, "n_skills"] == 3.0
    invalid = frame.loc[1]
    assert invalid["valid"] == "false"
    assert invalid[["role", "grade", "has_salary"]].isna().all()
    assert np.isnan(invalid["n_skills"])


@pytest.fixture(scope="module")
def telegram(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    html_dir = tmp_path_factory.mktemp("evidently")
    return analyse_source(BY_NAME["telegram"], FakeReader(), html_dir=html_dir), html_dir


def test_output_drift_finds_the_moved_fields(telegram: tuple[dict[str, Any], Path]) -> None:
    out, html_dir = telegram

    assert out["status"] == "ok"
    assert out["current_week"] == CURRENT.isoformat()
    features = {f["feature"]: f for f in out["drift"]["features"]}
    assert {name for name, f in features.items() if f["drifted"]} == {"role", "n_skills"}
    assert features["role"]["method"] == "jensenshannon"
    assert features["n_skills"]["method"] == "wasserstein"
    # employment_type is never stated, so it is all "null" in both windows: compared, not drifted.
    assert not features["employment_type"]["drifted"]
    valid = out["valid_json"]["current"]
    assert 0 < valid["n"] <= valid["denominator"] == out["extracted"]["current"]
    assert out["coverage"]["current"]["value"] == 1.0
    assert (html_dir / CURRENT.isoformat() / "output_telegram.html").stat().st_size > 0


def test_a_week_the_extraction_has_not_caught_up_with_is_not_compared() -> None:
    out = analyse_source(BY_NAME["trudvsem_snapshot"], FakeReader(current_coverage=0.4), html_dir=None)

    assert out["status"] == "extraction_incomplete"
    assert "drift" not in out
    current = out["coverage"]["current"]
    assert 0.3 < current["value"] < 0.5
    assert out["coverage"]["reference"]["value"] == 1.0


def test_metrics_follow_the_result_convention(telegram: tuple[dict[str, Any], Path]) -> None:
    incomplete = analyse_source(BY_NAME["habr"], FakeReader(current_coverage=0.4), html_dir=None)
    metrics = metrics_of({"telegram": telegram[0], "habr": incomplete, "superjob": {"status": "no_data"}})

    assert set(metrics) == {
        "telegram.coverage_current",
        "telegram.drifted_outputs",
        "telegram.valid_json_current",
        "habr.coverage_current",
    }
    for value in metrics.values():
        Metric.model_validate(value)


def test_extraction_model_is_the_marts_one(tmp_path: Path) -> None:
    project = tmp_path / "dbt_project.yml"
    project.write_text('vars:\n  extraction_model: "student-v1"\n', encoding="utf-8")
    assert extraction_model(project) == "student-v1"
    project.write_text('vars:\n  extraction_model: ""\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        extraction_model(project)
