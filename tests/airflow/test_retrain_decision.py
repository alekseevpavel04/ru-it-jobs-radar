"""The refit rule of radar_retrain over toy drift summaries shaped like results/monitor/drift.json."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from radar.models.salary_features import BLOCKS
from radar.monitor.drift import RECORD_CATEGORICAL, RECORD_NUMERICAL

INCLUDE = Path(__file__).resolve().parents[2] / "airflow" / "include"
spec = importlib.util.spec_from_file_location("airflow_include_retrain_rule", INCLUDE / "retrain_rule.py")
assert spec is not None and spec.loader is not None
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)

NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


def feature(name: str, score: float, spread: float, threshold: float = 0.1) -> dict[str, Any]:
    return {
        "feature": name,
        "score": score,
        "threshold": threshold,
        "drifted": score >= threshold,
        "reference_spread": spread,
        "beyond_spread": score > spread,
    }


def report(*features: dict[str, Any], week: str = "2026-10-12") -> dict[str, Any]:
    return {"status": "ok", "current_week": week, "drift": {"features": list(features)}}


def summary(created: str = "2026-10-19T10:20:00+00:00", **sources: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trudvsem": {"status": "insufficient_history"},
        "habr": {"status": "insufficient_history"},
        "superjob": {"status": "no_data"},
    }
    return {"created_at": created, "sources": base | sources}


def test_quiet_inputs_and_a_drifted_non_input_do_not_refit() -> None:
    # as on 2026-10-05: region drifted within its weekly spread, and the model reads region_group only
    doc = summary(trudvsem_snapshot=report(feature("region", 0.175, 0.215), feature("role_bucket", 0.071, 0.109)))
    d = rule.decide(doc, NOW)
    assert not d.retrain and d.week == "2026-10-12" and d.triggers == []
    assert "trudvsem_snapshot/role_bucket score 0.071" in d.reason
    assert "trudvsem insufficient_history" in d.reason and "superjob no_data" in d.reason


def test_input_drift_beyond_the_reference_spread_refits() -> None:
    doc = summary(
        trudvsem_snapshot=report(feature("role_bucket", 0.18, 0.11), feature("text_len", 0.5, 0.1)),
        habr=report(feature("n_skills", 0.3, 0.2), week="2026-10-05"),
    )
    d = rule.decide(doc, NOW)
    assert d.retrain and d.week == "2026-10-12"
    assert [(t["source"], t["feature"], t["block"]) for t in d.triggers] == [
        ("trudvsem_snapshot", "role_bucket", "role"),
        ("habr", "n_skills", "skills"),
    ]
    assert d.as_dict()["triggers"][0]["score"] == 0.18


@pytest.mark.parametrize(
    "drift",
    [
        feature("grade", 0.15, 0.2),  # past the threshold, within the weekly spread
        feature("grade", 0.08, 0.05),  # past the spread, under the threshold
        feature("salary_mid_rub", 0.4, 0.1),  # the target, not an input
    ],
)
def test_one_condition_or_a_non_input_is_not_enough(drift: dict[str, Any]) -> None:
    assert not rule.decide(summary(trudvsem_snapshot=report(drift)), NOW).retrain


def test_telegram_drift_does_not_refit_a_model_trained_without_telegram() -> None:
    doc = summary(trudvsem_snapshot=report(), telegram=report(feature("n_skills", 0.24, 0.06)))
    assert not rule.decide(doc, NOW).retrain


def test_last_weeks_summary_is_not_acted_on() -> None:
    doc = summary("2026-10-12T10:20:00+00:00", trudvsem_snapshot=report(feature("grade", 0.5, 0.1)))
    d = rule.decide(doc, NOW)
    assert not d.retrain and "2026-10-12" in d.reason and d.week is None


def test_no_report_for_any_salary_source() -> None:
    d = rule.decide(summary(), NOW)
    assert not d.retrain and d.reason.startswith("no source of the salary model has a drift report")


def test_mapped_features_are_drift_features_and_salary_inputs() -> None:
    assert set(rule.INPUT_BLOCK) <= set(RECORD_CATEGORICAL) | set(RECORD_NUMERICAL)
    assert set(rule.INPUT_BLOCK.values()) <= set(BLOCKS)
