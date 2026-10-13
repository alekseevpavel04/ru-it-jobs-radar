from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from radar.models.forecast import (
    Backtest,
    Config,
    Panel,
    backtest,
    boosting,
    bootstrap_index,
    compare,
    ets,
    load_config,
    mase,
    moving_average,
    naive,
    noise_floor,
    seasonal_naive,
    select_skills,
    served_method,
    smape,
    usable_weeks,
)


def config(**overrides: Any) -> Config:
    return replace(load_config(), **{"min_train": 20, "season": 6, "replicates": 200, **overrides})


def weeks(start: date, n: int) -> list[date]:
    return [start + timedelta(days=7 * i) for i in range(n)]


def test_usable_weeks_keep_the_last_complete_contiguous_run() -> None:
    w = weeks(date(2025, 1, 6), 10)
    coverage = pd.DataFrame({"week": w, "coverage_extractable": [0.9] * 10})
    coverage.loc[3, "coverage_extractable"] = 0.5  # breaks the run
    coverage.loc[1, "coverage_extractable"] = np.nan
    today = w[-1] + timedelta(days=3)  # the last week is not over
    assert usable_weeks(coverage, None, 0.85, today) == w[4:9]
    assert usable_weeks(coverage, w[6], 0.85, today) == w[6:9]
    assert usable_weeks(coverage, None, 0.85, w[-1] + timedelta(days=7)) == w[4:]


def test_baselines() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert naive(y, 3).tolist() == [6.0, 6.0, 6.0]
    assert moving_average(y, 2, 4).tolist() == [4.5, 4.5]
    assert seasonal_naive(y, 3, 4).tolist() == [3.0, 4.0, 5.0]
    assert np.isnan(seasonal_naive(y, 2, 7)).all()


def test_ets_on_a_flat_series_is_flat() -> None:
    assert ets(np.full(30, 0.2), 4).tolist() == [0.2] * 4
    rng = np.random.default_rng(0)
    noisy = 0.3 + rng.normal(0, 0.01, 60)
    assert np.allclose(ets(noisy, 4), 0.3, atol=0.02)


def test_mase_of_naive_on_a_trend_is_the_mean_horizon() -> None:
    # y = t: every one-step change is 1, the naive error at horizon j is j, so MASE = mean(1..4).
    Y = np.vstack([np.arange(40.0), 2 * np.arange(40.0)])
    bt = backtest(Y, config(), methods=("naive", "seasonal_naive"))
    assert bt.actual.shape == (2, 40 - 4 - 20 + 1, 4)
    assert np.allclose(mase(bt, "naive"), 2.5)


def test_seasonal_naive_is_exact_on_a_pure_season() -> None:
    Y = np.tile([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], 6)[None, :]
    bt = backtest(Y, config(), methods=("naive", "seasonal_naive"))
    assert np.allclose(mase(bt, "seasonal_naive"), 0.0)
    assert np.allclose(smape(bt, "seasonal_naive"), 0.0)
    assert (mase(bt, "naive") > 0.5).all()


def test_boosting_returns_a_forecast_per_series_and_horizon() -> None:
    rng = np.random.default_rng(1)
    Y = np.vstack([level + rng.normal(0, 0.01, 40) for level in (0.1, 0.2, 0.3)])
    cfg = config()
    out = boosting(Y, 4, replace(cfg, boosting=cfg.boosting | {"min_samples_leaf": 5}))
    assert out.shape == (3, 4)
    assert np.allclose(out, Y.mean(axis=1)[:, None], atol=0.05)


def test_bootstrap_draws_whole_blocks_of_origins() -> None:
    rng = np.random.default_rng(0)
    skills, origins = bootstrap_index(5, 10, 4, rng)
    assert len(skills) == 5 and len(origins) == 10
    assert ((skills >= 0) & (skills < 5)).all()
    assert (np.diff(origins[:4]) == 1).all() and (np.diff(origins[4:8]) == 1).all()


def test_paired_difference_and_served_method() -> None:
    rng = np.random.default_rng(2)
    Y = np.cumsum(rng.normal(0, 1, (4, 50)), axis=1) + 100
    cfg = config()
    bt = backtest(Y, cfg, methods=("naive", "moving_average"))
    out = compare(bt, cfg, ("naive", "moving_average"))
    diff = out["moving_average_minus_naive.mase"]
    assert diff["value"] == pytest.approx(out["moving_average.mase"]["value"] - out["naive.mase"]["value"], abs=1e-3)
    assert diff["ci_low"] <= diff["value"] <= diff["ci_high"]
    assert out["moving_average_minus_naive.skills_better"]["denominator"] == 4
    assert served_method({"naive.mase": {"value": 1.0}, "ets_minus_naive.mase": {"ci_high": 0.01}}) == "naive"
    better = {
        "ets.mase": {"value": 0.9},
        "ets_minus_naive.mase": {"ci_high": -0.01},
        "moving_average.mase": {"value": 0.8},
        "moving_average_minus_naive.mase": {"ci_high": -0.02},
    }
    assert served_method(better) == "moving_average"


def test_select_skills_by_median_and_exclusion() -> None:
    panel = Panel(
        source="telegram",
        weeks=weeks(date(2025, 1, 6), 3),
        skills=["A", "B", "HTTPS", "C"],
        mentions=np.array([[20.0, 25, 30], [40, 40, 1], [90, 90, 90], [5, 50, 50]]),
        it_units=np.array([100.0, 100, 100]),
    )
    assert [panel.skills[i] for i in select_skills(panel, 20, ("HTTPS",))] == ["C", "B", "A"]
    assert np.allclose(panel.target("share")[0], [0.2, 0.25, 0.3])


def test_noise_floor_is_the_binomial_mean_absolute_deviation() -> None:
    actual = np.full((1, 2, 4), 0.25)
    bt = Backtest(origins=[0, 1], actual=actual, forecast={}, scale=np.full((1, 2), 0.01))
    floor = noise_floor(bt, np.full(10, 300.0))
    assert np.allclose(floor, np.sqrt(2 / np.pi) * np.sqrt(0.25 * 0.75 / 300) / 0.01)
