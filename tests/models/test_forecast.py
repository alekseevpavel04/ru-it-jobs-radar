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
    coverage_by_year,
    ets,
    interval_check,
    link_check,
    load_config,
    mase,
    moving_average,
    naive,
    noise_floor,
    seasonal_naive,
    select_skills,
    served_method,
    site_forecast,
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


def test_coverage_by_year_counts_only_weeks_that_are_over() -> None:
    w = weeks(date(2021, 12, 20), 4)  # two weeks of 2021, two of 2022
    cov = pd.DataFrame({"week": w, "units_it": [10.0, 30, 500, 700], "coverage_extractable": [1.0, np.nan, 1, 1]})
    years = coverage_by_year(cov, 0.85, w[-1] + timedelta(days=3))
    assert years == {
        "2021": {"value": 20.0, "weeks": 2, "weeks_covered": 1},
        "2022": {"value": 500.0, "weeks": 1, "weeks_covered": 1},
    }


def test_link_check_shares_of_the_series_weeks() -> None:
    w = weeks(date(2025, 1, 6), 3)
    links = pd.DataFrame({"week": w, "it_units": [100, 100, 50], "link_only": [10, 20, 50], "named": [1, 1, 5]})
    entry = link_check(links, w[:2])
    assert entry == {"value": 0.15, "n": 30, "denominator": 200, "named_value": 0.01, "named_n": 2}


def test_interval_check_uses_only_errors_known_at_the_origin() -> None:
    err = np.ones((1, 60, 2))
    err[0, 59, 0] = 2.0
    err[0, :, 1] = np.arange(60.0)  # growing errors: a quantile of the past never covers the next one
    covered, cells = interval_check(err, 0.8, min_history=52)
    # horizon 1: origins 52..59, horizon 2: origins 53..59 (the error of origin o - 1 is not known yet)
    assert cells == 8 + 7
    assert covered == 7


def test_site_forecast_refits_the_method_on_the_usable_weeks() -> None:
    w = weeks(date(2024, 1, 1), 40)
    rng = np.random.default_rng(3)
    series = pd.DataFrame(
        [
            {"source": "telegram", "week": week, "skill": skill, "vacancies_mentioning": float(rng.poisson(lam))}
            for week in w
            for skill, lam in (("Python", 30), ("SQL", 20))
        ]
    )
    coverage = pd.DataFrame({"source": "telegram", "week": w, "units_it": 200.0, "coverage_extractable": 0.95})
    cfg = config(first_week={}, min_origins=4)
    today = w[-1] + timedelta(days=3)  # the last week is not over
    site = site_forecast(series, coverage, cfg, ["SQL", "Python"], "ets", today)
    assert site.weeks == w[:-1] and site.skills == ["SQL", "Python"]
    assert site.forecast_weeks == w[-1:] + [w[-1] + timedelta(days=7 * h) for h in (1, 2, 3)]
    assert site.origins == 39 - cfg.horizon - cfg.min_train + 1
    assert site.point.shape == site.low.shape == (2, cfg.horizon)
    assert (site.low <= site.point).all() and (site.point <= site.high).all()
    assert site.mentions[0].tolist() == series[series["skill"] == "SQL"]["vacancies_mentioning"].tolist()[:-1]
    with pytest.raises(ValueError, match="no weekly series"):
        site_forecast(series, coverage, cfg, ["Rust"], "ets", today)
    with pytest.raises(ValueError, match="usable weeks"):
        site_forecast(series, coverage, replace(cfg, min_train=60), ["SQL"], "ets", today)
