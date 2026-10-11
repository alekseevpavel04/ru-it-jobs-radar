import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter, KaplanMeierFitter  # type: ignore[import-untyped]
from lifelines.statistics import proportional_hazard_test  # type: ignore[import-untyped]

from radar.models.survival import (
    absence_runs,
    build_subjects,
    covariates,
    fit_cox,
    follow,
    grace_table,
    km_summary,
    observation_window,
    ph_test,
    piecewise_ratios,
    split_at,
    window_indices,
)

D0 = date(2026, 7, 1)


def day(i: int) -> date:
    return D0 + timedelta(days=i)


def sightings(rows: dict[str, tuple[list[int], int | None]]) -> pd.DataFrame:
    """entity -> (days seen open, creation day relative to D0)."""
    out = [
        {"entity_id": e, "dt": day(i), "published": day(pub) if pub is not None else None}
        for e, (days, pub) in rows.items()
        for i in days
    ]
    return pd.DataFrame(out)


def test_window_stops_at_the_first_long_hole_and_lists_every_hole() -> None:
    days = [day(i) for i in [0, 1, 2, 4, 5, 20, 21]]
    w = observation_window(days, max_step=2)
    assert w.days == tuple(day(i) for i in [0, 1, 2, 4, 5])
    assert w.holes == ((day(2), day(4)), (day(5), day(20)))
    assert w.days_after == 2


def test_return_within_grace_keeps_the_vacancy_open() -> None:
    # Seen 0-2, missing 3-5 (3 days), back 6-9; window 0..9, grace 4.
    assert follow([0, 1, 2, 6, 7, 8, 9], 10, 4) == (9, False, False)


def test_late_return_closes_on_the_first_missing_day() -> None:
    assert follow([0, 1, 7, 8], 10, 4) == (1, True, True)


def test_absence_at_the_end_closes_only_once_the_grace_is_observed() -> None:
    assert follow([0, 1, 2], 10, 4) == (2, True, False)
    assert follow([0, 1, 2, 3, 4, 5, 6], 10, 4) == (6, False, False)


def test_absence_runs_counts_missing_collected_days() -> None:
    assert absence_runs([0, 1, 4, 5, 9], 12) == ([2, 3], 2)


def test_subjects_time_since_creation_with_delayed_entry() -> None:
    calendar = [day(i) for i in range(20)]
    rows = {
        # already open on the first day, created 10 days before it; closes after day 5
        "prevalent": ([0, 1, 2, 3, 4, 5], -10),
        # appears on day 3, created on day 2; open to the end: censored
        "new_open": (list(range(3, 20)), 2),
        # hidden days 6-7 and back: one span, closes after day 12
        "hidden": ([*range(4, 6), *range(8, 13)], 4),
        # gone 5 days and back after the 3-day grace: closed after day 6, the return not followed
        "late": ([5, 6, 12, 13], 5),
        # last seen near the end, absence shorter than the grace: censored
        "tail": ([10, 11, 12, 13, 14, 15, 16, 17], 10),
    }
    w = observation_window(calendar)
    s = build_subjects(sightings(rows), w, grace=3).set_index("entity_id")

    p = s.loc["prevalent"]
    assert (p["entry"], p["duration"], p["event"], p["prevalent"]) == (10, 16, True, True)
    assert p["end"] == day(6)

    n = s.loc["new_open"]
    assert (n["entry"], n["duration"], n["event"], n["prevalent"]) == (1, 18, False, False)
    assert n["duration_incident"] == 17

    h = s.loc["hidden"]
    assert (h["entry"], h["duration"], h["event"], h["late_return"]) == (0, 9, True, False)

    late = s.loc["late"]
    assert (late["duration"], late["event"], late["late_return"]) == (2, True, True)

    t = s.loc["tail"]
    assert (t["duration"], t["event"]) == (8, False)


def test_reposts_of_one_entity_are_one_subject() -> None:
    calendar = [day(i) for i in range(15)]
    # Two records of one entity: the first seen 0-4 (created day 0), the repost 6-9 (created day 6).
    frame = pd.DataFrame(
        [{"entity_id": "e", "dt": day(i), "published": day(0)} for i in range(5)]
        + [{"entity_id": "e", "dt": day(i), "published": day(6)} for i in range(6, 10)]
    )
    s = build_subjects(frame, observation_window(calendar), grace=3)
    assert len(s) == 1
    row = s.iloc[0]
    assert (row["entry"], row["duration"], row["event"], row["late_return"]) == (0, 10, True, False)


def test_creation_date_after_first_sighting_is_not_trusted() -> None:
    calendar = [day(i) for i in range(10)]
    s = build_subjects(sightings({"e": ([2, 3], 5)}), observation_window(calendar), grace=3)
    assert s.iloc[0]["origin"] == day(2)
    assert s.iloc[0]["entry"] == 0


def test_grace_table_counts_closures_that_came_back() -> None:
    calendar = [day(i) for i in range(20)]
    frame = sightings({"a": ([0, 1, 5, 6], 0), "b": ([0, 10], 0), "c": ([0, 1, 2], 0)})
    w = observation_window(calendar)
    table = grace_table(window_indices(frame, w), len(w.days), [1, 3, 10])
    # a: run 3 then tail 13; b: run 9 then tail 9; c: tail 17
    assert table["by_grace"]["1"] == {"came_back": 2, "closures": 5}
    assert table["by_grace"]["3"] == {"came_back": 2, "closures": 5}
    assert table["by_grace"]["10"] == {"came_back": 0, "closures": 2}
    assert table["returned_run_days"]["3"] == 1
    assert table["returned_run_days"]["9"] == 1


def test_covariates_map_source_fields_to_levels() -> None:
    entities = pd.DataFrame(
        {
            "entity_id": ["a", "b", "c", "d"],
            "role_bucket": ["ml_ds", "backend", "not_it", "qa"],
            "experience_min_years": [0, 2, 4, None],
            "salary_from": [100000, None, None, 0],
            "salary_to": [None, None, 50000, None],
            "region_group": ["moscow", "remote", "unknown", "other"],
            "employer_key": ["1", None, "2", "2"],
        }
    )
    c = covariates(entities).set_index("entity_id")
    assert c["role"].tolist()[:2] == ["ml_ds_data", "dev"]
    assert pd.isna(c.loc["c", "role"])
    assert c["experience"].tolist()[:3] == ["none", "1-2", "3-5"]
    assert pd.isna(c.loc["d", "experience"])
    assert c["salary"].tolist() == ["given", "not_given", "given", "not_given"]
    assert pd.isna(c.loc["c", "region"])
    assert c.loc["b", "cluster"] == "b"


def test_km_with_delayed_entry_recovers_an_exponential_median() -> None:
    rng = np.random.default_rng(1)
    n = 20000
    life = rng.exponential(30.0, n)
    entry = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 60, n))
    seen = life > entry
    frame = pd.DataFrame({"entry": entry[seen], "duration": np.minimum(life[seen], entry[seen] + 40)})
    frame["event"] = life[seen] <= entry[seen] + 40
    out = km_summary(frame)
    assert abs(out["value"] - 30 * math.log(2)) < 1.5
    assert out["low"] <= out["value"] <= out["high"]


def test_km_reports_a_risk_set_that_empties_before_a_late_entry() -> None:
    frame = pd.DataFrame({"entry": [0, 5], "duration": [3, 8], "event": [True, True]})
    assert km_summary(frame)["value"] == "not estimable"


def cox_frame(n: int, entry_max: float, seed: int) -> pd.DataFrame:
    """Exponential lifetimes with a true log-HR of 0.7 for x and -0.4 for z, no ties, clustered ids."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    z = rng.integers(0, 2, n)
    life = rng.exponential(20.0 / np.exp(0.7 * x - 0.4 * z))
    entry = rng.uniform(0, entry_max, n) if entry_max else np.zeros(n)
    keep = life > entry
    censor = entry + rng.exponential(40.0, n)
    frame = pd.DataFrame(
        {
            "duration": np.minimum(life, censor),
            "event": (life <= censor).astype(int),
            "entry": entry,
            "cluster": rng.integers(0, n // 5, n).astype(str),
            "x": x,
            "z": z,
        }
    )
    return frame[keep].reset_index(drop=True)


def test_clustered_sandwich_matches_lifelines_without_entry() -> None:
    frame = cox_frame(1500, 0, seed=2)
    ours = fit_cox(frame)
    ref = CoxPHFitter().fit(frame.drop(columns=["entry"]), "duration", "event", cluster_col="cluster", robust=True)
    assert np.allclose(ours.beta, ref.params_[["x", "z"]].to_numpy(), atol=1e-6)
    assert np.allclose(np.sqrt(np.diag(ours.robust_variance)), ref.standard_errors_[["x", "z"]].to_numpy(), rtol=1e-4)


def test_cox_with_delayed_entry_recovers_the_hazard_ratio() -> None:
    frame = cox_frame(6000, 30, seed=3)
    fit = fit_cox(frame)
    se = np.sqrt(np.diag(fit.robust_variance))
    assert abs(fit.beta[0] - 0.7) < 3 * se[0]
    assert abs(fit.beta[1] + 0.4) < 3 * se[1]


def test_ph_test_matches_lifelines_without_entry() -> None:
    frame = cox_frame(1500, 0, seed=4)
    ours = ph_test(frame, fit_cox(frame))
    ref = CoxPHFitter().fit(frame.drop(columns=["entry", "cluster"]), "duration", "event")
    result = proportional_hazard_test(ref, frame.drop(columns=["entry", "cluster"]), time_transform="rank")
    for name in ("x", "z"):
        assert ours[name]["statistic"] == pytest.approx(float(result.summary.loc[name, "test_statistic"]), abs=1e-3)


def test_ph_test_flags_a_hazard_ratio_that_changes_with_time() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    x = rng.integers(0, 2, n)
    # x triples the hazard before day 10 and has no effect after it.
    early = rng.exponential(10.0 / np.where(x == 1, 3.0, 1.0))
    life = np.where(early < 10, early, 10 + rng.exponential(10.0, n))
    frame = pd.DataFrame({"duration": life, "event": 1, "entry": 0.0, "cluster": np.arange(n).astype(str), "x": x})
    assert ph_test(frame, fit_cox(frame))["x"]["p"] < 0.001

    split = split_at(frame, 10, ["x"])
    ratios = piecewise_ratios(fit_cox(split), split, ["x"], 10)["x"]
    assert ratios["early"]["low"] < 3.0 < ratios["early"]["high"]
    assert ratios["late"]["low"] < 1.0 < ratios["late"]["high"]
    assert ratios["change_p"] < 0.001
    assert ratios["subjects"] == int(x.sum())


def test_split_at_tiles_each_subject_and_keeps_its_event_last() -> None:
    frame = pd.DataFrame(
        {"duration": [5, 40, 50], "event": [1, 1, 0], "entry": [0, 2, 35], "cluster": ["a", "b", "c"], "x": [1, 0, 1]}
    )
    split = split_at(frame, 30, ["x"])
    by = split.sort_values(["cluster", "entry"]).groupby("cluster")
    spans = {k: list(zip(g["entry"], g["duration"], g["event"], g["x:late"], strict=True)) for k, g in by}
    assert spans == {
        "a": [(0, 5, 1, 0)],
        "b": [(2, 30, 0, 0), (30, 40, 1, 0)],
        "c": [(35, 50, 0, 1)],
    }


def test_km_summary_without_entry_counts_everyone_at_risk_at_zero() -> None:
    frame = pd.DataFrame({"duration_incident": [3, 5, 8], "event": [True, False, True]})
    out = km_summary(frame, None, "duration_incident")
    assert out["at_risk"]["0"] == 3
    assert out["subjects"] == 3
    assert out["events"] == 2
    kmf = KaplanMeierFitter().fit([3, 5, 8], [True, False, True])
    assert out["value"] == kmf.median_survival_time_
