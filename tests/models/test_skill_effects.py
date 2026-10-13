import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from radar.models.skill_effects import (
    Config,
    benjamini_hochberg,
    design,
    effects,
    holm,
    overlap_stats,
    treated_skills,
)


def test_treated_skills_need_rows_on_both_sides() -> None:
    lists = [["Python"]] * 8 + [["Python", "SQL"]] * 2 + [[]] * 2
    assert treated_skills(lists, 2) == ["Python", "SQL"]
    assert treated_skills(lists, 3) == []  # Python is absent from 2 rows only, SQL present in 2


def test_adjustments_match_statsmodels() -> None:
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.012])
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])
    assert np.allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """Python adds 0.2 to log salary; Excel adds nothing but goes with the low-paid role (a confounder)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        role = "backend" if rng.random() < 0.5 else "support"
        grade = rng.choice(["junior", "middle", "senior"])
        skills = []
        if rng.random() < (0.6 if role == "backend" else 0.3):
            skills.append("Python")
        if rng.random() < (0.2 if role == "backend" else 0.7):
            skills.append("Excel")
        if rng.random() < 0.3:
            skills.append("Git")
        log_mid = (
            11.5
            + 0.6 * (role == "backend")
            + {"junior": -0.3, "middle": 0, "senior": 0.3}[grade]
            + 0.2 * ("Python" in skills)
        )
        rows.append(
            {
                "group_id": f"g{i // 2}",
                "salary_mid": float(np.exp(log_mid + rng.normal(0, 0.2))),
                "role_bucket": role,
                "grade": grade,
                "region_group": "moscow",
                "work_format": "office",
                "employment_type": "unknown",
                "source": "trudvsem",
                "experience_min": float(rng.integers(0, 5)) if rng.random() < 0.8 else None,
                "skills": skills,
            }
        )
    return pd.DataFrame(rows)


def test_design_counts_skills_outside_the_treated_list() -> None:
    frame = synthetic(200)
    frame.at[0, "skills"] = ["Python", "Fortran"]
    des = design(frame, min_rows=20)
    assert "Fortran" not in des.skills
    assert des.controls.loc[0, "skills_outside_treated"] == 1
    assert des.controls["experience_known"].sum() < len(frame)


CFG = Config(
    feature_set="student",
    splits=("train",),
    min_rows=100,
    n_folds=3,
    n_rep=1,
    seed=1,
    learner={
        "max_iter": 200,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 20,
        "n_iter_no_change": 10,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "validation_fraction": 0.15,
    },
    trim_low=0.05,
    trim_high=0.95,
    min_kept=50,
    level=0.95,
    cf_y=0.03,
    cf_d=0.03,
)


@pytest.mark.filterwarnings("ignore")
def test_double_ml_removes_the_confounded_gap() -> None:
    frame = synthetic(1600)
    des = design(frame, min_rows=100)
    out = effects(des, CFG)
    python, excel = out["rows"]["Python"], out["rows"]["Excel"]
    assert python["ci_low"] < 0.2 < python["ci_high"]
    assert excel["naive_gap"] < -0.2  # the raw gap blames Excel for the role
    assert excel["ci_low"] < 0 < excel["ci_high"]
    assert python["bonferroni_ci_low"] <= python["ci_low"]
    assert out["summary"]["treatments"] == 3
    assert out["summary"]["estimated"]["denominator"] == 3
    assert python["rows_with_skill"] == int(sum("Python" in s for s in frame["skills"]))
    assert python["rows_kept"] == out["overlap"]["Python"]["n"] <= len(frame)
    assert out["overlap"]["Python"]["estimated"]
    assert python["rounds_m_median"] < 200  # early stopping kicked in


def test_overlap_band_and_logloss_ratio() -> None:
    d = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    p = np.array([0.01, 0.2, 0.5, 0.99, 0.3, 0.7])
    stats = overlap_stats(d, p, 0.05, 0.95)
    assert stats["band"].tolist() == [False, True, True, False, True, True]
    assert stats["auc"] == 1.0
    assert stats["logloss_ratio"] < 1  # informative probabilities beat the constant rate
    worst = overlap_stats(d, 1 - d * 0.999 - (1 - d) * 0.001, 0.05, 0.95)
    assert worst["logloss_ratio"] > 1  # confidently wrong probabilities lose to it


@pytest.mark.filterwarnings("ignore")
def test_a_skill_without_overlap_is_left_out_of_the_family() -> None:
    frame = synthetic(1600)
    # Excel now marks the support role exactly: no vacancy of one role is comparable with the other.
    frame["skills"] = [
        [s for s in skills if s != "Excel"] + (["Excel"] if role == "support" else [])
        for skills, role in zip(frame["skills"], frame["role_bucket"], strict=True)
    ]
    out = effects(design(frame, min_rows=100), CFG)
    assert not out["overlap"]["Excel"]["estimated"]
    assert out["overlap"]["Excel"]["kept_with_skill"] < CFG.min_kept
    assert "Excel" not in out["rows"]
    assert out["summary"]["estimated"]["n"] == len(out["rows"]) == 2
    assert out["summary"]["significant_holm"]["denominator"] == 2
