import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from radar.models.skill_effects import Config, benjamini_hochberg, design, effects, holm, treated_skills


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


@pytest.mark.filterwarnings("ignore")
def test_double_ml_removes_the_confounded_gap() -> None:
    frame = synthetic(1600)
    des = design(frame, min_rows=100)
    cfg = Config(
        feature_set="student",
        splits=("train",),
        min_rows=100,
        n_folds=3,
        n_rep=1,
        seed=1,
        learner={"max_iter": 60, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 20},
        level=0.95,
        cf_y=0.03,
        cf_d=0.03,
    )
    out = effects(des, cfg)
    python, excel = out["rows"]["Python"], out["rows"]["Excel"]
    assert python["ci_low"] < 0.2 < python["ci_high"]
    assert excel["naive_gap"] < -0.2  # the raw gap blames Excel for the role
    assert excel["ci_low"] < 0 < excel["ci_high"]
    assert python["bonferroni_ci_low"] <= python["ci_low"]
    assert out["summary"]["treatments"] == 3
    assert python["rows_with_skill"] == int(sum("Python" in s for s in frame["skills"]))
