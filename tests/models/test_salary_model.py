import math
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar.models.salary import (
    Config,
    Fitted,
    SalaryModel,
    choose,
    conformal_quantile,
    evaluate,
    fit_all,
    group_weights,
    importance,
    load_model,
    predict,
    score,
    shap_values,
    temporal_check,
)
from radar.models.salary_data import (
    CALIB,
    DEV,
    TEST,
    TRAIN,
    WITHHELD,
    SplitShares,
    assign,
    build_frame,
    components,
    is_part_time,
    midpoint,
    records_of,
    student_fields,
    v0_feature_row,
    with_features,
)
from radar.models.salary_features import BLOCKS, Encoder, OutOfScope, feature_row, region_group
from radar.models.salary_shap import block_shap, summary

ROLE_BASE = {"backend": 11.6, "sysadmin_support": 10.9, "ml_ds": 11.9, "qa": 11.2}
GRADE_SHIFT = {"junior": -0.4, "unknown": 0.0, "senior": 0.35}
REGION_SHIFT = {"moscow": 0.35, "other": 0.0, "remote": 0.15}


def cfg(**over: object) -> Config:
    base = {
        "min_monthly": 10000.0,
        "shares": SplitShares("t", 0.2, 0.2, 0.3),
        "level": 0.8,
        "cutoff": date(2026, 9, 1),
        "min_skill_count": 5,
        "min_cell": 5,
        "cv_folds": 3,
        "seed": 1,
        "max_iter": 100,
        "grid": {"learning_rate": [0.1], "max_leaf_nodes": [15], "min_samples_leaf": [10], "l2_regularization": [0.0]},
        "ridge_alpha": [1.0, 10.0],
        "importance_repeats": 2,
        "n_resamples": 200,
        "bootstrap_seed": 7,
        "confidence": 0.95,
    }
    return Config(**(base | over))  # type: ignore[arg-type]


def synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """Rows of the model frame whose log salary is a sum of role, grade, region and a Python bonus."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        bucket = rng.choice(list(ROLE_BASE))
        grade = rng.choice(list(GRADE_SHIFT))
        region = rng.choice(list(REGION_SHIFT))
        skills = ["Python"] if rng.random() < 0.4 else []
        log_mid = ROLE_BASE[bucket] + GRADE_SHIFT[grade] + REGION_SHIFT[region] + 0.2 * bool(skills)
        mid = float(np.exp(log_mid + rng.normal(0, 0.15)))
        row = feature_row(
            {
                "role": {"backend": "backend", "sysadmin_support": "sysadmin", "ml_ds": "ml_engineer", "qa": "qa"}[
                    bucket
                ],
                "grade": None if grade == "unknown" else grade,
                "cities": ["Москва"] if region == "moscow" else ["Тверь"],
                "work_format": "remote" if region == "remote" else "office",
                "skills": skills,
                "experience_years": {"min": float(rng.integers(0, 6)), "max": None},
            },
            "trudvsem",
        )
        u = i % 10
        split = TRAIN if u < 5 else CALIB if u < 7 else DEV if u < 9 else TEST
        rows.append(
            {
                "entity_id": f"e{i}",
                "group_id": f"g{i // 2}" if split == TRAIN else f"g{i}",
                "split": split,
                "first_seen": date(2026, 8, 1) + timedelta(days=i % 60),
                "salary_from": mid * 0.9,
                "salary_to": mid * 1.1,
                "salary_mid": mid,
                **row,
            }
        )
    return pd.DataFrame(rows)


def test_feature_row_maps_schema_fields() -> None:
    row = feature_row(
        {"role": "data_scientist", "grade": "middle", "cities": ["Казань"], "skills": ["SQL", "Python", "SQL"]}, "habr"
    )
    assert row["role_bucket"] == "ml_ds"
    assert row["role_group"] == "ml_ds_data"
    assert row["region_group"] == "other"
    assert row["work_format"] == "unknown"
    assert row["experience_min"] is None
    assert row["skills"] == ["Python", "SQL"]
    assert feature_row({}, "trudvsem")["role_bucket"] == "unknown"
    with pytest.raises(OutOfScope):
        feature_row({"role": "non_it"}, "trudvsem")


def test_region_group_follows_marts_order() -> None:
    assert region_group(["Москва"], "remote") == "remote"
    assert region_group(["Санкт-Петербург", "Москва"], "office") == "moscow"
    assert region_group(["Санкт-Петербург"], None) == "spb"
    assert region_group([], "hybrid") == "unknown"


def test_encoder_unseen_levels_and_rare_skills() -> None:
    rows = [feature_row({"role": "qa", "skills": ["Python"]}, "trudvsem") for _ in range(3)]
    rows.append(feature_row({"role": "qa", "skills": ["Rust"]}, "trudvsem"))
    enc = Encoder.fit(rows, min_skill_count=2)
    assert enc.skill_vocab == ("Python",)
    new = enc.encode([feature_row({"role": "backend", "skills": ["Rust", "Python"]}, "habr")])
    tree = enc.tree_matrix(new)
    assert math.isnan(tree[0, 0])  # role bucket unseen in training
    assert tree[0, -1] == 1.0
    linear = enc.linear_matrix(new, fill=np.array([1.0]))
    assert linear[0, : len(enc.levels["role_bucket"])].sum() == 0


def test_midpoint_and_part_time() -> None:
    assert midpoint(100.0, 200.0) == 150.0
    assert midpoint(None, 90.0) == 90.0
    assert math.isnan(midpoint(float("nan"), None))
    assert is_part_time("Частичная занятость", None)
    assert is_part_time(None, "Неполный рабочий день/неполная рабочая неделя")
    assert is_part_time("part_time", None)
    assert is_part_time(None, "Частичная занятость / Совместительство")
    assert not is_part_time("Полная занятость", "Полный рабочий день")


def test_v0_row_reads_no_experience_as_zero() -> None:
    rec = {
        "role_bucket": "backend",
        "grade": None,
        "region_group": "remote",
        "work_format": "remote",
        "employment": "Стажировка",
        "family": "habr",
        "experience_min_years": float("nan"),
        "no_experience": True,
        "skills": "SQL|Go",
    }
    row = v0_feature_row(rec)
    assert row["experience_min"] == 0.0
    assert row["grade"] == "unknown"
    assert row["employment_type"] == "internship"
    assert row["skills"] == ["Go", "SQL"]


def test_components_join_through_shared_clusters() -> None:
    comp = components([("e:a", "t:1"), ("e:b", "t:1"), ("e:c", "j:x|title")], ["a", "b", "c", "d"])
    assert comp["a"] == comp["b"] == "a"
    assert comp["c"] == "c"
    assert comp["d"] == "d"


def test_assign_pins_corpus_splits_before_the_hash() -> None:
    shares = SplitShares("s", 0.2, 0.2, 0.25)
    assert assign("x", {"train_pool", "test"}, shares) == TEST
    assert assign("x", {"dev"}, shares) == DEV
    assert assign("x", {"withheld"}, shares) == WITHHELD
    drawn = [assign(f"k{i}", set(), shares) for i in range(4000)]
    assert abs(drawn.count(TEST) / 4000 - 0.2) < 0.03
    assert abs(drawn.count(CALIB) / 4000 - 0.6 * 0.25) < 0.03


def test_build_frame_keeps_groups_in_one_split() -> None:
    n = 300
    entities = pd.DataFrame(
        {
            "entity_id": [f"e{i}" for i in range(n)],
            "family": "trudvsem",
            "latest_source": "trudvsem_snapshot",
            "latest_source_id": [f"t{i}" for i in range(n)],
            "first_seen": date(2026, 9, 1),
            "employer_key": [f"inn{i % 40}" for i in range(n)],
            "title": ["Программист 1С" if i % 3 else f"Инженер {i}" for i in range(n)],
            "salary_from": [50000.0 + i for i in range(n)],
            "salary_to": None,
            "role_bucket": "developer_1c",
            "grade": "unknown",
            "work_format": "unknown",
            "region_group": "other",
            "experience_min_years": 1.0,
            "no_experience": False,
            "skills": "1С",
            "employment": ["Частичная занятость" if i == 5 else "Полная занятость" for i in range(n)],
            "schedule": "Полный рабочий день",
        }
    )
    records = pd.DataFrame(
        {
            "entity_id": entities["entity_id"],
            "record_id": [f"trudvsem:{i}" for i in range(n)],
            "text_cluster_id": [f"c{i % 100}" for i in range(n)],
        }
    )
    # Record 3 is not extracted, record 4 is non-IT by the student, record 6 failed validation.
    student = pd.DataFrame(
        {
            "source": "trudvsem_snapshot",
            "source_id": [f"t{i}" for i in range(n) if i != 3],
            "valid": [i != 6 for i in range(n) if i != 3],
            "role": ["non_it" if i == 4 else "developer_1c" for i in range(n) if i != 3],
            "grade": "middle",
            "skills": "1С|SQL",
            "work_format": None,
            "cities": "Москва",
            "experience_min": 3.0,
            "experience_max": None,
            "employment_type": None,
        }
    )
    corpus = {"trudvsem:7": ("grp_a", "test"), "trudvsem:8": ("grp_b", "withheld")}
    frame, counts = build_frame(entities, records, student, corpus, SplitShares("s", 0.2, 0.2, 0.25), 10000)
    assert counts["part_time"] == 1
    assert counts["student_missing"] == 2
    assert counts["student_non_it"] == 1
    assert not {"e3", "e4", "e6"} & set(frame["entity_id"])
    view = with_features(frame, "student")
    assert view.iloc[0]["region_group"] == "moscow"
    assert view.iloc[0]["skills"] == ["1С", "SQL"]
    assert with_features(frame, "rules").iloc[0]["region_group"] == "other"
    assert not any(c.startswith(("rules_", "student_")) for c in view.columns)
    assert (frame.groupby("group_id")["split"].nunique() == 1).all()
    assert "e7" not in set(frame.loc[frame["split"] != TEST, "entity_id"])
    assert "e8" not in set(frame["entity_id"])
    assert counts["rows"] == len(frame) == sum(counts[f"rows_{s}"] for s in (TRAIN, CALIB, DEV, TEST))


def test_conformal_quantile_finite_sample() -> None:
    scores = np.arange(1.0, 10.0)  # 9 scores
    assert conformal_quantile(scores, 0.8) == 8.0  # ceil(10 * 0.8) = 8th smallest
    assert conformal_quantile(scores, 0.95) == math.inf


def test_group_weights_draw_whole_groups() -> None:
    W = group_weights(["a", "a", "b", "c"], 50, 3)
    assert W.shape == (50, 4)
    assert (W[:, 0] == W[:, 1]).all()


def test_score_reports_shares_with_denominator() -> None:
    W = group_weights(["a", "b", "c", "d"], 100, 1)
    out = score({"coverage": np.array([1.0, 0.0, 1.0, 1.0]), "mae_rub": np.array([1.0, 2.0, 3.0, 4.0])}, W, 0.95)
    assert out["coverage"]["n"] == 3
    assert out["coverage"]["denominator"] == 4
    assert out["mae_rub"]["value"] == 2.5
    assert out["mae_rub"]["ci_low"] <= 2.5 <= out["mae_rub"]["ci_high"]


@pytest.fixture(scope="module")
def fitted_run() -> tuple[Fitted, pd.DataFrame]:
    frame = synthetic(900)
    c = cfg()
    fitted = fit_all(frame[frame["split"] == TRAIN], frame[frame["split"] == CALIB], c)
    return fitted, frame


def test_models_learn_the_signal_and_cover_at_level(fitted_run: tuple[Fitted, pd.DataFrame]) -> None:
    fitted, frame = fitted_run
    dev = frame[frame["split"] == DEV]
    m = evaluate(fitted, dev, cfg(), DEV)["metrics"]
    assert m["dev_gbm_cqr_mae_rub"]["value"] < m["dev_global_median_mae_rub"]["value"] * 0.6
    assert m["dev_ridge_mape"]["value"] < 0.2
    assert 0.65 <= m["dev_gbm_cqr_coverage"]["value"] <= 0.95
    assert m["dev_gbm_cqr_coverage"]["denominator"] == len(dev)
    assert (
        m["dev_diff_gbm_cqr_minus_cell_median_mae_rub"]["ci_low"]
        <= m["dev_diff_gbm_cqr_minus_cell_median_mae_rub"]["ci_high"]
    )


def test_predict_round_trip(fitted_run: tuple[Fitted, pd.DataFrame], tmp_path: Path) -> None:
    fitted, _ = fitted_run
    model = fitted.model
    path = tmp_path / "salary.pkl"
    path.write_bytes(pickle.dumps(model))
    loaded = load_model(path)
    assert isinstance(loaded, SalaryModel)
    senior = predict(
        {"role": "ml_engineer", "grade": "senior", "cities": ["Москва"], "skills": ["Python"]}, model=loaded
    )
    junior = predict({"role": "sysadmin", "grade": "junior", "cities": ["Тверь"]}, model=loaded)
    assert senior["low"] <= senior["point"] <= senior["high"]
    assert senior["point"] > junior["point"]
    assert senior["point"] % 1000 == 0
    assert senior["n_similar"] > 0
    assert set(senior) >= {"point", "low", "high", "n_similar", "model_version"}
    with pytest.raises(OutOfScope):
        predict({"role": "non_it"}, model=loaded)


def test_importance_and_temporal_check(fitted_run: tuple[Fitted, pd.DataFrame]) -> None:
    fitted, frame = fitted_run
    dev = frame[frame["split"] == DEV]
    imp = importance(fitted.model, dev, cfg())
    assert set(imp) == set(BLOCKS)
    assert imp["role"]["mae_rise_rub"] > 0
    assert imp["employment_type"]["mae_rise_rub"] == 0
    temporal = temporal_check(frame, fitted, cfg())
    assert temporal["dev_rows_after_cutoff"] == int((dev["first_seen"] >= date(2026, 9, 1)).sum())
    assert temporal["refit_before_cutoff"]["coverage"]["denominator"] == temporal["dev_rows_after_cutoff"]


def test_typical_bounds_are_ordered(fitted_run: tuple[Fitted, pd.DataFrame]) -> None:
    fitted, frame = fitted_run
    out = fitted.model.predict_rows(records_of(frame))
    assert (out["from"] <= out["to"]).all()
    assert (out["low"] <= out["point"]).all() and (out["point"] <= out["high"]).all()


def test_student_fields_read_the_extraction_row() -> None:
    rec = {
        "role": "backend",
        "grade": None,
        "skills": "Go|SQL",
        "work_format": "remote",
        "cities": "",
        "experience_min": float("nan"),
        "experience_max": 3.0,
        "employment_type": None,
    }
    fields = student_fields(rec)
    assert fields["skills"] == ["Go", "SQL"]
    assert fields["cities"] == []
    assert fields["experience_years"] == {"min": None, "max": 3.0}
    assert student_fields(rec | {"experience_max": None})["experience_years"] is None
    assert feature_row(fields, "habr")["region_group"] == "remote"


def test_choose_follows_the_interval_then_the_tie_rule() -> None:
    assert choose({"value": -900.0, "ci_low": -1500.0, "ci_high": -100.0}, "rules")[0] == "student"
    assert choose({"value": 900.0, "ci_low": 100.0, "ci_high": 1500.0}, "student")[0] == "rules"
    assert choose({"value": -100.0, "ci_low": -600.0, "ci_high": 400.0}, "student")[0] == "student"
    assert choose({"value": -100.0, "ci_low": -600.0, "ci_high": 400.0}, "rules")[0] == "rules"


def test_linear_model_gets_its_textbook_shapley_values() -> None:
    rng = np.random.default_rng(0)
    w = np.array([1.0, -2.0, 0.5, 3.0])
    X, background = rng.normal(size=(20, 4)), rng.normal(size=(30, 4))
    phi, base = block_shap(lambda Z: Z @ w, X, background, [[0], [1, 2], [3]])
    centered = (X - background.mean(axis=0)) * w
    assert np.allclose(phi[:, 0], centered[:, 0])
    assert np.allclose(phi[:, 1], centered[:, 1] + centered[:, 2])
    assert np.allclose(phi.sum(axis=1) + base, X @ w)


def test_interaction_is_split_equally() -> None:
    X, background = np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]])
    phi, _ = block_shap(lambda Z: Z[:, 0] * Z[:, 1], X, background, [[0], [1]])
    assert np.allclose(phi, [[0.5, 0.5]])


def test_summary_orders_blocks_and_skips_rare_levels() -> None:
    phi = np.column_stack([np.r_[np.full(12, 0.2), np.full(3, -0.1)], np.zeros(15)])
    levels = {"grade": np.array(["senior"] * 12 + ["intern"] * 3)}
    out = summary(phi, ["grade", "role"], levels, {"grade": "grade"})
    assert list(out["blocks"]) == ["grade", "role"]
    assert list(out["per_level"]["grade"]) == ["senior"]
    assert out["per_level"]["grade"]["senior"]["pct"] == round(float(np.expm1(0.2)) * 100, 1)


def test_shap_of_the_fitted_model_adds_up(fitted_run: tuple[Fitted, pd.DataFrame]) -> None:
    fitted, frame = fitted_run
    out = shap_values(
        fitted.model, frame[frame["split"] == TRAIN], frame[frame["split"] == DEV], cfg(shap_background=20)
    )
    assert set(out["blocks"]) == set(BLOCKS)
    assert out["additivity_max_error_log"] < 1e-6
    assert out["blocks"]["employment_type"]["mean_abs_log"] == 0
    grade = out["per_level"]["grade"]
    assert grade["senior"]["mean_log"] > grade["junior"]["mean_log"]
