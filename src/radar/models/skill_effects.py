"""What moves the posted salary: double machine learning by skill.

For every frequent skill d the partially linear model log(salary) = theta * d + g(X) + e is fitted
with DoubleML (Chernozhukov et al. 2018): both the outcome and the skill indicator are predicted from
the controls by gradient boosting with cross-fitting, and theta is the slope of residual on residual.
Controls X: role bucket, grade, region, work format, employment type, source, minimum experience,
the other treated skills and the number of skills outside the treated list. Folds and standard errors
are clustered by the near-duplicate groups of the salary split, so one template posted in 70 regions
counts as one observation.

theta is a conditional association within vacancies that publish a salary: it equals the causal
effect of a skill on the posted salary only if no unobserved trait of the vacancy moves both (no
unobserved confounding), and it says nothing about vacancies without a salary. The result file
therefore carries, per skill, the naive gap and plain OLS next to it, multiple-testing corrections
over all skills, and the sensitivity bounds of DoubleML (how strong an unobserved confounder would
have to be to explain the estimate away).

    python -m radar.models.skill_effects             frame of radar.models.salary -> results/models/skill_effects.json
"""

import argparse
import logging
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from radar.config import ROOT
from radar.models.salary import frame_path
from radar.models.salary_data import FEATURE_SETS, with_features
from radar.results import write_result

log = logging.getLogger("radar.models.skill_effects")

CONFIG = ROOT / "configs" / "skill_effects.toml"
RESULT = "models/skill_effects.json"
CATEGORICAL = ("role_bucket", "grade", "region_group", "work_format", "employment_type", "source")

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    feature_set: str
    splits: tuple[str, ...]
    min_rows: int
    n_folds: int
    n_rep: int
    seed: int
    learner: dict[str, float]
    level: float
    cf_y: float
    cf_d: float


def load_config(path: Path = CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    d, m, i = raw["data"], raw["model"], raw["inference"]
    if d["feature_set"] not in FEATURE_SETS:
        raise ValueError(f"data.feature_set must be one of {FEATURE_SETS}")
    return Config(
        feature_set=d["feature_set"],
        splits=tuple(d["splits"]),
        min_rows=int(d["min_rows"]),
        n_folds=int(m["n_folds"]),
        n_rep=int(m["n_rep"]),
        seed=int(m["seed"]),
        learner={k: m[k] for k in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf")},
        level=float(i["level"]),
        cf_y=float(i["cf_y"]),
        cf_d=float(i["cf_d"]),
    )


@dataclass(frozen=True)
class Design:
    y: Array
    treatments: pd.DataFrame  # one 0/1 column per treated skill, columns named d0, d1, ...
    skills: list[str]  # skill name of each treatment column, same order
    controls: pd.DataFrame  # numeric controls for the learners (one-hot categories, experience, ...)
    groups: NDArray[np.str_]


def treated_skills(skill_lists: Sequence[Sequence[str]], min_rows: int) -> list[str]:
    """Skills present in at least `min_rows` rows and absent from at least `min_rows`, by frequency."""
    n = len(skill_lists)
    counts = Counter(s for skills in skill_lists for s in set(skills))
    keep = [s for s, c in counts.items() if c >= min_rows and n - c >= min_rows]
    return sorted(keep, key=lambda s: (-counts[s], s))


def design(frame: pd.DataFrame, min_rows: int) -> Design:
    skill_lists = [list(s) for s in frame["skills"]]
    skills = treated_skills(skill_lists, min_rows)
    index = {s: j for j, s in enumerate(skills)}
    D = np.zeros((len(frame), len(skills)))
    outside = np.zeros(len(frame))
    for i, row in enumerate(skill_lists):
        for s in set(row):
            if s in index:
                D[i, index[s]] = 1.0
            else:
                outside[i] += 1
    experience = pd.to_numeric(frame["experience_min"], errors="coerce").to_numpy(dtype=float)
    controls = pd.get_dummies(frame[list(CATEGORICAL)].astype(str), prefix=list(CATEGORICAL), dtype=float)
    controls["experience_known"] = np.isfinite(experience).astype(float)
    controls["experience_min"] = np.where(np.isfinite(experience), experience, -1.0)
    controls["skills_outside_treated"] = outside
    return Design(
        y=np.log(frame["salary_mid"].to_numpy(dtype=float)),
        treatments=pd.DataFrame(D, columns=[f"d{j}" for j in range(len(skills))], index=frame.index),
        skills=skills,
        controls=controls.reset_index(drop=True).set_index(frame.index),
        groups=frame["group_id"].astype(str).to_numpy(dtype=str),
    )


def holm(p: Array) -> Array:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adjusted
    return out


def benjamini_hochberg(p: Array) -> Array:
    order = np.argsort(p)
    m = len(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    out = np.empty(m)
    out[order] = adjusted
    return out


def naive_gaps(des: Design) -> Array:
    D = des.treatments.to_numpy()
    return np.array([des.y[D[:, j] == 1].mean() - des.y[D[:, j] == 0].mean() for j in range(D.shape[1])])


def ols(des: Design) -> tuple[Array, Array]:
    """OLS of log salary on all treatments and controls; coefficients and cluster-robust standard errors."""
    import statsmodels.api as sm  # type: ignore[import-untyped]

    # One level of every category is the reference, so the design has full rank.
    reference = [next(c for c in map(str, des.controls.columns) if c.startswith(f"{cat}_")) for cat in CATEGORICAL]
    controls = des.controls.drop(columns=reference)
    controls = controls.loc[:, controls.std() > 0]
    X = sm.add_constant(pd.concat([des.treatments, controls], axis=1).to_numpy(dtype=float), has_constant="add")
    codes = pd.factorize(pd.Series(des.groups))[0]
    fit = sm.OLS(des.y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    k = des.treatments.shape[1]
    return np.asarray(fit.params[1 : k + 1]), np.asarray(fit.bse[1 : k + 1])


def fit_dml(des: Design, cfg: Config) -> Any:
    import doubleml as dml  # type: ignore[import-untyped]

    df = pd.concat([des.treatments, des.controls], axis=1)
    df["y"] = des.y
    df["cluster"] = pd.factorize(pd.Series(des.groups))[0]
    data = dml.DoubleMLData(
        df,
        y_col="y",
        d_cols=list(des.treatments.columns),
        x_cols=list(des.controls.columns),
        cluster_cols="cluster",
        use_other_treat_as_covariate=True,
    )
    params = {
        "max_iter": int(cfg.learner["max_iter"]),
        "learning_rate": float(cfg.learner["learning_rate"]),
        "max_leaf_nodes": int(cfg.learner["max_leaf_nodes"]),
        "min_samples_leaf": int(cfg.learner["min_samples_leaf"]),
        "random_state": cfg.seed,
    }
    np.random.seed(cfg.seed)  # DoubleML draws its sample splits from the global generator
    model = dml.DoubleMLPLR(
        data,
        ml_l=HistGradientBoostingRegressor(**params),
        ml_m=HistGradientBoostingClassifier(**params),
        n_folds=cfg.n_folds,
        n_rep=cfg.n_rep,
    )
    model.fit(store_predictions=True)
    return model


def overlap(model: Any, des: Design) -> list[dict[str, float]]:
    """Per treatment: AUC of the skill predicted from the controls and the share of extreme propensities."""
    D = des.treatments.to_numpy()
    pred = model.predictions["ml_m"]  # (n, n_rep, n_treat)
    out = []
    for j in range(D.shape[1]):
        p = pred[:, :, j].mean(axis=1)
        out.append(
            {
                "auc": round(float(roc_auc_score(D[:, j], p)), 4),
                "extreme": int(((p < 0.02) | (p > 0.98)).sum()),
            }
        )
    return out


def effects(des: Design, cfg: Config) -> dict[str, Any]:
    model = fit_dml(des, cfg)
    theta, se = np.asarray(model.coef), np.asarray(model.se)
    p = np.asarray(model.pval)
    m = len(theta)
    z = NormalDist().inv_cdf(1 - (1 - cfg.level) / 2)
    z_bonf = NormalDist().inv_cdf(1 - (1 - cfg.level) / (2 * m))
    p_holm, q_bh = holm(p), benjamini_hochberg(p)
    naive = naive_gaps(des)
    ols_coef, ols_se = ols(des)
    model.sensitivity_analysis(cf_y=cfg.cf_y, cf_d=cfg.cf_d, level=cfg.level)
    sens = model.sensitivity_params
    losses = model.evaluate_learners()
    var_y = float(np.var(des.y))
    D = des.treatments.to_numpy()
    over = overlap(model, des)
    alpha = 1 - cfg.level

    def pct(x: float) -> float:
        return round(float(np.expm1(x)) * 100, 1)

    rows = {}
    for j, skill in enumerate(des.skills):
        rows[skill] = {
            "value": round(float(theta[j]), 4),
            "ci_low": round(float(theta[j] - z * se[j]), 4),
            "ci_high": round(float(theta[j] + z * se[j]), 4),
            "se": round(float(se[j]), 4),
            "pct": pct(theta[j]),
            "pct_ci_low": pct(theta[j] - z * se[j]),
            "pct_ci_high": pct(theta[j] + z * se[j]),
            "bonferroni_ci_low": round(float(theta[j] - z_bonf * se[j]), 4),
            "bonferroni_ci_high": round(float(theta[j] + z_bonf * se[j]), 4),
            "p": float(f"{p[j]:.3g}"),
            "p_holm": float(f"{p_holm[j]:.3g}"),
            "q_bh": float(f"{q_bh[j]:.3g}"),
            "rows_with_skill": int(D[:, j].sum()),
            "naive_gap": round(float(naive[j]), 4),
            "naive_pct": pct(naive[j]),
            "ols": round(float(ols_coef[j]), 4),
            "ols_pct": pct(ols_coef[j]),
            "ols_ci_low": round(float(ols_coef[j] - z * ols_se[j]), 4),
            "ols_ci_high": round(float(ols_coef[j] + z * ols_se[j]), 4),
            "rv_pct": round(float(sens["rv"][j]) * 100, 2),
            "rva_pct": round(float(sens["rva"][j]) * 100, 2),
            "sensitivity_theta_low": round(float(sens["theta"]["lower"][j]), 4),
            "sensitivity_theta_high": round(float(sens["theta"]["upper"][j]), 4),
            "sensitivity_ci_low": round(float(sens["ci"]["lower"][j]), 4),
            "sensitivity_ci_high": round(float(sens["ci"]["upper"][j]), 4),
            "outcome_r2": round(1 - float(losses["ml_l"][:, j].mean()) ** 2 / var_y, 4),
            "propensity_auc": over[j]["auc"],
            "extreme_propensity_rows": over[j]["extreme"],
        }
    significant_holm = int((p_holm < alpha).sum())
    significant_bh = int((q_bh < alpha).sum())
    robust = sum(1 for r in rows.values() if r["sensitivity_ci_low"] > 0 or r["sensitivity_ci_high"] < 0)
    return {
        "rows": rows,
        "summary": {
            "treatments": m,
            "significant_holm": {"value": significant_holm, "n": significant_holm, "denominator": m},
            "significant_bh": {"value": significant_bh, "n": significant_bh, "denominator": m},
            "robust_to_scenario": {"value": robust, "n": robust, "denominator": m},
        },
    }


DEFINITIONS = {
    "value": "theta of the partially linear model, log points of the posted monthly salary; pct = exp(theta) - 1.",
    "ci": "Pointwise interval at the configured level, cluster-robust over near-duplicate groups.",
    "bonferroni_ci": "Simultaneous interval over all treated skills (Bonferroni); the multiplier bootstrap "
    "of DoubleML is not implemented with clustered data.",
    "p_holm": "Holm adjustment over all treated skills (family-wise error).",
    "q_bh": "Benjamini-Hochberg adjustment over all treated skills (false discovery rate).",
    "naive_gap": "Mean log salary with the skill minus without it, no controls; naive_pct = exp(naive_gap) - 1.",
    "ols": "Coefficient of the skill in OLS of log salary on all treated skills and the same controls, "
    "cluster-robust interval; ols_pct = exp(ols) - 1.",
    "rv_pct": "Robustness value (DoubleML sensitivity analysis): the share of residual variance of both outcome "
    "and treatment an unobserved confounder would have to explain to move theta to zero; rva_pct for "
    "the interval to include zero.",
    "sensitivity": "Bounds of theta and of its interval under a confounder of strength cf_y, cf_d (config).",
    "outcome_r2": "Out-of-fold R^2 of the outcome learner (log salary from controls and other skills).",
    "propensity_auc": "Out-of-fold AUC of the skill indicator predicted from the controls; near 0.5 means "
    "little observed confounding, near 1 poor overlap.",
    "extreme_propensity_rows": "Rows with a predicted probability of the skill below 0.02 or above 0.98.",
}

LIMITS_RU = [
    "Оценка описывает вакансии с опубликованной зарплатой; компании, которые зарплату не пишут, могут "
    "платить за тот же навык иначе, и этот отбор моделью не исправляется.",
    "Причинное чтение требует, чтобы все общие причины навыка и вилки были среди контролей. Размер и "
    "отрасль работодателя, стек проекта, срочность найма в данных не наблюдаются; анализ "
    "чувствительности показывает, насколько сильным должен быть такой фактор, чтобы обнулить оценку.",
    "Навыки извлечены моделью из текста и содержат ошибки; ошибка в индикаторе навыка смещает оценку к нулю.",
    "Навыки одной вакансии связаны: оценка для навыка дана при прочих навыках из списка, а не в "
    "сравнении с вакансией без навыков вообще.",
    "Цель это середина опубликованной вилки в рублях, gross или net источники не указывают.",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.skill_effects", description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=Path, default=None, help="salary frame (parquet), default the saved one")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    path = args.frame or frame_path()
    frame = with_features(pd.read_parquet(path), cfg.feature_set)
    frame = frame[frame["split"].isin(cfg.splits)].reset_index(drop=True)
    des = design(frame, cfg.min_rows)
    log.info("%d rows, %d treated skills, %d controls", len(frame), len(des.skills), des.controls.shape[1])
    out = effects(des, cfg)
    metrics: dict[str, dict[str, Any]] = {
        "data_rows": {"value": len(frame)},
        "data_groups": {"value": int(frame["group_id"].nunique())},
        "data_controls": {"value": des.controls.shape[1]},
        "treatments": {"value": out["summary"]["treatments"]},
        "significant_holm": out["summary"]["significant_holm"],
        "significant_bh": out["summary"]["significant_bh"],
        "robust_to_scenario": out["summary"]["robust_to_scenario"],
    }
    # A marker in the docs cannot hold ":" (1С:Предприятие), so the key replaces it; `skill` keeps the name.
    metrics |= {f"effect.{skill.replace(':', '_')}": row | {"skill": skill} for skill, row in out["rows"].items()}
    written = write_result(
        RESULT,
        metrics,
        source_script="src/radar/models/skill_effects.py",
        definitions=DEFINITIONS,
        limits_ru=LIMITS_RU,
        feature_set=cfg.feature_set,
        splits=list(cfg.splits),
        frame=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        config=CONFIG.relative_to(ROOT).as_posix(),
        scenario={"cf_y": cfg.cf_y, "cf_d": cfg.cf_d, "level": cfg.level},
    )
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
