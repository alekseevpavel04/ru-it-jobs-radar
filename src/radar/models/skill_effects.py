"""What moves the posted salary: double machine learning by skill.

For every frequent skill d the partially linear model log(salary) = theta * d + g(X) + e is fitted
with DoubleML (Chernozhukov et al. 2018): both the outcome and the skill indicator are predicted from
the controls by gradient boosting with early stopping and cross-fitting, and theta is the slope of
residual on residual. Controls X: role bucket, grade, region, work format, employment type, source,
minimum experience, the other treated skills and the number of skills outside the treated list. Folds
and standard errors are clustered by the near-duplicate groups of the salary split, so one template
posted in 70 regions counts as one observation.

Overlap is checked before anything is read off: a first fit on all rows gives the out-of-fold
probability of every skill; rows outside the band of the config are trimmed per skill and theta is
estimated again on the rest, with the nuisances refitted there. A skill with too few kept rows on
either side is reported as not estimable and stays out of the multiple-testing family.

theta is a conditional association within vacancies that publish a salary: it equals the causal
effect of a skill on the posted salary only if no unobserved trait of the vacancy moves both (no
unobserved confounding), and it says nothing about vacancies without a salary. The result file
therefore carries, per skill, the naive gap, plain OLS and the untrimmed estimate next to it,
multiple-testing corrections over the estimated skills, and the sensitivity bounds of DoubleML (how
strong an unobserved confounder would have to be to explain the estimate away).

    python -m radar.models.skill_effects             frame of radar.models.salary -> results/models/skill_effects.json
    python -m radar.models.skill_effects --limit 3   probe: the first 3 skills, timings in the log, nothing written
"""

import argparse
import logging
import sys
import time
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
from sklearn.metrics import log_loss, roc_auc_score  # type: ignore[import-untyped]

from radar.config import ROOT
from radar.models.salary import frame_path
from radar.models.salary_data import FEATURE_SETS, with_features
from radar.results import write_result

log = logging.getLogger("radar.models.skill_effects")

CONFIG = ROOT / "configs" / "skill_effects.toml"
RESULT = "models/skill_effects.json"
CATEGORICAL = ("role_bucket", "grade", "region_group", "work_format", "employment_type", "source")
LEARNER_INT = ("max_iter", "max_leaf_nodes", "min_samples_leaf", "n_iter_no_change")
LEARNER_FLOAT = ("learning_rate", "l2_regularization", "validation_fraction")

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class Config:
    feature_set: str
    splits: tuple[str, ...]
    min_rows: int
    n_folds: int
    n_rep: int
    seed: int
    learner: dict[str, float]
    trim_low: float
    trim_high: float
    min_kept: int
    level: float
    cf_y: float
    cf_d: float


def load_config(path: Path = CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    d, m, o, i = raw["data"], raw["model"], raw["overlap"], raw["inference"]
    if d["feature_set"] not in FEATURE_SETS:
        raise ValueError(f"data.feature_set must be one of {FEATURE_SETS}")
    if not 0 <= o["low"] < o["high"] <= 1:
        raise ValueError("overlap needs 0 <= low < high <= 1")
    return Config(
        feature_set=d["feature_set"],
        splits=tuple(d["splits"]),
        min_rows=int(d["min_rows"]),
        n_folds=int(m["n_folds"]),
        n_rep=int(m["n_rep"]),
        seed=int(m["seed"]),
        learner={k: m[k] for k in LEARNER_INT + LEARNER_FLOAT},
        trim_low=float(o["low"]),
        trim_high=float(o["high"]),
        min_kept=int(o["min_kept"]),
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

    # One level of every category is the reference, so the design has full rank; region "remote" and
    # work format "remote" are the same column in this data, so exact duplicates go too.
    reference = [next(c for c in map(str, des.controls.columns) if c.startswith(f"{cat}_")) for cat in CATEGORICAL]
    controls = des.controls.drop(columns=reference)
    controls = controls.loc[:, controls.std() > 0]
    controls = controls.loc[:, ~controls.T.duplicated().to_numpy()]
    X = sm.add_constant(pd.concat([des.treatments, controls], axis=1).to_numpy(dtype=float), has_constant="add")
    codes = pd.factorize(pd.Series(des.groups))[0]
    fit = sm.OLS(des.y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    k = des.treatments.shape[1]
    return np.asarray(fit.params[1 : k + 1]), np.asarray(fit.bse[1 : k + 1])


def learner_params(cfg: Config) -> dict[str, Any]:
    params: dict[str, Any] = {k: int(v) for k, v in cfg.learner.items() if k in LEARNER_INT}
    params |= {k: float(v) for k, v in cfg.learner.items() if k in LEARNER_FLOAT}
    # The inner validation split is random, not by near-duplicate group: with 6.4k groups over 7.4k rows
    # a few duplicates straddle it and stopping comes slightly late, which only weakens the regularization.
    return params | {"early_stopping": True, "random_state": cfg.seed}


def fit_plr(des: Design, cfg: Config, rows: Mask, d_cols: list[str], x_cols: list[str]) -> Any:
    import doubleml as dml  # type: ignore[import-untyped]

    df = pd.concat([des.treatments, des.controls], axis=1).loc[rows]
    df["y"] = des.y[rows]
    df["cluster"] = pd.factorize(pd.Series(des.groups[rows]))[0]
    data = dml.DoubleMLData(
        df.reset_index(drop=True),
        y_col="y",
        d_cols=d_cols,
        x_cols=x_cols,
        cluster_cols="cluster",
        use_other_treat_as_covariate=True,
    )
    params = learner_params(cfg)
    np.random.seed(cfg.seed)  # DoubleML draws its sample splits from the global generator
    model = dml.DoubleMLPLR(
        data,
        ml_l=HistGradientBoostingRegressor(**params),
        ml_m=HistGradientBoostingClassifier(**params),
        n_folds=cfg.n_folds,
        n_rep=cfg.n_rep,
    )
    model.fit(store_predictions=True, store_models=True)
    return model


def rounds(model: Any, learner: str) -> list[int]:
    """Boosting rounds early stopping kept, per fitted fold model."""
    return [int(m.n_iter_) for per_d in model.models[learner].values() for rep in per_d for m in rep]


def overlap_stats(d: Array, p: Array, low: float, high: float) -> dict[str, Any]:
    band = (p >= low) & (p <= high)
    rate = float(d.mean())
    return {
        "band": band,
        "auc": round(float(roc_auc_score(d, p)), 4),
        # Out-of-fold log loss over that of the constant skill rate: below 1 the controls predict the
        # skill, far above 1 the probabilities are overconfident.
        "logloss_ratio": round(float(log_loss(d, np.clip(p, 1e-6, 1 - 1e-6)) / log_loss(d, np.full_like(p, rate))), 4),
    }


def _r2(model: Any, y: Array) -> float:
    rmse = float(np.asarray(model.evaluate_learners()["ml_l"]).mean())
    return round(1 - rmse**2 / float(np.var(y)), 4)


def effects(des: Design, cfg: Config, limit: int | None = None) -> dict[str, Any]:
    """Untrimmed fit of the first `limit` skills (all by default; the rest stay controls), then one
    trimmed fit per skill with enough overlap."""
    n = len(des.y)
    cols = list(des.treatments.columns)
    estimated_cols = cols[:limit] if limit else cols
    controls = list(des.controls.columns)
    started = time.perf_counter()
    full = fit_plr(des, cfg, np.ones(n, dtype=bool), estimated_cols, controls + cols[len(estimated_cols) :])
    log.info("untrimmed fit of %d skills: %.0fs", len(estimated_cols), time.perf_counter() - started)
    propensity = np.asarray(full.predictions["ml_m"]).mean(axis=1)  # (n, n_treat)
    full_theta, full_se = np.asarray(full.coef), np.asarray(full.se)
    D = des.treatments.to_numpy()
    z = NormalDist().inv_cdf(1 - (1 - cfg.level) / 2)
    naive = naive_gaps(des)
    ols_coef, ols_se = ols(des)

    def pct(x: float) -> float:
        return round(float(np.expm1(x)) * 100, 1)

    overlap: dict[str, dict[str, Any]] = {}
    trimmed: dict[str, dict[str, Any]] = {}
    for j, col in enumerate(estimated_cols):
        skill = des.skills[j]
        stats = overlap_stats(D[:, j], propensity[:, j], cfg.trim_low, cfg.trim_high)
        band = stats.pop("band")
        kept_with = int(D[band, j].sum())
        kept_without = int(band.sum()) - kept_with
        estimable = kept_with >= cfg.min_kept and kept_without >= cfg.min_kept
        overlap[skill] = stats | {
            "value": round(float(band.mean()), 4),
            "n": int(band.sum()),
            "denominator": n,
            "rows_with_skill": int(D[:, j].sum()),
            "kept_with_skill": kept_with,
            "kept_without_skill": kept_without,
            "estimated": estimable,
        }
        if not estimable:
            log.info("%s: %d of %d rows in the band, %d with the skill: not estimable", skill, band.sum(), n, kept_with)
            continue
        t0 = time.perf_counter()
        model = fit_plr(des, cfg, band, [col], controls + [c for c in cols if c != col])
        model.sensitivity_analysis(cf_y=cfg.cf_y, cf_d=cfg.cf_d, level=cfg.level)
        sens = model.sensitivity_params
        theta, se = float(model.coef[0]), float(model.se[0])
        trimmed[skill] = {
            "value": round(theta, 4),
            "se": round(se, 4),
            "p_raw": float(model.pval[0]),
            "ci_low": round(theta - z * se, 4),
            "ci_high": round(theta + z * se, 4),
            "pct": pct(theta),
            "pct_ci_low": pct(theta - z * se),
            "pct_ci_high": pct(theta + z * se),
            "rows_kept": int(band.sum()),
            "rows_kept_with_skill": kept_with,
            "rows_with_skill": int(D[:, j].sum()),
            "naive_gap": round(float(naive[j]), 4),
            "naive_pct": pct(naive[j]),
            "ols": round(float(ols_coef[j]), 4),
            "ols_pct": pct(ols_coef[j]),
            "ols_ci_low": round(float(ols_coef[j] - z * ols_se[j]), 4),
            "ols_ci_high": round(float(ols_coef[j] + z * ols_se[j]), 4),
            "untrimmed": round(float(full_theta[j]), 4),
            "untrimmed_pct": pct(full_theta[j]),
            "untrimmed_ci_low": round(float(full_theta[j] - z * full_se[j]), 4),
            "untrimmed_ci_high": round(float(full_theta[j] + z * full_se[j]), 4),
            "rv_pct": round(float(sens["rv"][0]) * 100, 2),
            "rva_pct": round(float(sens["rva"][0]) * 100, 2),
            "sensitivity_theta_low": round(float(sens["theta"]["lower"][0]), 4),
            "sensitivity_theta_high": round(float(sens["theta"]["upper"][0]), 4),
            "sensitivity_ci_low": round(float(sens["ci"]["lower"][0]), 4),
            "sensitivity_ci_high": round(float(sens["ci"]["upper"][0]), 4),
            "outcome_r2": _r2(model, des.y[band]),
            "propensity_auc": stats["auc"],
            "rounds_l_median": float(np.median(rounds(model, "ml_l"))),
            "rounds_m_median": float(np.median(rounds(model, "ml_m"))),
        }
        log.info(
            "%s: %d rows kept, %d with the skill, theta %.4f [%.4f; %.4f], %.0fs",
            skill,
            band.sum(),
            kept_with,
            theta,
            theta - z * se,
            theta + z * se,
            time.perf_counter() - t0,
        )

    m = len(trimmed)
    alpha = 1 - cfg.level
    if m:
        p = np.array([r.pop("p_raw") for r in trimmed.values()])
        z_bonf = NormalDist().inv_cdf(1 - alpha / (2 * m))
        for r, pj, hj, qj in zip(trimmed.values(), p, holm(p), benjamini_hochberg(p), strict=True):
            theta, se = r["value"], r["se"]
            r |= {
                "p": float(f"{pj:.3g}"),
                "p_holm": float(f"{hj:.3g}"),
                "q_bh": float(f"{qj:.3g}"),
                "bonferroni_ci_low": round(theta - z_bonf * se, 4),
                "bonferroni_ci_high": round(theta + z_bonf * se, 4),
            }
    significant_holm = sum(1 for r in trimmed.values() if r["p_holm"] < alpha)
    significant_bh = sum(1 for r in trimmed.values() if r["q_bh"] < alpha)
    robust = sum(1 for r in trimmed.values() if r["sensitivity_ci_low"] > 0 or r["sensitivity_ci_high"] < 0)
    k = len(estimated_cols)
    shares = [o["value"] for o in overlap.values()]
    return {
        "rows": trimmed,
        "overlap": overlap,
        "summary": {
            "treatments": k,
            "estimated": {"value": m, "n": m, "denominator": k},
            "significant_holm": {"value": significant_holm, "n": significant_holm, "denominator": m},
            "significant_bh": {"value": significant_bh, "n": significant_bh, "denominator": m},
            "robust_to_scenario": {"value": robust, "n": robust, "denominator": m},
            "overlap_share_median": round(float(np.median(shares)), 4),
            "overlap_share_min": round(float(np.min(shares)), 4),
            "untrimmed_auc_min": round(float(min(o["auc"] for o in overlap.values())), 4),
            "untrimmed_auc_max": round(float(max(o["auc"] for o in overlap.values())), 4),
            "untrimmed_rounds_l_median": float(np.median(rounds(full, "ml_l"))),
            "untrimmed_rounds_m_median": float(np.median(rounds(full, "ml_m"))),
        },
    }


DEFINITIONS = {
    "value": "theta of the partially linear model on the rows of the skill's overlap band, log points of the "
    "posted monthly salary; pct = exp(theta) - 1.",
    "overlap": "Share of rows whose out-of-fold probability of the skill (untrimmed fit) lies in the band of the "
    "config; n rows of denominator; estimated is false when the kept rows hold fewer than min_kept vacancies "
    "with or without the skill.",
    "ci": "Pointwise interval at the configured level, cluster-robust over near-duplicate groups.",
    "bonferroni_ci": "Simultaneous interval over the estimated skills (Bonferroni); the multiplier bootstrap "
    "of DoubleML is not implemented with clustered data.",
    "p_holm": "Holm adjustment over the estimated skills (family-wise error).",
    "q_bh": "Benjamini-Hochberg adjustment over the estimated skills (false discovery rate).",
    "naive_gap": "Mean log salary with the skill minus without it on all rows, no controls; "
    "naive_pct = exp(naive_gap) - 1.",
    "ols": "Coefficient of the skill in OLS of log salary on all treated skills and the same controls, all rows, "
    "cluster-robust interval; ols_pct = exp(ols) - 1.",
    "untrimmed": "theta of the same model on all rows, before trimming.",
    "rv_pct": "Robustness value (DoubleML sensitivity analysis): the share of residual variance of both outcome "
    "and treatment an unobserved confounder would have to explain to move theta to zero; rva_pct for "
    "the interval to include zero.",
    "sensitivity": "Bounds of theta and of its interval under a confounder of strength cf_y, cf_d (config).",
    "outcome_r2": "Out-of-fold R^2 of the outcome learner (log salary from controls and other skills), kept rows.",
    "auc": "Out-of-fold AUC of the skill indicator predicted from the controls, untrimmed fit.",
    "logloss_ratio": "Out-of-fold log loss of the skill probability over that of the constant skill rate.",
    "rounds": "Median boosting rounds kept by early stopping over the fold models.",
}

LIMITS_RU = [
    "Оценка описывает вакансии с опубликованной зарплатой; компании, которые зарплату не пишут, могут "
    "платить за тот же навык иначе, и этот отбор моделью не исправляется.",
    "После обрезки по перекрытию оценка относится только к вакансиям, где навык правдоподобен при их "
    "контролях, а не ко всем вакансиям выборки.",
    "Причинное чтение требует, чтобы все общие причины навыка и вилки были среди контролей. Размер и "
    "отрасль работодателя, стек проекта, срочность найма в данных не наблюдаются; анализ "
    "чувствительности показывает, насколько сильным должен быть такой фактор, чтобы обнулить оценку.",
    "Навыки извлечены моделью из текста и содержат ошибки; ошибка в индикаторе навыка смещает оценку к нулю.",
    "Навыки одной вакансии связаны: оценка для навыка дана при прочих навыках из списка, а не в "
    "сравнении с вакансией без навыков вообще.",
    "Цель это середина опубликованной вилки в рублях, gross или net источники не указывают.",
]


def key(skill: str) -> str:
    # A marker in the docs cannot hold ":" (1С:Предприятие), so the key replaces it; `skill` keeps the name.
    return skill.replace(":", "_")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.skill_effects", description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=Path, default=None, help="salary frame (parquet), default the saved one")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--limit", type=int, help="probe: estimate only the first skills, write nothing")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    path = args.frame or frame_path()
    frame = with_features(pd.read_parquet(path), cfg.feature_set)
    frame = frame[frame["split"].isin(cfg.splits)].reset_index(drop=True)
    des = design(frame, cfg.min_rows)
    log.info("%d rows, %d treated skills, %d controls", len(frame), len(des.skills), des.controls.shape[1])
    started = time.perf_counter()
    out = effects(des, cfg, args.limit)
    log.info("%d of %d skills estimated in %.0fs", len(out["rows"]), len(des.skills), time.perf_counter() - started)
    if args.limit:
        return 0
    summary = out["summary"]
    metrics: dict[str, dict[str, Any]] = {
        "data_rows": {"value": len(frame)},
        "data_groups": {"value": int(frame["group_id"].nunique())},
        "data_controls": {"value": des.controls.shape[1]},
        "treatments": {"value": summary.pop("treatments")},
    }
    metrics |= {name: v if isinstance(v, dict) else {"value": v} for name, v in summary.items()}
    metrics |= {f"overlap.{key(skill)}": row | {"skill": skill} for skill, row in out["overlap"].items()}
    metrics |= {f"effect.{key(skill)}": row | {"skill": skill} for skill, row in out["rows"].items()}
    written = write_result(
        RESULT,
        metrics,
        source_script="src/radar/models/skill_effects.py",
        definitions=DEFINITIONS,
        limits_ru=LIMITS_RU,
        feature_set=cfg.feature_set,
        splits=list(cfg.splits),
        frame=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        config=args.config.relative_to(ROOT).as_posix() if args.config.is_relative_to(ROOT) else str(args.config),
        learner=learner_params(cfg),
        overlap_rule={"low": cfg.trim_low, "high": cfg.trim_high, "min_kept": cfg.min_kept},
        scenario={"cf_y": cfg.cf_y, "cf_d": cfg.cf_d, "level": cfg.level},
    )
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
