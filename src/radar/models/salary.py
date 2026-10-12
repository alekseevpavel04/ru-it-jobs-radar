"""Salary estimate for an IT vacancy: point and interval in RUB per month (v0 features).

Data, target and the leak-free split: `radar.models.salary_data`. Features, defined on the extraction
schema so that the student's fields (6.7) replace the v0 rule fields without touching the models:
`radar.models.salary_features`.

Models, all on the log of the monthly midpoint and fitted on the `train` part only:
- global median and a cell median (role bucket x grade x region, backing off to coarser cells below
  `min_cell` vacancies) with the cell's empirical quantiles as its interval;
- ridge on one-hot features, interval by split conformal on absolute residuals;
- gradient boosting (sklearn HistGradientBoosting): a median model for the point and two quantile
  models, the interval conformalized on `calib` (CQR) to the nominal level.
Hyperparameters come from grouped cross-validation inside `train`; dev is scored once per run and
nothing is chosen on it. Confidence intervals of the metrics are percentile bootstraps over dev
groups (a group of near-duplicates is drawn as a whole), paired between models.

    python -m radar.models.salary                    query, save data/salary/frame_v0.parquet, fit, score dev,
                                                     write results and models/salary/
    python -m radar.models.salary --frame PATH       same on a saved frame instead of ClickHouse
    python -m radar.models.salary --final-test       saved models on test of the saved frame (once)
"""

import argparse
import itertools
import json
import logging
import math
import pickle
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

from radar.config import RESULTS, ROOT, data_dir, load_dotenv
from radar.models.salary_data import (
    CALIB,
    DEV,
    TEST,
    TRAIN,
    SplitShares,
    build_frame,
    ids_sha256,
    load_clickhouse,
    load_corpus_groups,
    records_of,
)
from radar.models.salary_features import BLOCKS, Encoder, feature_row
from radar.results import git_commit, invocation, write_result

log = logging.getLogger("radar.models.salary")

Array = NDArray[np.float64]

CONFIG = ROOT / "configs" / "salary.toml"
EVAL_CONFIG = ROOT / "configs" / "eval.toml"
MODEL_DIR = ROOT / "models" / "salary"
ARTIFACT = "salary.pkl"
# Everything fitted in the run, baselines included, so the test is scored on exactly these models.
EVAL_ARTIFACT = "salary_eval.pkl"
FEATURE_VERSION = "v0"
RESULT = "models/salary.json"
RESULT_TEST = "models/salary_test.json"
CELL_LEVELS: tuple[tuple[str, ...], ...] = (
    ("role_bucket", "grade", "region_group"),
    ("role_bucket", "region_group"),
    ("role_bucket",),
    ("role_group",),
    (),
)
ROUND_RUB = 1000


@dataclass(frozen=True)
class Config:
    min_monthly: float
    shares: SplitShares
    level: float
    cutoff: date
    min_skill_count: int
    min_cell: int
    cv_folds: int
    seed: int
    max_iter: int
    grid: dict[str, list[float]]
    ridge_alpha: list[float]
    importance_repeats: int
    n_resamples: int
    bootstrap_seed: int
    confidence: float


def load_config(path: Path = CONFIG, eval_path: Path = EVAL_CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    boot = tomllib.loads(eval_path.read_text(encoding="utf-8"))["bootstrap"]
    s, m = raw["split"], raw["model"]
    return Config(
        min_monthly=float(raw["data"]["min_monthly_rub"]),
        shares=SplitShares(s["salt"], s["dev_share"], s["test_share"], s["calib_share"]),
        level=float(raw["interval"]["level"]),
        cutoff=date.fromisoformat(raw["temporal"]["cutoff"]),
        min_skill_count=int(raw["features"]["min_skill_count"]),
        min_cell=int(raw["features"]["min_cell"]),
        cv_folds=int(m["cv_folds"]),
        seed=int(m["seed"]),
        max_iter=int(m["max_iter"]),
        grid={k: list(m[k]) for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")},
        ridge_alpha=list(m["ridge_alpha"]),
        importance_repeats=int(raw["importance"]["repeats"]),
        n_resamples=int(boot["n_resamples"]),
        bootstrap_seed=int(boot["seed"]),
        confidence=float(boot["confidence"]),
    )


def conformal_quantile(scores: Array, level: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n + 1) * level)-th smallest score."""
    n = len(scores)
    k = math.ceil((n + 1) * level)
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


@dataclass
class CellMedian:
    """Median and empirical quantiles of log salary per cell, backing off level by level."""

    level: float
    min_cell: int
    tables: list[dict[tuple[str, ...], tuple[int, float, float, float]]] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame, y: Array) -> "CellMedian":
        lo_q, hi_q = (1 - self.level) / 2, (1 + self.level) / 2
        self.tables = []
        for keys in CELL_LEVELS:
            table: dict[tuple[str, ...], tuple[int, float, float, float]] = {}
            labels = [tuple(r) for r in frame[list(keys)].astype(str).to_numpy()] if keys else [()] * len(frame)
            by: dict[tuple[str, ...], list[float]] = {}
            for label, v in zip(labels, y, strict=True):
                by.setdefault(label, []).append(float(v))
            for label, values in by.items():
                arr = np.array(values)
                table[label] = (len(arr), float(np.median(arr)), *np.quantile(arr, [lo_q, hi_q]).tolist())
            self.tables.append(table)
        return self

    def _lookup(self, row: Mapping[str, Any]) -> tuple[int, float, float, float]:
        for keys, table in zip(CELL_LEVELS, self.tables, strict=True):
            hit = table.get(tuple(str(row[k]) for k in keys))
            if hit and (hit[0] >= self.min_cell or not keys):
                return hit
        raise ValueError("the global cell is always present after fit")

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> tuple[Array, Array, Array]:
        hits = np.array([self._lookup(r)[1:] for r in rows])
        return hits[:, 0], hits[:, 1], hits[:, 2]

    def n_similar(self, row: Mapping[str, Any]) -> int:
        """Training vacancies with the same role bucket, grade and region group."""
        keys = CELL_LEVELS[0]
        hit = self.tables[0].get(tuple(str(row[k]) for k in keys))
        return hit[0] if hit else 0


def hgb(params: Mapping[str, float], cfg: Config, n_categorical: int, n_features: int, **loss: Any) -> Any:
    mask = [i < n_categorical for i in range(n_features)]
    return HistGradientBoostingRegressor(
        learning_rate=params["learning_rate"],
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=params["l2_regularization"],
        max_iter=cfg.max_iter,
        categorical_features=mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=cfg.seed,
        **loss,
    )


def grid_points(grid: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    names = sorted(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(grid[n] for n in names))]


def cv_select_gbm(
    X: Array, y: Array, groups: NDArray[np.str_], cfg: Config, n_cat: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Grid point with the lowest grouped-CV mean absolute error in log space, inside train only."""
    folds = list(GroupKFold(n_splits=cfg.cv_folds).split(X, y, groups))
    table = []
    for params in grid_points(cfg.grid):
        errors = []
        for fit_idx, val_idx in folds:
            model = hgb(params, cfg, n_cat, X.shape[1], loss="absolute_error").fit(X[fit_idx], y[fit_idx])
            errors.append(float(np.mean(np.abs(model.predict(X[val_idx]) - y[val_idx]))))
        table.append({**params, "cv_mae_log": round(float(np.mean(errors)), 5)})
        log.info("gbm %s cv mae(log) %.4f", params, np.mean(errors))
    best = min(table, key=lambda r: r["cv_mae_log"])
    return {k: v for k, v in best.items() if k != "cv_mae_log"}, table


def cv_select_ridge(X: Array, y: Array, groups: NDArray[np.str_], cfg: Config) -> tuple[float, list[dict[str, Any]]]:
    folds = list(GroupKFold(n_splits=cfg.cv_folds).split(X, y, groups))
    table = []
    for alpha in cfg.ridge_alpha:
        errors = [float(np.mean(np.abs(Ridge(alpha=alpha).fit(X[f], y[f]).predict(X[v]) - y[v]))) for f, v in folds]
        table.append({"alpha": alpha, "cv_mae_log": round(float(np.mean(errors)), 5)})
    best = min(table, key=lambda r: r["cv_mae_log"])
    return float(best["alpha"]), table


@dataclass
class SalaryModel:
    """What /salary loads: the encoder, the boosting models with their conformal margin, the cells."""

    encoder: Encoder
    point: Any
    lower: Any
    upper: Any
    bound_from: Any
    bound_to: Any
    margin: float
    level: float
    cells: CellMedian
    version: str
    trained_at: str

    def predict_rows(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Array]:
        X = self.encoder.tree_matrix(self.encoder.encode(rows))
        point = self.point.predict(X)
        low = np.minimum(self.lower.predict(X) - self.margin, point)
        high = np.maximum(self.upper.predict(X) + self.margin, point)
        # The bounds are two separate models fitted on different rows; they are ordered, not reconciled.
        bound_from, bound_to = self.bound_from.predict(X), self.bound_to.predict(X)
        return {
            "point": np.exp(point),
            "low": np.exp(low),
            "high": np.exp(high),
            "from": np.exp(np.minimum(bound_from, bound_to)),
            "to": np.exp(np.maximum(bound_from, bound_to)),
        }


def _round(value: float) -> int:
    return int(round(value / ROUND_RUB) * ROUND_RUB)


@lru_cache(maxsize=4)
def load_model(path: Path = MODEL_DIR / ARTIFACT) -> SalaryModel:
    with path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, SalaryModel):
        raise TypeError(f"{path} does not hold a SalaryModel")
    return model


def predict(fields: Mapping[str, Any], source: str = "trudvsem", model: SalaryModel | None = None) -> dict[str, Any]:
    """Monthly RUB estimate for extraction fields as the given source's market pays.

    `point` is the median estimate, `low` and `high` the conformal interval at the model's level,
    `typical_from` and `typical_to` the estimated stated bounds, `n_similar` the training vacancies
    with the same role bucket, grade and region group. A stated salary in `fields` is not read.
    Raises `OutOfScope` for a non-IT role.
    """
    model = model or load_model()
    row = feature_row(fields, source)
    out = model.predict_rows([row])
    return {
        "point": _round(float(out["point"][0])),
        "low": _round(float(out["low"][0])),
        "high": _round(float(out["high"][0])),
        "typical_from": _round(float(out["from"][0])),
        "typical_to": _round(float(out["to"][0])),
        "level": model.level,
        "n_similar": model.cells.n_similar(row),
        "source": source,
        "model_version": model.version,
    }


def group_weights(groups: Sequence[str], n_resamples: int, seed: int) -> Array:
    """(n_resamples, n_rows): how often each row is drawn when whole groups are resampled."""
    codes, inverse = np.unique(np.asarray(groups), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(codes), np.full(len(codes), 1.0 / len(codes)), size=n_resamples).astype(np.float64)
    return draws[:, inverse]


def _ci(point: float, samples: Array, confidence: float) -> dict[str, float]:
    ok = samples[~np.isnan(samples)]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(ok, [tail, 100 - tail]) if ok.size else (np.nan, np.nan)
    return {"value": round(point, 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4)}


def _wmean(W: Array, x: Array) -> Array:
    total = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (W @ x) / total, np.nan)


def point_errors(y: Array, pred: Array) -> dict[str, Array]:
    rel = np.abs(pred - y) / y
    return {"mae_rub": np.abs(pred - y), "mape": rel, "within_20pct": (rel <= 0.2).astype(np.float64)}


def interval_errors(y: Array, low: Array, high: Array, point: Array) -> dict[str, Array]:
    return {
        "coverage": ((y >= low) & (y <= high)).astype(np.float64),
        "width_rub": high - low,
        "width_rel": (high - low) / point,
    }


def score(
    columns: Mapping[str, Array], W: Array, confidence: float, mask: NDArray[np.bool_] | None = None
) -> dict[str, dict[str, Any]]:
    """Mean of each per-row column with its group-bootstrap interval; shares carry n and denominator."""
    keep = np.ones(W.shape[1], dtype=bool) if mask is None else mask
    Wk = W[:, keep]
    out: dict[str, dict[str, Any]] = {}
    for name, x in columns.items():
        xk = x[keep]
        block = _ci(float(np.mean(xk)), _wmean(Wk, xk), confidence)
        if name in ("coverage", "within_20pct"):
            block |= {"n": int(xk.sum()), "denominator": len(xk)}
        out[name] = block
    return out


def paired(a: Array, b: Array, W: Array, confidence: float) -> dict[str, float]:
    """Difference of means a - b on the same resampled groups."""
    return _ci(float(np.mean(a) - np.mean(b)), _wmean(W, a) - _wmean(W, b), confidence)


@dataclass
class Fitted:
    model: SalaryModel
    ridge: Any
    ridge_margin: float
    fill: Array
    global_median: float
    gbm_params: dict[str, float]
    gbm_cv: list[dict[str, Any]]
    ridge_alpha: float
    ridge_cv: list[dict[str, Any]]


def fit_all(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    cfg: Config,
    params: dict[str, float] | None = None,
    alpha: float | None = None,
) -> Fitted:
    """Every model on `train`, intervals calibrated on `calib`; given params and alpha skip the search."""
    rows_t, rows_c = records_of(train), records_of(calib)
    y_t, y_c = np.log(train["salary_mid"].to_numpy()), np.log(calib["salary_mid"].to_numpy())
    groups = train["group_id"].to_numpy().astype(str)
    enc = Encoder.fit(rows_t, cfg.min_skill_count)
    et, ec = enc.encode(rows_t), enc.encode(rows_c)
    Xt, Xc = enc.tree_matrix(et), enc.tree_matrix(ec)
    n_cat = len(enc.levels)

    gbm_cv: list[dict[str, Any]] = []
    if params is None:
        params, gbm_cv = cv_select_gbm(Xt, y_t, groups, cfg, n_cat)
    lo_q, hi_q = (1 - cfg.level) / 2, (1 + cfg.level) / 2
    point = hgb(params, cfg, n_cat, Xt.shape[1], loss="absolute_error").fit(Xt, y_t)
    lower = hgb(params, cfg, n_cat, Xt.shape[1], loss="quantile", quantile=lo_q).fit(Xt, y_t)
    upper = hgb(params, cfg, n_cat, Xt.shape[1], loss="quantile", quantile=hi_q).fit(Xt, y_t)
    scores = np.maximum(lower.predict(Xc) - y_c, y_c - upper.predict(Xc))
    margin = conformal_quantile(scores, cfg.level)

    bounds = []
    for column in ("salary_from", "salary_to"):
        has = train[column].notna().to_numpy() & (train[column].to_numpy(dtype=float) > 0)
        bounds.append(
            hgb(params, cfg, n_cat, Xt.shape[1], loss="absolute_error").fit(
                Xt[has], np.log(train[column].to_numpy(dtype=float)[has])
            )
        )

    fill = np.nanmedian(et.numeric, axis=0)
    Lt, Lc = enc.linear_matrix(et, fill), enc.linear_matrix(ec, fill)
    ridge_cv: list[dict[str, Any]] = []
    if alpha is None:
        alpha, ridge_cv = cv_select_ridge(Lt, y_t, groups, cfg)
    ridge = Ridge(alpha=alpha).fit(Lt, y_t)
    ridge_margin = conformal_quantile(np.abs(ridge.predict(Lc) - y_c), cfg.level)

    cells = CellMedian(cfg.level, cfg.min_cell).fit(train, y_t)
    model = SalaryModel(
        encoder=enc,
        point=point,
        lower=lower,
        upper=upper,
        bound_from=bounds[0],
        bound_to=bounds[1],
        margin=margin,
        level=cfg.level,
        cells=cells,
        version=f"salary-{FEATURE_VERSION}-{git_commit()}",
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return Fitted(model, ridge, ridge_margin, fill, float(np.median(y_t)), params, gbm_cv, alpha, ridge_cv)


def predictions(fitted: Fitted, frame: pd.DataFrame) -> dict[str, dict[str, Array]]:
    """Per model: point, and low and high when the model has an interval; RUB per month."""
    rows = records_of(frame)
    m = fitted.model
    enc = m.encoder
    e = enc.encode(rows)
    X = enc.tree_matrix(e)
    gbm_point = m.point.predict(X)
    q_lo, q_hi = m.lower.predict(X), m.upper.predict(X)
    cqr = m.predict_rows(rows)
    r_point = fitted.ridge.predict(enc.linear_matrix(e, fitted.fill))
    c_point, c_lo, c_hi = m.cells.predict(rows)
    n = len(rows)
    return {
        "global_median": {"point": np.full(n, math.exp(fitted.global_median))},
        "cell_median": {"point": np.exp(c_point), "low": np.exp(c_lo), "high": np.exp(c_hi)},
        "ridge": {
            "point": np.exp(r_point),
            "low": np.exp(r_point - fitted.ridge_margin),
            "high": np.exp(r_point + fitted.ridge_margin),
        },
        "gbm_quantile": {
            "point": np.exp(gbm_point),
            "low": np.exp(np.minimum(q_lo, gbm_point)),
            "high": np.exp(np.maximum(q_hi, gbm_point)),
        },
        "gbm_cqr": {"point": cqr["point"], "low": cqr["low"], "high": cqr["high"]},
        "gbm_bounds": {"from": cqr["from"], "to": cqr["to"]},
    }


def evaluate(fitted: Fitted, frame: pd.DataFrame, cfg: Config, split: str) -> dict[str, Any]:
    y = frame["salary_mid"].to_numpy(dtype=float)
    W = group_weights(frame["group_id"].astype(str).tolist(), cfg.n_resamples, cfg.bootstrap_seed)
    preds = predictions(fitted, frame)
    metrics: dict[str, dict[str, Any]] = {}
    per_row: dict[str, dict[str, Array]] = {}
    for name, p in preds.items():
        if name == "gbm_bounds":
            continue
        cols = point_errors(y, p["point"])
        if "low" in p:
            cols |= interval_errors(y, p["low"], p["high"], p["point"])
        per_row[name] = cols
        for metric, block in score(cols, W, cfg.confidence).items():
            metrics[f"{split}_{name}_{metric}"] = block
    for bound in ("from", "to"):
        column = frame[f"salary_{bound}"].to_numpy(dtype=float)
        has = np.isfinite(column) & (column > 0)
        err = np.abs(preds["gbm_bounds"][bound] - column)
        block = score({"mae_rub": np.where(has, err, 0.0)}, W, cfg.confidence, mask=has)["mae_rub"]
        metrics[f"{split}_gbm_{bound}_mae_rub"] = block | {"rows": int(has.sum())}

    diffs = {}
    for a, b in (
        ("gbm_cqr", "ridge"),
        ("gbm_cqr", "cell_median"),
        ("ridge", "cell_median"),
        ("cell_median", "global_median"),
    ):
        for metric in ("mae_rub", "mape"):
            key = f"{split}_diff_{a}_minus_{b}_{metric}"
            diffs[key] = paired(per_row[a][metric], per_row[b][metric], W, cfg.confidence)
    metrics |= diffs

    breakdown: dict[str, dict[str, Any]] = {}
    for dim in ("role_group", "grade", "source"):
        values = frame[dim].astype(str).to_numpy()
        table = {}
        for level in sorted(set(values)):
            mask = values == level
            table[level] = {"rows": int(mask.sum()), "groups": int(frame.loc[mask, "group_id"].nunique())} | {
                f"{model}_{metric}": block
                for model in ("gbm_cqr", "ridge", "cell_median")
                for metric, block in score(per_row[model], W, cfg.confidence, mask=mask).items()
                if metric in ("mae_rub", "mape", "coverage", "width_rub")
            }
        breakdown[dim] = table
    return {"metrics": metrics, "breakdown": breakdown, "rows": len(frame), "groups": int(frame["group_id"].nunique())}


def importance(model: SalaryModel, frame: pd.DataFrame, cfg: Config) -> dict[str, dict[str, float]]:
    """Rise of dev MAE (RUB) when the columns of one block are shuffled together across rows."""
    rows = records_of(frame)
    y = frame["salary_mid"].to_numpy(dtype=float)
    X = model.encoder.tree_matrix(model.encoder.encode(rows))
    base = float(np.mean(np.abs(np.exp(model.point.predict(X)) - y)))
    rng = np.random.default_rng(cfg.seed)
    out = {}
    for block in BLOCKS:
        cols = model.encoder.block_columns(block)
        rises = []
        for _ in range(cfg.importance_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            rises.append(float(np.mean(np.abs(np.exp(model.point.predict(Xp)) - y))) - base)
        out[block] = {"mae_rise_rub": round(float(np.mean(rises)), 1), "sd": round(float(np.std(rises)), 1)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mae_rise_rub"]))


def temporal_check(frame: pd.DataFrame, fitted: Fitted, cfg: Config) -> dict[str, Any]:
    """Refit on vacancies first seen before the cutoff; score dev vacancies first seen from it on."""
    seen = pd.to_datetime(frame["first_seen"], errors="coerce").dt.date
    before = seen.notna() & (seen < cfg.cutoff)
    after = seen.notna() & (seen >= cfg.cutoff)
    train = frame[before & (frame["split"] == TRAIN)]
    calib = frame[before & (frame["split"] == CALIB)]
    dev_late = frame[after & (frame["split"] == DEV)]
    refit = fit_all(train, calib, cfg, params=fitted.gbm_params, alpha=fitted.ridge_alpha)
    W = group_weights(dev_late["group_id"].astype(str).tolist(), cfg.n_resamples, cfg.bootstrap_seed)
    y = dev_late["salary_mid"].to_numpy(dtype=float)
    out: dict[str, Any] = {"cutoff": cfg.cutoff.isoformat(), "train_rows": len(train), "calib_rows": len(calib)}
    for name, fit in (("refit_before_cutoff", refit), ("main_model", fitted)):
        p = fit.model.predict_rows(records_of(dev_late))
        cols = point_errors(y, p["point"]) | interval_errors(y, p["low"], p["high"], p["point"])
        out[name] = score(cols, W, cfg.confidence)
    out["dev_rows_after_cutoff"] = len(dev_late)
    return out


def _share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


DEFINITIONS = {
    "target": "Midpoint of the stated range in RUB per month; a single stated bound stands for itself. "
    "Amounts as the source states them: no period or gross/net field exists in trudvsem or Habr Career.",
    "population": "IT dedup entities (rule role bucket not not_it) of trudvsem and Habr Career with a RUB "
    "salary, latest record of each entity, part-time vacancies left out.",
    "split": "Connected components over entity, dedup text cluster, extraction-corpus group and employer "
    "with one normalized title lie in one split; corpus gold test and dev groups pinned; calib is part of train.",
    "mae_rub": "Mean absolute error of the point against the midpoint, RUB per month.",
    "mape": "Mean of |point - midpoint| / midpoint.",
    "within_20pct": "Share of vacancies whose point is within 20% of the midpoint.",
    "coverage": "Share of vacancies whose midpoint lies in [low, high].",
    "width_rub": "Mean high - low, RUB per month.",
    "width_rel": "Mean (high - low) / point.",
    "ci": "Percentile bootstrap over dev groups (whole components), resamples and seed in configs/eval.toml; "
    "differences are paired on the same resamples.",
    "importance": "Rise of dev MAE of the boosting point when the columns of one input are shuffled across "
    "dev rows; shuffling breaks correlated inputs together only within a block.",
}

LIMITS_RU = [
    "Признаки v0 взяты из правил витрин (роль по заголовку, грейд по заголовку, навыки по словарю), "
    "а не из извлечения студентом; после разметки всего корпуса студентом модель переобучается на извлечённых полях.",
    "Gross или net источники не указывают, суммы взяты как опубликованы; источник стоит признаком.",
    "Почти все строки из trudvsem (портал «Работа России»), Хабр Карьера даёт десятки вакансий: оценка "
    "для source=habr опирается на малую выборку.",
    "Цель это середина опубликованной вилки; вилка «от X» без верхней границы считается как X.",
]


def run(frame: pd.DataFrame, counts: dict[str, int], cfg: Config, out_dir: Path) -> dict[str, Any]:
    train, calib, dev = (frame[frame["split"] == s] for s in (TRAIN, CALIB, DEV))
    fitted = fit_all(train, calib, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in ((ARTIFACT, fitted.model), (EVAL_ARTIFACT, fitted)):
        with (out_dir / name).open("wb") as f:
            pickle.dump(obj, f)
    log.info("saved %s", out_dir)

    result = evaluate(fitted, dev, cfg, DEV)
    metrics: dict[str, dict[str, Any]] = {f"data_{k}": {"value": v} for k, v in counts.items()}
    metrics["data_habr_share"] = _share(int((frame["source"] == "habr").sum()), len(frame))
    metrics["gbm_conformal_margin_log"] = {"value": round(fitted.model.margin, 5)}
    metrics |= result["metrics"]
    return {
        "metrics": metrics,
        "breakdown_dev": result["breakdown"],
        "importance_dev": importance(fitted.model, dev, cfg),
        "temporal_dev": temporal_check(frame, fitted, cfg),
        "selection": {
            "gbm_params": fitted.gbm_params,
            "gbm_cv": fitted.gbm_cv,
            "ridge_alpha": fitted.ridge_alpha,
            "ridge_cv": fitted.ridge_cv,
            "skill_features": len(fitted.model.encoder.skill_vocab),
        },
        "split_sha256": {s: ids_sha256(frame.loc[frame["split"] == s, "entity_id"]) for s in (TRAIN, CALIB, DEV, TEST)},
        "frame_sha256": frame_digest(frame),
        "model_version": fitted.model.version,
        "feature_version": FEATURE_VERSION,
        "level": cfg.level,
        "config": str(CONFIG.relative_to(ROOT).as_posix()),
    }


def log_test_run() -> None:
    path = RESULTS / "test_runs.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp}\tsalary\t-\t{git_commit()}\t{invocation()}\n")


def run_test(frame: pd.DataFrame, cfg: Config, model_dir: Path) -> dict[str, Any]:
    """Score the models saved by the dev run on test, unchanged."""
    with (model_dir / EVAL_ARTIFACT).open("rb") as f:
        fitted = pickle.load(f)
    if not isinstance(fitted, Fitted):
        raise TypeError(f"{model_dir / EVAL_ARTIFACT} does not hold fitted salary models")
    log_test_run()
    result = evaluate(fitted, frame[frame["split"] == TEST], cfg, TEST)
    return {"metrics": result["metrics"], "breakdown_test": result["breakdown"], "model_version": fitted.model.version}


def frame_digest(frame: pd.DataFrame) -> str:
    """Digest of what the models see: entity, split and target of every row."""
    return ids_sha256(
        f"{e}\t{s}\t{m:.2f}" for e, s, m in zip(frame["entity_id"], frame["split"], frame["salary_mid"], strict=True)
    )


def frame_path() -> Path:
    return data_dir() / "salary" / f"frame_{FEATURE_VERSION}.parquet"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.salary", description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=Path, help="saved frame (parquet) instead of a ClickHouse query")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--final-test", action="store_true", help="the one scoring of the test split")
    args = ap.parse_args(argv)
    load_dotenv()
    cfg = load_config(args.config)

    # The test is scored on the frame the dev run saved: a fresh query would move vacancies between splits.
    if args.frame or args.final_test:
        path = args.frame or frame_path()
        frame = pd.read_parquet(path)
        counts = json.loads(path.with_suffix(".counts.json").read_text(encoding="utf-8"))
    else:
        entities, records = load_clickhouse()
        corpus = load_corpus_groups(data_dir() / "corpus" / "split.parquet")
        frame, counts = build_frame(entities, records, corpus, cfg.shares, cfg.min_monthly)
        path = frame_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        path.with_suffix(".counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        log.info("frame %s: %s", path, counts)

    if args.final_test:
        doc = run_test(frame, cfg, args.model_dir)
        path = write_result(RESULT_TEST, doc.pop("metrics"), source_script="src/radar/models/salary.py", **doc)
    else:
        doc = run(frame, counts, cfg, args.model_dir)
        path = write_result(
            RESULT,
            doc.pop("metrics"),
            source_script="src/radar/models/salary.py",
            definitions=DEFINITIONS,
            limits_ru=LIMITS_RU,
            **doc,
        )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    # Run through the importable module, so pickled models refer to radar.models.salary, not __main__.
    from radar.models import salary

    sys.exit(salary.main())
