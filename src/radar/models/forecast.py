"""Weekly skill demand forecast with a rolling-origin backtest against naive baselines.

Series: marts.mart_extracted_skills_weekly, the skills the fine-tuned extractor read from the texts.
The primary target is the share of IT posts (vacancies) of the week that name the skill; the plain
count is reported next to it. A source enters the backtest only if its run of usable weeks (extracted,
complete, contiguous) is long enough for the configured history, horizon and number of origins; on
the current data that is Telegram alone.

At every origin each method sees the weeks before it and forecasts the next `horizon` weeks:
naive (last value), seasonal naive (value a season back, where the history allows), moving average,
simple exponential smoothing (statsmodels) and one gradient boosting model over all skills on lags.
Errors are MASE (scaled by the in-sample one-step naive error of the training weeks) and sMAPE per
skill and origin; intervals and paired differences against naive come from a two-way bootstrap that
draws skills and blocks of consecutive origins. Nothing is tuned on the backtest: every setting is
fixed in the config before the run.

    python -m radar.models.forecast        marts -> results/models/forecast.json
"""

import argparse
import logging
import sys
import time
import tomllib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # type: ignore[import-untyped]

from radar.config import ROOT, load_dotenv
from radar.results import write_result

if TYPE_CHECKING:
    from radar.stream.clickhouse import ClickHouse

log = logging.getLogger("radar.models.forecast")

CONFIG = ROOT / "configs" / "forecast.toml"
RESULT = "models/forecast.json"
METHODS = ("naive", "seasonal_naive", "moving_average", "ets", "boosting")
BASELINES = ("naive", "seasonal_naive", "moving_average")
MODELS = ("ets", "boosting")

Array = NDArray[np.float64]

SERIES_SQL = """
SELECT source, week, skill, vacancies_mentioning
FROM marts.mart_extracted_skills_weekly
WHERE source IN {sources:Array(String)}
FORMAT TSVWithNames
"""

COVERAGE_SQL = """
SELECT source, week, units, units_with_fields, units_it, coverage_extractable
FROM marts.mart_extraction_coverage
WHERE level = 'week' AND source IN {sources:Array(String)}
FORMAT TSVWithNames
"""

# IT units of the week whose extracted HTTP or HTTPS the marts dropped because the text names them only
# in links (configs/skills.toml [text_check]), and the units that keep the reported skill.
LINK_SQL = """
SELECT u.source AS source, u.week AS week, countIf(u.is_it) AS it_units,
    countIf(u.is_it AND f.skills_link_only_n > 0) AS link_only,
    countIf(u.is_it AND has(u.skills, {skill:String})) AS named
FROM staging.int_extracted_weekly_units AS u
LEFT JOIN staging.int_extracted_fields AS f ON f.source = u.record_source AND f.source_id = u.record_source_id
WHERE u.source IN {sources:Array(String)}
GROUP BY source, week
FORMAT TSVWithNames
SETTINGS join_use_nulls = 1
"""


@dataclass(frozen=True)
class Config:
    sources: tuple[str, ...]
    first_week: dict[str, date]
    min_coverage: float
    targets: tuple[str, ...]
    min_median_mentions: float
    exclude: tuple[str, ...]
    reference_source: str
    link_report: str
    horizon: int
    min_train: int
    min_origins: int
    season: int
    ma_window: int
    lags: int
    boosting: dict[str, float]
    replicates: int
    block: int
    level: float
    seed: int
    interval_quantile: float

    @property
    def min_weeks(self) -> int:
        return self.min_train + self.horizon + self.min_origins - 1


def load_config(path: Path = CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    d, s, b, base, g, bs = (
        raw["data"],
        raw["skills"],
        raw["backtest"],
        raw["baselines"],
        raw["boosting"],
        raw["bootstrap"],
    )
    targets = tuple(d["targets"])
    if not set(targets) <= {"share", "count"} or not targets:
        raise ValueError("data.targets takes share and/or count")
    return Config(
        sources=tuple(d["sources"]),
        first_week={k: date.fromisoformat(v) for k, v in d.get("first_week", {}).items()},
        min_coverage=float(d["min_coverage"]),
        targets=targets,
        min_median_mentions=float(s["min_median_mentions"]),
        exclude=tuple(s["exclude"]),
        reference_source=s["reference_source"],
        link_report=s["link_report"],
        horizon=int(b["horizon"]),
        min_train=int(b["min_train"]),
        min_origins=int(b["min_origins"]),
        season=int(b["season"]),
        ma_window=int(base["ma_window"]),
        lags=int(g["lags"]),
        boosting={
            k: g[k] for k in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")
        },
        replicates=int(bs["replicates"]),
        block=int(bs["block"]),
        level=float(bs["level"]),
        seed=int(bs["seed"]),
        interval_quantile=float(raw["serve"]["interval_quantile"]),
    )


# --- data ----------------------------------------------------------------------------------------


def usable_weeks(coverage: pd.DataFrame, first_week: date | None, min_coverage: float, today: date) -> list[date]:
    """Weeks of one source that are over, extracted enough and after first_week; the last contiguous run."""
    ok = coverage[
        (coverage["coverage_extractable"].fillna(0.0) >= min_coverage)
        & (coverage["week"].map(lambda w: w + timedelta(days=7) <= today))
    ]
    if first_week is not None:
        ok = ok[ok["week"] >= first_week]
    weeks = sorted(ok["week"])
    run: list[date] = []
    for w in weeks:
        if run and w - run[-1] != timedelta(days=7):
            run = []
        run.append(w)
    return run


@dataclass(frozen=True)
class Panel:
    source: str
    weeks: list[date]
    skills: list[str]
    mentions: Array  # skills x weeks
    it_units: Array  # weeks

    def target(self, name: str) -> Array:
        return self.mentions / self.it_units if name == "share" else self.mentions


def build_panel(source: str, series: pd.DataFrame, coverage: pd.DataFrame, weeks: list[date]) -> Panel:
    cov = coverage.set_index("week").loc[weeks]
    rows = series[series["week"].isin(weeks)]
    wide = rows.pivot_table(index="skill", columns="week", values="vacancies_mentioning", aggfunc="sum")
    wide = wide.reindex(columns=weeks).fillna(0.0)
    return Panel(
        source=source,
        weeks=weeks,
        skills=[str(s) for s in wide.index],
        mentions=wide.to_numpy(dtype=float),
        it_units=cov["units_it"].to_numpy(dtype=float),
    )


def select_skills(panel: Panel, min_median: float, exclude: tuple[str, ...]) -> list[int]:
    """Rows of the skills with enough weekly mentions, by total mentions, without the excluded ones."""
    median = np.median(panel.mentions, axis=1)
    total = panel.mentions.sum(axis=1)
    keep = [i for i, s in enumerate(panel.skills) if median[i] >= min_median and s not in exclude]
    return sorted(keep, key=lambda i: (-total[i], panel.skills[i]))


# --- methods ---------------------------------------------------------------------------------------


def naive(y: Array, h: int) -> Array:
    return np.full(h, y[-1])


def seasonal_naive(y: Array, h: int, m: int) -> Array:
    if len(y) < m or h > m:
        return np.full(h, np.nan)
    n = len(y)
    return np.array([y[n - m + j] for j in range(h)])


def moving_average(y: Array, h: int, k: int) -> Array:
    return np.full(h, y[-k:].mean())


def ets(y: Array, h: int) -> Array:
    """Simple exponential smoothing, ETS(A,N,N): smoothing and initial level fitted on the training weeks."""
    if np.ptp(y) == 0:
        return np.full(h, y[-1])
    with warnings.catch_warnings():
        # A boundary optimum (alpha near 0 or 1) is a valid fit; statsmodels warns about it anyway.
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
    return np.asarray(fit.forecast(h), dtype=float)


def lag_features(y: Array, t: int, lags: int, scale: float) -> list[float]:
    window = y[t - lags + 1 : t + 1] / scale
    return [*window, y[t - 3 : t + 1].mean() / scale, y[t - 12 : t + 1].mean() / scale]


def boosting(Y: Array, h: int, cfg: Config) -> Array:
    """One direct model over all series: change from the last value at horizon j, on each series' scale."""
    S, n = Y.shape
    start = max(cfg.lags, 13) - 1
    scales = Y.mean(axis=1)
    scales[scales == 0] = 1.0
    X: list[list[float]] = []
    target: list[float] = []
    for s in range(S):
        for t in range(start, n - 1):
            feats = lag_features(Y[s], t, cfg.lags, scales[s])
            for j in range(1, h + 1):
                if t + j < n:
                    X.append([*feats, float(j)])
                    target.append((Y[s, t + j] - Y[s, t]) / scales[s])
    params = cfg.boosting
    model = HistGradientBoostingRegressor(
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=cfg.seed,
    )
    model.fit(np.array(X), np.array(target))
    query = np.array(
        [[*lag_features(Y[s], n - 1, cfg.lags, scales[s]), float(j)] for s in range(S) for j in range(1, h + 1)]
    )
    change = np.asarray(model.predict(query), dtype=float).reshape(S, h)
    return np.asarray(Y[:, -1:] + change * scales[:, None], dtype=float)


def per_series(fn: Callable[[Array, int], Array]) -> Callable[[Array, int, Config], Array]:
    def run(Y: Array, h: int, cfg: Config) -> Array:
        return np.vstack([fn(y, h) for y in Y])

    return run


def method_table(cfg: Config) -> dict[str, Callable[[Array, int, Config], Array]]:
    return {
        "naive": per_series(naive),
        "seasonal_naive": per_series(lambda y, h: seasonal_naive(y, h, cfg.season)),
        "moving_average": per_series(lambda y, h: moving_average(y, h, cfg.ma_window)),
        "ets": per_series(ets),
        "boosting": boosting,
    }


# --- backtest --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Backtest:
    origins: list[int]  # number of training weeks at each origin
    actual: Array  # skills x origins x horizon
    forecast: dict[str, Array]  # method -> skills x origins x horizon
    scale: Array  # skills x origins, in-sample mean absolute one-step change


def backtest(Y: Array, cfg: Config, methods: tuple[str, ...] = METHODS) -> Backtest:
    T = Y.shape[1]
    h = cfg.horizon
    origins = list(range(cfg.min_train, T - h + 1))
    table = method_table(cfg)
    actual = np.stack([Y[:, n : n + h] for n in origins], axis=1)
    scale = np.stack([np.abs(np.diff(Y[:, :n], axis=1)).mean(axis=1) for n in origins], axis=1)
    forecast: dict[str, Array] = {}
    for name in methods:
        started = time.perf_counter()
        forecast[name] = np.stack([table[name](Y[:, :n], h, cfg) for n in origins], axis=1)
        log.info("%s: %d origins in %.1fs", name, len(origins), time.perf_counter() - started)
    return Backtest(origins=origins, actual=actual, forecast=forecast, scale=scale)


def mase(bt: Backtest, method: str) -> Array:
    """Skills x origins; NaN where the method has no forecast or the training weeks never change."""
    err = np.abs(bt.actual - bt.forecast[method]).mean(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bt.scale > 0, err / bt.scale, np.nan)


def mase_by_horizon(bt: Backtest, method: str) -> Array:
    err = np.abs(bt.actual - bt.forecast[method])
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(bt.scale[:, :, None] > 0, err / bt.scale[:, :, None], np.nan)
    return np.asarray(np.nanmean(scaled, axis=(0, 1)), dtype=float)


def smape(bt: Backtest, method: str) -> Array:
    a, f = bt.actual, bt.forecast[method]
    denom = np.abs(a) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        cell = np.where(denom > 0, 200.0 * np.abs(a - f) / denom, 0.0)
    cell = np.where(np.isnan(f), np.nan, cell)
    return np.asarray(cell.mean(axis=2), dtype=float)


def bootstrap_index(
    n_skills: int, n_origins: int, block: int, rng: np.random.Generator
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    skills = rng.integers(0, n_skills, n_skills)
    block = min(block, n_origins)
    starts = rng.integers(0, n_origins - block + 1, -(-n_origins // block))
    origins = (starts[:, None] + np.arange(block)).ravel()[:n_origins]
    return skills, origins


def bootstrap_means(matrices: dict[str, Array], cfg: Config) -> dict[str, Array]:
    """Replicates of the NaN-aware mean of every matrix, all on the same draws of skills and origins."""
    rng = np.random.default_rng(cfg.seed)
    n_skills, n_origins = next(iter(matrices.values())).shape
    out = {k: np.empty(cfg.replicates) for k in matrices}
    for r in range(cfg.replicates):
        s, o = bootstrap_index(n_skills, n_origins, cfg.block, rng)
        for k, m in matrices.items():
            out[k][r] = np.nanmean(m[np.ix_(s, o)])
    return out


def summary(point: Array, reps: Array, level: float) -> dict[str, Any]:
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(reps, [alpha, 1 - alpha])
    return {
        "value": round(float(np.nanmean(point)), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "cells": int(np.isfinite(point).sum()),
    }


def noise_floor(bt: Backtest, units: Array) -> Array:
    """MASE of an oracle that knows each week's true share: binomial sampling noise alone, skills x origins.

    E|p_hat - p| is about sqrt(2/pi) * sqrt(p(1-p)/m) for m posts; the realized share stands in for p.
    Posts of a week are not independent draws, so the true floor is higher than this.
    """
    m = np.stack([units[n : n + bt.actual.shape[2]] for n in bt.origins])  # origins x horizon
    sd = np.sqrt(bt.actual * (1 - bt.actual) / m[None, :, :])
    err = np.sqrt(2 / np.pi) * sd.mean(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bt.scale > 0, err / bt.scale, np.nan)


def compare(
    bt: Backtest, cfg: Config, methods: tuple[str, ...], extra: dict[str, Array] | None = None
) -> dict[str, dict[str, Any]]:
    """Mean error of every method and paired differences, keys `<method>.<metric>` and `<a>_minus_<b>.<metric>`.

    `extra` adds error matrices summarized on the same bootstrap draws (the noise floor).
    """
    errors = {"mase": {m: mase(bt, m) for m in methods}, "smape": {m: smape(bt, m) for m in methods}}
    pairs = [(m, "naive") for m in methods if m != "naive"]
    pairs += [(m, "moving_average") for m in MODELS if m in methods and "moving_average" in methods]
    pairs += [("boosting", "ets")] if {"boosting", "ets"} <= set(methods) else []
    matrices: dict[str, Array] = {}
    for metric, by_method in errors.items():
        for m, e in by_method.items():
            matrices[f"{m}.{metric}"] = e
        for a, b in pairs:
            matrices[f"{a}_minus_{b}.{metric}"] = by_method[a] - by_method[b]
    matrices |= extra or {}
    reps = bootstrap_means(matrices, cfg)
    out = {k: summary(m, reps[k], cfg.level) for k, m in matrices.items()}
    for m in methods:
        out[f"{m}.mase"]["by_horizon"] = [round(float(v), 4) for v in mase_by_horizon(bt, m)]
    for a, b in pairs:
        per_skill = np.nanmean(errors["mase"][a], axis=1) < np.nanmean(errors["mase"][b], axis=1)
        valid = np.isfinite(np.nanmean(errors["mase"][a], axis=1))
        n, d = int((per_skill & valid).sum()), int(valid.sum())
        out[f"{a}_minus_{b}.skills_better"] = {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}
    return out


def served_method(results: dict[str, dict[str, Any]]) -> str:
    """Lowest mean MASE among the methods whose paired interval against naive lies below zero, else naive."""
    better = [
        m
        for m in METHODS
        if m != "naive" and f"{m}_minus_naive.mase" in results and results[f"{m}_minus_naive.mase"]["ci_high"] < 0
    ]
    return min(better, key=lambda m: results[f"{m}.mase"]["value"]) if better else "naive"


def next_weeks(Y: Array, bt: Backtest, method: str, cfg: Config, target: str) -> tuple[Array, Array, Array]:
    """Point forecast from all weeks and an interval from the method's backtest errors per skill and horizon."""
    point = method_table(cfg)[method](Y, cfg.horizon, cfg)
    q = np.nanquantile(np.abs(bt.actual - bt.forecast[method]), cfg.interval_quantile, axis=1)
    upper_cap = 1.0 if target == "share" else np.inf
    return point, np.clip(point - q, 0.0, upper_cap), np.clip(point + q, 0.0, upper_cap)


# --- site ------------------------------------------------------------------------------------------

# The interval check needs a year of earlier backtest errors before it judges an origin.
INTERVAL_CHECK_MIN_ORIGINS = 52


def interval_check(err: Array, quantile: float, min_history: int = INTERVAL_CHECK_MIN_ORIGINS) -> tuple[int, int]:
    """How often an interval built only from errors already known would have covered the actual value.

    `err` is skills x origins x horizon absolute errors, origins one week apart. The error of origin o
    at horizon h is known from origin o + h on, so the interval at origin o and horizon h takes the
    quantile of origins up to o - h. Returns (covered, cells) over the origins with at least
    `min_history` such errors.
    """
    _, n_origins, horizon = err.shape
    covered = cells = 0
    for j in range(horizon):
        h = j + 1
        for o in range(min_history + h - 1, n_origins):
            past = err[:, : o - h + 1, j]
            q = np.nanquantile(past, quantile, axis=1)
            now = err[:, o, j]
            ok = np.isfinite(now) & np.isfinite(q)
            covered += int((now[ok] <= q[ok]).sum())
            cells += int(ok.sum())
    return covered, cells


@dataclass(frozen=True)
class SiteForecast:
    """The served method refit on the weeks the site export sees, with its own backtest interval."""

    method: str
    weeks: list[date]
    skills: list[str]
    mentions: Array  # skills x weeks
    it_units: Array  # weeks
    forecast_weeks: list[date]
    point: Array  # skills x horizon
    low: Array
    high: Array
    origins: int
    interval_quantile: float
    interval_covered: int
    interval_cells: int


def site_forecast(
    series: pd.DataFrame,
    coverage: pd.DataFrame,
    cfg: Config,
    skills: list[str],
    method: str,
    today: date,
    source: str = "telegram",
) -> SiteForecast:
    """Share target of one source: the usable weeks as the backtest defines them, the given skills.

    Skills and method come from the result file, so the site forecasts what the backtest judged; the
    weeks are the ones over by `today`, so the forecast moves on as weeks complete.
    """
    cov = coverage[coverage["source"] == source]
    weeks = usable_weeks(cov, cfg.first_week.get(source), cfg.min_coverage, today)
    if len(weeks) < cfg.min_weeks:
        raise ValueError(f"{source}: {len(weeks)} usable weeks, the backtest needs {cfg.min_weeks}")
    panel = build_panel(source, series[series["source"] == source], cov, weeks)
    missing = [s for s in skills if s not in panel.skills]
    if missing:
        raise ValueError(f"{source}: no weekly series for {missing}")
    rows = [panel.skills.index(s) for s in skills]
    Y = panel.target("share")[rows]
    bt = backtest(Y, cfg, methods=(method,))
    point, low, high = next_weeks(Y, bt, method, cfg, "share")
    covered, cells = interval_check(np.abs(bt.actual - bt.forecast[method]), cfg.interval_quantile)
    return SiteForecast(
        method=method,
        weeks=weeks,
        skills=list(skills),
        mentions=panel.mentions[rows],
        it_units=panel.it_units,
        forecast_weeks=[weeks[-1] + timedelta(days=7 * h) for h in range(1, cfg.horizon + 1)],
        point=point,
        low=low,
        high=high,
        origins=len(bt.origins),
        interval_quantile=cfg.interval_quantile,
        interval_covered=covered,
        interval_cells=cells,
    )


# --- run -------------------------------------------------------------------------------------------


def _tsv(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), sep="\t", na_values=["\\N"], keep_default_na=False)
    frame["week"] = pd.to_datetime(frame["week"]).dt.date
    return frame


def _array_arg(sources: tuple[str, ...]) -> str:
    return "[" + ",".join(f"'{s}'" for s in sources) + "]"


def fetch_series(ch: "ClickHouse", sources: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly skill mentions and extraction coverage of the sources, marts only."""
    arg = _array_arg(sources)
    return _tsv(ch.query(SERIES_SQL, sources=arg)), _tsv(ch.query(COVERAGE_SQL, sources=arg))


def load(sources: tuple[str, ...], link_report: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        series, coverage = fetch_series(ch, sources)
        links = _tsv(ch.query(LINK_SQL, sources=_array_arg(sources), skill=link_report))
    return series, coverage, links


def _share(n: float, d: float) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": int(n), "denominator": int(d)}


def coverage_by_year(cov: pd.DataFrame, min_coverage: float, today: date) -> dict[str, dict[str, Any]]:
    """Per calendar year of the week's Monday: weeks over, weeks extracted enough, median weekly IT units."""
    over = cov[cov["week"].map(lambda w: w + timedelta(days=7) <= today)]
    years: dict[str, dict[str, Any]] = {}
    for year, g in over.groupby(over["week"].map(lambda w: w.year)):
        years[str(year)] = {
            "value": float(np.median(g["units_it"])),
            "weeks": len(g),
            "weeks_covered": int((g["coverage_extractable"].fillna(0.0) >= min_coverage).sum()),
        }
    return years


def link_check(links: pd.DataFrame, weeks: list[date]) -> dict[str, Any]:
    rows = links[links["week"].isin(weeks)]
    it = rows["it_units"].sum()
    named = _share(rows["named"].sum(), it)
    return _share(rows["link_only"].sum(), it) | {f"named_{k}": v for k, v in named.items() if k != "denominator"}


DEFINITIONS = {
    "share": "Posts (vacancies) of the week naming the skill / IT posts (vacancies) of the week with valid fields.",
    "count": "Posts (vacancies) of the week naming the skill.",
    "mase": "Mean absolute error over the horizon divided by the mean absolute one-step change of the skill's "
    "training weeks (in-sample naive), averaged over skills and origins. Below 1: better than the in-sample naive.",
    "smape": "Mean over the horizon of 200 |actual - forecast| / (|actual| + |forecast|), in percent, averaged over "
    "skills and origins.",
    "ci": "Percentile interval of a two-way bootstrap: skills with replacement, origins in moving blocks of "
    "consecutive origins; the same draws for every method, so differences are paired.",
    "minus": "<a>_minus_<b>: mean over the cells both methods forecast of a's error minus b's error. Below zero: a "
    "is better.",
    "skills_better": "Skills whose mean MASE over origins is lower with a than with b.",
    "seasonal_naive": "Defined only at origins with a full season of history, so its means and its differences "
    "cover those origins alone (cells).",
    "by_horizon": "MASE at horizon 1, 2, 3, 4 weeks, averaged over skills and origins.",
    "noise_floor": "Share target only: MASE of an oracle that knows the true weekly share and misses only by the "
    "binomial sampling noise of the week's IT posts, sqrt(2/pi) sqrt(p(1-p)/m); a lower bound under independent "
    "posts.",
}

LIMITS_RU = [
    "Ряд Telegram начинается 3 января 2022 года: раньше IT-постов в неделю заметно меньше, и шум доли "
    "навыка в ранних неделях был бы другим, чем в поздних.",
    "Набор каналов менялся (каналы появлялись и затихали), доля навыка отражает и состав каналов, а не "
    "только спрос рынка.",
    "Навыки извлечены моделью из текста и содержат ошибки, ошибка извлечения входит в ряд как шум.",
    "Порталы собираются меньше 20 недель, для бэктеста с горизонтом 4 недели и полугодовой историей этого мало.",
    "Метод для прогноза на следующие недели выбран по тем же точкам бэктеста, по которым он оценён, поэтому "
    "его ошибка на будущих неделях может оказаться выше.",
]


def run_target(
    panel: Panel, rows: list[int], target: str, cfg: Config
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    Y = panel.target(target)[rows]
    bt = backtest(Y, cfg)
    extra = {"noise_floor.mase": noise_floor(bt, panel.it_units)} if target == "share" else None
    results = compare(bt, cfg, METHODS, extra)
    method = served_method(results)
    point, low, high = next_weeks(Y, bt, method, cfg, target)
    weeks = [str(panel.weeks[-1] + timedelta(days=7 * j)) for j in range(1, cfg.horizon + 1)]
    digits = 4 if target == "share" else 1
    served = {
        "method": method,
        "weeks": weeks,
        "skills": {
            panel.skills[r]: {
                "last": round(float(Y[i, -1]), digits),
                "forecast": [round(float(v), digits) for v in point[i]],
                "low": [round(float(v), digits) for v in low[i]],
                "high": [round(float(v), digits) for v in high[i]],
            }
            for i, r in enumerate(rows)
        },
    }
    metrics = {f"{target}.{k}": v for k, v in results.items()}
    metrics[f"{target}.served"] = {"value": method}
    seasonal_origins = int(np.isfinite(bt.forecast["seasonal_naive"][0, :, 0]).sum())
    metrics[f"{target}.origins"] = {"value": len(bt.origins), "seasonal_naive": seasonal_origins}
    return metrics, served


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.forecast", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    load_dotenv()
    series, coverage, links = load(cfg.sources, cfg.link_report)
    today = datetime.now(ZoneInfo("Europe/Moscow")).date()
    run_started = time.perf_counter()

    metrics: dict[str, dict[str, Any]] = {}
    panels: dict[str, Panel] = {}
    for source in cfg.sources:
        cov = coverage[coverage["source"] == source]
        weeks = usable_weeks(cov, cfg.first_week.get(source), cfg.min_coverage, today)
        eligible = len(weeks) >= cfg.min_weeks
        metrics[f"data.{source}.weeks"] = {
            "value": len(weeks),
            "first_week": str(weeks[0]) if weeks else None,
            "last_week": str(weeks[-1]) if weeks else None,
            "eligible": eligible,
            "min_weeks": cfg.min_weeks,
        }
        if source in cfg.first_week:
            # Why the series starts there: extraction coverage and weekly volume of every year.
            for year, entry in coverage_by_year(cov, cfg.min_coverage, today).items():
                metrics[f"data.{source}.years.{year}"] = entry
        if weeks:
            panels[source] = build_panel(source, series[series["source"] == source], cov, weeks)
        log.info("%s: %d usable weeks, eligible %s", source, len(weeks), eligible)

    served: dict[str, Any] = {}
    selected: dict[str, list[str]] = {}
    for source, panel in panels.items():
        if not metrics[f"data.{source}.weeks"]["eligible"]:
            continue
        rows = select_skills(panel, cfg.min_median_mentions, cfg.exclude)
        selected[source] = [panel.skills[r] for r in rows]
        metrics[f"data.{source}.skills"] = {"value": len(rows), "min_median_mentions": cfg.min_median_mentions}
        metrics[f"data.{source}.it_units_median"] = {"value": float(np.median(panel.it_units))}
        for skill in cfg.exclude:
            if skill in panel.skills:
                i = panel.skills.index(skill)
                entry = _share(panel.mentions[i].sum(), panel.it_units.sum())
                ref = panels.get(cfg.reference_source)
                if ref is not None and skill in ref.skills:
                    r = _share(ref.mentions[ref.skills.index(skill)].sum(), ref.it_units.sum())
                    entry |= {"reference_source": cfg.reference_source} | {f"reference_{k}": v for k, v in r.items()}
                metrics[f"excluded.{source}.{skill}"] = entry
        for name in (source, cfg.reference_source):
            if name in panels:
                metrics[f"link_check.{name}"] = link_check(links[links["source"] == name], panels[name].weeks)
        for target in cfg.targets:
            started = time.perf_counter()
            target_metrics, served[f"{source}.{target}"] = run_target(panel, rows, target, cfg)
            metrics |= {f"{source}.{k}": v for k, v in target_metrics.items()}
            log.info("%s %s: done in %.0fs", source, target, time.perf_counter() - started)

    metrics["runtime_seconds"] = {"value": round(time.perf_counter() - run_started)}
    written = write_result(
        RESULT,
        metrics,
        source_script="src/radar/models/forecast.py",
        definitions=DEFINITIONS,
        limits_ru=LIMITS_RU,
        config=args.config.relative_to(ROOT).as_posix() if args.config.is_relative_to(ROOT) else str(args.config),
        run_date=str(today),
        backtest={
            "horizon": cfg.horizon,
            "min_train": cfg.min_train,
            "min_origins": cfg.min_origins,
            "season": cfg.season,
            "ma_window": cfg.ma_window,
            "lags": cfg.lags,
            "min_coverage": cfg.min_coverage,
        },
        bootstrap={"replicates": cfg.replicates, "block": cfg.block, "level": cfg.level, "seed": cfg.seed},
        interval_quantile=cfg.interval_quantile,
        methods=list(METHODS),
        skills=selected,
        served=served,
    )
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
