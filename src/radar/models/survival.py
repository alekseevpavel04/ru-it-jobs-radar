"""Vacancy lifetimes on the trudvsem snapshots: Kaplan-Meier and Cox PH.

Subject: a dedup entity with at least one open sighting in the snapshot, so a vacancy
posted again under a new id stays one vacancy. Only the snapshot counts: it is a full daily dump of
the portal, so absence is evidence. Habr and the API slice start on 2026-10-08; the API never shows
a closure and Habr has days, not weeks, so both stay out until they have history.

Window: collected complete snapshot days (hist_calendar) from the first one up to the first step of
more than MAX_STEP_DAYS between collected days. A vacancy missing across a long hole cannot be told
from one that closed, so days after it are not used until the hole is filled.

Event: the entity has no open sighting on GRACE_DAYS consecutive collected days; it closed on the
first of them. Deleted and unpublished rows are not open sightings. Back within the grace window,
it was open all along (the portal hides vacancies for a few days); back later, it closed on its
first missing day and the return is counted, not followed.

Time: days since the creation date at the source (published_at, earliest over the entity's
records). Every subject enters the risk set on its first sighting (left truncation): a vacancy
open on the first collected day enters at its age then, a new one at its age when it appeared.
Right censoring: no closure confirmed by the end of the window; censored the day after the last
sighting. The incident cohort (first seen after the first day, time from first sighting) is
reported next to it as a check that needs no creation date.

    python -m radar.models.survival            results/models/survival.json and KM figures
"""

import argparse
import csv
import logging
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter  # type: ignore[import-untyped]
from lifelines.exceptions import StatError  # type: ignore[import-untyped]
from lifelines.utils import median_survival_times  # type: ignore[import-untyped]
from numpy.typing import NDArray

from radar.config import RESULTS, load_dotenv
from radar.results import write_result

log = logging.getLogger("radar.models.survival")

Array = NDArray[np.float64]

GRACE_DAYS = 7
GRACE_SENSITIVITY = (1, 3, 7, 14)
MAX_STEP_DAYS = 2
RUN_HIST_MAX = 14
AT_RISK_DAYS = (0, 7, 14, 30, 60, 90)
ALPHA = 0.05
Z95 = 1.959963984540054
PH_ALPHA = 0.01
# Covariates that fail the PH test get one hazard ratio up to this age and one after it. 30 days
# separates closing before the spike of closures at 30-32 days from closing at or after it.
SPLIT_DAY = 30
# Closures pile up on these days since creation; a 30-day publication term of the portal would
# explain it; not confirmed against the portal's rules.
SPIKE_AGES = (30, 31, 32)
PLOT_MAX_DAYS = 120
TOP_EMPLOYERS = 3
FIGURES = RESULTS / "models" / "figures"

# Marts role buckets (macros/marts.sql) in the groups of the gold-sample proxy (extract/role_proxy.py).
ROLE_GROUPS = {
    "ml_ds": "ml_ds_data",
    "data_engineering": "ml_ds_data",
    "data_analytics": "ml_ds_data",
    "software_engineer": "dev",
    "backend": "dev",
    "frontend": "dev",
    "fullstack": "dev",
    "mobile": "dev",
    "embedded": "dev",
    "developer_1c": "dev",
    "business_system_analysis": "analyst",
    "qa": "qa_devops_sys",
    "devops": "qa_devops_sys",
    "sysadmin_support": "qa_devops_sys",
    "security": "qa_devops_sys",
    "design": "other_it",
    "management": "other_it",
    "other_it": "other_it",
}

# Covariate: (levels, reference level of the Cox model). The reference is the largest level.
COVARIATES: dict[str, tuple[tuple[str, ...], str]] = {
    "role": (("dev", "ml_ds_data", "analyst", "qa_devops_sys", "other_it"), "dev"),
    "experience": (("none", "1-2", "3-5", "6+"), "1-2"),
    "salary": (("given", "not_given"), "given"),
    "region": (("moscow", "spb", "other", "remote"), "other"),
}

LABELS_RU = {
    "role": "Роль",
    "experience": "Требуемый опыт, лет",
    "salary": "Зарплата",
    "region": "Регион",
    "ml_ds_data": "ML, DS, данные",
    "dev": "разработка",
    "analyst": "аналитики",
    "qa_devops_sys": "QA, DevOps, админы, ИБ",
    "other_it": "прочие IT",
    "none": "без опыта",
    "given": "указана",
    "not_given": "не указана",
    "moscow": "Москва",
    "spb": "Санкт-Петербург",
    "other": "другие регионы",
    "remote": "удалённо",
}

SIGHTINGS_SQL = """
SELECT r.entity_id AS entity_id, s.dt AS dt, toDate(s.published_at) AS published
FROM staging.hist_snapshot_days AS s
INNER JOIN (
    SELECT source_id, entity_id FROM staging.int_vacancy_records WHERE source = 'trudvsem_snapshot'
) AS r ON s.source_id = r.source_id
WHERE NOT s.is_closed_state
FORMAT TabSeparatedWithNames
"""

CALENDAR_SQL = """
SELECT dt FROM staging.hist_calendar WHERE source = 'trudvsem_snapshot' AND is_complete ORDER BY dt
FORMAT TabSeparatedWithNames
"""

ENTITIES_SQL = """
SELECT entity_id, role_bucket, experience_min_years, salary_from, salary_to, region_group, employer_key
FROM staging.int_vacancy_entities
WHERE family = 'trudvsem'
FORMAT TabSeparatedWithNames
"""


@dataclass(frozen=True)
class Window:
    days: tuple[date, ...]
    holes: tuple[tuple[date, date], ...]
    days_after: int


def observation_window(collected: Iterable[date], max_step: int = MAX_STEP_DAYS) -> Window:
    """Collected days up to the first step longer than max_step; every hole is listed."""
    days = sorted(set(collected))
    holes = tuple((a, b) for a, b in pairwise(days) if (b - a).days > 1)
    end = len(days)
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days > max_step:
            end = i
            break
    return Window(tuple(days[:end]), holes, len(days) - end)


def absence_runs(idx: Sequence[int], n_days: int) -> tuple[list[int], int]:
    """Collected days missing between sightings that were followed by a return, and after the last one."""
    returned = [b - a - 1 for a, b in pairwise(idx) if b - a > 1]
    return returned, n_days - 1 - idx[-1]


def follow(idx: Sequence[int], n_days: int, grace: int) -> tuple[int, bool, bool]:
    """Last sighting of the followed span, whether its closure is confirmed, whether it came back later."""
    last = idx[0]
    for i in idx[1:]:
        if i - last - 1 >= grace:
            return last, True, True
        last = i
    return last, n_days - 1 - last >= grace, False


def window_indices(sightings: pd.DataFrame, window: Window) -> dict[str, list[int]]:
    """Sorted positions in the window of each entity's open sightings; entities seen only outside are dropped."""
    pos = {d: i for i, d in enumerate(window.days)}
    found: dict[str, set[int]] = {}
    for entity_id, dt in zip(sightings["entity_id"], sightings["dt"], strict=True):
        i = pos.get(dt)
        if i is not None:
            found.setdefault(entity_id, set()).add(i)
    return {k: sorted(v) for k, v in sorted(found.items())}


def build_subjects(sightings: pd.DataFrame, window: Window, grace: int = GRACE_DAYS) -> pd.DataFrame:
    """One row per entity: entry and duration in days since creation, event, and the incident-cohort time."""
    indices = window_indices(sightings, window)
    published = sightings.dropna(subset=["published"]).groupby("entity_id")["published"].min().to_dict()
    n_days = len(window.days)
    rows = []
    for entity_id, idx in indices.items():
        last, event, late = follow(idx, n_days, grace)
        first_seen = window.days[idx[0]]
        last_seen = window.days[last]
        end = window.days[last + 1] if event else last_seen + timedelta(days=1)
        origin = min(published.get(entity_id, first_seen), first_seen)
        rows.append(
            {
                "entity_id": entity_id,
                "origin": origin,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "end": end,
                "event": event,
                "late_return": late,
                "prevalent": idx[0] == 0,
                "entry": (first_seen - origin).days,
                "duration": (end - origin).days,
                "duration_incident": (end - first_seen).days,
            }
        )
    return pd.DataFrame(rows)


def grace_table(indices: dict[str, list[int]], n_days: int, graces: Sequence[int]) -> dict[str, Any]:
    """For each grace: closures the rule would call, and how many of them the vacancy later came back from."""
    returned: list[int] = []
    tails: list[int] = []
    for idx in indices.values():
        runs, last_tail = absence_runs(idx, n_days)
        returned.extend(runs)
        tails.append(last_tail)
    back = np.array(returned, dtype=np.int64)
    tail_days = np.array(tails, dtype=np.int64)
    table = {}
    for g in graces:
        n_back = int((back >= g).sum())
        closures = n_back + int((tail_days >= g).sum())
        table[str(g)] = {"came_back": n_back, "closures": closures}
    hist = {str(k): int((back == k).sum()) for k in range(1, RUN_HIST_MAX + 1)}
    hist[f">{RUN_HIST_MAX}"] = int((back > RUN_HIST_MAX).sum())
    return {"by_grace": table, "returned_runs": len(returned), "returned_run_days": hist}


def experience_group(years: float) -> str | None:
    if pd.isna(years):
        return None
    if years == 0:
        return "none"
    if years <= 2:
        return "1-2"
    if years <= 5:
        return "3-5"
    return "6+"


def covariates(entities: pd.DataFrame) -> pd.DataFrame:
    """Entity fields of the marts (latest record) as the model's categories; None where unknown."""
    salary = (entities["salary_from"].fillna(0) > 0) | (entities["salary_to"].fillna(0) > 0)
    region = entities["region_group"].where(entities["region_group"].isin(COVARIATES["region"][0]))
    return pd.DataFrame(
        {
            "entity_id": entities["entity_id"],
            "role_bucket": entities["role_bucket"],
            "role": entities["role_bucket"].map(ROLE_GROUPS),
            "experience": entities["experience_min_years"].map(experience_group),
            "salary": np.where(salary, "given", "not_given"),
            "region": region,
            "cluster": entities["employer_key"].fillna(entities["entity_id"]),
        }
    )


def at_risk(frame: pd.DataFrame, day: int, entry: str = "entry", duration: str = "duration") -> int:
    """Subjects under observation on a day: entered by it and not ended before it."""
    return int(((frame[entry] <= day) & (frame[duration] > day)).sum())


def km_fitter(frame: pd.DataFrame, label: str, entry: str | None = "entry", duration: str = "duration") -> Any:
    """Kaplan-Meier with delayed entry; None when a risk set empties between entries (S(t) would stay 0)."""
    try:
        return KaplanMeierFitter(alpha=ALPHA).fit(
            frame[duration], frame["event"], entry=frame[entry] if entry else None, label=label
        )
    except StatError:
        return None


def km_summary(frame: pd.DataFrame, entry: str | None = "entry", duration: str = "duration") -> dict[str, Any]:
    in_frame = frame.assign(_entry=frame[entry] if entry else 0)
    out: dict[str, Any] = {
        "subjects": len(frame),
        "events": int(frame["event"].sum()),
        "at_risk": {str(d): at_risk(in_frame, d, "_entry", duration) for d in AT_RISK_DAYS},
    }
    kmf = km_fitter(frame, "km", entry, duration)
    if kmf is None:
        return {"value": "not estimable", "reason": "a risk set empties after the last early entry", **out}
    median = float(kmf.median_survival_time_)
    low, high = (float(b) for b in median_survival_times(kmf.confidence_interval_).to_numpy().ravel())
    return {
        "value": median if np.isfinite(median) else "not reached",
        "low": low if np.isfinite(low) else None,
        "high": high if np.isfinite(high) else None,
        **out,
        "survival": {str(d): round(float(kmf.survival_function_at_times(d).iloc[0]), 4) for d in AT_RISK_DAYS if d > 0},
    }


def design(subjects: pd.DataFrame) -> pd.DataFrame:
    """Duration, event, entry, cluster and one indicator per non-reference level."""
    frame = subjects[["duration", "event", "entry", "cluster"]].copy()
    frame["event"] = frame["event"].astype(int)
    for name, (levels, reference) in COVARIATES.items():
        for level in levels:
            if level != reference:
                frame[f"{name}={level}"] = (subjects[name] == level).astype(int)
    return frame


def split_at(frame: pd.DataFrame, day: int, columns: Sequence[str]) -> pd.DataFrame:
    """Episodes before and after `day`, with `<column>:late` = column after it (piecewise hazard ratio)."""
    early = frame[frame["entry"] < day].copy()
    early_end = early["duration"] > day
    early.loc[early_end, "event"] = 0
    early["duration"] = early["duration"].clip(upper=day)
    late = frame[frame["duration"] > day].copy()
    late["entry"] = late["entry"].clip(lower=day)
    parts = []
    for part, is_late in ((early, 0), (late, 1)):
        for column in columns:
            part[f"{column}:late"] = part[column] * is_late
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class CoxFit:
    columns: list[str]
    beta: Array
    variance: Array
    robust_variance: Array
    clusters: int


@dataclass(frozen=True)
class RiskSets:
    """Breslow risk-set means at each distinct event time; the risk set at t is entry < t <= duration."""

    x: Array
    risk: Array
    times: Array
    masks: list[NDArray[np.bool_]]
    means: Array
    totals: Array


def risk_sets(frame: pd.DataFrame, columns: Sequence[str], beta: Array) -> RiskSets:
    x = frame[list(columns)].to_numpy(dtype=np.float64)
    duration = frame["duration"].to_numpy(dtype=np.float64)
    entry = frame["entry"].to_numpy(dtype=np.float64)
    risk = np.exp(x @ beta)
    times = np.unique(duration[frame["event"].to_numpy(dtype=bool)])
    masks = [(entry < t) & (duration >= t) for t in times]
    totals = np.array([risk[m].sum() for m in masks])
    means = np.array([risk[m] @ x[m] / s for m, s in zip(masks, totals, strict=True)]).reshape(len(times), x.shape[1])
    return RiskSets(x, risk, times, masks, means, totals)


def schoenfeld_residuals(frame: pd.DataFrame, sets: RiskSets) -> tuple[Array, Array]:
    """Event times and covariate minus its risk-set mean, one row per event."""
    events = np.flatnonzero(frame["event"].to_numpy(dtype=bool))
    times = frame["duration"].to_numpy(dtype=np.float64)[events]
    at = np.searchsorted(sets.times, times)
    return times, sets.x[events] - sets.means[at]


def score_residuals(frame: pd.DataFrame, sets: RiskSets) -> Array:
    """Per-subject score (Lin-Wei) with delayed entry: its own event term minus its share of every
    event it was at risk for. lifelines builds these without entry times, hence its robust errors
    are wrong for left-truncated data and this module computes its own."""
    duration = frame["duration"].to_numpy(dtype=np.float64)
    event = frame["event"].to_numpy(dtype=bool)
    resid = np.zeros_like(sets.x)
    _, ev_resid = schoenfeld_residuals(frame, sets)
    resid[np.flatnonzero(event)] = ev_resid
    deaths = np.array([int(((duration == t) & event).sum()) for t in sets.times])
    for mask, mean, total, d in zip(sets.masks, sets.means, sets.totals, deaths, strict=True):
        resid[mask] -= (d * sets.risk[mask] / total)[:, None] * (sets.x[mask] - mean)
    return resid


def fit_cox(frame: pd.DataFrame) -> CoxFit:
    """Cox PH with delayed entry (lifelines, Efron ties) and a sandwich covariance clustered by `cluster`."""
    columns = [c for c in frame.columns if c not in ("duration", "event", "entry", "cluster")]
    cph = CoxPHFitter(alpha=ALPHA).fit(
        frame.drop(columns=["cluster"]), duration_col="duration", event_col="event", entry_col="entry"
    )
    beta = cph.params_[columns].to_numpy(dtype=np.float64)
    variance = cph.variance_matrix_.loc[columns, columns].to_numpy(dtype=np.float64)
    delta = score_residuals(frame, risk_sets(frame, columns, beta)) @ variance
    by_cluster = pd.DataFrame(delta).groupby(frame["cluster"].to_numpy()).sum().to_numpy()
    return CoxFit(columns, beta, variance, by_cluster.T @ by_cluster, int(frame["cluster"].nunique()))


def hazard_ratios(fit: CoxFit, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for i, name in enumerate(fit.columns):
        column = frame[name]
        sizes = frame.loc[column == 1, "cluster"].value_counts()
        out[name] = {
            **_ratio(float(fit.beta[i]), float(fit.robust_variance[i, i])),
            "se_robust": round(math.sqrt(fit.robust_variance[i, i]), 4),
            "se_model": round(math.sqrt(fit.variance[i, i]), 4),
            "subjects": int((column == 1).sum()),
            "events": int(((column == 1) & (frame["event"] == 1)).sum()),
            "employers": len(sizes),
            "top3_employer_subjects": int(sizes.head(3).sum()),
        }
    return out


def _ratio(b: float, var: float) -> dict[str, float]:
    s = math.sqrt(var)
    return {
        "value": round(math.exp(b), 4),
        "low": round(math.exp(b - Z95 * s), 4),
        "high": round(math.exp(b + Z95 * s), 4),
        "p": math.erfc(abs(b / s) / math.sqrt(2)),
    }


def piecewise_ratios(fit: CoxFit, split: pd.DataFrame, names: Sequence[str], day: int) -> dict[str, dict[str, Any]]:
    """Hazard ratio of each name up to `day` and after it, from a model with `<name>:late` terms (split_at)."""
    at = {c: i for i, c in enumerate(fit.columns)}
    v = fit.robust_variance
    out = {}
    for name in names:
        a, b = at[name], at[f"{name}:late"]
        late = fit.beta[a] + fit.beta[b]
        out[name] = {
            "split_day": day,
            "early": _ratio(float(fit.beta[a]), float(v[a, a])),
            "late": _ratio(float(late), float(v[a, a] + v[b, b] + 2 * v[a, b])),
            "change_p": _ratio(float(fit.beta[b]), float(v[b, b]))["p"],
            "value": round(math.exp(late), 4),
            "subjects": int((split.loc[split[f"{name}:late"] == 0, name] == 1).sum()),
        }
    return out


def ph_test(frame: pd.DataFrame, fit: CoxFit) -> dict[str, dict[str, float]]:
    """Grambsch-Therneau test per covariate: scaled Schoenfeld residuals against the rank of event
    time, with the model-based covariance as in lifelines' proportional_hazard_test. lifelines has
    no residuals under delayed entry, so they come from risk_sets."""
    times, resid = schoenfeld_residuals(frame, risk_sets(frame, fit.columns, fit.beta))
    d = len(times)
    scaled = d * resid @ fit.variance
    rank = pd.Series(times).rank(method="average").to_numpy()
    rank -= rank.mean()
    stats = (rank @ scaled) ** 2 / (d * np.diag(fit.variance) * (rank @ rank))
    return {
        name: {"statistic": round(float(s), 3), "p": math.erfc(math.sqrt(float(s) / 2))}
        for name, s in zip(fit.columns, stats, strict=True)
    }


def plot_km(subjects: pd.DataFrame, name: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels, _ = COVARIATES[name]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    for level in levels:
        part = subjects[subjects[name] == level]
        label = f"{LABELS_RU.get(level, level)} ({len(part)})"
        kmf = km_fitter(part, label) if len(part) else None
        if kmf is None:
            ax.plot([], [], " ", label=f"{label}: не оценивается")
        else:
            kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.legend(fontsize=8)
    ax.set_xlim(0, PLOT_MAX_DAYS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("дней с даты публикации")
    ax.set_ylabel("доля ещё открытых")
    ax.set_title(f"{LABELS_RU[name]}: Каплан-Мейер с поздним входом, закрытие после {GRACE_DAYS} дней отсутствия")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _tsv(text: str) -> pd.DataFrame:
    return pd.read_csv(
        StringIO(text), sep="\t", quoting=csv.QUOTE_NONE, na_values=["\\N"], keep_default_na=False, dtype=str
    )


def _dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def load() -> tuple[pd.DataFrame, list[date], pd.DataFrame]:
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        sightings = _tsv(ch.query(SIGHTINGS_SQL))
        calendar = _tsv(ch.query(CALENDAR_SQL))
        entities = _tsv(ch.query(ENTITIES_SQL))
    sightings["dt"] = _dates(sightings["dt"])
    sightings["published"] = _dates(sightings["published"])
    for column in ("experience_min_years", "salary_from", "salary_to"):
        entities[column] = pd.to_numeric(entities[column])
    return sightings, list(_dates(calendar["dt"])), entities


def _share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def analyse(sightings: pd.DataFrame, calendar: Sequence[date], entities: pd.DataFrame) -> dict[str, Any]:
    window = observation_window(calendar)
    n_days = len(window.days)
    indices = window_indices(sightings, window)
    subjects_all = build_subjects(sightings, window).merge(covariates(entities), on="entity_id", how="left")
    is_it = subjects_all["role"].notna()
    subjects = subjects_all[is_it].reset_index(drop=True)
    log.info(
        "window %s..%s, %d days; %d entities, %d IT",
        window.days[0],
        window.days[-1],
        n_days,
        len(subjects_all),
        len(subjects),
    )

    metrics: dict[str, dict[str, Any]] = {
        "window_days": {"value": n_days, "first_dt": str(window.days[0]), "last_dt": str(window.days[-1])},
        "collected_days_after_window": {"value": window.days_after},
        "entities_seen": {"value": len(subjects_all)},
        "it_subjects": _share(len(subjects), len(subjects_all)),
        "not_it_excluded": _share(int((~is_it).sum()), len(subjects_all)),
        "no_role_bucket": _share(int(subjects_all["role_bucket"].isna().sum()), len(subjects_all)),
        "closed": _share(int(subjects["event"].sum()), len(subjects)),
        "prevalent": _share(int(subjects["prevalent"].sum()), len(subjects)),
        "late_return": _share(int(subjects["late_return"].sum()), int(subjects["event"].sum())),
        f"closed_at_age_{SPIKE_AGES[0]}_{SPIKE_AGES[-1]}": _share(
            int((subjects["event"] & subjects["duration"].isin(SPIKE_AGES)).sum()), int(subjects["event"].sum())
        ),
        "entry_after_creation_over_7_days_incident": _share(
            int(((~subjects["prevalent"]) & (subjects["entry"] > 7)).sum()), int((~subjects["prevalent"]).sum())
        ),
    }

    grace = grace_table({k: indices[k] for k in subjects["entity_id"]}, n_days, GRACE_SENSITIVITY)
    for g, row in grace["by_grace"].items():
        metrics[f"closure_came_back_grace_{g}"] = _share(row["came_back"], row["closures"])
        others = build_subjects(sightings[sightings["entity_id"].isin(subjects["entity_id"])], window, int(g))
        metrics[f"median_days_grace_{g}"] = km_summary(others)

    metrics["median_days_overall"] = km_summary(subjects)
    incident = subjects[~subjects["prevalent"]]
    metrics["median_days_incident_from_first_sighting"] = km_summary(incident, None, "duration_incident")

    for name, (levels, _) in COVARIATES.items():
        metrics[f"unknown_{name}"] = _share(int(subjects[name].isna().sum()), len(subjects))
        for level in levels:
            part = subjects[subjects[name] == level]
            if len(part):
                metrics[f"median_days_{name}_{level}"] = km_summary(part)
                metrics[f"median_days_incident_{name}_{level}"] = km_summary(
                    part[~part["prevalent"]], None, "duration_incident"
                )

    complete = subjects.dropna(subset=list(COVARIATES)).reset_index(drop=True)
    frame = design(complete)
    metrics["cox_subjects"] = _share(len(complete), len(subjects))
    metrics["cox_events"] = _share(int(frame["event"].sum()), len(complete))
    fit = fit_cox(frame)
    metrics["cox_clusters"] = {"value": fit.clusters}
    for name, hr in hazard_ratios(fit, frame).items():
        metrics[f"hr_{name}"] = hr
    ph = ph_test(frame, fit)
    violated = sorted(k for k, v in ph.items() if v["p"] < PH_ALPHA)
    metrics["ph_violations"] = {"value": len(violated), "tested": len(ph), "names": violated}

    # A few employers post hundreds of vacancies; the same model without the largest ones shows
    # which ratios are theirs rather than the group's.
    top = frame["cluster"].value_counts().head(TOP_EMPLOYERS).index
    rest = frame[~frame["cluster"].isin(top)].reset_index(drop=True)
    metrics[f"cox_subjects_without_top{TOP_EMPLOYERS}_employers"] = _share(len(rest), len(frame))
    for name, hr in hazard_ratios(fit_cox(rest), rest).items():
        metrics[f"hr_without_top{TOP_EMPLOYERS}_employers_{name}"] = hr

    piecewise: dict[str, Any] = {}
    if violated:
        split = split_at(frame, SPLIT_DAY, violated)
        piecewise = piecewise_ratios(fit_cox(split), split, violated, SPLIT_DAY)
        for name, row in piecewise.items():
            metrics[f"hr_piecewise_{name}"] = row

    for name in COVARIATES:
        plot_km(subjects, name, FIGURES / f"survival_{name}.png")

    return {
        "metrics": metrics,
        "ph_test": ph,
        "piecewise_split_day": SPLIT_DAY if violated else None,
        "grace": grace,
        "window": {
            "first_dt": str(window.days[0]),
            "last_dt": str(window.days[-1]),
            "holes": [[str(a), str(b)] for a, b in window.holes],
            "collected_days_after_window": window.days_after,
        },
    }


DEFINITIONS = {
    "subject": "dedup entity (vacancy_entity_id, else the portal id) with an open trudvsem snapshot sighting, "
    "IT roles only (marts role_bucket not not_it)",
    "event": f"no open sighting on {GRACE_DAYS} consecutive collected snapshot days; closed on the first of them",
    "time_origin": "creation date at the source (published_at, earliest over the entity's records)",
    "entry": "first sighting (left truncation): prevalent vacancies enter at their age on the first day",
    "right_censoring": "closure not confirmed by the end of the window; censored the day after the last sighting",
    "late_return": "back after the grace window: closed on its first missing day, the return is not followed",
    "window": f"collected complete days up to the first step longer than {MAX_STEP_DAYS} days between them",
    "km_ci": "Greenwood, exponential (lifelines), 95%",
    "cox": "lifelines CoxPHFitter with delayed entry (Efron ties); sandwich covariance clustered by employer "
    "(INN, else the entity) from score residuals with delayed entry, computed in this module",
    "ph_test": f"scaled Schoenfeld residuals against rank of time; violated at p < {PH_ALPHA}",
}

LIMITS_RU = [
    "Закрытие здесь это исчезновение вакансии из ежедневной выгрузки портала, а не найм. Вакансию снимают "
    "и по другим причинам, и по выгрузке их не различить.",
    "Во всех группах закрытия резко учащаются на 30-32 день с даты создания (ступенька на графиках). "
    "Похоже на срок публикации портала, но правилами портала это не проверено. Медиана около месяца "
    "во многих группах попадает на эту ступеньку и говорит скорее о том, продлевают ли вакансию, "
    "чем о скорости найма.",
    "Только trudvsem (портал службы занятости). Большинство вакансий в выборке из регионов вне Москвы "
    "и Петербурга; насколько состав работодателей и ролей отличается от hh.ru и Хабра, не измерялось, "
    "поэтому на рынок в целом выводы не переносятся. Хабр и срез API собираются с 08.10 и в анализ "
    "пока не входят.",
    "IT-фильтр снимка широкий, роль определяется правилом по названию, вакансии вне IT исключены. "
    "Вакансия, которая после смены названия вышла из фильтра, читается как закрытая.",
    "Дата создания у портала не меняется и не бывает позже первого появления, но у части новых "
    "вакансий она раньше первого появления в выгрузке больше чем на неделю, причина не установлена. "
    "Поэтому рядом приведена когорта новых вакансий со временем от первого появления.",
    "Несколько работодателей публикуют сотни вакансий. Интервалы отношений рисков учитывают это "
    "(кластеры по работодателю), у каждого отношения указано, сколько вакансий группы дают три её "
    "крупнейших работодателя, и рядом та же модель без трёх крупнейших работодателей выборки.",
    "Признаки взяты из последней записи вакансии. Отношения рисков описывают связь, а не влияние: "
    "роль, регион, опыт и зарплата связаны с работодателем и между собой.",
    "Окно наблюдения заканчивается на первом пропуске в календаре снимков длиннее одного дня. "
    "Пока бэкфилл не закрыл пропуск, дни после него не используются.",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.survival", description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="models/survival.json", help="path under results/")
    args = ap.parse_args(argv)
    load_dotenv()
    sightings, calendar, entities = load()
    doc = analyse(sightings, calendar, entities)
    path = write_result(
        args.out,
        doc.pop("metrics"),
        source_script="src/radar/models/survival.py",
        definitions=DEFINITIONS,
        limits_ru=LIMITS_RU,
        figures=[f"models/figures/survival_{name}.png" for name in COVARIATES],
        **doc,
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
