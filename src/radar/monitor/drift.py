"""Weekly drift and data quality of the incoming vacancies, per source, with Evidently.

What is compared. The unit is a vacancy a source shows for the first time: for trudvsem (snapshots
and API), Habr and SuperJob a record of staging.int_vacancy_records whose first_seen (the first day
the history saw it open) falls in the week, for Telegram a vacancy post published in the week. Each
vacancy counts once: a record whose dedup entity already has an earlier record of the same source
(a repost under a new id) and a Telegram post that repeats an earlier post of its week are
duplicates, which the quality block counts and the drift leaves out. Features are those the marts
use, as of the record's latest staged day: role bucket, grade, work format, region, salary (stated
RUB amount and the midpoint of the stated bounds), experience, dictionary skills found, text length.

Windows. Weeks run Monday to Sunday, and a week counts when the source collected from its Monday to
its Sunday. The reference is the source's first four complete weeks and stays fixed once they exist;
until then it is every complete week but the latest (two at least). The current window is the
latest complete week. Telegram's archive reaches back to 2016 through one channel, so its weeks
start with the project's collection (the first snapshot day), where the other sources start.

Drift is Evidently's per column: normed Wasserstein distance for numbers, Jensen-Shannon distance
for categories, both drifted at 0.1 and above. Fixed methods rather than Evidently's size-dependent
defaults (p-value tests below 1000 rows), so weeks of different volume are judged alike.

Data quality per day of the current week against the reference days: the row count (a day is
anomalous more than 3 robust standard deviations, 1.4826 * MAD, floored at 5% of the median, from
the reference median; a day without rows is missing), the share of IT candidates among the rows, and
IT rows without text, title or publication date. Null shares of the features come from the frames.

Reads: aggregated or one-row-per-vacancy queries on the staging database, with readonly = 2 on the
session. The read-only account of the agent (infra/clickhouse/agent) sees marts only, and the marts
have no per-vacancy rows, so this uses the main account restricted on the server side.

Output: the summary in results/monitor/drift.json (aggregates only, publishable) and an Evidently
HTML report per source in reports/evidently/<current week>/, which embeds histograms of the
features rather than rows and stays local for its size (about 4 MB each, mostly the bundled JS).

    python -m radar.monitor.drift
    python -m radar.monitor.drift --sources habr telegram --no-html

The extraction model's side, on the same units and windows: radar.monitor.drift_output (drift of
the student's fields) and radar.monitor.drift_quality (agreement with the teacher on a subsample).
"""

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report  # type: ignore[import-untyped]
from evidently.presets import DataDriftPreset, DataSummaryPreset  # type: ignore[import-untyped]

from radar.config import ROOT, load_dotenv
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse, ClickHouseError

log = logging.getLogger("radar.monitor.drift")

RESULT = "monitor/drift.json"
REPORTS = ROOT / "reports" / "evidently"

REFERENCE_WEEKS = 4
MIN_REFERENCE_WEEKS = 2
NUM_METHOD = "wasserstein"
CAT_METHOD = "jensenshannon"
DRIFT_THRESHOLD = 0.1
ANOMALY_Z = 3.0
SIGMA_FLOOR = 0.05
MAD_TO_SIGMA = 1.4826
# A feature whose null share grows by more than this between the windows is flagged.
NULL_JUMP = 0.05

# The server applies these to every query of the session; readonly = 2 refuses writes and DDL but
# still lets the session set the limits after it.
SESSION_SETTINGS = {
    "readonly": "2",
    "max_threads": "2",
    "max_memory_usage": "1500000000",
    "join_use_nulls": "1",
}

RECORD_CATEGORICAL = (
    "role_bucket",
    "grade",
    "work_format",
    "region_group",
    "region",
    "has_salary_rub",
    "no_experience",
)
RECORD_NUMERICAL = ("salary_mid_rub", "experience_min_years", "n_skills", "text_len")
TELEGRAM_CATEGORICAL = ("role_bucket", "grade", "work_format", "no_experience", "is_multi_vacancy", "channel_topic")
TELEGRAM_NUMERICAL = ("n_skills",)


@dataclass(frozen=True)
class Source:
    name: str
    kind: Literal["records", "telegram"]
    table: str
    # SQL of the IT flag over the staging table; None where the source has no such notion.
    is_it: str | None
    categorical: tuple[str, ...]
    numerical: tuple[str, ...]


SOURCES = (
    Source(
        "trudvsem_snapshot", "records", "stg_trudvsem_snapshot", "is_it_candidate", RECORD_CATEGORICAL, RECORD_NUMERICAL
    ),
    Source("trudvsem", "records", "stg_trudvsem_api", "is_it_candidate", RECORD_CATEGORICAL, RECORD_NUMERICAL),
    Source("habr", "records", "stg_habr", "true", RECORD_CATEGORICAL, RECORD_NUMERICAL),
    Source("superjob", "records", "stg_superjob", "is_it_candidate", RECORD_CATEGORICAL, RECORD_NUMERICAL),
    Source("telegram", "telegram", "stg_telegram", None, TELEGRAM_CATEGORICAL, TELEGRAM_NUMERICAL),
)
BY_NAME = {s.name: s for s in SOURCES}

# Daily rows of the IT part that miss a field; the denominator is it_rows.
MISSING_FIELDS = ("it_without_text", "it_without_title", "it_without_published_at")


class Extent(Protocol):
    def extent(self, source: Source) -> tuple[date, date] | None:
        """First and last day of the source's collection; None when it has no rows."""


class Reader(Extent, Protocol):
    def daily(self, source: Source, first: date, last: date) -> pd.DataFrame:
        """One row per day with rows: day, rows and, for records, it_rows and MISSING_FIELDS."""

    def inflow(self, source: Source, first: date, last: date) -> pd.DataFrame:
        """One row per new vacancy of the period: week (Monday), is_duplicate and the features."""


def monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def complete_weeks(first: date, last: date) -> list[date]:
    """Mondays of the weeks the collection covers from Monday to Sunday."""
    week = monday(first) if first.weekday() == 0 else monday(first) + timedelta(days=7)
    weeks = []
    while week + timedelta(days=6) <= last:
        weeks.append(week)
        week += timedelta(days=7)
    return weeks


def choose_windows(weeks: Sequence[date], reference_weeks: int = REFERENCE_WEEKS) -> tuple[list[date], date] | None:
    if len(weeks) < MIN_REFERENCE_WEEKS + 1:
        return None
    return list(weeks[: min(reference_weeks, len(weeks) - 1)]), weeks[-1]


def week_days(week: date) -> list[date]:
    return [week + timedelta(days=i) for i in range(7)]


def share(n: int, denominator: int) -> dict[str, Any]:
    return {"value": round(n / denominator, 4) if denominator else 0.0, "n": int(n), "denominator": int(denominator)}


def volume_anomalies(reference_rows: pd.Series, current: pd.Series, days: Sequence[date]) -> dict[str, Any]:
    """Days of the current week whose row count is far from the reference days, or that have no rows.

    reference_rows: rows per collected day of the reference; current: rows indexed by day.
    """
    median = float(reference_rows.median()) if len(reference_rows) else 0.0
    mad = float((reference_rows - median).abs().median()) if len(reference_rows) else 0.0
    sigma = max(MAD_TO_SIGMA * mad, SIGMA_FLOOR * median)
    flagged = []
    for day in days:
        rows = int(current.get(day, 0))
        if rows == 0:
            flagged.append({"day": day.isoformat(), "rows": 0, "kind": "missing"})
            continue
        # Without reference rows there is no expectation, only the missing days above.
        z = (rows - median) / sigma if sigma > 0 else 0.0
        if abs(z) > ANOMALY_Z:
            flagged.append({"day": day.isoformat(), "rows": rows, "kind": "low" if z < 0 else "high", "z": round(z, 2)})
    return {
        "reference_median_rows": median,
        "reference_sigma_rows": round(sigma, 1),
        "days": share(len(flagged), len(days)),
        "flagged": flagged,
    }


def quality_summary(
    source: Source, daily: pd.DataFrame, inflow: pd.DataFrame, reference: Sequence[date], current: date
) -> dict[str, Any]:
    reference_days = {d for w in reference for d in week_days(w)}
    current_days = week_days(current)
    by_day = daily.set_index("day")
    ref_daily = by_day[by_day.index.isin(reference_days)]
    cur_daily = by_day[by_day.index.isin(current_days)]

    out: dict[str, Any] = {
        "days_collected": {
            "reference": share(len(ref_daily), len(reference_days)),
            "current": share(len(cur_daily), len(current_days)),
        },
        "rows": {"reference": int(ref_daily["rows"].sum()), "current": int(cur_daily["rows"].sum())},
        "volume": volume_anomalies(ref_daily["rows"], cur_daily["rows"], current_days),
    }
    if source.is_it is not None:
        out["it_share"] = {
            "reference": share(int(ref_daily["it_rows"].sum()), int(ref_daily["rows"].sum())),
            "current": share(int(cur_daily["it_rows"].sum()), int(cur_daily["rows"].sum())),
        }
        for field in MISSING_FIELDS:
            out[field] = {
                "reference": share(int(ref_daily[field].sum()), int(ref_daily["it_rows"].sum())),
                "current": share(int(cur_daily[field].sum()), int(cur_daily["it_rows"].sum())),
            }

    in_ref, in_cur = inflow["week"].isin(reference), inflow["week"] == current
    out["new_vacancies"] = {
        "reference_per_week": round(int((in_ref & ~inflow["is_duplicate"]).sum()) / len(reference), 1),
        "current": int((in_cur & ~inflow["is_duplicate"]).sum()),
    }
    out["duplicates"] = {
        "reference": share(int((in_ref & inflow["is_duplicate"]).sum()), int(in_ref.sum())),
        "current": share(int((in_cur & inflow["is_duplicate"]).sum()), int(in_cur.sum())),
    }
    return out


def null_shares(reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        ref = share(int(reference[column].isna().sum()), len(reference))
        cur = share(int(current[column].isna().sum()), len(current))
        rows.append(
            {
                "feature": column,
                "reference": ref,
                "current": cur,
                "jump": cur["value"] - ref["value"] > NULL_JUMP,
            }
        )
    return rows


def category(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    return str(value)


def feature_frame(inflow: pd.DataFrame, categorical: Sequence[str], numerical: Sequence[str]) -> pd.DataFrame:
    """Categories as strings ("true"/"false" for flags), numbers as floats, nulls as NaN."""
    frame = pd.DataFrame(index=inflow.index)
    for column in categorical:
        frame[column] = inflow[column].map(category)
    for column in numerical:
        frame[column] = pd.to_numeric(inflow[column], errors="raise").astype(float)
    return frame.reset_index(drop=True)


def usable_columns(reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Columns with at least one value in both windows: a field the source does not have is all null."""
    return [c for c in columns if reference[c].notna().any() and current[c].notna().any()]


def run_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    categorical: Sequence[str],
    numerical: Sequence[str],
    summary: bool = True,
) -> Any:
    columns = [*numerical, *categorical]
    definition = DataDefinition(numerical_columns=list(numerical), categorical_columns=list(categorical))
    presets = [
        DataDriftPreset(columns=columns, num_method=NUM_METHOD, cat_method=CAT_METHOD, threshold=DRIFT_THRESHOLD)
    ]
    if summary:
        presets.append(DataSummaryPreset(columns=columns))
    return Report(presets).run(
        Dataset.from_pandas(current[columns], data_definition=definition),
        Dataset.from_pandas(reference[columns], data_definition=definition),
    )


def value_drifts(snapshot: Any) -> dict[str, dict[str, Any]]:
    """Method, score and threshold per column from the ValueDrift metrics of a report."""
    out = {}
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith(":ValueDrift"):
            out[config["column"]] = {
                "method": config["method"],
                "score": float(metric["value"]),
                "threshold": float(config["threshold"]),
            }
    return out


def reference_spread(
    reference: pd.DataFrame, weeks: pd.Series, categorical: Sequence[str], numerical: Sequence[str]
) -> dict[str, float]:
    """The largest score of one reference week against the other reference weeks, per column.

    A distance between two samples of one distribution is not zero, and with dozens of regions and
    a few hundred vacancies a week it can pass the threshold by itself. This is that floor as the
    data shows it, so a score of the current week is read against it as well as against the threshold.
    """
    spread: dict[str, float] = {}
    for week in sorted(set(weeks)):
        held = (weeks == week).to_numpy()
        one, rest = reference[held], reference[~held]
        columns = usable_columns(rest, one, [*numerical, *categorical])
        scores = value_drifts(
            run_report(
                rest,
                one,
                [c for c in categorical if c in columns],
                [c for c in numerical if c in columns],
                summary=False,
            )
        )
        for column, drift in scores.items():
            spread[column] = max(spread.get(column, 0.0), drift["score"])
    return spread


def drift_summary(snapshot: Any, categorical: Sequence[str], spread: dict[str, float]) -> dict[str, Any]:
    """Per-column scores and Evidently's count of drifted columns, from the report's metrics."""
    features = []
    for column, drift in value_drifts(snapshot).items():
        features.append(
            {
                "feature": column,
                "type": "categorical" if column in categorical else "numerical",
                "method": drift["method"],
                "score": round(drift["score"], 4),
                "threshold": drift["threshold"],
                "drifted": drift["score"] >= drift["threshold"],
                "reference_spread": round(spread[column], 4) if column in spread else None,
                "beyond_spread": column in spread and drift["score"] > spread[column],
            }
        )
    counted = next(
        (m["value"] for m in snapshot.dict()["metrics"] if m["config"]["type"].endswith(":DriftedColumnsCount")),
        None,
    )
    if counted is None:
        raise RuntimeError("the report has no DriftedColumnsCount")
    n_drifted = sum(f["drifted"] for f in features)
    # Both distances drift at score >= threshold; a mismatch means Evidently changed the rule.
    if n_drifted != int(counted["count"]):
        raise RuntimeError(f"{n_drifted} columns over threshold, Evidently counts {counted['count']}")
    return {
        "drifted_features": share(n_drifted, len(features)),
        "drifted_beyond_spread": share(sum(f["drifted"] and f["beyond_spread"] for f in features), len(features)),
        "features": features,
    }


@dataclass(frozen=True)
class Windows:
    reference: list[date]
    current: date
    base: dict[str, Any]

    @property
    def last_day(self) -> date:
        return self.current + timedelta(days=6)


def source_windows(source: Source, reader: Extent, reference_weeks: int = REFERENCE_WEEKS) -> Windows | dict[str, Any]:
    """The reference weeks and the current week of a source, or the status that says why there are none."""
    extent = reader.extent(source)
    if extent is None:
        return {"status": "no_data"}
    first, last = extent
    weeks = complete_weeks(first, last)
    base: dict[str, Any] = {"first_day": first.isoformat(), "last_day": last.isoformat(), "complete_weeks": len(weeks)}
    windows = choose_windows(weeks, reference_weeks)
    if windows is None:
        return {"status": "insufficient_history", **base, "needed_weeks": MIN_REFERENCE_WEEKS + 1}
    return Windows(windows[0], windows[1], base)


def analyse_source(
    source: Source, reader: Reader, reference_weeks: int = REFERENCE_WEEKS, html_dir: Path | None = REPORTS
) -> dict[str, Any]:
    windows = source_windows(source, reader, reference_weeks)
    if not isinstance(windows, Windows):
        return windows
    reference, current, base = windows.reference, windows.current, windows.base
    period_last = windows.last_day
    daily = reader.daily(source, reference[0], period_last)
    inflow = reader.inflow(source, reference[0], period_last)
    out: dict[str, Any] = {
        **base,
        "reference_weeks": [w.isoformat() for w in reference],
        "current_week": current.isoformat(),
        "quality": quality_summary(source, daily, inflow, reference, current),
    }

    units = inflow[~inflow["is_duplicate"]]
    ref_units = units[units["week"].isin(reference)]
    ref = feature_frame(ref_units, source.categorical, source.numerical)
    cur = feature_frame(units[units["week"] == current], source.categorical, source.numerical)
    out["units"] = {"reference": len(ref), "current": len(cur)}
    if ref.empty or cur.empty:
        return {"status": "empty_window", **out}

    columns = usable_columns(ref, cur, [*source.numerical, *source.categorical])
    categorical = [c for c in source.categorical if c in columns]
    numerical = [c for c in source.numerical if c in columns]
    out["quality"]["feature_nulls"] = null_shares(ref, cur, columns)
    snapshot = run_report(ref, cur, categorical, numerical)
    spread = reference_spread(ref, ref_units["week"].reset_index(drop=True), categorical, numerical)
    out.update(
        status="ok",
        not_available=[c for c in (*source.numerical, *source.categorical) if c not in columns],
        drift=drift_summary(snapshot, categorical, spread),
    )
    if html_dir is not None:
        path = html_dir / current.isoformat() / f"{source.name}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))
        out["report_html"] = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return out


def metrics_of(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for name, res in results.items():
        if res["status"] != "ok":
            continue
        quality = res["quality"]
        metrics[f"{name}.drifted_features"] = res["drift"]["drifted_features"]
        metrics[f"{name}.volume_anomaly_days"] = quality["volume"]["days"]
        metrics[f"{name}.duplicates_current"] = quality["duplicates"]["current"]
        metrics[f"{name}.new_vacancies_current"] = {"value": quality["new_vacancies"]["current"]}
    return metrics


class ClickHouseReader:
    def __init__(self) -> None:
        self.ch = ClickHouse()
        self.ch.client.params = self.ch.client.params.merge(SESSION_SETTINGS)

    def close(self) -> None:
        self.ch.close()

    def _frame(self, sql: str, **params: str) -> pd.DataFrame:
        response = self.ch.client.post(
            "/",
            content=f"{sql}\nFORMAT Parquet".encode(),
            params={f"param_{k}": v for k, v in params.items()},
        )
        if response.status_code != 200:
            raise ClickHouseError(response.text.strip()[:2000])
        return pd.read_parquet(BytesIO(response.content))

    def extent(self, source: Source) -> tuple[date, date] | None:
        if source.kind == "telegram":
            sql = (
                "SELECT (SELECT min(dt) FROM staging.stg_trudvsem_snapshot) AS first, max(dt) AS last, count() AS n "
                "FROM staging.stg_telegram"
            )
        else:
            sql = f"SELECT min(dt) AS first, max(dt) AS last, count() AS n FROM staging.{source.table}"
        row = self._frame(sql).iloc[0]
        if int(row["n"]) == 0:
            return None
        return pd.Timestamp(row["first"]).date(), pd.Timestamp(row["last"]).date()

    def daily(self, source: Source, first: date, last: date) -> pd.DataFrame:
        if source.kind == "telegram":
            sql = """
                SELECT toDate(published_at, 'Europe/Moscow') AS day, count() AS rows
                FROM staging.stg_telegram
                WHERE NOT is_week_duplicate AND day BETWEEN {first:Date} AND {last:Date}
                GROUP BY day
                ORDER BY day
            """
        else:
            it = source.is_it
            sql = f"""
                SELECT
                    dt AS day,
                    count() AS rows,
                    countIf({it}) AS it_rows,
                    countIf({it} AND NOT has_text) AS it_without_text,
                    countIf({it} AND coalesce(title, '') = '') AS it_without_title,
                    countIf({it} AND published_at IS NULL) AS it_without_published_at
                FROM staging.{source.table}
                WHERE dt BETWEEN {{first:Date}} AND {{last:Date}}
                GROUP BY dt
                ORDER BY dt
            """
        frame = self._frame(sql, first=first.isoformat(), last=last.isoformat())
        frame["day"] = pd.to_datetime(frame["day"]).dt.date
        return frame

    def inflow(self, source: Source, first: date, last: date) -> pd.DataFrame:
        if source.kind == "telegram":
            sql = """
                SELECT
                    published_week AS week,
                    is_week_duplicate AS is_duplicate,
                    role_bucket, grade, work_format,
                    no_experience_stated AS no_experience,
                    is_multi_vacancy, channel_topic,
                    length(skills) AS n_skills
                FROM staging.stg_telegram
                WHERE published_week BETWEEN {first:Date} AND {last:Date}
            """
            frame = self._frame(sql, first=first.isoformat(), last=last.isoformat())
        else:
            # first_seen comes from the history, the features from the record's latest staged day,
            # and the text length from the staging row of that day.
            sql = f"""
                WITH units AS (
                    SELECT
                        source_id, dt, first_seen, role_bucket, grade, work_format, region_group, region,
                        salary_from, salary_to, salary_currency_code, experience_min_years, no_experience, skills,
                        row_number() OVER (PARTITION BY entity_id ORDER BY first_seen, source_id) > 1 AS is_duplicate
                    FROM staging.int_vacancy_records
                    WHERE source = {{source:String}} AND first_seen IS NOT NULL
                ),
                win AS (
                    SELECT * FROM units WHERE first_seen BETWEEN {{first:Date}} AND {{last:Date}}
                ),
                texts AS (
                    SELECT source_id, dt, lengthUTF8(coalesce(text, '')) AS text_len
                    FROM staging.{source.table}
                    WHERE dt IN (SELECT DISTINCT dt FROM win) AND (source_id, dt) IN (SELECT source_id, dt FROM win)
                )
                SELECT
                    toMonday(w.first_seen) AS week,
                    w.is_duplicate AS is_duplicate,
                    w.role_bucket AS role_bucket,
                    w.grade AS grade,
                    w.work_format AS work_format,
                    w.region_group AS region_group,
                    w.region AS region,
                    coalesce(w.salary_currency_code = 'RUB', false)
                        AND (w.salary_from IS NOT NULL OR w.salary_to IS NOT NULL) AS has_salary_rub,
                    -- the middle of the stated bounds, or the one bound stated
                    if(
                        has_salary_rub,
                        (coalesce(w.salary_from, w.salary_to) + coalesce(w.salary_to, w.salary_from)) / 2,
                        NULL
                    ) AS salary_mid_rub,
                    w.experience_min_years AS experience_min_years,
                    w.no_experience AS no_experience,
                    length(w.skills) AS n_skills,
                    t.text_len AS text_len
                FROM win AS w
                LEFT JOIN texts AS t ON w.source_id = t.source_id AND w.dt = t.dt
            """
            frame = self._frame(sql, source=source.name, first=first.isoformat(), last=last.isoformat())
        frame["week"] = pd.to_datetime(frame["week"]).dt.date
        frame["is_duplicate"] = frame["is_duplicate"].astype(bool)
        return frame


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    for name, res in results.items():
        if res["status"] != "ok":
            print(f"{name:18} {res['status']} ({res.get('complete_weeks', 0)} complete weeks)")
            continue
        drifted = res["drift"]["drifted_features"]
        days = res["quality"]["volume"]["days"]
        names = ", ".join(f["feature"] for f in res["drift"]["features"] if f["drifted"]) or "none"
        beyond = res["drift"]["drifted_beyond_spread"]["n"]
        print(
            f"{name:18} week {res['current_week']}: drifted {drifted['n']} of {drifted['denominator']} ({names}), "
            f"{beyond} beyond the reference spread; "
            f"anomalous days {days['n']} of {days['denominator']}; "
            f"new {res['units']['current']} vs reference {res['units']['reference']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", nargs="+", choices=list(BY_NAME), default=list(BY_NAME))
    parser.add_argument("--reference-weeks", type=int, default=REFERENCE_WEEKS)
    parser.add_argument("--no-html", action="store_true", help="skip the Evidently HTML reports")
    args = parser.parse_args(argv)
    if args.reference_weeks < MIN_REFERENCE_WEEKS:
        parser.error(f"--reference-weeks must be at least {MIN_REFERENCE_WEEKS}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()

    reader = ClickHouseReader()
    try:
        results = {}
        for name in args.sources:
            log.info("source %s", name)
            results[name] = analyse_source(
                BY_NAME[name], reader, args.reference_weeks, None if args.no_html else REPORTS
            )
    finally:
        reader.close()

    path = write_result(
        RESULT,
        metrics_of(results),
        source_script="src/radar/monitor/drift.py",
        settings={
            "reference_weeks": args.reference_weeks,
            "min_reference_weeks": MIN_REFERENCE_WEEKS,
            "num_method": NUM_METHOD,
            "cat_method": CAT_METHOD,
            "drift_threshold": DRIFT_THRESHOLD,
            "anomaly_z": ANOMALY_Z,
            "sigma_floor": SIGMA_FLOOR,
            "null_jump": NULL_JUMP,
        },
        sources=results,
    )
    print_summary(results)
    print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
