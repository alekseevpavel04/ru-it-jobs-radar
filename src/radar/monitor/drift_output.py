"""Weekly drift of the extraction model's output per source, with Evidently.

The units and windows are those of radar.monitor.drift: a vacancy a source shows for the first time,
duplicates left out, the source's first four complete weeks against its latest complete week. What
is compared is not the source's own fields but what the student (the model_id the marts read,
dbt var extraction_model) extracted from the unit's latest version, from staging.int_vacancy_fields.

Features. Categorical: valid (the answer passed the schema), role, grade, work_format,
employment_type, lang, has_salary (a bound stated), has_experience (a bound stated); a field the
student left null is the category "null", so a model that stops seeing grades moves the
distribution. Numerical: number of skills and of cities. Fields are those of valid answers; an
invalid answer counts in `valid` only. Methods and threshold are drift.py's (normed Wasserstein for
numbers, Jensen-Shannon for categories, drifted at 0.1), with the same reference spread beside them.

Coverage. Units without an extracted version (Telegram posts the corpus rules drop, vacancies
collected after the corpus run) are counted, not compared. When the current week's share of
extracted units falls more than COVERAGE_GAP below the reference's, the extraction has not caught
up with the week and the drift is not computed (status extraction_incomplete): the extracted part
of such a week is its first days, not a sample of it.

Output: results/monitor/drift_output.json (aggregates only) and an Evidently HTML report per source
in reports/evidently/<current week>/output_<source>.html (local).

    python -m radar.monitor.drift_output
    python -m radar.monitor.drift_output --sources telegram --no-html
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from radar.config import ROOT, load_dotenv
from radar.monitor.drift import (
    BY_NAME,
    CAT_METHOD,
    DRIFT_THRESHOLD,
    MIN_REFERENCE_WEEKS,
    NUM_METHOD,
    REFERENCE_WEEKS,
    REPORTS,
    ClickHouseReader,
    Extent,
    Source,
    Windows,
    drift_summary,
    reference_spread,
    run_report,
    share,
    source_windows,
    usable_columns,
)
from radar.results import write_result

log = logging.getLogger("radar.monitor.drift_output")

RESULT = "monitor/drift_output.json"
DBT_PROJECT = ROOT / "dbt" / "dbt_project.yml"
COVERAGE_GAP = 0.05

FIELD_CATEGORICAL = ("role", "grade", "work_format", "employment_type", "lang", "has_salary", "has_experience")
CATEGORICAL = ("valid", *FIELD_CATEGORICAL)
NUMERICAL = ("n_skills", "n_cities")
NULL = "null"


class Reader(Extent, Protocol):
    def outputs(self, source: Source, first: date, last: date) -> pd.DataFrame:
        """One row per new vacancy of the period: week, is_duplicate, source_id, content_hash,
        is_extracted, valid, the fields of FIELD_CATEGORICAL (null where not stated or not
        extracted) and n_skills, n_cities."""


def extraction_model(path: Path = DBT_PROJECT) -> str:
    model = str(yaml.safe_load(path.read_text(encoding="utf-8"))["vars"]["extraction_model"])
    if not model:
        raise SystemExit(f"{path}: vars.extraction_model is empty, the marts read no extracted fields")
    return model


def output_frame(units: pd.DataFrame) -> pd.DataFrame:
    """Extracted units as Evidently reads them: the fields of a valid answer as strings with "null"
    for a field not stated, an invalid answer with every field missing but `valid`."""
    extracted = units[units["is_extracted"].astype(bool)]
    valid = extracted["valid"].astype(bool).to_numpy()
    frame = pd.DataFrame({"valid": ["true" if v else "false" for v in valid]})
    for column in FIELD_CATEGORICAL:
        values: list[str | None] = []
        for ok, value in zip(valid, extracted[column], strict=True):
            if not ok:
                values.append(None)
            elif isinstance(value, bool | np.bool_):
                values.append("true" if value else "false")
            else:
                values.append(NULL if value is None or pd.isna(value) else str(value))
        frame[column] = values
    for column in NUMERICAL:
        frame[column] = pd.to_numeric(extracted[column], errors="raise").astype(float).where(valid).to_numpy()
    return frame


def coverage(units: pd.DataFrame) -> dict[str, Any]:
    return share(int(units["is_extracted"].astype(bool).sum()), len(units))


def analyse_source(
    source: Source, reader: Reader, reference_weeks: int = REFERENCE_WEEKS, html_dir: Path | None = REPORTS
) -> dict[str, Any]:
    windows = source_windows(source, reader, reference_weeks)
    if not isinstance(windows, Windows):
        return windows
    out: dict[str, Any] = {
        **windows.base,
        "reference_weeks": [w.isoformat() for w in windows.reference],
        "current_week": windows.current.isoformat(),
    }
    frame = reader.outputs(source, windows.reference[0], windows.last_day)
    units = frame[~frame["is_duplicate"].astype(bool)]
    ref_units = units[units["week"].isin(windows.reference)].reset_index(drop=True)
    cur_units = units[units["week"] == windows.current].reset_index(drop=True)
    out["coverage"] = {"reference": coverage(ref_units), "current": coverage(cur_units)}
    if out["coverage"]["current"]["value"] < out["coverage"]["reference"]["value"] - COVERAGE_GAP:
        return {"status": "extraction_incomplete", **out, "coverage_gap": COVERAGE_GAP}

    ref, cur = output_frame(ref_units), output_frame(cur_units)
    out["extracted"] = {"reference": len(ref), "current": len(cur)}
    if ref.empty or cur.empty:
        return {"status": "empty_window", **out}
    out["valid_json"] = {
        "reference": share(int((ref["valid"] == "true").sum()), len(ref)),
        "current": share(int((cur["valid"] == "true").sum()), len(cur)),
    }
    columns = usable_columns(ref, cur, [*NUMERICAL, *CATEGORICAL])
    categorical = [c for c in CATEGORICAL if c in columns]
    numerical = [c for c in NUMERICAL if c in columns]
    snapshot = run_report(ref, cur, categorical, numerical)
    ref_weeks = ref_units.loc[ref_units["is_extracted"].astype(bool), "week"].reset_index(drop=True)
    spread = reference_spread(ref, ref_weeks, categorical, numerical)
    out.update(
        status="ok",
        not_available=[c for c in (*NUMERICAL, *CATEGORICAL) if c not in columns],
        drift=drift_summary(snapshot, categorical, spread),
    )
    if html_dir is not None:
        path = html_dir / windows.current.isoformat() / f"output_{source.name}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))
        out["report_html"] = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return out


def metrics_of(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for name, res in results.items():
        if "coverage" in res:
            metrics[f"{name}.coverage_current"] = res["coverage"]["current"]
        if res["status"] != "ok":
            continue
        metrics[f"{name}.drifted_outputs"] = res["drift"]["drifted_features"]
        metrics[f"{name}.valid_json_current"] = res["valid_json"]["current"]
    return metrics


# The view joins with join_use_nulls = 0 (dbt's session), and a query reading it must say the same;
# an unmatched unit then has is_extracted = false and NULL fields.
_RECORDS_SQL = """
WITH units AS (
    SELECT
        source_id, first_seen,
        row_number() OVER (PARTITION BY entity_id ORDER BY first_seen, source_id) > 1 AS is_duplicate
    FROM staging.int_vacancy_records
    WHERE source = {source:String} AND first_seen IS NOT NULL
),
fields AS (
    SELECT * FROM staging.int_vacancy_fields WHERE source = {source:String}
)
SELECT
    toMonday(u.first_seen) AS week,
    u.is_duplicate AS is_duplicate,
    u.source_id AS source_id,
    %(FIELDS)s
FROM units AS u
LEFT JOIN fields AS f ON u.source_id = f.source_id
WHERE u.first_seen BETWEEN {first:Date} AND {last:Date}
SETTINGS join_use_nulls = 0
"""

# Posts with several vacancies are not extracted (int_vacancy_fields leaves them out), so they are
# not units here.
_TELEGRAM_SQL = """
WITH fields AS (
    SELECT * FROM staging.int_vacancy_fields WHERE source = 'telegram'
)
SELECT
    t.published_week AS week,
    t.is_week_duplicate AS is_duplicate,
    t.source_id AS source_id,
    %(FIELDS)s
FROM staging.stg_telegram AS t
LEFT JOIN fields AS f ON t.source_id = f.source_id
WHERE NOT t.is_multi_vacancy AND t.published_week BETWEEN {first:Date} AND {last:Date}
SETTINGS join_use_nulls = 0
"""

_FIELDS = """f.content_hash AS content_hash,
    f.is_extracted AS is_extracted,
    f.valid AS valid,
    CAST(f.role AS Nullable(String)) AS role,
    CAST(f.grade AS Nullable(String)) AS grade,
    CAST(f.work_format AS Nullable(String)) AS work_format,
    CAST(f.employment_type AS Nullable(String)) AS employment_type,
    CAST(f.lang AS Nullable(String)) AS lang,
    f.salary_from IS NOT NULL OR f.salary_to IS NOT NULL AS has_salary,
    f.experience_min IS NOT NULL OR f.experience_max IS NOT NULL AS has_experience,
    length(f.skills) AS n_skills,
    length(f.cities) AS n_cities"""


class OutputReader(ClickHouseReader):
    def outputs(self, source: Source, first: date, last: date) -> pd.DataFrame:
        sql = (_TELEGRAM_SQL if source.kind == "telegram" else _RECORDS_SQL) % {"FIELDS": _FIELDS}
        frame = self._frame(sql, source=source.name, first=first.isoformat(), last=last.isoformat())
        frame["week"] = pd.to_datetime(frame["week"]).dt.date
        for column in ("is_duplicate", "is_extracted", "valid"):
            frame[column] = frame[column].astype(bool)
        return frame


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    for name, res in results.items():
        cov = res.get("coverage", {}).get("current")
        shown = f", extracted {cov['n']} of {cov['denominator']} current units" if cov else ""
        if res["status"] != "ok":
            print(f"{name:18} {res['status']}{shown}")
            continue
        drifted = res["drift"]["drifted_features"]
        names = ", ".join(f["feature"] for f in res["drift"]["features"] if f["drifted"]) or "none"
        valid = res["valid_json"]["current"]
        print(
            f"{name:18} week {res['current_week']}: drifted {drifted['n']} of {drifted['denominator']} ({names}); "
            f"valid JSON {valid['n']} of {valid['denominator']}{shown}"
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
    model = extraction_model()

    reader = OutputReader()
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
        source_script="src/radar/monitor/drift_output.py",
        extraction_model=model,
        settings={
            "reference_weeks": args.reference_weeks,
            "min_reference_weeks": MIN_REFERENCE_WEEKS,
            "num_method": NUM_METHOD,
            "cat_method": CAT_METHOD,
            "drift_threshold": DRIFT_THRESHOLD,
            "coverage_gap": COVERAGE_GAP,
        },
        sources=results,
    )
    print_summary(results)
    print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
