"""The dataset repo for Hugging Face, exported to storage/hf/dataset; nothing is uploaded.

    python -m radar.site.hf_dataset [--model-id ID]

What leaves the project follows DECISIONS 2 and 3 and the publication rules of docs/sources.md:

- trudvsem (API and daily snapshots of the open data set), one row per vacancy, its latest extracted
  version: title, text, link and the portal's own fields with the extracted fields. A vacancy the
  portal marks "Вакансия интернет ресурса" was reposted from another site: its title and text stay
  out, the fields stay. An employer who is a person (ИП, a 12-digit INN) has no name in the export.
- SuperJob: the extracted fields only, with the date. No text, no title, no vacancy id (an id leads
  back to the text), no link, no employer.
- Telegram: aggregates only, no row of any kind. Weekly counts over all channels per value of a
  field, the same dimensions and the same MIN_CELL as Habr. No channel dimension: per channel the
  floor dropped the field of almost half the posts, and a small channel's week came close to its posts.
- Habr Career: aggregates only, no row of any kind (DECISIONS 2 of 12.10). Monthly counts per role,
  grade, work format, employment type and skill; a cell below MIN_CELL vacancies is left out, so no
  cell describes a single vacancy.

The extracted fields come from extract.vacancy_fields for one model_id: the latest
extracted version of every vacancy or post. A row whose answer broke the schema stays with
`extraction_valid` false and empty fields. Every exported title and text is scanned for personal
data once more (radar.pii.text); a finding is replaced by its placeholder and counted.

Rows are split by time: one Parquet file per year of publication (the version's raw partition date
when the source gives none) under data/<source>/<YYYY>.parquet; trudvsem keeps vacancies open since
2015, so monthly files would be hundreds of tiny ones. The export is rerunnable:
each run rewrites the tree from what the table holds now, so it follows the corpus extraction while
that is still running. Writes the card (hf/dataset/README.md.in filled from results/) and
results/public/hf_dataset.json.
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.config import load_dotenv
from radar.pii.employer import is_person_employer
from radar.pii.text import clean_text
from radar.results import write_result
from radar.site.hf_common import HF, STORAGE, load_config, render_card, shown
from radar.stream.clickhouse import ClickHouse

log = logging.getLogger("radar.site.hf_dataset")

OUT = STORAGE / "dataset"
RESULT = "public/hf_dataset.json"
REPOSTED = "Вакансия интернет ресурса"
MIN_CELL = 5
TRUDVSEM_FEEDS = {"trudvsem": "staging.stg_trudvsem_api", "trudvsem_snapshot": "staging.stg_trudvsem_snapshot"}
ROW_SOURCES = ("trudvsem", "superjob")
AGGREGATE_DIMENSIONS = ("role", "grade", "work_format", "employment_type")

FIELD_COLUMNS = (
    "role",
    "grade",
    "skills",
    "salary_from",
    "salary_to",
    "salary_currency",
    "salary_gross_net",
    "salary_period",
    "work_format",
    "cities",
    "experience_min",
    "experience_max",
    "employment_type",
    "lang",
)


def schema(columns: list[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema(columns)


FIELDS_SCHEMA: list[tuple[str, pa.DataType]] = [
    ("extraction_valid", pa.bool_()),
    ("role", pa.string()),
    ("grade", pa.string()),
    ("skills", pa.list_(pa.string())),
    ("salary_from", pa.int64()),
    ("salary_to", pa.int64()),
    ("salary_currency", pa.string()),
    ("salary_gross_net", pa.string()),
    ("salary_period", pa.string()),
    ("work_format", pa.string()),
    ("cities", pa.list_(pa.string())),
    ("experience_min", pa.float64()),
    ("experience_max", pa.float64()),
    ("employment_type", pa.string()),
    ("lang", pa.string()),
]

SCHEMAS = {
    "trudvsem": schema(
        [
            ("vacancy_id", pa.string()),
            ("feed", pa.string()),
            ("published_date", pa.date32()),
            ("version_date", pa.date32()),
            ("url", pa.string()),
            ("title", pa.string()),
            ("text", pa.string()),
            ("text_withheld", pa.string()),
            ("source_type", pa.string()),
            ("region", pa.string()),
            ("employer_name", pa.string()),
            ("stated_salary_from", pa.float64()),
            ("stated_salary_to", pa.float64()),
            *FIELDS_SCHEMA,
        ]
    ),
    "superjob": schema([("published_date", pa.date32()), *FIELDS_SCHEMA]),
}

AGGREGATE_SCHEMA = schema(
    [
        ("month", pa.string()),
        ("dimension", pa.string()),
        ("value", pa.string()),
        ("vacancies", pa.int64()),
        ("vacancies_in_month", pa.int64()),
    ]
)

TELEGRAM_SCHEMA = schema(
    [
        ("week", pa.date32()),
        ("dimension", pa.string()),
        ("value", pa.string()),
        ("posts", pa.int64()),
        ("posts_in_week", pa.int64()),
    ]
)

_FIELDS_SQL = ", ".join(f"f.{c} AS {c}" for c in FIELD_COLUMNS) + ", f.valid AS extraction_valid"

# The latest extracted version of each source_id; dt orders versions, content_hash breaks a tie.
_LATEST = """
SELECT * FROM extract.vacancy_fields FINAL
WHERE model_id = {{model:String}} AND source = '{source}'
ORDER BY dt DESC, content_hash DESC
LIMIT 1 BY source_id
"""

_TRUDVSEM_SQL = """
SELECT f.source AS feed, f.source_id AS vacancy_id, toString(f.dt) AS version_day,
       toString(toDate(r.published_at)) AS published_day, r.url AS url, r.title AS title, r.text AS text,
       s.source_type AS source_type, s.region AS region, s.employer_name AS employer_name,
       s.employer_inn AS employer_inn, s.salary_from AS stated_salary_from, s.salary_to AS stated_salary_to,
       {fields}
FROM ({latest}) AS f
LEFT JOIN (
    SELECT source_id, dt, content_hash, published_at, url, title, text
    FROM raw.vacancies
    WHERE source = '{source}' AND (source_id, dt, content_hash) IN (
        SELECT source_id, dt, content_hash FROM ({latest})
    )
    ORDER BY fetched_at DESC
    LIMIT 1 BY source_id, dt, content_hash
) AS r ON r.source_id = f.source_id AND r.dt = f.dt AND r.content_hash = f.content_hash
LEFT JOIN (
    SELECT source_id, dt, content_hash, source_type, region, employer_name, employer_inn, salary_from, salary_to
    FROM {staging}
    WHERE (source_id, dt, content_hash) IN (SELECT source_id, dt, content_hash FROM ({latest}))
    LIMIT 1 BY source_id, dt, content_hash
) AS s ON s.source_id = f.source_id AND s.dt = f.dt AND s.content_hash = f.content_hash
FORMAT JSONEachRow
"""

# Fields and a publication date only; the id and the text never leave the query.
_FIELDS_ONLY_SQL = """
SELECT {channel} AS channel, toString(f.dt) AS version_day, toString(toDate(r.published_at)) AS published_day,
       {fields}
FROM ({latest}) AS f
LEFT JOIN (
    SELECT source_id, content_hash, min(published_at) AS published_at
    FROM raw.vacancies
    WHERE source = '{source}' AND (source_id, content_hash) IN (SELECT source_id, content_hash FROM ({latest}))
    GROUP BY source_id, content_hash
) AS r ON r.source_id = f.source_id AND r.content_hash = f.content_hash
ORDER BY published_day, version_day, cityHash64(f.source_id)
FORMAT JSONEachRow
"""


def json_lines(body: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in body.splitlines() if line]


def latest_sql(source: str) -> str:
    return _LATEST.format(source=source)


def fields_part(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"extraction_valid": bool(row["extraction_valid"])}
    for c in FIELD_COLUMNS:
        out[c] = row[c]
    for c in ("salary_from", "salary_to"):
        out[c] = int(out[c]) if out[c] is not None else None
    return out


def month_of(row: Mapping[str, Any]) -> str:
    day = row["published_day"] or row["version_day"]
    return str(day)[:7]


def year_of(month: str) -> str:
    return month[:4]


def day_or_none(value: str | None) -> datetime | None:
    if not value or value.startswith("1970-01-01"):
        return None
    return datetime.fromisoformat(value)


def scrub(text: str | None, found: Counter[str]) -> str | None:
    if text is None:
        return None
    cleaned, counts = clean_text(text)
    found.update(counts)
    return cleaned


def trudvsem_rows(ch: ClickHouse, model_id: str, found: Counter[str]) -> list[dict[str, Any]]:
    """One row per vacancy: the API and the snapshots carry the same vacancy id, the later version wins."""
    best: dict[str, dict[str, Any]] = {}
    for feed, staging in TRUDVSEM_FEEDS.items():
        latest = latest_sql(feed)
        sql = _TRUDVSEM_SQL.format(fields=_FIELDS_SQL, latest=latest, source=feed, staging=staging)
        for row in json_lines(ch.query(sql, model=model_id)):
            held = best.get(row["vacancy_id"])
            # Same day in both feeds: the API version, which the portal served directly.
            if held is None or (row["version_day"], row["feed"] == "trudvsem") > (
                held["version_day"],
                held["feed"] == "trudvsem",
            ):
                best[row["vacancy_id"]] = row
    rows = []
    for row in best.values():
        reposted = row["source_type"] == REPOSTED
        person = is_person_employer(row["employer_name"], row["employer_inn"])
        found["employer_person"] += person
        rows.append(
            {
                "vacancy_id": row["vacancy_id"],
                "feed": row["feed"],
                "published_date": day_or_none(row["published_day"]),
                "version_date": day_or_none(row["version_day"]),
                "url": row["url"],
                "title": None if reposted else scrub(row["title"], found),
                "text": None if reposted else scrub(row["text"], found),
                "text_withheld": "reposted_from_other_site" if reposted else None,
                "source_type": row["source_type"],
                "region": row["region"],
                "employer_name": None if person else row["employer_name"],
                "stated_salary_from": row["stated_salary_from"],
                "stated_salary_to": row["stated_salary_to"],
                **fields_part(row),
                "_month": month_of(row),
            }
        )
    return sorted(rows, key=lambda r: (r["_month"], r["vacancy_id"]))


def fields_only_rows(ch: ClickHouse, model_id: str, source: str) -> list[dict[str, Any]]:
    channel = "splitByChar('/', f.source_id)[1]" if source == "telegram" else "NULL"
    sql = _FIELDS_ONLY_SQL.format(channel=channel, fields=_FIELDS_SQL, latest=latest_sql(source), source=source)
    rows = []
    for row in json_lines(ch.query(sql, model=model_id)):
        out: dict[str, Any] = {"published_date": day_or_none(row["published_day"])}
        if source == "telegram":
            out["channel"] = row["channel"]
        day = out["published_date"] or day_or_none(row["version_day"])
        rows.append({**out, **fields_part(row), "_month": month_of(row), "_day": day})
    return rows


def field_cells(
    rows: Sequence[Mapping[str, Any]], group: Callable[[Mapping[str, Any]], tuple[Any, ...]]
) -> tuple[Counter[tuple[Any, ...]], Counter[tuple[Any, ...]]]:
    """(rows per group, rows per group and value of a field); a row that broke the schema counts in the group only."""
    per_group: Counter[tuple[Any, ...]] = Counter()
    cells: Counter[tuple[Any, ...]] = Counter()
    for r in rows:
        key = group(r)
        per_group[key] += 1
        if not r["extraction_valid"]:
            continue
        for dim in AGGREGATE_DIMENSIONS:
            cells[(*key, dim, r[dim] or "null")] += 1
        for skill in set(r["skills"]):
            cells[(*key, "skill", skill)] += 1
    return per_group, cells


def habr_aggregates(ch: ClickHouse, model_id: str) -> tuple[list[dict[str, Any]], int, int]:
    """Monthly counts of Habr vacancies per value of a field; cells under MIN_CELL are dropped."""
    rows = fields_only_rows(ch, model_id, "habr")
    per_month, cells = field_cells(rows, lambda r: (r["_month"],))
    out = [
        {"month": m, "dimension": d, "value": v, "vacancies": n, "vacancies_in_month": per_month[(m,)]}
        for (m, d, v), n in sorted(cells.items())
        if n >= MIN_CELL
    ]
    return out, sum(1 for n in cells.values() if n < MIN_CELL), len(rows)


def week_of(row: Mapping[str, Any]) -> date:
    """Monday of the week of publication, of the raw version when the source gives no date."""
    day: date = row["_day"].date()
    return day - timedelta(days=day.weekday())


def telegram_aggregates(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Weekly counts of posts over all channels per value of a field; cells under MIN_CELL are dropped."""
    per_week, cells = field_cells(rows, lambda r: (week_of(r),))
    out = [
        {"week": w, "dimension": d, "value": v, "posts": n, "posts_in_week": per_week[(w,)]}
        for (w, d, v), n in sorted(cells.items())
        if n >= MIN_CELL
    ]
    return out, sum(1 for n in cells.values() if n < MIN_CELL)


def role_kept(rows: Sequence[Mapping[str, Any]], group: Callable[[Mapping[str, Any]], tuple[Any, ...]]) -> int:
    """Posts whose role lands in a cell of at least MIN_CELL under this grouping."""
    _, cells = field_cells(rows, group)
    return sum(n for key, n in cells.items() if key[-2] == "role" and n >= MIN_CELL)


def write_parts(source: str, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    folder = OUT / "data" / source
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    schema = SCHEMAS[source]
    by_year: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_year.setdefault(year_of(r["_month"]), []).append(r)
    files = []
    for year, part in sorted(by_year.items()):
        table = pa.Table.from_pylist([{k: r[k] for k in schema.names} for r in part], schema=schema)
        path = folder / f"{year}.parquet"
        pq.write_table(table, path, compression="zstd")
        files.append(path)
    return files


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def date_range(rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    months = sorted({r["_month"] for r in rows})
    return months[0], months[-1]


def export(model_id: str) -> dict[str, Any]:
    load_dotenv()
    ch = ClickHouse()
    found: Counter[str] = Counter()
    rows = {
        "trudvsem": trudvsem_rows(ch, model_id, found),
        "superjob": fields_only_rows(ch, model_id, "superjob"),
    }
    files = {source: write_parts(source, rows[source]) for source in ROW_SOURCES}
    habr, habr_suppressed, habr_vacancies = habr_aggregates(ch, model_id)
    agg_dir = OUT / "aggregates"
    agg_dir.mkdir(parents=True, exist_ok=True)
    habr_file = agg_dir / "habr_monthly.parquet"
    pq.write_table(pa.Table.from_pylist(habr, schema=AGGREGATE_SCHEMA), habr_file, compression="zstd")
    posts = fields_only_rows(ch, model_id, "telegram")
    telegram, telegram_suppressed = telegram_aggregates(posts)
    telegram_file = agg_dir / "telegram_weekly.parquet"
    pq.write_table(pa.Table.from_pylist(telegram, schema=TELEGRAM_SCHEMA), telegram_file, compression="zstd")
    stale = OUT / "data" / "telegram"
    if stale.exists():
        shutil.rmtree(stale)

    tv = rows["trudvsem"]
    metrics: dict[str, dict[str, Any]] = {}
    for source in ROW_SOURCES:
        rs = rows[source]
        valid = sum(r["extraction_valid"] for r in rs)
        metrics[f"{source}.rows"] = {"value": len(rs)}
        metrics[f"{source}.extraction_valid"] = share(valid, len(rs))
        metrics[f"{source}.files"] = {"value": len(files[source])}
        metrics[f"{source}.mib"] = {"value": round(sum(f.stat().st_size for f in files[source]) / 2**20, 1)}
        first, last = date_range(rs)
        metrics[f"{source}.first_month"] = {"value": first}
        metrics[f"{source}.last_month"] = {"value": last}
    with_text = sum(r["text"] is not None for r in tv)
    metrics["trudvsem.with_text"] = share(with_text, len(tv))
    metrics["trudvsem.text_withheld_reposted"] = share(sum(r["text_withheld"] is not None for r in tv), len(tv))
    # Not withheld, the portal sent no text: with_text and the reposted ones then add up to all rows.
    metrics["trudvsem.no_text_at_source"] = share(
        sum(r["text"] is None and r["text_withheld"] is None for r in tv), len(tv)
    )
    metrics["trudvsem.employer_name_removed_person"] = {"value": found.pop("employer_person", 0)}
    metrics["trudvsem.from_api_feed"] = share(sum(r["feed"] == "trudvsem" for r in tv), len(tv))
    metrics["habr.cells"] = {"value": len(habr)}
    metrics["habr.cells_dropped_below_min"] = {"value": habr_suppressed}
    metrics["habr.min_cell"] = {"value": MIN_CELL}
    metrics["habr.vacancies"] = {"value": habr_vacancies}
    metrics["telegram.posts"] = {"value": len(posts)}
    metrics["telegram.extraction_valid"] = share(sum(r["extraction_valid"] for r in posts), len(posts))
    metrics["telegram.channels"] = {"value": len({r["channel"] for r in posts})}
    metrics["telegram.cells"] = {"value": len(telegram)}
    metrics["telegram.cells_dropped_below_min"] = {"value": telegram_suppressed}
    # How much of the field survives the cell floor: a post's role is in at most one published cell.
    valid_posts = metrics["telegram.extraction_valid"]["n"]
    metrics["telegram.role_in_published_cells"] = share(
        sum(c["posts"] for c in telegram if c["dimension"] == "role"), valid_posts
    )
    # The same floor per channel and week, the layout these aggregates replaced; kept for the comparison.
    metrics["telegram.role_in_cells_per_channel_week"] = share(
        role_kept(posts, lambda r: (week_of(r), r["channel"])), valid_posts
    )
    weeks = sorted({week_of(r) for r in posts})
    metrics["telegram.first_week"] = {"value": weeks[0].isoformat()}
    metrics["telegram.last_week"] = {"value": weeks[-1].isoformat()}
    metrics["pii_replaced_on_export"] = {"value": sum(found.values()), "by_kind": dict(found)}
    metrics["rows_total"] = {"value": sum(len(rows[s]) for s in ROW_SOURCES)}

    template = (HF / "dataset" / "README.md.in").read_text(encoding="utf-8")
    extra = {
        "model_id": model_id,
        "files": {s: [shown(f) for f in fs] for s, fs in files.items()}
        | {"habr": [shown(habr_file)], "telegram": [shown(telegram_file)]},
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_result(RESULT, metrics, source_script="src/radar/site/hf_dataset.py", **extra)
    card = render_card(template)
    (OUT / "README.md").write_text(card.published, encoding="utf-8", newline="\n")
    shutil.copy2(HF / "dataset" / "LICENSE.md", OUT / "LICENSE.md")
    log.info(
        "dataset -> %s: trudvsem %d (with text %d), superjob %d, telegram cells %d of %d posts, habr cells %d; "
        "card %d numbers",
        shown(OUT),
        len(tv),
        with_text,
        len(rows["superjob"]),
        len(telegram),
        len(posts),
        len(habr),
        card.numbers,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.site.hf_dataset", description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", help="extract.vacancy_fields model_id; default [dataset] model_id of hf/hf.toml")
    args = ap.parse_args(argv)
    export(args.model_id or load_config()["dataset"]["model_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
