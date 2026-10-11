"""Aggregates of the marts for the public site, written to site/src/data (DECISIONS 2).

    python -m radar.site.export [--out site/src/data]

Reads only marts.* and writes only aggregates: counts per week, source, role bucket, skill, format
and salary cell, each with its denominator. No vacancy row of any source, no title, text or link
leaves ClickHouse here; every row model forbids fields it does not declare, and tests pin the field
lists. Rerunnable: the files are rewritten whole, rows in a fixed order, so a daily rerun changes
only what the data changed. Also writes results/site/export.json (rows and bytes per file).
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from radar.config import ROOT, load_dotenv
from radar.results import git_commit, write_result
from radar.stream.clickhouse import ClickHouse

OUT = ROOT / "site" / "src" / "data"
# Telegram history reaches back years, as far as each channel goes; the site plots two years of it.
TELEGRAM_WEEKS = 104
# A skill of a role bucket mentioned fewer times than this is left out of skills_by_role (noise and
# size); the per-source totals in skills.json keep every skill.
SKILL_MIN_MENTIONS = 3

VacancySource = Literal["trudvsem_snapshot", "trudvsem", "habr"]
Family = Literal["trudvsem", "habr", "telegram"]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WeeklyVacancies(Row):
    week: date
    source: VacancySource
    role_bucket: str
    days_collected: int
    complete_days: int
    first_collected_week: bool
    active: int
    new: int
    closed: int | None


class WeeklyTelegram(Row):
    week: date
    role_bucket: str
    posts: int
    posts_before_dedup: int
    channels: int


class SkillTotal(Row):
    source: Family
    skill: str
    category: str
    vacancies_mentioning: int
    vacancies_total: int


class SkillByRole(Row):
    source: Family
    role_bucket: str
    skill: str
    category: str
    vacancies_mentioning: int
    vacancies_in_bucket: int


class WorkFormat(Row):
    source: Family
    role_bucket: str
    work_format: Literal["office", "hybrid", "remote", "unknown"]
    vacancies: int
    vacancies_in_bucket: int


class Junior(Row):
    source: Family
    role_bucket: str
    vacancies: int
    open_to_junior: int
    graded_intern: int
    graded_junior: int
    no_experience_required: int
    grade_known: int
    experience_field_stated: int | None


class Salary(Row):
    source: Literal["trudvsem", "habr"]
    role_bucket: str
    grade: str
    region_group: str
    vacancies: int
    with_salary_rub: int
    with_salary_other_currency: int
    with_from: int
    with_to: int
    salary_from_p25: int | None
    salary_from_median: int | None
    salary_from_p75: int | None
    salary_to_p25: int | None
    salary_to_median: int | None
    salary_to_p75: int | None


class Period(Row):
    period_from: date
    period_to: date


class Meta(Row):
    exported_at: datetime
    commit: str
    data_through: date
    periods: dict[str, Period]
    snapshot_first_week: date
    snapshot_last_week: date
    telegram_first_week_shown: date
    telegram_weeks_shown: int
    skill_min_mentions: int
    files: dict[str, int]


MART_COLUMNS: dict[str, tuple[str, ...]] = {
    "mart_weekly_dynamics": (
        "week",
        "source",
        "role_bucket",
        "days_collected",
        "complete_days",
        "is_first_collected_week",
        "vacancies_active",
        "vacancies_new",
        "vacancies_closed",
        "telegram_posts",
        "telegram_posts_before_dedup",
        "telegram_channels",
    ),
    "mart_skills": (
        "source",
        "role_bucket",
        "grade",
        "skill",
        "skill_category",
        "vacancies_mentioning",
        "period_from",
        "period_to",
    ),
    "mart_work_format": ("source", "role_bucket", "work_format", "vacancies", "vacancies_in_bucket"),
    "mart_junior": tuple(Junior.model_fields),
    "mart_salaries": tuple(Salary.model_fields),
}

MartRows = Mapping[str, list[dict[str, Any]]]


def fetch_marts(ch: ClickHouse) -> dict[str, list[dict[str, Any]]]:
    marts = {}
    for table, columns in MART_COLUMNS.items():
        body = ch.query(
            f"SELECT {', '.join(columns)} FROM marts.{table} "
            "SETTINGS output_format_json_quote_64bit_integers = 0 FORMAT JSONEachRow"
        )
        marts[table] = [json.loads(line) for line in body.splitlines() if line]
    return marts


def weekly_vacancies(rows: Iterable[dict[str, Any]]) -> list[WeeklyVacancies]:
    out = [
        WeeklyVacancies(
            week=r["week"],
            source=r["source"],
            role_bucket=r["role_bucket"],
            days_collected=r["days_collected"],
            complete_days=r["complete_days"],
            first_collected_week=bool(r["is_first_collected_week"]),
            active=r["vacancies_active"],
            new=r["vacancies_new"],
            closed=r["vacancies_closed"],
        )
        for r in rows
        if r["source"] != "telegram"
    ]
    return sorted(out, key=lambda r: (r.source, r.week, r.role_bucket))


def weekly_telegram(rows: Iterable[dict[str, Any]], weeks: int = TELEGRAM_WEEKS) -> list[WeeklyTelegram]:
    posts = [r for r in rows if r["source"] == "telegram"]
    if not posts:
        return []
    last = max(date.fromisoformat(r["week"]) for r in posts)
    first = last - timedelta(weeks=weeks - 1)
    out = [
        WeeklyTelegram(
            week=r["week"],
            role_bucket=r["role_bucket"],
            posts=r["telegram_posts"],
            posts_before_dedup=r["telegram_posts_before_dedup"],
            channels=r["telegram_channels"],
        )
        for r in posts
        if date.fromisoformat(r["week"]) >= first
    ]
    return sorted(out, key=lambda r: (r.week, r.role_bucket))


def bucket_sizes(junior: Iterable[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Vacancies (Telegram: posts) per source and bucket, mentioning a skill or not.

    mart_skills carries the size of a bucket x grade cell only on rows of a mentioned skill, so a cell
    where nothing was mentioned would be missing from a sum over its rows; mart_junior counts the
    same units (int_vacancy_entities and the Telegram window, not_it left out) for every bucket.
    """
    return {(r["source"], r["role_bucket"]): r["vacancies"] for r in junior}


def skills(
    rows: Iterable[dict[str, Any]], sizes: Mapping[tuple[str, str], int], min_mentions: int = SKILL_MIN_MENTIONS
) -> tuple[list[SkillTotal], list[SkillByRole]]:
    by_role: dict[tuple[str, str, str], int] = defaultdict(int)
    category: dict[str, str] = {}
    for r in rows:
        by_role[r["source"], r["role_bucket"], r["skill"]] += r["vacancies_mentioning"]
        category[r["skill"]] = r["skill_category"]

    per_source: dict[tuple[str, str], int] = defaultdict(int)
    for (source, _, skill), n in by_role.items():
        per_source[source, skill] += n
    source_sizes: dict[str, int] = defaultdict(int)
    for (source, _), n in sizes.items():
        source_sizes[source] += n

    totals = [
        SkillTotal(
            source=source,
            skill=skill,
            category=category[skill],
            vacancies_mentioning=n,
            vacancies_total=source_sizes[source],
        )
        for (source, skill), n in per_source.items()
    ]
    roles = [
        SkillByRole(
            source=source,
            role_bucket=bucket,
            skill=skill,
            category=category[skill],
            vacancies_mentioning=n,
            vacancies_in_bucket=sizes[source, bucket],
        )
        for (source, bucket, skill), n in by_role.items()
        if n >= min_mentions
    ]
    checks = [(t.source, t.skill, t.vacancies_mentioning, t.vacancies_total) for t in totals]
    checks += [(r.source, r.skill, r.vacancies_mentioning, r.vacancies_in_bucket) for r in roles]
    for source, skill, n, denominator in checks:
        if n > denominator:
            raise ValueError(f"{source} {skill}: {n} mentions of {denominator}")
    totals.sort(key=lambda r: (r.source, -r.vacancies_mentioning, r.skill))
    roles.sort(key=lambda r: (r.source, r.role_bucket, -r.vacancies_mentioning, r.skill))
    return totals, roles


def periods(skill_rows: Iterable[dict[str, Any]]) -> dict[str, Period]:
    found: dict[str, Period] = {}
    for r in skill_rows:
        found.setdefault(r["source"], Period(period_from=r["period_from"], period_to=r["period_to"]))
    return dict(sorted(found.items()))


def build(marts: MartRows, exported_at: datetime, commit: str) -> dict[str, Any]:
    """File name -> rows (or Meta for meta.json), all from the mart rows given."""
    weekly = marts["mart_weekly_dynamics"]
    sizes = bucket_sizes(marts["mart_junior"])
    skill_totals, skill_roles = skills(marts["mart_skills"], sizes)
    files: dict[str, list[Any]] = {
        "weekly_vacancies.json": weekly_vacancies(weekly),
        "weekly_telegram.json": weekly_telegram(weekly),
        "skills.json": skill_totals,
        "skills_by_role.json": skill_roles,
        "work_format.json": sorted(
            (WorkFormat(**r) for r in marts["mart_work_format"]),
            key=lambda r: (r.source, r.role_bucket, r.work_format),
        ),
        "junior.json": sorted((Junior(**r) for r in marts["mart_junior"]), key=lambda r: (r.source, r.role_bucket)),
        "salaries.json": sorted(
            (Salary(**r) for r in marts["mart_salaries"]),
            key=lambda r: (r.source, r.role_bucket, r.grade, r.region_group),
        ),
    }
    snapshot_weeks = [r.week for r in files["weekly_vacancies.json"] if r.source == "trudvsem_snapshot"]
    telegram_weeks = [r.week for r in files["weekly_telegram.json"]]
    source_periods = periods(marts["mart_skills"])
    meta = Meta(
        exported_at=exported_at,
        commit=commit,
        data_through=max(p.period_to for p in source_periods.values()),
        periods=source_periods,
        snapshot_first_week=min(snapshot_weeks),
        snapshot_last_week=max(snapshot_weeks),
        telegram_first_week_shown=min(telegram_weeks),
        telegram_weeks_shown=len(set(telegram_weeks)),
        skill_min_mentions=SKILL_MIN_MENTIONS,
        files={name: len(rows) for name, rows in files.items()},
    )
    return {"meta.json": meta, **files}


def render(content: Any) -> str:
    """One row per line: small daily diffs in git and still valid JSON."""
    if isinstance(content, BaseModel):
        return json.dumps(content.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    lines = [json.dumps(r.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) for r in content]
    return "[\n" + ",\n".join(lines) + "\n]\n"


def write(docs: Mapping[str, Any], out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, content in docs.items():
        data = render(content).encode("utf-8")
        (out / name).write_bytes(data)
        written[name] = len(data)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    load_dotenv()
    with ClickHouse() as ch:
        marts = fetch_marts(ch)
    docs = build(marts, datetime.now(UTC).replace(microsecond=0), git_commit())
    sizes = write(docs, args.out)
    meta: Meta = docs["meta.json"]
    metrics: dict[str, dict[str, Any]] = {}
    for name, size in sizes.items():
        stem = name.removesuffix(".json")
        if name in meta.files:
            metrics[f"rows_{stem}"] = {"value": meta.files[name]}
        metrics[f"bytes_{stem}"] = {"value": size}
    metrics["data_through"] = {"value": meta.data_through.isoformat()}
    out = args.out.resolve()
    write_result(
        "site/export.json",
        metrics,
        source_script="src/radar/site/export.py",
        out=out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out.name,
    )
    for name, size in sizes.items():
        print(f"{name}\t{meta.files.get(name, '-')} rows\t{size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
