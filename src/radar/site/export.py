"""Aggregates of the marts for the public site, written to site/src/data (DECISIONS 2).

    python -m radar.site.export [--out site/src/data]

Reads only marts.* and writes only aggregates: counts per week, source, role bucket, skill, format
and salary cell, each with its denominator. No vacancy row of any source, no title, text or link
leaves ClickHouse here; every row model forbids fields it does not declare, and tests pin the field
lists. Inputs besides the marts, each copied with its path and provenance:
results/models/survival.json (vacancy lifetimes) as medians with intervals and counts and
the survival curves; results/models/forecast.json (skill demand forecast), whose skills and
served method are refit here on the weekly skill shares of Telegram, so the forecast and its history
come from the same weeks; results/market/facts.json (the Telegram history by year),
results/models/skill_effects.json (skill and salary, double ML) and results/dbt/history.json
(closures that came back), as the counts and estimates these files hold. Rerunnable: the files are
rewritten whole, rows in a fixed order, so a daily rerun changes only what the data changed. Also
writes results/site/export.json (rows and bytes per file).

    python -m radar.site.export --skill-trends-only

rewrites skill_trends.json alone from results/market/skill_trends.json: the daily GitHub Actions run
(.github/workflows/autopilot.yml) has no ClickHouse, and the other files stay as last exported.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from radar.config import RESULTS, ROOT, load_dotenv
from radar.results import git_commit, write_result
from radar.stream.clickhouse import ClickHouse

# The forecast pulls in numpy, pandas, scikit-learn and statsmodels; the daily site run
# (--skill-trends-only) has none of them.
if TYPE_CHECKING:
    from radar.models.forecast import SiteForecast

OUT = ROOT / "site" / "src" / "data"
SURVIVAL = RESULTS / "models" / "survival.json"
FORECAST = RESULTS / "models" / "forecast.json"
MARKET_FACTS = RESULTS / "market" / "facts.json"
SKILL_EFFECTS = RESULTS / "models" / "skill_effects.json"
# The same estimate without one-bound vacancies and placeholder midpoints (radar.models.skill_effects --sensitivity)
SKILL_EFFECTS_SENSITIVITY = RESULTS / "models" / "skill_effects_sensitivity.json"
HISTORY = RESULTS / "dbt" / "history.json"
SKILL_TRENDS = RESULTS / "market" / "skill_trends.json"
FORECAST_SOURCE = "telegram"
FORECAST_TARGET = f"{FORECAST_SOURCE}.share"
JUNIOR_GRADES = ("intern", "junior")
# The site shows a salary quantile of a cell only when the cell is large enough for its median to
# say something; see salary_min_n.
MEDIAN_CI_LEVEL = 0.95
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
    channels_week: int


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


SalaryLevel = Literal[
    "cell",
    "bucket_grade",
    "bucket",
    "bucket_region",
    "group_grade",
    "group",
    "grade",
    "grade_region",
    "region",
    "source",
]


class Salary(Row):
    """One row of mart_salaries at one level; a dimension the level does not split by reads 'all'.

    Rows of one level cover every vacancy of the source once: counts add up within a level only, and
    quantiles of a group are the mart's own, never combined from cells.
    """

    source: Literal["trudvsem", "habr"]
    level: SalaryLevel
    role_group: str
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


class ExtractedSalary(Row):
    """A Telegram row of mart_extracted_salaries: salary, role and grade as the extraction model read
    them from the post. Posts have no salary fields of their own, so mart_salaries has no Telegram."""

    level: Literal["cell", "role", "grade", "source"]
    role: str
    grade: str
    vacancies: int
    with_salary: int
    with_salary_rub_month: int
    with_from: int
    with_to: int
    salary_from_p25: int | None
    salary_from_median: int | None
    salary_from_p75: int | None
    salary_to_p25: int | None
    salary_to_median: int | None
    salary_to_p75: int | None
    period_from: date
    period_to: date


class SkillJunior(Row):
    source: Family
    skill: str
    category: str
    vacancies_mentioning: int
    vacancies_graded_junior: int


class LifetimeMedian(Row):
    group: Literal["overall", "role", "experience", "salary", "region"]
    level: str
    cohort: Literal["prevalent", "incident"]
    median_days: float | None
    low: float | None
    high: float | None
    subjects: int
    events: int


class SurvivalPoint(Row):
    day: int
    survival: float
    at_risk: int


class SurvivalCurve(Row):
    group: Literal["role", "experience", "salary", "region"]
    level: str
    points: list[SurvivalPoint]


class Lifetime(Row):
    source_file: str
    source_script: str
    commit: str
    created_at: str
    window_from: date
    window_to: date
    event: str
    grace_days: int
    medians: list[LifetimeMedian]
    overall_curve: list[SurvivalPoint]
    group_curves: list[SurvivalCurve]
    limits: list[str]


class LifetimeObserved(Row):
    """mart_lifetime: durations of the closures observed in the collection window, not survival-corrected."""

    source: Literal["trudvsem_snapshot", "habr"]
    level: Literal["group", "source"]
    role_group: str
    vacancies: int
    still_open: int
    closed: int
    closed_in_gap: int
    censored: int
    left_censored: int
    closed_by_repost: int
    closed_in_quantiles: int
    duration_days_p25: float | None
    duration_days_median: float | None
    duration_days_p75: float | None
    duration_days_p90: float | None
    period_from: date
    period_to: date
    window_days: int


class ForecastHistory(Row):
    """Telegram posts of the week naming the skill, of the week's IT posts with fields: the forecast series."""

    week: date
    skill: str
    posts_mentioning: int
    posts_it: int


class ForecastPoint(Row):
    week: date
    skill: str
    horizon: int
    forecast: float
    low: float
    high: float


class Forecast(Row):
    """Skill demand forecast of the site; provenance is the result file that chose the method and skills."""

    source_file: str
    source_script: str
    commit: str
    created_at: str
    method: str
    series_first_week: date
    last_week: date
    weeks: int
    history_first_week: date
    horizon: int
    origins: int
    interval_quantile: float
    interval_check_min_origins: int
    interval_check_covered: int
    interval_check_cells: int
    points: list[ForecastPoint]
    limits: list[str]


class Provenance(Row):
    source_file: str
    source_script: str
    commit: str
    created_at: str


class TelegramYear(Row):
    """IT posts of a calendar year (of the week's Monday) and the ML, junior and graded posts among them."""

    scope: Literal["all", "panel"]
    year: int
    it_posts: int
    ml_posts: int
    junior_posts: int
    graded_posts: int
    ml_junior_posts: int
    ml_graded_posts: int


class TelegramSkillYear(Row):
    scope: Literal["all", "panel"]
    year: int
    skill: str
    posts_mentioning: int
    ml_posts: int


class TelegramFormatYear(Row):
    scope: Literal["it", "ml"]
    year: int
    group: Literal["junior", "other"]
    work_format: Literal["office", "hybrid", "remote"]
    posts: int
    posts_with_format: int


class TelegramYears(Provenance):
    """The Telegram history by year of results/market/facts.json: all channels, and the panel of
    channels with IT posts in every panel year, because the channel mix changes over the years."""

    last_week: date
    ml_roles: list[str]
    panel_years: list[int]
    panel_channels: int
    channels: int
    years: list[TelegramYear]
    skills: list[TelegramSkillYear]
    work_format: list[TelegramFormatYear]


class SkillEffect(Row):
    skill: str
    rows_with_skill: int
    rows_kept: int  # the comparable vacancies the estimate rests on: the overlap band, with and without
    rows_kept_with_skill: int
    pct: float
    pct_ci_low: float  # the false coverage rate interval at SkillEffects.fcr_level, not the pointwise one
    pct_ci_high: float
    p_holm: float
    q_bh: float
    # The sensitivity run: True when it confirms the effect (same sign, q below the rate), False when it does
    # not, None when it could not estimate the skill.
    robust: bool | None = None


class SkillEffects(Provenance):
    """Skill and posted salary of results/models/skill_effects.json: the double ML estimate at equal role,
    grade, region, experience and other skills, with its interval and the comparable vacancies behind it."""

    rows: int
    treatments: int
    estimated: int
    significant_holm: int
    significant_bh: int
    fdr: float
    fcr_level: float
    effects: list[SkillEffect]
    limits: list[str]


SkillGroup = str  # "<direction>_<grade>", see radar.site.skill_trends


class SkillGroupYear(Row):
    group: SkillGroup
    year: int | None  # None: the pooled recent years
    posts: int


class SkillShare(Row):
    group: SkillGroup
    skill: str
    year: int | None
    posts_mentioning: int
    posts: int


class SkillChange(Row):
    """Change of a skill's share since the base year, measured within the channels of both periods."""

    group: SkillGroup
    skill: str
    diff: float
    p: float
    q_bh: float  # Benjamini-Hochberg over the cell's skills; the site marks a change beyond noise below 0.05


class MixYear(Row):
    """A grade's or a direction's share in a year, at fixed channel weights over the panel channels."""

    year: int
    key: str
    share: float


class MixChange(Row):
    """Change of that share from the first year to the last, with its q over the grades or directions."""

    key: str
    diff: float
    q_bh: float


class Mix(Row):
    panel_channels: int
    channels: int
    panel_posts: int  # posts of the panel channels, of all_posts
    all_posts: int
    years: list[MixYear]
    changes: list[MixChange]


class GradedYear(Row):
    year: int
    graded: int  # IT posts of the year that name a grade
    total: int


class SkillTrends(Provenance):
    """Skills of Telegram posts by year and group from results/market/skill_trends.json; only the skills
    and years that pass the file's thresholds are there."""

    recent_years: list[int]
    base_year: int
    min_cell_posts: int
    min_mentions: int
    min_year_posts: int
    groups: list[SkillGroupYear]
    shares: list[SkillShare]  # the pooled recent years
    changes: list[SkillChange]
    grades: Mix
    graded: list[GradedYear]
    directions: Mix


class VacancyHistory(Provenance):
    """Closures of portal vacancies (snapshots) that were not final, from results/dbt/history.json."""

    vacancies_tracked: int
    closed: int
    reopened: int
    reposted: int


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
    salary_min_n: int
    salary_min_n_coverage: float
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
        "telegram_channels_week",
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
    "mart_lifetime": tuple(LifetimeObserved.model_fields),
    "mart_extracted_salaries": ("source", *ExtractedSalary.model_fields),
}

MartRows = Mapping[str, list[dict[str, Any]]]

# SuperJob is in the marts since 2026-10-12 with a single collected day: it joins the site once its
# weekly series and closures mean something.
SITE_EXCLUDED = frozenset({"superjob"})


def fetch_marts(ch: ClickHouse) -> dict[str, list[dict[str, Any]]]:
    marts = {}
    for table, columns in MART_COLUMNS.items():
        body = ch.query(
            f"SELECT {', '.join(columns)} FROM marts.{table} "
            "SETTINGS output_format_json_quote_64bit_integers = 0 FORMAT JSONEachRow"
        )
        rows = (json.loads(line) for line in body.splitlines() if line)
        marts[table] = [r for r in rows if not SITE_EXCLUDED.intersection({r.get("source"), r.get("family")})]
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
            channels_week=r["telegram_channels_week"],
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


def skills_junior(rows: Iterable[dict[str, Any]], junior: Iterable[dict[str, Any]]) -> list[SkillJunior]:
    """Skills of the vacancies graded intern or junior, per source.

    The base is graded_intern + graded_junior of mart_junior (the same units as mart_skills). The
    "no experience required" part of the junior slice has no skill rows: mart_skills is split by
    grade, not by experience.
    """
    base: dict[str, int] = defaultdict(int)
    for r in junior:
        base[r["source"]] += r["graded_intern"] + r["graded_junior"]
    mentions: dict[tuple[str, str], int] = defaultdict(int)
    category: dict[str, str] = {}
    for r in rows:
        if r["grade"] in JUNIOR_GRADES:
            mentions[r["source"], r["skill"]] += r["vacancies_mentioning"]
            category[r["skill"]] = r["skill_category"]
    out = []
    for (source, skill), n in mentions.items():
        if n > base[source]:
            raise ValueError(f"{source} {skill}: {n} mentions of {base[source]} junior-graded vacancies")
        out.append(
            SkillJunior(
                source=source,
                skill=skill,
                category=category[skill],
                vacancies_mentioning=n,
                vacancies_graded_junior=base[source],
            )
        )
    return sorted(out, key=lambda r: (r.source, -r.vacancies_mentioning, r.skill))


def median_ci_coverage(n: int) -> float:
    """Coverage of [x(2), x(n-1)] as an interval for the median of n values (binomial, p = 1/2)."""
    return 1 - (1 + n) / (1 << (n - 1))


def salary_min_n(level: float = MEDIAN_CI_LEVEL) -> int:
    """Smallest cell whose 95% distribution-free interval for the median drops the min and the max.

    Below it the only 95% interval the order statistics give is the whole range of the cell (and
    below 6 not even that), so its median says next to nothing about the cell.
    """
    n = 3
    while median_ci_coverage(n) < level:
        n += 1
    return n


_MEDIAN_KEY = re.compile(
    r"^median_days_(?P<incident>incident_)?(?P<group>role|experience|salary|region)_(?P<level>.+)$"
)
_GRACE = re.compile(r"no open sighting on (?P<days>\d+) consecutive")
_OVERALL = {"median_days_overall": "prevalent", "median_days_incident_from_first_sighting": "incident"}


def survival_points(metric: Mapping[str, Any]) -> list[SurvivalPoint]:
    points = [
        SurvivalPoint(day=int(day), survival=s, at_risk=metric["at_risk"][day]) for day, s in metric["survival"].items()
    ]
    return sorted(points, key=lambda p: p.day)


def lifetime(doc: Mapping[str, Any], source_file: str) -> Lifetime:
    """Kaplan-Meier medians of results/models/survival.json, as computed there, with their counts."""
    medians = []
    curves = []
    for key, m in doc["metrics"].items():
        found = _MEDIAN_KEY.match(key)
        if key in _OVERALL:
            group, level, cohort = "overall", "all", _OVERALL[key]
        elif found:
            group, level = found["group"], found["level"]
            cohort = "incident" if found["incident"] else "prevalent"
            # a group whose median is not estimable has no curve in the result
            if cohort == "prevalent" and "survival" in m:
                curves.append(SurvivalCurve(group=group, level=level, points=survival_points(m)))
        else:
            continue
        estimable = isinstance(m["value"], int | float)
        medians.append(
            LifetimeMedian(
                group=group,
                level=level,
                cohort=cohort,
                median_days=m["value"] if estimable else None,
                low=m.get("low") if estimable else None,
                high=m.get("high") if estimable else None,
                subjects=m["subjects"],
                events=m["events"],
            )
        )
    grace = _GRACE.search(doc["definitions"]["event"])
    if grace is None:
        raise ValueError(f"no grace days in the event definition: {doc['definitions']['event']}")
    return Lifetime(
        source_file=source_file,
        source_script=doc["source_script"],
        commit=doc["commit"],
        created_at=doc["created_at"],
        window_from=doc["window"]["first_dt"],
        window_to=doc["window"]["last_dt"],
        event=doc["definitions"]["event"],
        grace_days=int(grace["days"]),
        medians=sorted(medians, key=lambda m: (m.group, m.cohort, m.level)),
        overall_curve=survival_points(doc["metrics"]["median_days_overall"]),
        group_curves=sorted(curves, key=lambda c: (c.group, c.level)),
        limits=doc["limits_ru"],
    )


def forecast_files(
    result: Mapping[str, Any], site: "SiteForecast", source_file: str, weeks: int = TELEGRAM_WEEKS
) -> tuple[Forecast, list[ForecastHistory]]:
    """The forecast document and the last `weeks` weeks of its series, shares as counts with their base."""
    from radar.models import forecast

    shown = range(max(0, len(site.weeks) - weeks), len(site.weeks))
    history = [
        ForecastHistory(
            week=site.weeks[t],
            skill=skill,
            posts_mentioning=int(site.mentions[i, t]),
            posts_it=int(site.it_units[t]),
        )
        for i, skill in enumerate(site.skills)
        for t in shown
    ]
    for row in history:
        if row.posts_mentioning > row.posts_it:
            raise ValueError(f"{row.skill} {row.week}: {row.posts_mentioning} mentions of {row.posts_it} posts")
    points = [
        ForecastPoint(
            week=week,
            skill=skill,
            horizon=j + 1,
            forecast=round(float(site.point[i, j]), 4),
            low=round(float(site.low[i, j]), 4),
            high=round(float(site.high[i, j]), 4),
        )
        for i, skill in enumerate(site.skills)
        for j, week in enumerate(site.forecast_weeks)
    ]
    doc = Forecast(
        source_file=source_file,
        source_script=result["source_script"],
        commit=result["commit"],
        created_at=result["created_at"],
        method=site.method,
        series_first_week=site.weeks[0],
        last_week=site.weeks[-1],
        weeks=len(site.weeks),
        history_first_week=site.weeks[shown.start],
        horizon=len(site.forecast_weeks),
        origins=site.origins,
        interval_quantile=site.interval_quantile,
        interval_check_min_origins=forecast.INTERVAL_CHECK_MIN_ORIGINS,
        interval_check_covered=site.interval_covered,
        interval_check_cells=site.interval_cells,
        points=points,
        limits=result["limits_ru"],
    )
    return doc, sorted(history, key=lambda r: (r.skill, r.week))


def provenance(doc: Mapping[str, Any], source_file: str) -> dict[str, str]:
    return {
        "source_file": source_file,
        "source_script": doc["source_script"],
        "commit": doc["commit"],
        "created_at": doc["created_at"],
    }


_YEAR_KEY = re.compile(r"^(?P<prefix>telegram|telegram_panel)_year\.(?P<year>\d{4})\.(?P<name>\w+)$")
_SKILL_KEY = re.compile(r"^(?P<prefix>telegram|telegram_panel)_ml_skill\.(?P<year>\d{4})\.(?P<skill>.+)$")
_FORMAT_KEY = re.compile(
    r"^telegram(?P<ml>_ml)?_work_format\.(?P<year>\d{4})\.(?P<group>junior|other)\.(?P<format>office|hybrid|remote)$"
)


def telegram_years(doc: Mapping[str, Any], source_file: str) -> TelegramYears:
    """The counts behind the shares of results/market/facts.json, so every share keeps its base."""
    metrics = doc["metrics"]
    years: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    skill_rows: list[TelegramSkillYear] = []
    formats: list[TelegramFormatYear] = []
    for key, m in metrics.items():
        if found := _YEAR_KEY.match(key):
            row = years["panel" if found["prefix"] == "telegram_panel" else "all", int(found["year"])]
            name = found["name"]
            if name == "it_posts":
                row["it_posts"] = m["value"]
            elif name == "ml_posts":
                row["ml_posts"] = m["n"]
            elif name == "junior_of_graded":
                row["junior_posts"], row["graded_posts"] = m["n"], m["denominator"]
            elif name == "ml_junior_of_graded":
                row["ml_junior_posts"], row["ml_graded_posts"] = m["n"], m["denominator"]
        elif found := _SKILL_KEY.match(key):
            skill_rows.append(
                TelegramSkillYear(
                    scope="panel" if found["prefix"] == "telegram_panel" else "all",
                    year=int(found["year"]),
                    skill=found["skill"],
                    posts_mentioning=m["n"],
                    ml_posts=m["denominator"],
                )
            )
        elif found := _FORMAT_KEY.match(key):
            formats.append(
                TelegramFormatYear(
                    scope="ml" if found["ml"] else "it",
                    year=int(found["year"]),
                    group=found["group"],
                    work_format=found["format"],
                    posts=m["n"],
                    posts_with_format=m["denominator"],
                )
            )
    panel = metrics["telegram_panel.channels"]
    return TelegramYears(
        **provenance(doc, source_file),
        last_week=metrics["telegram_year.last_week"]["value"],
        ml_roles=doc["ml_roles"],
        panel_years=doc["panel_years"],
        panel_channels=panel["n"],
        channels=panel["denominator"],
        years=sorted(
            (TelegramYear(scope=scope, year=year, **row) for (scope, year), row in years.items()),
            key=lambda r: (r.scope, r.year),
        ),
        skills=sorted(skill_rows, key=lambda r: (r.scope, r.skill, r.year)),
        work_format=sorted(formats, key=lambda r: (r.scope, r.year, r.group, r.work_format)),
    )


def skill_effects(
    doc: Mapping[str, Any], source_file: str, sensitivity: Mapping[str, Any] | None = None
) -> SkillEffects:
    """The intervals are the false coverage rate ones of Benjamini and Yekutieli (2005) for the skills the
    Benjamini-Hochberg test selects: level 1 - R * alpha / m with R selected of m. At that level an interval
    leaves out zero exactly when the test selects the skill, so the band agrees with the block it sits in;
    the pointwise 95 % band of a skill that the test does not select can miss zero after 44 tries."""
    metrics = doc["metrics"]
    fdr = 1 - doc["scenario"]["level"]
    estimated = metrics["estimated"]["n"]
    selected = metrics["significant_bh"]["n"]
    level = 1 - max(selected, 1) * fdr / estimated
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)

    def pct(theta: float) -> float:
        return round(100 * (math.exp(theta) - 1), 1)

    checked = {m["skill"]: m for k, m in (sensitivity or {}).get("metrics", {}).items() if k.startswith("effect.")}

    def robust(m: Mapping[str, Any]) -> bool | None:
        other = checked.get(m["skill"])
        if other is None:
            return None
        return bool(other["q_bh"] < fdr and (other["value"] > 0) == (m["value"] > 0))

    effects = [
        SkillEffect(
            skill=m["skill"],
            rows_with_skill=m["rows_with_skill"],
            rows_kept=m["rows_kept"],
            rows_kept_with_skill=m["rows_kept_with_skill"],
            pct=m["pct"],
            pct_ci_low=pct(m["value"] - z * m["se"]),
            pct_ci_high=pct(m["value"] + z * m["se"]),
            p_holm=m["p_holm"],
            q_bh=m["q_bh"],
            robust=robust(m) if sensitivity else None,
        )
        for key, m in metrics.items()
        if key.startswith("effect.")
    ]
    if len(effects) != estimated:
        raise ValueError(f"{len(effects)} effects, the result reports {estimated} estimated")
    return SkillEffects(
        **provenance(doc, source_file),
        rows=metrics["data_rows"]["value"],
        treatments=metrics["treatments"]["value"],
        estimated=estimated,
        significant_holm=metrics["significant_holm"]["n"],
        significant_bh=selected,
        fdr=round(fdr, 4),
        fcr_level=round(level, 4),
        effects=sorted(effects, key=lambda e: (-e.rows_with_skill, e.skill)),
        limits=doc["limits_ru"],
    )


def vacancy_history(doc: Mapping[str, Any], source_file: str) -> VacancyHistory:
    metrics = doc["metrics"]
    source = "trudvsem_snapshot"
    return VacancyHistory(
        **provenance(doc, source_file),
        vacancies_tracked=metrics[f"vacancies_tracked_{source}"]["value"],
        closed=metrics[f"closed_{source}"]["n"],
        reopened=metrics[f"reopened_{source}"]["n"],
        reposted=metrics[f"reposts_{source}"]["n"],
    )


def mix(metrics: Mapping[str, Any], kind: str) -> Mix:
    years, changes = [], []
    for key, m in metrics.items():
        k, *rest = key.split(".")
        if k != kind or rest[0].startswith("panel") or rest[-1] == "graded":
            continue
        if rest[0] == "change":
            changes.append(MixChange(key=rest[1], diff=m["value"], q_bh=m["q_bh"]))
        else:
            years.append(MixYear(year=int(rest[0]), key=rest[1], share=m["value"]))
    channels, posts = metrics[f"{kind}.panel_channels"], metrics[f"{kind}.panel_posts"]
    return Mix(
        panel_channels=channels["n"],
        channels=channels["denominator"],
        panel_posts=posts["n"],
        all_posts=posts["denominator"],
        years=sorted(years, key=lambda r: (r.year, r.key)),
        changes=sorted(changes, key=lambda r: r.key),
    )


def skill_trends(doc: Mapping[str, Any], source_file: str) -> SkillTrends:
    groups, shares, changes, graded = [], [], [], []
    for key, m in doc["metrics"].items():
        kind, *rest = key.split(".")
        if kind in ("directions", "grades"):
            if kind == "grades" and rest[-1] == "graded":
                graded.append(GradedYear(year=int(rest[0]), graded=m["n"], total=m["denominator"]))
            continue
        if kind == "panel":
            continue
        if kind == "change":
            changes.append(
                SkillChange(group=rest[0], skill=".".join(rest[1:]), diff=m["value"], p=m["p"], q_bh=m["q_bh"])
            )
            continue
        year = None if rest[-1] == "recent" else int(rest[-1])
        if kind == "posts":
            groups.append(SkillGroupYear(group=rest[0], year=year, posts=m["value"]))
        elif kind == "share" and year is None:
            # the site shows the pooled recent years and the change; yearly shares stay in results/
            skill = ".".join(rest[1:-1])
            shares.append(
                SkillShare(
                    group=rest[0],
                    skill=skill,
                    year=year,
                    posts_mentioning=m["n"],
                    posts=m["denominator"],
                )
            )
    return SkillTrends(
        **provenance(doc, source_file),
        recent_years=doc["recent_years"],
        base_year=doc["base_year"],
        min_cell_posts=doc["min_cell_posts"],
        min_mentions=doc["min_mentions"],
        min_year_posts=doc["min_year_posts"],
        groups=sorted(groups, key=lambda r: (r.group, r.year or 0)),
        shares=sorted(shares, key=lambda r: (r.group, r.skill, r.year or 0)),
        changes=sorted(changes, key=lambda r: (r.group, r.skill)),
        grades=mix(doc["metrics"], "grades"),
        graded=sorted(graded, key=lambda r: r.year),
        directions=mix(doc["metrics"], "directions"),
    )


def periods(skill_rows: Iterable[dict[str, Any]]) -> dict[str, Period]:
    found: dict[str, Period] = {}
    for r in skill_rows:
        found.setdefault(r["source"], Period(period_from=r["period_from"], period_to=r["period_to"]))
    return dict(sorted(found.items()))


def build(
    marts: MartRows,
    survival: Mapping[str, Any],
    exported_at: datetime,
    commit: str,
    *,
    forecast_doc: Forecast,
    forecast_history: list[ForecastHistory],
    results: Mapping[Path, Mapping[str, Any]],
) -> dict[str, Any]:
    """File name -> rows, or one document for meta.json and each file built from results/.

    results maps MARKET_FACTS, SKILL_EFFECTS, HISTORY and SKILL_TRENDS to their parsed result files.
    """
    weekly = marts["mart_weekly_dynamics"]
    sizes = bucket_sizes(marts["mart_junior"])
    skill_totals, skill_roles = skills(marts["mart_skills"], sizes)
    files: dict[str, list[Any]] = {
        "weekly_vacancies.json": weekly_vacancies(weekly),
        "weekly_telegram.json": weekly_telegram(weekly),
        "skills.json": skill_totals,
        "skills_by_role.json": skill_roles,
        "skills_junior.json": skills_junior(marts["mart_skills"], marts["mart_junior"]),
        "work_format.json": sorted(
            (WorkFormat(**r) for r in marts["mart_work_format"]),
            key=lambda r: (r.source, r.role_bucket, r.work_format),
        ),
        "junior.json": sorted((Junior(**r) for r in marts["mart_junior"]), key=lambda r: (r.source, r.role_bucket)),
        "salaries.json": sorted(
            (Salary(**r) for r in marts["mart_salaries"]),
            key=lambda r: (r.source, r.level, r.role_group, r.role_bucket, r.grade, r.region_group),
        ),
        "salaries_telegram.json": sorted(
            (
                ExtractedSalary(**{k: v for k, v in r.items() if k != "source"})
                for r in marts["mart_extracted_salaries"]
                if r["source"] == "telegram"
            ),
            key=lambda r: (r.level, r.role, r.grade),
        ),
        "lifetime_observed.json": sorted(
            (LifetimeObserved(**r) for r in marts["mart_lifetime"]),
            key=lambda r: (r.source, r.level, r.role_group),
        ),
        "forecast_history.json": forecast_history,
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
        salary_min_n=salary_min_n(),
        salary_min_n_coverage=round(median_ci_coverage(salary_min_n()), 4),
        files={name: len(rows) for name, rows in files.items()},
    )
    rel = {
        path: path.relative_to(ROOT).as_posix()
        for path in (SURVIVAL, MARKET_FACTS, SKILL_EFFECTS, HISTORY, SKILL_TRENDS)
    }
    return {
        "meta.json": meta,
        "lifetime.json": lifetime(survival, rel[SURVIVAL]),
        "forecast.json": forecast_doc,
        "telegram_years.json": telegram_years(results[MARKET_FACTS], rel[MARKET_FACTS]),
        "skill_effects.json": skill_effects(
            results[SKILL_EFFECTS], rel[SKILL_EFFECTS], results.get(SKILL_EFFECTS_SENSITIVITY)
        ),
        "vacancy_history.json": vacancy_history(results[HISTORY], rel[HISTORY]),
        "skill_trends.json": skill_trends(results[SKILL_TRENDS], rel[SKILL_TRENDS]),
        **files,
    }


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
    parser.add_argument(
        "--skill-trends-only",
        action="store_true",
        help="rewrite only skill_trends.json from results/, without ClickHouse (the daily site run)",
    )
    args = parser.parse_args()
    if args.skill_trends_only:
        doc = json.loads(SKILL_TRENDS.read_text(encoding="utf-8"))
        sizes = write({"skill_trends.json": skill_trends(doc, SKILL_TRENDS.relative_to(ROOT).as_posix())}, args.out)
        print(f"skill_trends.json	{sizes['skill_trends.json']} bytes")
        return 0

    from radar.models import forecast

    load_dotenv()
    with ClickHouse() as ch:
        marts = fetch_marts(ch)
        series, coverage = forecast.fetch_series(ch, (FORECAST_SOURCE,))
    survival = json.loads(SURVIVAL.read_text(encoding="utf-8"))
    result = json.loads(FORECAST.read_text(encoding="utf-8"))
    site = forecast.site_forecast(
        series,
        coverage,
        forecast.load_config(),
        result["skills"][FORECAST_SOURCE],
        result["served"][FORECAST_TARGET]["method"],
        datetime.now(ZoneInfo("Europe/Moscow")).date(),
        FORECAST_SOURCE,
    )
    forecast_doc, history = forecast_files(result, site, FORECAST.relative_to(ROOT).as_posix())
    docs = build(
        marts,
        survival,
        datetime.now(UTC).replace(microsecond=0),
        git_commit(),
        forecast_doc=forecast_doc,
        forecast_history=history,
        results={
            path: json.loads(path.read_text(encoding="utf-8"))
            for path in (MARKET_FACTS, SKILL_EFFECTS, SKILL_EFFECTS_SENSITIVITY, HISTORY, SKILL_TRENDS)
        },
    )
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
