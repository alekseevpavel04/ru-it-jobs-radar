"""Market facts: what the market asks of a junior in ML and DS.

    python -m radar.site.market_facts

Writes results/market/facts.json, aggregates only, every share with its denominator:

- ml_ds.*, ml_ds_skill.*, ml_ds_junior_skill.*: the ML/DS bucket per source in the units of the site
  (marts.mart_junior, marts.mart_skills: role bucket, grade and skills by rule, Telegram over the
  cross-sectional window of the marts);
- salary.*, salary_ml_cell.*: posted RUB monthly salaries by extracted grade, and how many ML
  vacancies of a junior grade state one at all (marts.mart_extracted_salaries);
- telegram_year.*, telegram_ml_skill.*: the Telegram history by calendar year of the week's Monday on
  the extracted fields: IT posts, ML posts (roles data_scientist, ml_engineer, mlops), posts graded
  intern or junior, skills named in ML posts. The channel mix changes over the years (channels on ML
  start in 2023-2025), so every series is repeated for the panel of channels with IT posts in each
  year of PANEL_YEARS (telegram_panel_*). The channel is not in the marts: these counts are grouped
  in ClickHouse over staging.int_extracted_weekly_units joined to staging.stg_telegram, and only
  sums per year leave it.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from radar.config import load_dotenv
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse

SOURCES = ("telegram", "trudvsem", "habr", "superjob")
BUCKET = "ml_ds"
JUNIOR_GRADES = ("intern", "junior")
SENIOR_GRADES = ("senior", "lead")
SALARY_GRADES = ("intern", "junior", "middle", "senior")
SALARY_SOURCES = ("telegram", "trudvsem", "habr")
ML_ROLES = ("data_scientist", "ml_engineer", "mlops")
TOP_SKILLS = 15
TOP_JUNIOR_SKILLS = 10
YEARS = tuple(range(2019, 2027))
PANEL_YEARS = tuple(range(2022, 2027))
TRACKED_SKILLS = (
    "Python",
    "SQL",
    "pandas",
    "NumPy",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Spark",
    "Docker",
    "Kubernetes",
    "Airflow",
    "MLOps",
    "FastAPI",
    "RAG",
    "LangChain",
)

Rows = list[dict[str, Any]]

JUNIOR_SQL = f"""
SELECT source, vacancies, open_to_junior, graded_intern, graded_junior, grade_known
FROM marts.mart_junior WHERE role_bucket = '{BUCKET}'
"""
# A grade cell of mart_skills carries its size only on rows of a mentioned skill; grade_split checks
# that the cells add up to the bucket of mart_junior before it uses them.
GRADE_SQL = f"""
SELECT source, grade, max(vacancies_in_cell) AS vacancies
FROM marts.mart_skills WHERE role_bucket = '{BUCKET}' GROUP BY source, grade
"""
SKILL_SQL = f"""
SELECT source, grade, skill, sum(vacancies_mentioning) AS vacancies_mentioning
FROM marts.mart_skills WHERE role_bucket = '{BUCKET}' GROUP BY source, grade, skill
"""
WINDOW_SQL = """
SELECT source, toString(min(period_from)) AS period_from, toString(max(period_to)) AS period_to
FROM marts.mart_skills GROUP BY source
"""
LAST_WEEK_SQL = "SELECT toString(max(week)) AS week FROM staging.int_extracted_weekly_units WHERE source = 'telegram'"
SALARY_SQL = """
SELECT source, level, role, grade, vacancies, with_salary_rub_month, with_from, with_to,
    salary_from_median, salary_to_median
FROM marts.mart_extracted_salaries WHERE level IN ('grade', 'cell')
"""
_UNITS = f"""
WITH units AS (
    SELECT toYear(u.week) AS year, t.channel AS channel, u.role IN {ML_ROLES} AS is_ml,
        u.grade IN {JUNIOR_GRADES} AS is_junior, u.grade IS NOT NULL AS is_graded, u.skills AS skills
    FROM staging.int_extracted_weekly_units AS u
    INNER JOIN (SELECT source_id, any(channel) AS channel FROM staging.stg_telegram GROUP BY source_id) AS t
        ON u.record_source_id = t.source_id
    WHERE u.source = 'telegram' AND u.is_it
)
"""
WORK_FORMAT_SQL = f"""
SELECT toYear(week) AS year, role IN {ML_ROLES} AS ml, grade IN {JUNIOR_GRADES} AS junior, work_format,
    count() AS posts
FROM staging.int_extracted_weekly_units
WHERE source = 'telegram' AND is_it AND grade IS NOT NULL AND work_format IS NOT NULL
GROUP BY year, ml, junior, work_format
"""
# The 2025-2026 Telegram cut on the extracted fields: the window of the marts holds 97 ML posts of a
# junior grade, two years hold several times more, which skill shares and medians of a junior need.
RECENT_YEARS = (2025, 2026)
RECENT_TOP_SKILLS = 20
MIN_SALARY_ROWS = 9
_RECENT = f"source = 'telegram' AND is_it AND toYear(week) IN {RECENT_YEARS}"
RECENT_SIZE_SQL = f"""
SELECT role IN {ML_ROLES} AS ml, grade IN {JUNIOR_GRADES} AS junior, count() AS posts
FROM staging.int_extracted_weekly_units WHERE {_RECENT}
GROUP BY ml, junior
"""
RECENT_SKILL_SQL = f"""
SELECT role IN {ML_ROLES} AS ml, grade IN {JUNIOR_GRADES} AS junior, skill, count() AS posts
FROM staging.int_extracted_weekly_units ARRAY JOIN skills AS skill WHERE {_RECENT}
GROUP BY ml, junior, skill
"""
# The salary rule of mart_extracted_salaries: RUB per month, or no period stated.
_RECENT_SALARY_AGG = """grade, count() AS posts, countIf(in_q) AS stated,
    countIf(in_q AND salary_from IS NOT NULL) AS with_from, countIf(in_q AND salary_to IS NOT NULL) AS with_to,
    quantileExactInclusive(0.5)(if(in_q, salary_from, NULL)) AS from_median,
    quantileExactInclusive(0.5)(if(in_q, salary_to, NULL)) AS to_median"""
RECENT_SALARY_SQL = f"""
WITH posts AS (
    SELECT u.role IN {ML_ROLES} AS ml, u.grade AS grade, f.salary_from AS salary_from, f.salary_to AS salary_to,
        (f.salary_from IS NOT NULL OR f.salary_to IS NOT NULL) AND coalesce(f.salary_currency = 'RUB', false)
            AND coalesce(f.salary_period, 'month') = 'month' AS in_q
    FROM staging.int_extracted_weekly_units AS u
    INNER JOIN staging.int_extracted_fields AS f ON f.source = 'telegram' AND f.source_id = u.record_source_id
    WHERE u.source = 'telegram' AND u.is_it AND toYear(u.week) IN {RECENT_YEARS} AND u.grade IN {SALARY_GRADES}
)
SELECT 'it' AS scope, {_RECENT_SALARY_AGG} FROM posts GROUP BY grade
UNION ALL
SELECT 'ml' AS scope, {_RECENT_SALARY_AGG} FROM posts WHERE ml GROUP BY grade
"""
CHANNEL_YEAR_SQL = (
    _UNITS
    + """
SELECT year, channel, count() AS it_posts, countIf(is_ml) AS ml_posts, countIf(is_junior) AS junior_posts,
    countIf(is_graded) AS graded_posts, countIf(is_ml AND is_junior) AS ml_junior_posts,
    countIf(is_ml AND is_graded) AS ml_graded_posts
FROM units GROUP BY year, channel
"""
)
CHANNEL_SKILL_SQL = (
    _UNITS
    + f"""
SELECT year, channel, skill, count() AS posts
FROM units ARRAY JOIN skills AS skill
WHERE is_ml AND skill IN {TRACKED_SKILLS}
GROUP BY year, channel, skill
"""
)


def share(n: int, denominator: int) -> dict[str, Any]:
    return {"value": round(n / denominator, 4) if denominator else 0.0, "n": int(n), "denominator": int(denominator)}


def ml_ds_buckets(junior: Iterable[Mapping[str, Any]], cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Size, junior openness and grade split of the ML/DS bucket per source."""
    by_grade: dict[str, dict[str, int]] = defaultdict(dict)
    for r in cells:
        by_grade[r["source"]][r["grade"]] = int(r["vacancies"])
    metrics: dict[str, Any] = {}
    for r in junior:
        s = r["source"]
        metrics[f"ml_ds.{s}.vacancies"] = {"value": int(r["vacancies"])}
        metrics[f"ml_ds.{s}.open_to_junior"] = share(r["open_to_junior"], r["vacancies"])
        metrics[f"ml_ds.{s}.graded_junior"] = share(r["graded_intern"] + r["graded_junior"], r["grade_known"])
        grades = by_grade.get(s, {})
        if sum(grades.values()) != r["vacancies"]:
            raise ValueError(f"{s}: grade cells add up to {sum(grades.values())}, the bucket has {r['vacancies']}")
        senior = sum(grades.get(g, 0) for g in SENIOR_GRADES)
        metrics[f"ml_ds.{s}.graded_senior"] = share(senior, r["grade_known"])
    return metrics


def ml_ds_skills(skill_rows: Iterable[Mapping[str, Any]], junior: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Top skills of the ML/DS bucket per source, of all its vacancies and of those graded intern or junior."""
    junior = list(junior)
    base = {r["source"]: int(r["vacancies"]) for r in junior}
    base_junior = {r["source"]: int(r["graded_intern"] + r["graded_junior"]) for r in junior}
    all_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    junior_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in skill_rows:
        all_n[r["source"]][r["skill"]] += int(r["vacancies_mentioning"])
        if r["grade"] in JUNIOR_GRADES:
            junior_n[r["source"]][r["skill"]] += int(r["vacancies_mentioning"])

    def top(counts: Mapping[str, int], k: int) -> list[str]:
        return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

    metrics: dict[str, Any] = {}
    for source in base:
        # Both slices get the union of the two top lists, so they read side by side; a skill no junior
        # vacancy names reads 0 there.
        chosen = dict.fromkeys(top(all_n[source], TOP_SKILLS) + top(junior_n[source], TOP_JUNIOR_SKILLS))
        for prefix, counts, bases in (("ml_ds_skill", all_n, base), ("ml_ds_junior_skill", junior_n, base_junior)):
            for skill in chosen:
                n = counts[source].get(skill, 0)
                if n > bases[source]:
                    raise ValueError(f"{source} {skill}: {n} mentions of {bases[source]}")
                metrics[f"{prefix}.{source}.{skill}"] = share(n, bases[source])
    return metrics


def salaries(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Median posted salary by grade, and how many vacancies state a RUB monthly amount."""
    metrics: dict[str, Any] = {}
    for r in rows:
        s, grade = r["source"], r["grade"]
        if r["level"] == "grade" and s in SALARY_SOURCES and grade in SALARY_GRADES:
            key = f"salary.{s}.{grade}"
            metrics[f"{key}.stated"] = share(r["with_salary_rub_month"], r["vacancies"])
            for end in ("from", "to"):
                median = r[f"salary_{end}_median"]
                if median is not None:
                    metrics[f"{key}.{end}_median"] = {"value": round(median), "rows": int(r[f"with_{end}"])}
        elif r["level"] == "cell" and r["role"] in ML_ROLES and grade in JUNIOR_GRADES:
            stated = share(r["with_salary_rub_month"], r["vacancies"])
            metrics[f"salary_ml_cell.{s}.{r['role']}.{grade}.stated"] = stated
    return metrics


def panel_channels(channel_years: Iterable[Mapping[str, Any]], years: tuple[int, ...] = PANEL_YEARS) -> set[str]:
    """Channels with at least one IT post in every year of the panel."""
    seen: dict[str, set[int]] = defaultdict(set)
    for r in channel_years:
        if r["it_posts"] > 0:
            seen[r["channel"]].add(int(r["year"]))
    return {c for c, ys in seen.items() if set(years) <= ys}


def telegram_years(
    channel_years: Iterable[Mapping[str, Any]],
    channel_skills: Iterable[Mapping[str, Any]],
    years: tuple[int, ...] = YEARS,
    panel_years: tuple[int, ...] = PANEL_YEARS,
) -> dict[str, Any]:
    """Yearly Telegram series for all channels and for the panel of channels present in every panel year."""
    channel_years = list(channel_years)
    panel = panel_channels(channel_years, panel_years)
    channels = {r["channel"] for r in channel_years if r["it_posts"] > 0}
    metrics: dict[str, Any] = {"telegram_panel.channels": share(len(panel), len(channels))}

    totals: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in channel_years:
        scopes = ["all", "panel"] if r["channel"] in panel else ["all"]
        for scope in scopes:
            for k in ("it_posts", "ml_posts", "junior_posts", "graded_posts", "ml_junior_posts", "ml_graded_posts"):
                totals[scope, int(r["year"])][k] += int(r[k])
    skills: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in channel_skills:
        scopes = ["all", "panel"] if r["channel"] in panel else ["all"]
        for scope in scopes:
            skills[scope, int(r["year"])][r["skill"]] += int(r["posts"])

    for scope, prefix, scope_years in (("all", "telegram", years), ("panel", "telegram_panel", panel_years)):
        for y in scope_years:
            t = totals[scope, y]
            metrics[f"{prefix}_year.{y}.it_posts"] = {"value": t["it_posts"]}
            metrics[f"{prefix}_year.{y}.ml_posts"] = share(t["ml_posts"], t["it_posts"])
            metrics[f"{prefix}_year.{y}.junior_posts"] = share(t["junior_posts"], t["it_posts"])
            metrics[f"{prefix}_year.{y}.junior_of_graded"] = share(t["junior_posts"], t["graded_posts"])
            metrics[f"{prefix}_year.{y}.ml_junior_of_graded"] = share(t["ml_junior_posts"], t["ml_graded_posts"])
            for skill in TRACKED_SKILLS:
                n = skills[scope, y].get(skill, 0)
                if n > t["ml_posts"]:
                    raise ValueError(f"{scope} {y} {skill}: {n} mentions of {t['ml_posts']} ML posts")
                metrics[f"{prefix}_ml_skill.{y}.{skill}"] = share(n, t["ml_posts"])
    return metrics


def fetch(ch: ClickHouse, sql: str) -> Rows:
    body = ch.query(sql + " SETTINGS output_format_json_quote_64bit_integers = 0 FORMAT JSONEachRow")
    return [json.loads(line) for line in body.splitlines() if line]


def work_formats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Share of each work format among graded Telegram IT posts with a stated format, juniors against the rest,
    for all IT posts and for the ML roles alone."""
    cells: dict[tuple[str, int, str], Counter[str]] = defaultdict(Counter)
    for r in rows:
        group = "junior" if r["junior"] else "other"
        scopes = ["telegram_work_format", "telegram_ml_work_format"] if r.get("ml") else ["telegram_work_format"]
        for scope in scopes:
            cells[(scope, int(r["year"]), group)][str(r["work_format"])] += int(r["posts"])
    out: dict[str, Any] = {}
    for (scope, year, group), formats in sorted(cells.items()):
        total = sum(formats.values())
        for fmt, n in sorted(formats.items()):
            out[f"{scope}.{year}.{group}.{fmt}"] = share(n, total)
    return out


def _recent_group(ml: Any, junior: Any) -> list[str]:
    groups = ["it_all"] + (["it_junior"] if junior else [])
    return groups + (["ml_all"] + (["ml_junior"] if junior else []) if ml else [])


def recent_skills(sizes: Iterable[Mapping[str, Any]], skills: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Skill shares of the 2025-2026 Telegram posts for ML and all IT, juniors and everyone, over the
    tracked skills and the top skills of the two junior groups."""
    posts: Counter[str] = Counter()
    for r in sizes:
        for g in _recent_group(r["ml"], r["junior"]):
            posts[g] += int(r["posts"])
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in skills:
        for g in _recent_group(r["ml"], r["junior"]):
            counts[g][r["skill"]] += int(r["posts"])
    chosen = set(TRACKED_SKILLS)
    chosen |= {s for g in ("ml_junior", "it_junior") for s, _ in counts[g].most_common(RECENT_TOP_SKILLS)}
    metrics: dict[str, Any] = {f"telegram_recent.{g}.posts": {"value": n} for g, n in sorted(posts.items())}
    for g in sorted(posts):
        for s in sorted(chosen):
            metrics[f"telegram_recent_skill.{g}.{s}"] = share(counts[g][s], posts[g])
    return metrics


def recent_salaries(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Posted RUB monthly salaries of the 2025-2026 Telegram posts by grade, for the ML roles and for all
    IT. A median is kept only over MIN_SALARY_ROWS bounds or more, the threshold of the site."""
    metrics: dict[str, Any] = {}
    for r in rows:
        key = f"telegram_recent_salary.{r['scope']}.{r['grade']}"
        metrics[f"{key}.stated"] = share(r["stated"], r["posts"])
        for end in ("from", "to"):
            if r[f"with_{end}"] >= MIN_SALARY_ROWS:
                metrics[f"{key}.{end}_median"] = {"value": round(r[f"{end}_median"]), "rows": int(r[f"with_{end}"])}
    return metrics


def build(ch: ClickHouse) -> dict[str, Any]:
    junior = fetch(ch, JUNIOR_SQL)
    metrics = ml_ds_buckets(junior, fetch(ch, GRADE_SQL))
    metrics |= ml_ds_skills(fetch(ch, SKILL_SQL), junior)
    metrics |= salaries(fetch(ch, SALARY_SQL))
    metrics |= telegram_years(fetch(ch, CHANNEL_YEAR_SQL), fetch(ch, CHANNEL_SKILL_SQL))
    metrics |= work_formats(fetch(ch, WORK_FORMAT_SQL))
    metrics |= recent_skills(fetch(ch, RECENT_SIZE_SQL), fetch(ch, RECENT_SKILL_SQL))
    metrics |= recent_salaries(fetch(ch, RECENT_SALARY_SQL))
    for r in fetch(ch, WINDOW_SQL):
        metrics[f"window.{r['source']}.from"] = {"value": r["period_from"]}
        metrics[f"window.{r['source']}.to"] = {"value": r["period_to"]}
    last = fetch(ch, LAST_WEEK_SQL)
    metrics["telegram_year.last_week"] = {"value": last[0]["week"]}
    return metrics


DEFINITIONS = {
    "ml_ds": "Rule bucket ml_ds of mart_junior: vacancies (Telegram: posts) of the collection window; "
    "graded_junior and graded_senior are of the vacancies whose grade the title rule found (grade_known).",
    "ml_ds_skill": "Vacancies of the bucket naming the skill (rule dictionary), of the bucket; "
    "ml_ds_junior_skill of those graded intern or junior.",
    "salary": "mart_extracted_salaries, level grade: stated = vacancies with a RUB monthly amount of the grade's "
    "vacancies; from_median and to_median are medians of the posted bounds, rows = vacancies with that bound.",
    "salary_ml_cell": "mart_extracted_salaries, level cell: ML role x junior grade, "
    "vacancies stating a RUB monthly amount.",
    "telegram_year": "Single-vacancy IT posts after the weekly cross-post dedup, by year of the week's Monday, "
    "extracted fields; ml_posts: role data_scientist, ml_engineer or mlops; junior_posts: grade intern or junior; "
    "junior_of_graded: of the posts with a grade; ml_junior_of_graded: the same within the ML roles.",
    "telegram_ml_skill": "ML posts of the year naming the skill (extracted skills), of the year's ML posts.",
    "telegram_work_format": "Single-vacancy IT posts with an extracted grade and work format, by year of the week's "
    "Monday; junior: grade intern or junior, other: the other grades; share of the group's posts with a format; "
    "telegram_ml_work_format: the same within the ML roles.",
    "telegram_recent": f"Telegram single-vacancy IT posts of the weeks whose Monday falls in {RECENT_YEARS[0]}-"
    f"{RECENT_YEARS[-1]}, extracted fields; groups it_all, it_junior (grade intern or junior), ml_all (ML roles), "
    "ml_junior; telegram_recent_skill: posts of the group naming the skill, over the tracked skills and the top "
    f"{RECENT_TOP_SKILLS} skills of the two junior groups; telegram_recent_salary: posts of the grade stating a RUB "
    f"monthly amount, medians of the posted bounds when {MIN_SALARY_ROWS} or more posts state the bound.",
    "window": "Collection window of the cross-sectional marts per source (mart_skills period_from, period_to).",
    "telegram_panel": f"The same over the channels with IT posts in every year {PANEL_YEARS[0]}-{PANEL_YEARS[-1]}.",
}


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    load_dotenv()
    with ClickHouse() as ch:
        metrics = build(ch)
    path = write_result(
        "market/facts.json",
        metrics,
        source_script="src/radar/site/market_facts.py",
        definitions=DEFINITIONS,
        ml_roles=list(ML_ROLES),
        panel_years=list(PANEL_YEARS),
        salary_years=list(RECENT_YEARS),
    )
    print(f"{len(metrics)} metrics -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
