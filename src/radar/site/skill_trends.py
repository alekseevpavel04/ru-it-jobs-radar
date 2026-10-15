"""Skills of the site: what Telegram vacancy posts name, by year, for a direction of IT and a grade.

    python -m radar.site.skill_trends

Writes results/market/skill_trends.json. The skills, roles and grades are the extracted ones
(staging.int_extracted_weekly_units, single-vacancy IT posts after the weekly cross-post dedup), the same as
the yearly series of results/market/facts.json. A cell is a direction (all IT or one of DIRECTIONS, by role)
and a grade (all, or one of GRADES; posts without a grade count in "all" only), named "<direction>_<grade>".
Only shares a reader can lean on leave the file:

- a cell is kept when it has MIN_CELL_POSTS posts or more in RECENT_YEARS;
- a skill is listed for a cell when MIN_MENTIONS or more posts of RECENT_YEARS name it, the TOP_SKILLS
  most named of each cell, plus the GROWTH_SKILLS outside that top whose share gained most points since
  BASE_YEAR among those with at least GROWTH_MIN_SHARE of the cell's posts (points, not a ratio: a ratio
  makes a skill with a near-zero base look like the fastest grower);
- a year of a cell's series is kept when the cell has MIN_YEAR_POSTS posts or more that year; without
  BASE_YEAR a cell has its recent shares only, with no change to show.

It also writes the grade mix of all IT by year (each grade of GRADES as a share of the posts that name a
grade, with how many posts name one at all) and the direction mix from BASE_YEAR on: each direction's share
of all IT posts, over the panel of channels with IT posts in every year from BASE_YEAR, because channels
that start later (ML ones in 2023-2025) would move the shares by themselves.
"""

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from radar.config import load_dotenv
from radar.results import write_result
from radar.site.market_facts import JUNIOR_GRADES, ML_ROLES, RECENT_YEARS, YEARS, fetch, share
from radar.stream.clickhouse import ClickHouse

DIRECTIONS: dict[str, tuple[str, ...]] = {
    "backend": ("backend",),
    "frontend": ("frontend",),
    "qa": ("qa",),
    "analytics": ("data_analyst", "business_analyst", "system_analyst"),
    "ml": ML_ROLES,
    "dataeng": ("data_engineer",),
    "devops": ("devops", "sysadmin"),
    "mobile": ("mobile",),
}
# Roles of the direction mix only (no skill cells: their skills are mostly tools and soft skills); every
# other IT role goes to "other", so the mix adds up to all IT posts.
MIX_ONLY: dict[str, tuple[str, ...]] = {
    "management": ("manager",),
    "design": ("designer",),
    "fullstack": ("fullstack",),
    "security": ("security",),
}
GRADES: dict[str, tuple[str, ...]] = {"junior": JUNIOR_GRADES, "middle": ("middle",), "senior": ("senior", "lead")}
MIN_CELL_POSTS = 200
MIN_MENTIONS = 20
MIN_YEAR_POSTS = 300
TOP_SKILLS = 30
# From 2022 the channel set is steady; in 2019 a few frontend channels made up much of the sample.
BASE_YEAR = 2022
GROWTH_SKILLS = 10
GROWTH_MIN_SHARE = 0.005

SIZE_SQL = f"""
SELECT toYear(week) AS year, role, grade, count() AS posts
FROM staging.int_extracted_weekly_units WHERE source = 'telegram' AND is_it AND toYear(week) IN {YEARS}
GROUP BY year, role, grade
"""
DIRECTION_SQL = f"""
SELECT toYear(u.week) AS year, t.channel AS channel, u.role AS role, count() AS posts
FROM staging.int_extracted_weekly_units AS u
INNER JOIN (SELECT source_id, any(channel) AS channel FROM staging.stg_telegram GROUP BY source_id) AS t
    ON u.record_source_id = t.source_id
WHERE u.source = 'telegram' AND u.is_it AND toYear(u.week) IN {YEARS}
GROUP BY year, channel, role
"""
SKILL_SQL = f"""
SELECT toYear(week) AS year, role, grade, skill, count() AS posts
FROM staging.int_extracted_weekly_units ARRAY JOIN skills AS skill
WHERE source = 'telegram' AND is_it AND toYear(week) IN {YEARS}
GROUP BY year, role, grade, skill
"""


def cells_of(role: str | None, grade: str | None) -> list[str]:
    directions = ["it"] + [d for d, roles in DIRECTIONS.items() if role in roles]
    grades = ["all"] + [g for g, names in GRADES.items() if grade in names]
    return [f"{d}_{g}" for d in directions for g in grades]


def growing(
    cell: str,
    posts: Mapping[str, Counter[int]],
    counts: Mapping[str, Mapping[int, Counter[str]]],
    recent: Counter[str],
    skip: Iterable[str] = (),
) -> list[str]:
    """The skills outside skip whose share gained most points since BASE_YEAR; none without that year."""
    base_posts = posts[cell][BASE_YEAR]
    recent_posts = sum(posts[cell][y] for y in RECENT_YEARS)
    if base_posts < MIN_YEAR_POSTS or not recent_posts:
        return []
    base = counts[cell][BASE_YEAR]

    def growth(skill: str) -> float:
        return recent[skill] / recent_posts - base[skill] / base_posts

    skipped = set(skip)
    candidates = [
        s for s, n in recent.items() if s not in skipped and n >= MIN_MENTIONS and n / recent_posts >= GROWTH_MIN_SHARE
    ]
    return sorted(candidates, key=lambda s: (-growth(s), -recent[s]))[:GROWTH_SKILLS]


def grade_mix(sizes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Each grade's share of the IT posts that name a grade, by year, and how many name one."""
    named: dict[int, Counter[str]] = defaultdict(Counter)
    total: Counter[int] = Counter()
    for r in sizes:
        year = int(r["year"])
        total[year] += int(r["posts"])
        for g, names in GRADES.items():
            if r["grade"] in names:
                named[year][g] += int(r["posts"])
    metrics: dict[str, Any] = {}
    for year in sorted(total):
        graded = sum(named[year].values())
        metrics[f"grades.{year}.graded"] = share(graded, total[year])
        for g in GRADES:
            metrics[f"grades.{year}.{g}"] = share(named[year][g], graded)
    return metrics


def direction_mix(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Each direction's share of the IT posts of the panel channels, by year from BASE_YEAR."""
    rows = list(rows)
    years = [y for y in YEARS if y >= BASE_YEAR]
    seen: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        seen[r["channel"]].add(int(r["year"]))
    panel = {c for c, ys in seen.items() if set(years) <= ys}
    direction_of = {role: d for d, roles in (DIRECTIONS | MIX_ONLY).items() for role in roles}
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    total: Counter[int] = Counter()
    for r in rows:
        year = int(r["year"])
        if year in years and r["channel"] in panel:
            total[year] += int(r["posts"])
            counts[year][direction_of.get(r["role"], "other")] += int(r["posts"])
    metrics: dict[str, Any] = {"directions.panel_channels": share(len(panel), len(seen))}
    for year in years:
        for d in [*DIRECTIONS, *MIX_ONLY, "other"]:
            metrics[f"directions.{year}.{d}"] = share(counts[year][d], total[year])
    return metrics


def trends(sizes: Iterable[Mapping[str, Any]], skills: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sizes = list(sizes)
    posts: dict[str, Counter[int]] = defaultdict(Counter)
    for r in sizes:
        for c in cells_of(r["role"], r["grade"]):
            posts[c][int(r["year"])] += int(r["posts"])
    counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for r in skills:
        for c in cells_of(r["role"], r["grade"]):
            counts[c][int(r["year"])][r["skill"]] += int(r["posts"])

    cells = [c for c in posts if sum(posts[c][y] for y in RECENT_YEARS) >= MIN_CELL_POSTS]
    recent = {c: sum((counts[c][y] for y in RECENT_YEARS), Counter()) for c in cells}
    own = {c: [s for s, n in recent[c].most_common() if n >= MIN_MENTIONS][:TOP_SKILLS] for c in cells}
    for c in cells:
        own[c] += growing(c, posts, counts, recent[c], skip=own[c])

    metrics: dict[str, Any] = {}
    for c in sorted(cells):
        recent_posts = sum(posts[c][y] for y in RECENT_YEARS)
        kept_years = [y for y in YEARS if posts[c][y] >= MIN_YEAR_POSTS]
        metrics[f"posts.{c}.recent"] = {"value": recent_posts}
        for y in kept_years:
            metrics[f"posts.{c}.{y}"] = {"value": posts[c][y]}
        for s in own[c]:
            metrics[f"share.{c}.{s}.recent"] = share(recent[c][s], recent_posts)
            for y in kept_years:
                metrics[f"share.{c}.{s}.{y}"] = share(counts[c][y][s], posts[c][y])
    return metrics | grade_mix(sizes)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    load_dotenv()
    with ClickHouse() as ch:
        metrics = trends(fetch(ch, SIZE_SQL), fetch(ch, SKILL_SQL)) | direction_mix(fetch(ch, DIRECTION_SQL))
    path = write_result(
        "market/skill_trends.json",
        metrics,
        source_script="src/radar/site/skill_trends.py",
        directions={d: list(roles) for d, roles in DIRECTIONS.items()},
        grades={g: list(names) for g, names in GRADES.items()},
        recent_years=list(RECENT_YEARS),
        min_cell_posts=MIN_CELL_POSTS,
        min_mentions=MIN_MENTIONS,
        min_year_posts=MIN_YEAR_POSTS,
        top_skills=TOP_SKILLS,
        base_year=BASE_YEAR,
        growth_skills=GROWTH_SKILLS,
        growth_min_share=GROWTH_MIN_SHARE,
        definitions={
            "cells": "<direction>_<grade>: Telegram single-vacancy IT posts of the direction's roles (it: all IT) "
            "and of the grade (all: every post, a post without a grade included).",
            "directions": "directions.<year>.<direction>: posts of the direction's roles of all IT posts of the panel "
            "channels (IT posts in every year from the base year); directions.panel_channels: panel channels of all.",
            "grades": "grades.<year>.<grade>: posts of the grade of the IT posts that name a grade; "
            "grades.<year>.graded: IT posts that name a grade of all IT posts.",
            "share": "posts of the cell naming the skill (extracted skills) of the cell's posts; recent pools "
            f"{RECENT_YEARS[0]}-{RECENT_YEARS[-1]}, a year by the week's Monday.",
        },
    )
    print(f"{len(metrics)} metrics -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
