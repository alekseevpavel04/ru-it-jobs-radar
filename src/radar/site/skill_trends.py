"""Skills of the site: what Telegram vacancy posts name, by year, for a direction of IT and a grade.

    python -m radar.site.skill_trends                                       (local ClickHouse)
    python -m radar.site.skill_trends --from-aggregates aggregates/telegram   (the daily run, no ClickHouse)

Writes results/market/skill_trends.json. The skills, roles and grades are the extracted ones
(staging.int_extracted_weekly_units, single-vacancy IT posts after the weekly cross-post dedup), the same as
the yearly series of results/market/facts.json. Both inputs give the same rows (clickhouse_rows,
aggregate_rows): the weekly counts of aggregates/telegram are those tables summed by week and channel.
A cell is a direction (all IT or one of DIRECTIONS, by role) and a grade (all, or one of GRADES; posts
without a grade count in "all" only), named "<direction>_<grade>".
Only shares a reader can lean on leave the file:

- a cell is kept when it has MIN_CELL_POSTS posts or more in RECENT_YEARS;
- a skill is listed for a cell when MIN_MENTIONS or more posts of RECENT_YEARS name it, the TOP_SKILLS
  most named of each cell, plus the GROWTH_SKILLS outside that top whose share gained most points since
  BASE_YEAR among those with at least GROWTH_MIN_SHARE of the cell's posts (points, not a ratio: a ratio
  makes a skill with a near-zero base look like the fastest grower);
- a year of a cell's series is kept when the cell has MIN_YEAR_POSTS posts or more that year;
- the change since BASE_YEAR (change.<cell>.<skill>) is measured within channels (radar.site.within): only
  channels with posts of the cell in BASE_YEAR and in RECENT_YEARS, each with the same weight in both, so
  that channels that started posting since then do not move it (they gave 18 % of the IT posts of
  2025-2026). Its p is the larger of a post-level and a channel-bootstrap p, q_bh the Benjamini-Hochberg
  adjustment over the cell's listed skills. A cell has it when it has MIN_YEAR_POSTS posts in BASE_YEAR,
  all of them and those of the panel channels; otherwise it has its recent shares only.
  panel.<cell>.base / .recent: the posts of those channels of all the cell's posts.

It also writes the grade mix (each grade of GRADES as a share of the IT posts that name a grade) and the
direction mix (each direction's share of all IT posts) by year from BASE_YEAR, both over the panel of
channels with such posts in every year and at fixed channel weights (radar.site.within.fixed_weight_series):
channels that start later (ML ones in 2023-2025) or grow several times (junior-only channels, topic
channels) would move pooled shares by themselves. Pooled, the junior share of graded posts rose from 13 %
to 26 %; at fixed weights it stays about 20 %.
"""

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from radar.autopilot import aggregates
from radar.config import load_dotenv
from radar.results import write_result
from radar.site.market_facts import JUNIOR_GRADES, ML_ROLES, RECENT_YEARS, YEARS, fetch, share
from radar.site.within import benjamini_hochberg, fixed_weight_series, panel, within_change
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
    "fullstack": ("fullstack",),
    "security": ("security",),
}
# Roles of the direction mix only, without skill cells: managers' posts name few skills (Agile in 4 %), and
# "designer" mixes interface designers with game artists. Every other IT role goes to "other", so the mix
# adds up to all IT posts.
MIX_ONLY: dict[str, tuple[str, ...]] = {"management": ("manager",), "design": ("designer",)}
GRADES: dict[str, tuple[str, ...]] = {"junior": JUNIOR_GRADES, "middle": ("middle",), "senior": ("senior", "lead")}
MIN_CELL_POSTS = 200
MIN_MENTIONS = 20
MIN_YEAR_POSTS = 300
TOP_SKILLS = 30
# From 2022 the channel set is steady; in 2019 a few frontend channels made up much of the sample.
BASE_YEAR = 2022
GROWTH_SKILLS = 10
GROWTH_MIN_SHARE = 0.005

# Every row carries the channel of its posts; trends() sums over channels.
_CHANNEL_JOIN = """
FROM staging.int_extracted_weekly_units AS u
INNER JOIN (SELECT source_id, any(channel) AS channel FROM staging.stg_telegram GROUP BY source_id) AS t
    ON u.record_source_id = t.source_id
"""
_IT_UNITS = f"u.source = 'telegram' AND u.is_it AND toYear(u.week) IN {YEARS}"
SIZE_SQL = f"""
SELECT toYear(u.week) AS year, t.channel AS channel, u.role AS role, u.grade AS grade, count() AS posts
{_CHANNEL_JOIN}
WHERE {_IT_UNITS}
GROUP BY year, channel, role, grade
"""
DIRECTION_SQL = f"""
SELECT toYear(u.week) AS year, t.channel AS channel, u.role AS role, count() AS posts
{_CHANNEL_JOIN}
WHERE {_IT_UNITS}
GROUP BY year, channel, role
"""
SKILL_SQL = f"""
SELECT toYear(u.week) AS year, t.channel AS channel, u.role AS role, u.grade AS grade, skill, count() AS posts
{_CHANNEL_JOIN}
ARRAY JOIN u.skills AS skill
WHERE {_IT_UNITS}
GROUP BY year, channel, role, grade, skill
"""

Rows = list[dict[str, Any]]


def ordered(rows: Iterable[Mapping[str, Any]]) -> Rows:
    """Rows in a fixed order: trends() takes skills tied in mentions in the order it met them, so the
    30th skill of a cell would otherwise depend on the order ClickHouse or the files return them."""
    keys = ("year", "channel", "role", "grade", "skill")
    return sorted((dict(r) for r in rows), key=lambda r: tuple(str(r.get(k) or "") for k in keys))


def clickhouse_rows() -> tuple[Rows, Rows, Rows]:
    """(sizes, skills, directions) from the dbt tables of the local stack."""
    load_dotenv()
    with ClickHouse() as ch:
        return ordered(fetch(ch, SIZE_SQL)), ordered(fetch(ch, SKILL_SQL)), ordered(fetch(ch, DIRECTION_SQL))


def aggregate_rows(root: Path) -> tuple[Rows, Rows, Rows]:
    """(sizes, skills, directions) from the weekly counts of aggregates/telegram (radar.autopilot)."""
    sizes, skills, directions = aggregates.trend_rows(aggregates.load(root))
    return ordered(sizes), ordered(skills), ordered(directions)


def period_of(year: int) -> str | None:
    return "base" if year == BASE_YEAR else "recent" if year in RECENT_YEARS else None


def cells_of(role: str | None, grade: str | None) -> list[str]:
    directions = ["it"] + [d for d, roles in DIRECTIONS.items() if role in roles]
    grades = ["all"] + [g for g, names in GRADES.items() if grade in names]
    return [f"{d}_{g}" for d in directions for g in grades]


def _series(size: Mapping[str, Mapping[str, int]], named: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Yearly shares at fixed channel weights over the panel (radar.site.within), with the change from the
    first year to the last and its Benjamini-Hochberg q over the keys."""
    years = sorted({y for n in size.values() for y in n})
    channels, found = fixed_weight_series(size, named, years)
    keys = sorted(found)
    q = dict(zip(keys, benjamini_hochberg([found[k][1] for k in keys]), strict=True))
    metrics: dict[str, Any] = {
        f"{kind}.panel_channels": share(len(channels), len(size)),
        f"{kind}.panel_posts": share(
            sum(sum(size[c].values()) for c in channels), sum(sum(n.values()) for n in size.values())
        ),
    }
    for k in keys:
        line, p = found[k]
        for y, v in zip(years, line, strict=True):
            metrics[f"{kind}.{y}.{k}"] = {"value": round(v, 4)}
        metrics[f"{kind}.change.{k}"] = {
            "value": round(line[-1] - line[0], 4),
            "p": float(f"{p:.3g}"),
            "q_bh": float(f"{q[k]:.3g}"),
        }
    return metrics


def grade_mix(sizes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Each grade's share of the IT posts that name a grade, by year from BASE_YEAR at fixed channel weights,
    and how many IT posts name a grade at all (pooled, all channels)."""
    grade_of = {name: g for g, names in GRADES.items() for name in names}
    size: dict[str, Counter[str]] = defaultdict(Counter)
    named: dict[str, dict[str, Counter[str]]] = {g: defaultdict(Counter) for g in GRADES}
    graded: Counter[int] = Counter()
    total: Counter[int] = Counter()
    for r in sizes:
        year, g = int(r["year"]), grade_of.get(r["grade"])
        total[year] += int(r["posts"])
        if g:
            graded[year] += int(r["posts"])
            if year >= BASE_YEAR:
                size[r["channel"]][str(year)] += int(r["posts"])
                named[g][r["channel"]][str(year)] += int(r["posts"])
    metrics = {f"grades.{y}.graded": share(graded[y], total[y]) for y in sorted(total) if y >= BASE_YEAR}
    return metrics | _series(size, named, "grades")


def direction_mix(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Each direction's share of IT posts by year from BASE_YEAR, over the channels with IT posts in every
    year, at fixed channel weights: a channel that grows several times (topic channels did) would move the
    pooled shares by itself."""
    direction_of = {role: d for d, roles in (DIRECTIONS | MIX_ONLY).items() for role in roles}
    size: dict[str, Counter[str]] = defaultdict(Counter)
    named: dict[str, dict[str, Counter[str]]] = {d: defaultdict(Counter) for d in [*DIRECTIONS, *MIX_ONLY, "other"]}
    for r in rows:
        year = int(r["year"])
        if year >= BASE_YEAR:
            size[r["channel"]][str(year)] += int(r["posts"])
            named[direction_of.get(r["role"], "other")][r["channel"]][str(year)] += int(r["posts"])
    return _series(size, named, "directions")


def trends(sizes: Iterable[Mapping[str, Any]], skills: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sizes = list(sizes)
    posts: dict[str, Counter[int]] = defaultdict(Counter)
    # cell -> channel -> period -> posts, and cell -> skill -> channel -> period -> posts naming the skill,
    # for the change since BASE_YEAR, which is measured within channels (radar.site.within)
    by_channel: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for r in sizes:
        year, period = int(r["year"]), period_of(int(r["year"]))
        for c in cells_of(r["role"], r["grade"]):
            posts[c][year] += int(r["posts"])
            if period:
                by_channel[c][r["channel"]][period] += int(r["posts"])
    counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    named: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for r in skills:
        year, period = int(r["year"]), period_of(int(r["year"]))
        for c in cells_of(r["role"], r["grade"]):
            counts[c][year][r["skill"]] += int(r["posts"])
            if period:
                named[c][r["skill"]][r["channel"]][period] += int(r["posts"])

    cells = [c for c in posts if sum(posts[c][y] for y in RECENT_YEARS) >= MIN_CELL_POSTS]
    recent = {c: sum((counts[c][y] for y in RECENT_YEARS), Counter()) for c in cells}
    metrics: dict[str, Any] = {}
    for c in sorted(cells):
        recent_posts = sum(posts[c][y] for y in RECENT_YEARS)
        # ties in mentions go by name, so the 30th skill does not depend on the order of the input rows
        by_mentions = sorted(recent[c].items(), key=lambda kv: (-kv[1], kv[0]))
        listed = [s for s, n in by_mentions if n >= MIN_MENTIONS][:TOP_SKILLS]
        channels = panel(by_channel[c], ("base", "recent")) if posts[c][BASE_YEAR] >= MIN_YEAR_POSTS else []
        compared = sum(by_channel[c][ch]["base"] for ch in channels) >= MIN_YEAR_POSTS
        changes: dict[str, tuple[float, float, float]] = {}
        if compared:
            candidates = [
                s for s, n in by_mentions if n >= MIN_MENTIONS and (s in listed or n / recent_posts >= GROWTH_MIN_SHARE)
            ]
            changes = within_change(by_channel[c], {s: named[c][s] for s in candidates}, "base", "recent")
            # plus the skills outside the top whose share gained most points within channels
            growth = sorted((s for s in changes if s not in listed), key=lambda s: (changes[s][0] - changes[s][1], s))
            listed += growth[:GROWTH_SKILLS]
            q_values = dict(zip(listed, benjamini_hochberg([changes[s][2] for s in listed]), strict=True))
            metrics[f"panel.{c}.base"] = share(sum(by_channel[c][ch]["base"] for ch in channels), posts[c][BASE_YEAR])
            metrics[f"panel.{c}.recent"] = share(sum(by_channel[c][ch]["recent"] for ch in channels), recent_posts)
        kept_years = [y for y in YEARS if posts[c][y] >= MIN_YEAR_POSTS]
        metrics[f"posts.{c}.recent"] = {"value": recent_posts}
        for y in kept_years:
            metrics[f"posts.{c}.{y}"] = {"value": posts[c][y]}
        for s in listed:
            metrics[f"share.{c}.{s}.recent"] = share(recent[c][s], recent_posts)
            for y in kept_years:
                metrics[f"share.{c}.{s}.{y}"] = share(counts[c][y][s], posts[c][y])
            if compared:
                was, now, p = changes[s]
                metrics[f"change.{c}.{s}"] = {
                    "value": round(now - was, 4),
                    "p": float(f"{p:.3g}"),
                    "q_bh": float(f"{q_values[s]:.3g}"),
                    "base": round(was, 4),
                    "recent": round(now, 4),
                }
    return metrics | grade_mix(sizes)


def metrics_of(sizes: Rows, skills: Rows, directions: Rows) -> dict[str, Any]:
    return trends(sizes, skills) | direction_mix(directions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-aggregates",
        type=Path,
        metavar="DIR",
        help="read the weekly counts of DIR (aggregates/telegram) instead of ClickHouse",
    )
    args = parser.parse_args(argv)
    if args.from_aggregates is None:
        metrics, source = metrics_of(*clickhouse_rows()), "clickhouse: staging.int_extracted_weekly_units"
    else:
        metrics, source = (
            metrics_of(*aggregate_rows(args.from_aggregates)),
            f"aggregates: {args.from_aggregates.as_posix()}",
        )
    path = write_result(
        "market/skill_trends.json",
        metrics,
        source_script="src/radar/site/skill_trends.py",
        input=source,
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
