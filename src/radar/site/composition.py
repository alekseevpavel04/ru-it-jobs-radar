"""How much the make-up of the channels moved the numbers before the site measured changes within channels.

    python -m radar.site.composition

Writes results/market/composition.json, the evidence behind the README's account of that mistake:

- pooled.<cell>.<skill> against within.<cell>.<skill>: the change of a skill's share from BASE_YEAR to
  RECENT_YEARS over all posts pooled, and within channels (radar.site.within), for the skills the first
  version of the site led with;
- grades.junior.pooled / .fixed: the junior share of graded posts in BASE_YEAR and in the last year, pooled
  and at fixed channel weights over the panel;
- placebo.pooled_test: the share of skills a plain two-proportion test over posts flags when the panel
  channels of BASE_YEAR are split at random into two halves, so that only the channels differ;
- resume_posts.<year>: Telegram posts that the collector took for vacancies under rules version 2 and that
  the résumé rule (radar.collect.telegram_rules.RESUME_HEAD, rules version 3) drops.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from statistics import NormalDist
from typing import Any

from radar.config import load_dotenv
from radar.results import write_result
from radar.site.market_facts import RECENT_YEARS, fetch, share
from radar.site.skill_trends import BASE_YEAR, GRADES, cells_of, clickhouse_rows, period_of
from radar.site.within import SEED, fixed_weight_series, panel, within_change
from radar.stream.clickhouse import ClickHouse

SHOWN = {"it_all": ("Python", "CI/CD", "JavaScript"), "frontend_all": ("React", "TypeScript", "CSS")}
PLACEBO_SPLITS = 200
PLACEBO_SKILLS = 60
RESUME_SQL = """
SELECT toYear(published_at) AS year, count() AS posts
FROM (
    SELECT source_id, argMax(text, fetched_at) AS text, argMax(published_at, fetched_at) AS published_at
    FROM raw.vacancies WHERE source = 'telegram' AND is_vacancy GROUP BY source_id
)
WHERE match(lowerUTF8(text), '^[^a-zа-яё0-9]*(резюме|лучшие резюме|топ резюме|обзор за|ищу работу)')
GROUP BY year ORDER BY year
"""


def changes(sizes: list[dict[str, Any]], skills: list[dict[str, Any]]) -> dict[str, Any]:
    size: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    named: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for r in sizes:
        if period := period_of(int(r["year"])):
            for c in cells_of(r["role"], r["grade"]):
                size[c][r["channel"]][period] += int(r["posts"])
    for r in skills:
        if (period := period_of(int(r["year"]))) and r["skill"] in {s for v in SHOWN.values() for s in v}:
            for c in cells_of(r["role"], r["grade"]):
                named[c][r["skill"]][r["channel"]][period] += int(r["posts"])
    metrics: dict[str, Any] = {}
    for cell, shown in SHOWN.items():
        total = {p: sum(n[p] for n in size[cell].values()) for p in ("base", "recent")}
        found = within_change(size[cell], {s: named[cell][s] for s in shown}, "base", "recent")
        for s in shown:
            hits = {p: sum(n[p] for n in named[cell][s].values()) for p in ("base", "recent")}
            metrics[f"pooled.{cell}.{s}"] = {
                "value": round(hits["recent"] / total["recent"] - hits["base"] / total["base"], 4)
            }
            metrics[f"within.{cell}.{s}"] = {
                "value": round(found[s][1] - found[s][0], 4),
                "p": float(f"{found[s][2]:.3g}"),
            }
        channels = panel(size[cell], ("base", "recent"))
        metrics[f"new_channels_share.{cell}"] = share(
            total["recent"] - sum(size[cell][c]["recent"] for c in channels), total["recent"]
        )
    return metrics


def junior(sizes: list[dict[str, Any]]) -> dict[str, Any]:
    grade_of = {name: g for g, names in GRADES.items() for name in names}
    last = str(max(RECENT_YEARS))
    years = [str(y) for y in range(BASE_YEAR, max(RECENT_YEARS) + 1)]
    size: dict[str, Counter[str]] = defaultdict(Counter)
    named: dict[str, Counter[str]] = defaultdict(Counter)
    for r in sizes:
        if grade_of.get(r["grade"]) and str(r["year"]) in years:
            size[r["channel"]][str(r["year"])] += int(r["posts"])
            if grade_of[r["grade"]] == "junior":
                named[r["channel"]][str(r["year"])] += int(r["posts"])
    pooled = {y: sum(named[c][y] for c in size) / sum(size[c][y] for c in size) for y in (years[0], last)}
    _, found = fixed_weight_series(size, {"junior": named}, years)
    line = found["junior"][0]
    return {
        "grades.junior.pooled.base": {"value": round(pooled[years[0]], 4)},
        "grades.junior.pooled.last": {"value": round(pooled[last], 4)},
        "grades.junior.fixed.base": {"value": round(line[0], 4)},
        "grades.junior.fixed.last": {"value": round(line[-1], 4)},
    }


def placebo(sizes: list[dict[str, Any]], skills: list[dict[str, Any]]) -> dict[str, Any]:
    posts: Counter[str] = Counter()
    for r in sizes:
        if int(r["year"]) == BASE_YEAR and cells_of(r["role"], r["grade"]):
            posts[r["channel"]] += int(r["posts"])
    named: dict[str, Counter[str]] = defaultdict(Counter)
    mentions: Counter[str] = Counter()
    for r in skills:
        if int(r["year"]) == BASE_YEAR and cells_of(r["role"], r["grade"]):
            named[r["skill"]][r["channel"]] += int(r["posts"])
            mentions[r["skill"]] += int(r["posts"])
    listed = sorted(mentions, key=lambda s: (-mentions[s], s))[:PLACEBO_SKILLS]
    channels = sorted(posts)
    flagged = tests = 0
    for split in range(PLACEBO_SPLITS):
        half = set(random.Random(SEED + split).sample(channels, len(channels) // 2))
        n1 = sum(posts[c] for c in half)
        n2 = sum(posts[c] for c in channels if c not in half)
        for s in listed:
            k1 = sum(named[s][c] for c in half)
            k2 = sum(named[s][c] for c in channels if c not in half)
            p = (k1 + k2) / (n1 + n2)
            z = abs(k1 / n1 - k2 / n2) / (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
            flagged += 2 * (1 - NormalDist().cdf(z)) < 0.05
            tests += 1
    return {"placebo.pooled_test": {"value": round(flagged / tests, 4), "n": flagged, "denominator": tests}}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    sizes, skills, _ = clickhouse_rows()
    metrics = changes(sizes, skills) | junior(sizes) | placebo(sizes, skills)
    load_dotenv()
    with ClickHouse() as ch:
        metrics |= {f"resume_posts.{r['year']}": {"value": int(r["posts"])} for r in fetch(ch, RESUME_SQL)}
    path = write_result(
        "market/composition.json",
        metrics,
        source_script="src/radar/site/composition.py",
        base_year=BASE_YEAR,
        recent_years=list(RECENT_YEARS),
        placebo_splits=PLACEBO_SPLITS,
        placebo_skills=PLACEBO_SKILLS,
    )
    print(f"{len(metrics)} metrics -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
