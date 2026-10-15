"""Skills of the site: what Telegram vacancy posts name, by year, for all IT, IT juniors, ML and ML juniors.

    python -m radar.site.skill_trends

Writes results/market/skill_trends.json. The skills are the extracted ones (staging.int_extracted_weekly_units,
single-vacancy IT posts after the weekly cross-post dedup), the same as the yearly series of
results/market/facts.json. Only shares a reader can lean on leave the file:

- a skill is listed for a group when MIN_MENTIONS or more posts of RECENT_YEARS name it, the TOP_SKILLS
  most named of each group; all IT and ML also get the skills listed for their juniors, so a junior share
  always has the share of everyone next to it;
- a year of a group's series is kept when the group has MIN_YEAR_POSTS posts or more that year, so the
  ML series starts later and ML juniors have the two-year cut only.
"""

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from radar.config import load_dotenv
from radar.results import write_result
from radar.site.market_facts import JUNIOR_GRADES, ML_ROLES, RECENT_YEARS, YEARS, _recent_group, fetch, share
from radar.stream.clickhouse import ClickHouse

GROUPS = ("it_all", "it_junior", "ml_all", "ml_junior")
JUNIORS_OF = {"it_all": "it_junior", "ml_all": "ml_junior"}
MIN_MENTIONS = 20
MIN_YEAR_POSTS = 300
TOP_SKILLS = 30

_UNITS = f"FROM staging.int_extracted_weekly_units WHERE source = 'telegram' AND is_it AND toYear(week) IN {YEARS}"
SIZE_SQL = f"""
SELECT toYear(week) AS year, role IN {ML_ROLES} AS ml, grade IN {JUNIOR_GRADES} AS junior, count() AS posts
{_UNITS}
GROUP BY year, ml, junior
"""
SKILL_SQL = f"""
SELECT toYear(week) AS year, role IN {ML_ROLES} AS ml, grade IN {JUNIOR_GRADES} AS junior, skill, count() AS posts
FROM staging.int_extracted_weekly_units ARRAY JOIN skills AS skill
WHERE source = 'telegram' AND is_it AND toYear(week) IN {YEARS}
GROUP BY year, ml, junior, skill
"""


def trends(sizes: Iterable[Mapping[str, Any]], skills: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    posts: dict[str, Counter[int]] = defaultdict(Counter)
    for r in sizes:
        for g in _recent_group(r["ml"], r["junior"]):
            posts[g][int(r["year"])] += int(r["posts"])
    counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for r in skills:
        for g in _recent_group(r["ml"], r["junior"]):
            counts[g][int(r["year"])][r["skill"]] += int(r["posts"])

    recent = {g: sum((counts[g][y] for y in RECENT_YEARS), Counter()) for g in GROUPS}
    listed = {g: [s for s, n in recent[g].most_common() if n >= MIN_MENTIONS][:TOP_SKILLS] for g in GROUPS}
    for parent, junior in JUNIORS_OF.items():
        listed[parent] += [s for s in listed[junior] if s not in listed[parent]]

    metrics: dict[str, Any] = {}
    for g in GROUPS:
        recent_posts = sum(posts[g][y] for y in RECENT_YEARS)
        kept_years = [y for y in YEARS if posts[g][y] >= MIN_YEAR_POSTS]
        metrics[f"posts.{g}.recent"] = {"value": recent_posts}
        for y in kept_years:
            metrics[f"posts.{g}.{y}"] = {"value": posts[g][y]}
        for s in listed[g]:
            metrics[f"share.{g}.{s}.recent"] = share(recent[g][s], recent_posts)
            for y in kept_years:
                metrics[f"share.{g}.{s}.{y}"] = share(counts[g][y][s], posts[g][y])
    return metrics


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    load_dotenv()
    with ClickHouse() as ch:
        metrics = trends(fetch(ch, SIZE_SQL), fetch(ch, SKILL_SQL))
    path = write_result(
        "market/skill_trends.json",
        metrics,
        source_script="src/radar/site/skill_trends.py",
        groups=list(GROUPS),
        recent_years=list(RECENT_YEARS),
        min_mentions=MIN_MENTIONS,
        min_year_posts=MIN_YEAR_POSTS,
        top_skills=TOP_SKILLS,
        definitions={
            "groups": "it_all: Telegram single-vacancy IT posts; it_junior: graded intern or junior; ml_all: roles "
            f"{', '.join(ML_ROLES)}; ml_junior: both.",
            "share": "posts of the group naming the skill (extracted skills) of the group's posts; recent pools "
            f"{RECENT_YEARS[0]}-{RECENT_YEARS[-1]}, a year by the week's Monday.",
        },
    )
    print(f"{len(metrics)} metrics -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
