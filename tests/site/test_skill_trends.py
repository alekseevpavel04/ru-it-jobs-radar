"""radar.site.skill_trends: thresholds on mentions and on yearly posts, junior skills listed for everyone."""

from collections import Counter, defaultdict
from typing import Any

from radar.site import skill_trends


def size(year: int, ml: int, junior: int, posts: int) -> dict[str, Any]:
    return {"year": year, "ml": ml, "junior": junior, "posts": posts}


def mention(year: int, ml: int, junior: int, skill: str, posts: int) -> dict[str, object]:
    return {"year": year, "ml": ml, "junior": junior, "skill": skill, "posts": posts}


def test_thresholds_and_junior_skills_for_everyone() -> None:
    sizes = [size(2022, 1, 0, 100), size(2025, 1, 0, 400), size(2025, 1, 1, 50), size(2026, 0, 0, 1000)]
    skills = [
        mention(2025, 1, 0, "Docker", 60),
        mention(2025, 1, 1, "pandas", 25),
        mention(2025, 1, 0, "pandas", 5),
        mention(2025, 1, 1, "Rare", 3),
    ]
    m = skill_trends.trends(sizes, skills)
    assert m["posts.ml_all.recent"] == {"value": 450}
    assert m["share.ml_junior.pandas.recent"] == {"value": 0.5, "n": 25, "denominator": 50, "own": True}
    assert m["share.ml_all.pandas.recent"]["n"] == 30
    assert "share.ml_junior.Rare.recent" not in m
    assert "posts.ml_all.2022" not in m and "share.ml_all.Docker.2022" not in m
    assert m["share.ml_all.Docker.2025"] == {"value": 0.1333, "n": 60, "denominator": 450}
    assert m["share.ml_all.Docker.recent"]["own"] is True
    assert "posts.ml_junior.2025" not in m


def test_fast_growers_join_the_top_and_new_ones_lead() -> None:
    sizes = [size(2022, 0, 0, 1000), size(2025, 0, 0, 1000)]
    skills = [mention(2022, 0, 0, f"S{i}", 300 - i) for i in range(35)]
    skills += [mention(2025, 0, 0, f"S{i}", 300 - i) for i in range(35)]
    skills += [mention(2022, 0, 0, "Grown", 10), mention(2025, 0, 0, "Grown", 40), mention(2025, 0, 0, "Fresh", 30)]
    grown = skill_trends.growing("it_all", *_counts(sizes, skills))
    assert grown[:2] == ["Fresh", "Grown"]


def _counts(
    sizes: list[dict[str, Any]], skills: list[dict[str, Any]]
) -> tuple[dict[str, Counter[int]], dict[str, dict[int, Counter[str]]], Counter[str]]:
    posts: dict[str, Counter[int]] = defaultdict(Counter)
    counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for r in sizes:
        posts["it_all"][r["year"]] += r["posts"]
    for r in skills:
        counts["it_all"][r["year"]][r["skill"]] += r["posts"]
    recent: Counter[str] = sum((counts["it_all"][y] for y in skill_trends.RECENT_YEARS), Counter())
    return posts, counts, recent
