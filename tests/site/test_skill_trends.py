"""radar.site.skill_trends: cells by direction and grade, thresholds on posts and mentions, growers by points."""

from collections import Counter, defaultdict
from typing import Any

from radar.site import skill_trends


def size(year: int, role: str | None, grade: str | None, posts: int) -> dict[str, Any]:
    return {"year": year, "role": role, "grade": grade, "posts": posts}


def mention(year: int, role: str | None, grade: str | None, skill: str, posts: int) -> dict[str, Any]:
    return {"year": year, "role": role, "grade": grade, "skill": skill, "posts": posts}


def test_cells_cover_all_it_the_direction_and_the_grade() -> None:
    assert skill_trends.cells_of("ml_engineer", "intern") == ["it_all", "it_junior", "ml_all", "ml_junior"]
    assert skill_trends.cells_of("system_analyst", "lead") == [
        "it_all",
        "it_senior",
        "analytics_all",
        "analytics_senior",
    ]
    assert skill_trends.cells_of("designer", None) == ["it_all"]


def test_cells_need_enough_posts_and_mentions() -> None:
    sizes = [
        size(2022, "ml_engineer", "middle", 100),
        size(2025, "ml_engineer", "middle", 400),
        size(2025, "ml_engineer", "junior", 250),
        size(2025, "mobile", "junior", 150),
    ]
    skills = [
        mention(2025, "ml_engineer", "middle", "Docker", 60),
        mention(2025, "ml_engineer", "junior", "pandas", 25),
        mention(2025, "ml_engineer", "middle", "pandas", 5),
        mention(2025, "ml_engineer", "junior", "Rare", 3),
    ]
    m = skill_trends.trends(sizes, skills)
    assert m["posts.ml_all.recent"] == {"value": 650}
    assert m["share.ml_junior.pandas.recent"] == {"value": 0.1, "n": 25, "denominator": 250}
    assert m["share.ml_all.pandas.recent"]["n"] == 30
    assert "share.ml_junior.Rare.recent" not in m
    # 150 recent posts: the mobile junior cell is too small to show
    assert "posts.mobile_junior.recent" not in m
    # 100 posts in 2022: no base year for ML, recent shares only
    assert "posts.ml_all.2022" not in m and "share.ml_all.Docker.2022" not in m
    assert m["share.ml_all.Docker.2025"] == {"value": 0.0923, "n": 60, "denominator": 650}


def test_fast_growers_rank_by_points_not_by_ratio() -> None:
    sizes = [size(2022, "backend", "middle", 1000), size(2025, "backend", "middle", 1000)]
    skills = [mention(2022, "backend", "middle", f"S{i}", 300 - i) for i in range(35)]
    skills += [mention(2025, "backend", "middle", f"S{i}", 300 - i) for i in range(35)]
    # Tiny: 1 -> 20 posts, a twentyfold ratio but 1.9 points; Grown: 100 -> 160 posts, 6 points.
    skills += [mention(2022, "backend", "middle", "Tiny", 1), mention(2025, "backend", "middle", "Tiny", 20)]
    skills += [mention(2022, "backend", "middle", "Grown", 100), mention(2025, "backend", "middle", "Grown", 160)]
    grown = skill_trends.growing("backend_middle", *_counts(sizes, skills))
    assert grown.index("Grown") < grown.index("Tiny")


def _counts(
    sizes: list[dict[str, Any]], skills: list[dict[str, Any]]
) -> tuple[dict[str, Counter[int]], dict[str, dict[int, Counter[str]]], Counter[str]]:
    posts: dict[str, Counter[int]] = defaultdict(Counter)
    counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for r in sizes:
        posts["backend_middle"][r["year"]] += r["posts"]
    for r in skills:
        counts["backend_middle"][r["year"]][r["skill"]] += r["posts"]
    recent: Counter[str] = sum((counts["backend_middle"][y] for y in skill_trends.RECENT_YEARS), Counter())
    return posts, counts, recent


def test_grade_mix_counts_only_posts_that_name_a_grade() -> None:
    sizes = [
        size(2025, "backend", "junior", 30),
        size(2025, "qa", "intern", 10),
        size(2025, "backend", "middle", 20),
        size(2025, "backend", "lead", 40),
        size(2025, "backend", None, 100),
    ]
    m = skill_trends.grade_mix(sizes)
    assert m["grades.2025.graded"] == {"value": 0.5, "n": 100, "denominator": 200}
    assert m["grades.2025.junior"] == {"value": 0.4, "n": 40, "denominator": 100}
    assert m["grades.2025.senior"]["n"] == 40


def test_direction_mix_keeps_only_channels_of_every_year() -> None:
    rows = [{"year": y, "channel": "a", "role": "backend", "posts": 10} for y in range(2022, 2027)]
    rows += [{"year": y, "channel": "a", "role": "designer", "posts": 10} for y in range(2022, 2027)]
    # channel b starts in 2024: its ML posts must not lift the ML share
    rows += [{"year": y, "channel": "b", "role": "ml_engineer", "posts": 50} for y in range(2024, 2027)]
    m = skill_trends.direction_mix(rows)
    assert m["directions.panel_channels"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert m["directions.2026.backend"] == {"value": 0.5, "n": 10, "denominator": 20}
    assert m["directions.2026.ml"]["n"] == 0
    assert m["directions.2026.design"]["n"] == 10
    assert sum(v["n"] for k, v in m.items() if k.startswith("directions.2026.")) == 20
