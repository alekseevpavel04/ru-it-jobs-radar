"""radar.site.skill_trends: cells, thresholds on posts and mentions, change within channels, growers by points."""

from typing import Any

from radar.site import skill_trends


def size(year: int, role: str | None, grade: str | None, posts: int, channel: str = "c1") -> dict[str, Any]:
    return {"year": year, "channel": channel, "role": role, "grade": grade, "posts": posts}


def mention(
    year: int, role: str | None, grade: str | None, skill: str, posts: int, channel: str = "c1"
) -> dict[str, Any]:
    return {"year": year, "channel": channel, "role": role, "grade": grade, "skill": skill, "posts": posts}


def test_cells_cover_all_it_the_direction_and_the_grade() -> None:
    assert skill_trends.cells_of("ml_engineer", "intern") == ["it_all", "it_junior", "ml_all", "ml_junior"]
    assert skill_trends.cells_of("system_analyst", "lead") == [
        "it_all",
        "it_senior",
        "analytics_all",
        "analytics_senior",
    ]
    assert skill_trends.cells_of("backend", None) == ["it_all", "backend_all"]
    assert skill_trends.cells_of("designer", "senior") == [] and skill_trends.cells_of("manager", None) == []


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
    m = skill_trends.trends(sizes, skills)
    listed = [k.split(".")[2] for k in m if k.startswith("share.backend_middle.") and k.endswith(".recent")]
    assert listed[:30] == [f"S{i}" for i in range(30)]
    assert listed[30:32] == ["Grown", "Tiny"]
    assert m["change.backend_middle.Grown"]["value"] == 0.06


def test_a_new_channel_does_not_move_the_change() -> None:
    # c2 posts only in 2025, none of them with Python: pooled, Python would fall from 30 % to 15 %.
    sizes = [size(2022, "backend", "middle", 1000), size(2025, "backend", "middle", 1000)]
    sizes += [size(2025, "backend", "middle", 1000, channel="c2")]
    skills = [mention(2022, "backend", "middle", "Python", 300), mention(2025, "backend", "middle", "Python", 300)]
    m = skill_trends.trends(sizes, skills)
    assert m["share.backend_middle.Python.recent"]["value"] == 0.15
    assert m["change.backend_middle.Python"] == {"value": 0.0, "p": 1.0, "q_bh": 1.0, "base": 0.3, "recent": 0.3}
    assert m["panel.backend_middle.recent"] == {"value": 0.5, "n": 1000, "denominator": 2000}


def test_skills_tied_in_mentions_go_by_name() -> None:
    sizes = [size(2025, "backend", "middle", 1000)]
    skills = [mention(2025, "backend", "middle", f"S{i:02d}", 300 - i) for i in range(29)]
    skills += [mention(2025, "backend", "middle", name, 50) for name in ("Zeta", "Alpha")]
    m = skill_trends.trends(sizes, skills)
    assert "share.backend_middle.Alpha.recent" in m and "share.backend_middle.Zeta.recent" not in m


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
    assert m["grades.2025.junior"] == {"value": 0.4}
    assert m["grades.2025.senior"] == {"value": 0.4}


def test_a_growing_junior_channel_does_not_lift_the_junior_share() -> None:
    # Channel j posts juniors only and grows tenfold; inside each channel nothing changes.
    sizes = [size(y, "backend", "middle", 100, channel="m") for y in (2022, 2026)]
    sizes += [size(2022, "backend", "junior", 10, channel="j"), size(2026, "backend", "junior", 100, channel="j")]
    m = skill_trends.grade_mix(sizes)
    assert m["grades.2022.junior"] == m["grades.2026.junior"]
    assert m["grades.change.junior"]["value"] == 0.0


def test_direction_mix_keeps_only_channels_of_every_year() -> None:
    rows = [{"year": y, "channel": "a", "role": "backend", "posts": 10} for y in range(2022, 2027)]
    rows += [{"year": y, "channel": "a", "role": "designer", "posts": 10} for y in range(2022, 2027)]
    # channel b starts in 2024: its ML posts must not lift the ML share
    rows += [{"year": y, "channel": "b", "role": "ml_engineer", "posts": 50} for y in range(2024, 2027)]
    m = skill_trends.direction_mix(rows)
    assert m["directions.panel_channels"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert m["directions.2026.backend"] == {"value": 0.5}
    assert m["directions.2026.ml"] == {"value": 0.0}
    assert m["directions.2026.design"] == {"value": 0.5}
    assert round(sum(v["value"] for k, v in m.items() if k.startswith("directions.2026.")), 4) == 1.0
