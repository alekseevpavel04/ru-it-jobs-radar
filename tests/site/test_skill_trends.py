"""radar.site.skill_trends: thresholds on mentions and on yearly posts, junior skills listed for everyone."""

from radar.site import skill_trends


def size(year: int, ml: int, junior: int, posts: int) -> dict[str, int]:
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
    assert m["share.ml_junior.pandas.recent"] == {"value": 0.5, "n": 25, "denominator": 50}
    assert m["share.ml_all.pandas.recent"]["n"] == 30
    assert "share.ml_junior.Rare.recent" not in m
    assert "posts.ml_all.2022" not in m and "share.ml_all.Docker.2022" not in m
    assert m["share.ml_all.Docker.2025"] == {"value": 0.1333, "n": 60, "denominator": 450}
    assert "posts.ml_junior.2025" not in m
