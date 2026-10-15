"""radar.autopilot.aggregates: year files, the cross-post hash, and skill_trends read from the files."""

from datetime import UTC, date, datetime
from pathlib import Path

from radar.autopilot import aggregates
from radar.autopilot.aggregates import Counts, moscow_week, text_hash
from radar.site import skill_trends


def test_a_copy_signed_by_another_channel_has_the_same_hash() -> None:
    first = "Python-разработчик (Middle)\nУдалёнка! Пишите @hr_anna https://t.me/jobs_a/1 #python #remote"
    copy = "python разработчик middle\n\nУдаленка  Пишите @hr_boris https://t.me/jobs_b/77 #вакансия"
    assert text_hash(first) == text_hash(copy)
    assert text_hash(first) != text_hash("Java-разработчик (Middle)")
    assert text_hash(None) == text_hash("")


def test_the_week_is_the_moscow_monday() -> None:
    # Sunday 22:30 UTC is already Monday in Moscow
    assert moscow_week(datetime(2026, 10, 11, 22, 30, tzinfo=UTC)) == date(2026, 10, 12)
    assert moscow_week(datetime(2026, 10, 11, 20, 59, tzinfo=UTC)) == date(2026, 10, 5)


def test_counts_go_to_sorted_year_files_and_back(tmp_path: Path) -> None:
    counts = Counts()
    counts.add(date(2026, 1, 5), "jobs_b", "backend", None, ["SQL", "Python", "SQL"])
    counts.add(date(2026, 1, 5), "jobs_a", "backend", "middle", ["Python"])
    counts.add(date(2025, 12, 29), "jobs_a", "qa", "junior", [])
    aggregates.save(counts, tmp_path)
    assert sorted(p.name for p in (tmp_path / "posts").iterdir()) == ["2025.csv", "2026.csv"]
    assert (tmp_path / "posts" / "2026.csv").read_text(encoding="utf-8").splitlines() == [
        "week,channel,role,grade,posts",
        "2026-01-05,jobs_a,backend,middle,1",
        "2026-01-05,jobs_b,backend,,1",
    ]
    # a skill named twice in a post counts once
    assert "2026-01-05,jobs_b,backend,,SQL,1" in (tmp_path / "skills" / "2026.csv").read_text(encoding="utf-8")
    assert aggregates.load(tmp_path) == counts
    assert aggregates.load(tmp_path, [2025]).posts == {("2025-12-29", "jobs_a", "qa", "junior"): 1}


def test_skill_trends_from_the_files_equals_the_same_rows_from_clickhouse(tmp_path: Path) -> None:
    counts = Counts()
    for i in range(320):
        week = date(2022, 3, 7) if i < 310 else date(2025, 3, 3)
        counts.add(week, f"ch{i % 3}", "backend", "middle" if i % 2 else None, ["Go"] if i % 4 else ["Go", "SQL"])
    for i in range(400):
        counts.add(date(2025, 6, 2), f"ch{i % 3}", "ml_engineer", "junior", ["Python"] if i % 5 else ["PyTorch"])
    aggregates.save(counts, tmp_path)
    from_files = skill_trends.metrics_of(*skill_trends.aggregate_rows(tmp_path))

    sizes, skills, directions = aggregates.trend_rows(counts)
    # what clickhouse_rows returns: the same rows, NULL for an empty grade, in another order
    reference = skill_trends.metrics_of(
        skill_trends.ordered(reversed(sizes)), skill_trends.ordered(reversed(skills)), skill_trends.ordered(directions)
    )
    assert from_files == reference
    assert from_files["posts.it_all.recent"] == {"value": 410}
    assert from_files["share.ml_junior.PyTorch.recent"] == {"value": 0.2, "n": 80, "denominator": 400}


def test_only_the_open_weeks_keep_their_hashes(tmp_path: Path) -> None:
    hashes = {"2026-09-28": {"a"}, "2026-10-05": {"b"}, "2026-10-12": {"c", "d"}}
    aggregates.write_week_hashes(hashes, date(2026, 10, 15), tmp_path)
    assert aggregates.read_week_hashes(tmp_path) == {"2026-10-05": {"b"}, "2026-10-12": {"c", "d"}}
