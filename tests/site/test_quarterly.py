from datetime import date

from radar.site.quarterly import last_finished, notes, period_of, report, span, window


def test_the_window_is_the_year_to_the_quarter_and_the_year_before() -> None:
    assert window("2026Q3") == (date(2024, 10, 1), date(2025, 10, 1), date(2026, 10, 1))
    assert window("2026Q4") == (date(2025, 1, 1), date(2026, 1, 1), date(2027, 1, 1))
    assert period_of(date(2026, 9, 28), "2026Q3") == "now"
    assert period_of(date(2025, 9, 29), "2026Q3") == "before"
    assert period_of(date(2026, 10, 5), "2026Q3") is None
    assert span(date(2025, 10, 1), date(2026, 10, 1)) == "с октября 2025 по сентябрь 2026"
    assert last_finished(date(2026, 10, 15)) == "2026Q3"
    assert last_finished(date(2027, 2, 1)) == "2026Q4"


def rows(week: str, channel: str, posts: int, python: int, sql: int) -> tuple[list[dict], list[dict]]:
    size = [{"week": week, "channel": channel, "role": "backend", "grade": "middle", "posts": str(posts)}]
    named = [
        {"week": week, "channel": channel, "role": "backend", "grade": "middle", "skill": skill, "posts": str(n)}
        for skill, n in (("Python", python), ("SQL", sql))
    ]
    return size, named


def collect(*parts: tuple[list[dict], list[dict]]) -> tuple[list[dict], list[dict]]:
    return [r for p, _ in parts for r in p], [r for _, s in parts for r in s]


def test_report_calls_out_a_real_rise_and_keeps_noise_quiet() -> None:
    parts = []
    for i, channel in enumerate(("c1", "c2", "c3", "c4", "c5", "c6")):
        parts += [
            rows("2025-01-06", channel, 2000, 300 + i, 400),
            rows("2026-01-05", channel, 2000, 500 + i, 410),
        ]
    parts.append(rows("2026-10-05", "c1", 2000, 2000, 0))  # after the window
    rep = report(*collect(*parts), "2026Q3")
    assert [c["direction"] for c in rep["cells"]] == ["it", "backend"]
    by_skill = {r["skill"]: r for r in rep["cells"][0]["skills"]}
    assert round(by_skill["Python"]["diff"], 4) == 0.1 and by_skill["Python"]["q"] < 0.05
    assert by_skill["SQL"]["q"] > 0.05
    text = notes(rep, image="https://example.org/2026Q3.png")
    assert text.startswith("# Навыки в IT за год с октября 2025 по сентябрь 2026")
    assert "Сильнее всего за год вырос Python во всём IT." in text
    assert "![Что изменилось за год](https://example.org/2026Q3.png)" in text
    # the summary table names each skill once, the direction tables list only the moves beyond noise
    assert text.count("| Python | Всё IT |") + text.count("| Python | Backend |") == 1
    assert "| Python | 15,1 % | 25,1 % | ▲ +10,0 п. п. |" in text
    assert "Без заметных изменений: SQL." in text
    assert "(с октября 2024 по сентябрь 2025)" in text
    assert "—" not in text


def test_a_new_channel_does_not_move_the_shares() -> None:
    # c2 starts posting in the last year with no Python at all: pooled, Python would fall from 25 % to 12.5 %.
    posts, skills = collect(
        rows("2025-01-06", "c1", 2000, 500, 400),
        rows("2026-01-05", "c1", 2000, 500, 400),
        rows("2026-01-05", "c2", 2000, 0, 400),
    )
    rep = report(posts, skills, "2026Q3")
    python = {r["skill"]: r for r in rep["cells"][0]["skills"]}["Python"]
    assert (python["before"], python["now"], python["diff"]) == (0.25, 0.25, 0.0)
    assert rep["cells"][0]["channels"] == 1
    assert rep["metrics"]["posts.it.both_channels_now"] == {"value": 0.5, "n": 2000, "denominator": 4000}


def test_a_cell_below_the_post_floor_is_left_out() -> None:
    posts, skills = collect(rows("2025-01-06", "c1", 200, 30, 40), rows("2026-01-05", "c1", 2000, 500, 410))
    assert report(posts, skills, "2026Q3")["cells"] == []
