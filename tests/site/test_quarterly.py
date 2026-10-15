from datetime import date

from radar.site.quarterly import benjamini_hochberg, last_finished, notes, quarter_of, report, year_before


def test_quarters_follow_the_monday_and_compare_with_the_year_before() -> None:
    assert quarter_of(date(2026, 9, 28)) == "2026Q3"
    assert quarter_of(date(2026, 10, 5)) == "2026Q4"
    assert year_before("2026Q3") == "2025Q3"
    assert last_finished(date(2026, 10, 15)) == "2026Q3"
    assert last_finished(date(2027, 2, 1)) == "2026Q4"


def test_benjamini_hochberg_matches_the_step_up_rule() -> None:
    assert [round(q, 4) for q in benjamini_hochberg([0.01, 0.04, 0.03, 0.5])] == [0.04, 0.0533, 0.0533, 0.5]


def rows(week: str, channel: str, posts: int, python: int, sql: int) -> tuple[list[dict], list[dict]]:
    size = [{"week": week, "channel": channel, "role": "backend", "grade": "middle", "posts": str(posts)}]
    named = [
        {
            "week": week,
            "channel": channel,
            "role": "backend",
            "grade": "middle",
            "skill": "Python",
            "posts": str(python),
        },
        {"week": week, "channel": channel, "role": "backend", "grade": "middle", "skill": "SQL", "posts": str(sql)},
    ]
    return size, named


def collect(*parts: tuple[list[dict], list[dict]]) -> tuple[list[dict], list[dict]]:
    return [r for p, _ in parts for r in p], [r for _, s in parts for r in s]


def test_report_calls_out_a_real_rise_and_keeps_noise_quiet() -> None:
    posts, skills = collect(
        rows("2025-07-07", "c1", 2000, 300, 400),
        rows("2026-07-06", "c1", 2000, 500, 410),
        rows("2026-10-05", "c1", 2000, 2000, 0),  # the next quarter stays out
    )
    rep = report(posts, skills, "2026Q3")
    assert [c["direction"] for c in rep["cells"]] == ["it", "backend"]
    by_skill = {r["skill"]: r for r in rep["cells"][0]["skills"]}
    assert round(by_skill["Python"]["diff"], 4) == 0.1 and by_skill["Python"]["q"] < 0.05
    assert by_skill["SQL"]["q"] > 0.05
    assert rep["metrics"]["share.it.Python.now"] == {"value": 0.25}
    text = notes(rep)
    assert "Python 15,0 % → 25,0 % (+10,0 п. п.)" in text
    assert "Без заметных изменений 1 из 2 навыков." in text
    assert "III квартал 2026 против III квартала 2025" in text
    assert "—" not in text


def test_a_new_channel_does_not_move_the_shares() -> None:
    # c2 starts posting in 2026 with no Python at all: pooled, Python would fall from 25 % to 12.5 %.
    posts, skills = collect(
        rows("2025-07-07", "c1", 2000, 500, 400),
        rows("2026-07-06", "c1", 2000, 500, 400),
        rows("2026-07-06", "c2", 2000, 0, 400),
    )
    rep = report(posts, skills, "2026Q3")
    python = {r["skill"]: r for r in rep["cells"][0]["skills"]}["Python"]
    assert (python["before"], python["now"], python["diff"]) == (0.25, 0.25, 0.0)
    assert rep["cells"][0]["channels"] == 1
    assert rep["metrics"]["posts.it.both_channels_now"] == {"value": 0.5, "n": 2000, "denominator": 4000}


def test_a_cell_below_the_post_floor_is_left_out() -> None:
    posts, skills = collect(rows("2025-07-07", "c1", 200, 30, 40), rows("2026-07-06", "c1", 2000, 500, 410))
    assert report(posts, skills, "2026Q3")["cells"] == []
