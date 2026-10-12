import json
import re
import shlex
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from radar.config import ROOT
from radar.site import digest, export
from radar.site.digest import (
    build,
    change,
    figures,
    latest_complete_week,
    main,
    parse_week,
    previous_issue,
    queries,
    render,
    render_index,
    week_label,
    write,
)

W40 = date(2026, 9, 28)


def plain(text: str) -> str:
    """The page with ordinary spaces where numbers carry non-breaking ones."""
    return text.replace(" ", " ")


W39 = date(2026, 9, 21)
NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def weekly_row(week: date, source: str, bucket: str, **counts: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "week": week.isoformat(),
        "source": source,
        "role_bucket": bucket,
        "days_collected": None,
        "complete_days": None,
        "is_first_collected_week": 0,
        "vacancies_active": None,
        "vacancies_new": None,
        "vacancies_closed": None,
        "telegram_posts": None,
        "telegram_posts_before_dedup": None,
        "telegram_channels_week": None,
    }
    row.update(counts)
    return row


def snapshot(week: date, bucket: str, active: int, new: int, closed: int) -> dict[str, Any]:
    # ClickHouse quotes 64-bit integers in JSON; the read-only account cannot switch that off.
    return weekly_row(
        week,
        "trudvsem_snapshot",
        bucket,
        days_collected=7,
        complete_days=7,
        vacancies_active=str(active),
        vacancies_new=str(new),
        vacancies_closed=str(closed),
    )


def telegram(week: date, bucket: str, posts: int, before: int, channels: int = 30) -> dict[str, Any]:
    return weekly_row(
        week,
        "telegram",
        bucket,
        telegram_posts=str(posts),
        telegram_posts_before_dedup=str(before),
        telegram_channels_week=str(channels),
    )


def skill_rows(
    week: date, counts: dict[str, int], base: int, source: str = "trudvsem_snapshot"
) -> list[dict[str, Any]]:
    return [
        {
            "week": week.isoformat(),
            "source": source,
            "skill": skill,
            "vacancies_mentioning": str(n),
            "vacancies_in_week": str(base),
        }
        for skill, n in counts.items()
    ]


def salary_row(
    source: str, level: str, group: str, n_from: int, from_median: float | None, **rest: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "level": level,
        "role_group": group,
        "vacancies": "500",
        "with_salary_rub": "450",
        "with_from": str(n_from),
        "salary_from_median": from_median,
        "with_to": str(n_from),
        "salary_to_median": None if from_median is None else from_median + 50000,
    }
    row.update(rest)
    return row


def results(skill_counts: dict[str, int] | None = None, base: int = 1000) -> dict[str, list[dict[str, Any]]]:
    skill_counts = skill_counts or {"Python": 300, "SQL": 250, "Docker": 15}
    return {
        "weekly": [
            snapshot(W39, "backend", 1000, 100, 90),
            snapshot(W39, "ml_ds", 200, 20, 10),
            snapshot(W40, "backend", 1030, 110, 80),
            snapshot(W40, "frontend", 50, 5, 1),
            snapshot(W40, "ml_ds", 210, 25, 15),
            weekly_row(
                W40,
                "habr",
                "backend",
                days_collected=3,
                complete_days=3,
                is_first_collected_week=1,
                vacancies_active="400",
                vacancies_new="20",
                vacancies_closed="0",
            ),
            telegram(W39, "backend", 100, 110),
            telegram(W39, "security", 1, 1),
            telegram(W40, "backend", 120, 130, channels=41),
            telegram(W40, "security", 13, 14, channels=41),
            telegram(W40, "multi_post", 30, 30, channels=41),
        ],
        "last_week": [
            {"source": "habr", "last_week": W40.isoformat()},
            {"source": "telegram", "last_week": W40.isoformat()},
            {"source": "trudvsem_snapshot", "last_week": W40.isoformat()},
        ],
        # complete days through 10 October: W40 closures of the snapshot are known (through 4 October)
        "closure_window": [
            {"source": "habr", "last_complete_day": "2026-10-10"},
            {"source": "trudvsem_snapshot", "last_complete_day": "2026-10-10"},
        ],
        # W39 shares: Python 28.0%, SQL 26.0%, Docker 1.5%; W40 from the arguments over `base`
        "skills": [
            *skill_rows(W39, {"Python": 280, "SQL": 260, "Docker": 15}, 1000),
            *skill_rows(W40, skill_counts, base),
        ],
        "periods": [
            {"source": "trudvsem", "period_from": "2026-06-30", "period_to": "2026-10-04"},
            {"source": "telegram", "period_from": "2026-07-01", "period_to": "2026-10-08"},
        ],
        "salaries": [
            salary_row("trudvsem", "group", "dev", 40, 150000.0),
            salary_row("trudvsem", "group", "ml_ds_data", 8, 60000.0),
            salary_row("trudvsem", "source", "all", 48, 140000.0),
            salary_row("habr", "source", "all", 3, None),
        ],
    }


def test_latest_complete_week_follows_moscow_time() -> None:
    # Sunday 11 October: 23:00 in Moscow the week is still running, at 00:30 on Monday it is over.
    assert latest_complete_week(datetime(2026, 10, 11, 20, 0, tzinfo=UTC)) == W40
    assert latest_complete_week(datetime(2026, 10, 11, 21, 30, tzinfo=UTC)) == date(2026, 10, 5)


def test_iso_week_labels_across_the_year_boundary() -> None:
    assert parse_week("2026-W01") == date(2025, 12, 29)
    assert week_label(date(2025, 12, 29)) == "2026-W01"
    assert week_label(parse_week("2026-W40")) == "2026-W40"


def test_queries_read_only_the_marts_of_the_week() -> None:
    sqls = queries(W40)
    assert set(sqls) == {"weekly", "last_week", "closure_window", "skills", "periods", "salaries"}
    for sql in sqls.values():
        assert set(re.findall(r"\bFROM\s+(\w+)\.", sql)) == {"marts"}
        assert sql.endswith("FORMAT JSON")
    assert "'2026-09-21', '2026-09-28'" in sqls["weekly"]
    assert "FROM marts.mart_skills_weekly WHERE week IN ('2026-09-21', '2026-09-28')" in sqls["skills"]
    # cells and the other levels of mart_salaries would count every vacancy once per level
    assert "WHERE level IN ('group', 'source')" in sqls["salaries"]


def test_build_sums_buckets_into_role_groups() -> None:
    issue = build(W40, results(), NOW)
    snap = issue.vacancies[0]
    assert snap.source == "trudvsem_snapshot"
    assert snap.current is not None and snap.previous is not None
    assert snap.current.total.active == 1030 + 50 + 210
    assert snap.current.groups["Разработка"].active == 1080
    assert sum(g.new for g in snap.current.groups.values()) == snap.current.total.new
    assert snap.previous.total.active == 1200
    assert issue.telegram.current is not None
    assert issue.telegram.current.groups["Несколько вакансий в посте"] == 30


def test_salaries_are_the_group_and_source_rows() -> None:
    groups = build(W40, results(), NOW).salaries
    assert [(g.role_group, g.with_from, g.from_median) for g in groups["trudvsem"]] == [
        ("dev", 40, 150000),
        ("ml_ds_data", 8, 60000),
        ("all", 48, 140000),
    ]
    page = render(build(W40, results(), NOW))
    # below the threshold the median is not printed, its n is
    assert "| ML, DS, данные | 450 из 500 (90,0 %) | мало данных | 8 | мало данных | 8 |" in plain(page)
    assert "| Разработка, включая 1С | 450 из 500 (90,0 %) | 150 000 ₽ | 40 | 200 000 ₽ | 40 |" in plain(page)


def test_salary_threshold_is_the_site_rule() -> None:
    assert export.salary_min_n() == digest.SALARY_MIN_N


def test_mentions_above_the_base_stop_the_issue() -> None:
    with pytest.raises(ValueError, match="above the base"):
        build(W40, results({"Python": 1200}, base=1000), NOW)


def test_skills_move_week_over_week_from_the_weekly_mart() -> None:
    page = render(build(W40, results({"Python": 360, "SQL": 250, "Docker": 15}, base=1100), NOW))
    # Python 28.0% -> 32.7%, SQL 26.0% -> 22.7%; Docker (15 mentions) stays under the floor of a mover.
    assert "Python +4,7 п. п. (было 280 из 1 000 (28,0 %), стало 360 из 1 100 (32,7 %))" in plain(page)
    assert "SQL -3,3 п. п." in plain(page)
    assert "Docker" not in page.split("Сдвиги к прошлой неделе")[1].split("\n")[0]
    assert "- «Работа в России», снимки портала, вакансий недели 1 100, дней сбора 7:" in plain(page)
    assert "- Хабр Карьера: за эту неделю данных нет." in plain(page)


def test_change_prints_a_percent_only_from_a_base_that_carries_it() -> None:
    assert change(13, 1) == "+12"
    assert change(120, 100) == "+20 (+20,0 %)"
    assert change(90, 100) == "-10 (-10,0 %)"
    assert change(5, 5) == "0"


def test_page_text() -> None:
    page = render(build(W40, results(), NOW))
    assert "# Неделя с 28 сентября по 4 октября 2026" in page
    # Week over week with both weeks in the text, new vacancies with their base.
    assert "Неделей раньше 1 200 (дней сбора 7): +90 (+7,5 %)." in page
    assert "Новых 140 из 1 290 (10,9 %)." in page
    assert "| Разработка | 1 080 | 1 000 | +80 (+8,0 %) | 115 | 81 |" in page
    assert "это первая неделя сбора, вакансии первого дня в новые не входят" in page
    assert "Постов с вакансиями 163: это 163 из 174 (93,7 %) опубликованных" in plain(page)
    assert "Посты пришли из 41 канала. Неделей раньше 101 из 30 каналов" in plain(page)
    assert "Число каналов в неделе изменилось" in plain(page)
    assert "| Информационная безопасность | 13 | 1 | +12 |" in page
    assert "Python 300 из 1 000 (30,0 %)" in plain(page)
    assert "Сравнения зарплат с прошлым выпуском нет" in page
    assert "| Все IT-вакансии | 450 из 500 (90,0 %) | мало данных | 3 | мало данных | 3 |" in plain(page)
    assert "—" not in page and "–" not in page
    assert "None" not in page


def test_a_week_without_data_names_the_last_one() -> None:
    data = results()
    data["weekly"] = [r for r in data["weekly"] if r["source"] != "trudvsem_snapshot"]
    data["last_week"][2]["last_week"] = "2026-09-07"
    page = render(build(W40, data, NOW))
    assert "За эту неделю данных нет. Последняя неделя с данными началась 7 сентября 2026." in page


def test_closures_of_a_week_the_data_cannot_confirm_yet_are_not_printed() -> None:
    # An issue made on Monday 5 October: snapshot days complete through Sunday 4 October, so its
    # closures are known through 28 September (7 missing days) and the week of 28 September is open.
    data = results()
    data["closure_window"] = [
        {"source": "habr", "last_complete_day": "2026-10-04"},
        {"source": "trudvsem_snapshot", "last_complete_day": "2026-10-04"},
    ]
    issue = build(W40, data, NOW)
    assert issue.vacancies[0].closures_known_through == "2026-09-28"
    page = plain(render(issue))
    assert "Закрыто 96" not in page
    assert (
        "Закрытия этой недели ещё не известны: закрытой вакансия считается после 7 полных дней сбора без неё, "
        "а полные дни сбора есть по 4 октября 2026. Неделей раньше закрыто 100." in page
    )
    assert "| Группа ролей | Открыто | Неделей раньше | Изменение | Новые | Закрыты неделей раньше |" in page
    assert "| Разработка | 1 080 | 1 000 | +80 (+8,0 %) | 115 | 90 |" in page
    # Habr: its first week, three days missing close a vacancy, nothing earlier to fall back on
    assert "после 3 полных дней сбора без неё" in page


def test_closures_of_a_settled_week_are_printed() -> None:
    page = plain(render(build(W40, results(), NOW)))
    assert "Закрыто 96." in page
    assert "| Новые | Закрыты |" in page


def test_absent_days_are_the_dbt_macro() -> None:
    macro = (ROOT / "dbt" / "macros" / "history.sql").read_text(encoding="utf-8")
    body = re.search(r"macro history_absent_days_to_close.*?multiIf\((.*?)\)\s*\n", macro, re.S)
    grace = re.search(r"var\('history_grace_days', (\d+)\)", macro)
    assert body is not None and grace is not None
    found = dict(re.findall(r"= '(\w+)', (\d+|\{\{ history_grace_days\(\) \}\})", body.group(1)))
    found = {k: int(grace.group(1)) if "grace" in v else int(v) for k, v in found.items()}
    assert found == digest.ABSENT_DAYS_TO_CLOSE


def test_next_issue_compares_salaries_with_the_previous(tmp_path: Path) -> None:
    write(build(W40, results(), NOW), tmp_path)
    after = results()
    after["salaries"][0]["salary_from_median"] = 155000.0
    w41 = date(2026, 10, 5)
    previous = previous_issue(tmp_path, week_label(w41))
    assert previous is not None and previous["week"] == "2026-W40"

    issue = build(w41, after, NOW, previous)
    assert issue.compared_with == "2026-W40"
    page = render(issue)
    assert "| +5 000 ₽ |" in plain(page)
    # a group under the threshold in either issue is not compared
    assert "| мало данных | 8 | нет сравнения |" in plain(page)
    assert "salaries_previous" not in figures(issue)


def test_index_lists_issues_newest_first() -> None:
    index = render_index(
        [
            {"week": "2026-W40", "week_start": "2026-09-28", "week_end": "2026-10-04"},
            {"week": "2026-W41", "week_start": "2026-10-05", "week_end": "2026-10-11"},
        ]
    )
    assert index.index("2026-W41") < index.index("2026-W40")
    assert "- [2026-W41, с 5 по 11 октября 2026](./2026-W41)" in index


def test_render_from_a_bundle_is_repeatable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"week": "2026-W40", "results": results()}), encoding="utf-8")
    out = tmp_path / "digest"
    assert main(["render", "--bundle", str(bundle), "--out", str(out)]) == 0
    first = (out / "2026-W40.md").read_bytes()
    assert main(["render", "--bundle", str(bundle), "--out", str(out)]) == 0
    assert (out / "2026-W40.md").read_bytes() == first
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == ["2026-W40.json", "2026-W40.md", "index.md"]
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["week"] == "2026-W40" and summary["issues"] == 1


def test_queries_command_prints_the_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["queries", "--week", "2026-W40"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["week"] == "2026-W40"
    assert [q["name"] for q in plan["queries"]] == list(queries(W40))


def _js_object(source: str, name: str) -> dict[str, str]:
    body = re.search(rf"export const {name} = \{{(.*?)\}};", source, re.S)
    assert body is not None, name
    return {k.strip('"'): v for k, v in re.findall(r'^\s*("[^"]+"|\w+): "([^"]*)"', body.group(1), re.M)}


def test_labels_are_the_site_labels() -> None:
    source = (ROOT / "site" / "src" / "components" / "labels.js").read_text(encoding="utf-8")
    assert _js_object(source, "ROLE") == digest.ROLE
    assert _js_object(source, "ROLE_GROUP") == digest.ROLE_GROUP
    assert _js_object(source, "GRADE") == digest.GRADE
    assert _js_object(source, "REGION") == digest.REGION
    assert _js_object(source, "MART_GROUP") == digest.MART_GROUP
    assert _js_object(source, "SKILL") == digest.SKILL


def test_workflow_calls_the_module_and_is_wired() -> None:
    workflow = json.loads((ROOT / digest.WORKFLOW).read_text(encoding="utf-8"))
    names = {n["name"] for n in workflow["nodes"]}
    for source, outputs in workflow["connections"].items():
        assert source in names
        assert all(link["node"] in names for branch in outputs["main"] for link in branch)
    commands = [n["parameters"]["command"] for n in workflow["nodes"] if n["type"] == "n8n-nodes-base.executeCommand"]
    assert [shlex.split(c)[:4] for c in commands] == [
        ["python3", "-m", "radar.site.digest", "queries"],
        ["python3", "-m", "radar.site.digest", "render"],
    ]
    http = next(n for n in workflow["nodes"] if n["type"] == "n8n-nodes-base.httpRequest")
    assert http["parameters"]["url"] == "http://clickhouse:8123/"
    import_script = (ROOT / "scripts" / "n8n_import.sh").read_text(encoding="utf-8")
    assert http["credentials"]["httpBasicAuth"]["id"] in import_script
    assert workflow["id"] in import_script


def _issue_from_figures(doc: dict[str, Any], previous: dict[str, Any] | None) -> digest.Issue:
    def vacancy_week(d: dict[str, Any] | None) -> digest.VacancyWeek | None:
        if d is None:
            return None
        groups = {g: digest.Counts(**c) for g, c in d["groups"].items()}
        return digest.VacancyWeek(**{**d, "total": digest.Counts(**d["total"]), "groups": groups})

    def telegram_week(d: dict[str, Any] | None) -> digest.TelegramWeek | None:
        return None if d is None else digest.TelegramWeek(**d)

    def skill_week(d: dict[str, Any] | None) -> digest.SkillWeek | None:
        return None if d is None else digest.SkillWeek(**d)

    def salaries(d: dict[str, Any]) -> dict[str, list[digest.SalaryGroup]]:
        return {family: [digest.SalaryGroup(**g) for g in groups] for family, groups in d["salaries"].items()}

    plain_fields = ("week", "week_start", "week_end", "generated_at", "generated_by", "salary_periods")
    return digest.Issue(
        **{k: doc[k] for k in plain_fields},
        vacancies=[
            digest.VacancySection(
                **{**s, "current": vacancy_week(s["current"]), "previous": vacancy_week(s["previous"])}
            )
            for s in doc["vacancies"]
        ],
        telegram=digest.TelegramSection(
            **{
                **doc["telegram"],
                "current": telegram_week(doc["telegram"]["current"]),
                "previous": telegram_week(doc["telegram"]["previous"]),
            }
        ),
        skills=[
            digest.SkillSection(**{**s, "current": skill_week(s["current"]), "previous": skill_week(s["previous"])})
            for s in doc["skills"]
        ],
        salaries=salaries(doc),
        compared_with=doc["compared_with"],
        salaries_previous={} if previous is None else salaries(previous),
    )


def test_committed_issues_are_their_figures() -> None:
    # Every number of a published issue page comes from its data/<week>.json, and the index lists them all.
    issues = {str(d["week"]): d for d in digest.load_issues(digest.OUT)}
    assert issues
    for week, doc in issues.items():
        previous = issues[doc["compared_with"]] if doc["compared_with"] else None
        page = (digest.OUT / f"{week}.md").read_text(encoding="utf-8")
        assert render(_issue_from_figures(doc, previous)) == page, week
    assert (digest.OUT / "index.md").read_text(encoding="utf-8") == render_index(issues.values())
