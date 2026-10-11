"""Weekly market digest: one short Russian page per ISO week in site/src/digest, built from the marts.

    python -m radar.site.digest queries [--week 2026-W40]
    python -m radar.site.digest render --bundle bundle.json [--out site/src/digest]
    python -m radar.site.digest run [--week 2026-W40] [--out site/src/digest]

The n8n workflow (infra/n8n/workflows/weekly_digest.json) calls `queries` for the SELECTs of the
latest complete week, sends each to ClickHouse as the read-only account, saves the answers as one
bundle and calls `render`. `run` does the same from the host without n8n. Both paths use the standard
library only up to the HTTP call: inside the n8n container there is a bare CPython and no venv.

Only aggregates of marts.* are read and written (DECISIONS 2). Week over week works for what the marts
split by week (vacancies, Telegram posts). Skills and salaries are period aggregates, so every issue
keeps its own figures in data/<week>.json and the next issue compares its figures with them.
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from radar.config import ROOT

OUT = ROOT / "site" / "src" / "digest"
WORKFLOW = "infra/n8n/workflows/weekly_digest.json"
# Moscow has kept UTC+3 without daylight saving since 2014; a fixed offset spares the container tzdata.
MSK = timezone(timedelta(hours=3), "MSK")

# The API slice of trudvsem is a daily feed of modified records, not the open stock, so the digest
# follows the full portal snapshots and Habr (mart_weekly_dynamics description).
VACANCY_SOURCES = ("trudvsem_snapshot", "habr")
FAMILIES = ("trudvsem", "habr", "telegram")
TOP_SKILLS = 5
# Below this base a relative change ("+1200 %" from 1 to 13) says nothing; the counts stay.
CHANGE_MIN_BASE = 20
TOP_MOVERS = 3
# A skill moves in the digest only when both issues count at least this many mentions of it.
MOVER_MIN_MENTIONS = 20
# Same rule as radar.site.export.salary_min_n() (the smallest cell whose 95% interval for the median
# drops the min and the max); a test pins the two together, export itself needs pydantic.
SALARY_MIN_N = 9
TOP_SALARY_CELLS = 5

# Russian names of the mart codes, as on the site (site/src/components/labels.js; a test keeps them equal).
ROLE = {
    "ml_ds": "ML и Data Science",
    "data_engineering": "Инженерия данных",
    "data_analytics": "Аналитика данных",
    "developer_1c": "1С",
    "business_system_analysis": "Бизнес- и системный анализ",
    "security": "Информационная безопасность",
    "devops": "DevOps",
    "qa": "Тестирование",
    "mobile": "Мобильная разработка",
    "fullstack": "Fullstack",
    "backend": "Backend",
    "frontend": "Frontend",
    "embedded": "Встраиваемые системы",
    "sysadmin_support": "Администрирование и поддержка",
    "design": "Дизайн",
    "management": "Управление",
    "software_engineer": "Программист без уточнения",
    "other_it": "Прочее IT",
    "multi_post": "Несколько вакансий в посте",
}
ROLE_GROUP = {
    "backend": "Разработка",
    "frontend": "Разработка",
    "fullstack": "Разработка",
    "mobile": "Разработка",
    "embedded": "Разработка",
    "software_engineer": "Разработка",
    "developer_1c": "1С",
    "ml_ds": "Данные и ML",
    "data_engineering": "Данные и ML",
    "data_analytics": "Данные и ML",
    "sysadmin_support": "Администрирование, поддержка, DevOps",
    "devops": "Администрирование, поддержка, DevOps",
    "security": "Информационная безопасность",
    "qa": "Тестирование",
    "business_system_analysis": "Анализ и управление",
    "management": "Анализ и управление",
    "design": "Дизайн и прочее IT",
    "other_it": "Дизайн и прочее IT",
    "multi_post": "Несколько вакансий в посте",
}
GRADE = {
    "intern": "стажёр",
    "junior": "джуниор",
    "middle": "мидл",
    "senior": "сеньор",
    "lead": "лид",
    "unknown": "грейд не указан",
}
REGION = {
    "moscow": "Москва",
    "spb": "Санкт-Петербург",
    "other": "другие города",
    "remote": "удалённо",
    "unknown": "место не указано",
}
SOURCE = {
    "trudvsem_snapshot": "«Работа в России», снимки портала",
    "trudvsem": "«Работа в России»",
    "habr": "Хабр Карьера",
    "telegram": "Telegram-каналы",
}
MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
NBSP = " "


# --- weeks -----------------------------------------------------------------------------------------


def parse_week(label: str) -> date:
    """Monday of an ISO week given as 2026-W40."""
    year, _, week = label.partition("-W")
    return date.fromisocalendar(int(year), int(week), 1)


def week_label(monday: date) -> str:
    year, week, _ = monday.isocalendar()
    return f"{year}-W{week:02d}"


def latest_complete_week(now: datetime) -> date:
    """Monday of the last ISO week that ended before `now` in Moscow."""
    today = now.astimezone(MSK).date()
    return today - timedelta(days=today.weekday() + 7)


# --- queries ---------------------------------------------------------------------------------------


def queries(monday: date) -> dict[str, str]:
    """The SELECTs of one issue, all on marts.*; dates are formatted here, nothing else is spliced."""
    current, previous = monday.isoformat(), (monday - timedelta(days=7)).isoformat()
    return {
        "weekly": (
            "SELECT toString(week) AS week, source, role_bucket, days_collected, complete_days, "
            "is_first_collected_week, vacancies_active, vacancies_new, vacancies_closed, telegram_posts, "
            "telegram_posts_before_dedup FROM marts.mart_weekly_dynamics "
            f"WHERE week IN ('{previous}', '{current}') AND source IN ('trudvsem_snapshot', 'habr', 'telegram') "
            "ORDER BY source, week, role_bucket FORMAT JSON"
        ),
        "last_week": (
            "SELECT source, toString(max(week)) AS last_week FROM marts.mart_weekly_dynamics "
            f"WHERE week <= '{current}' GROUP BY source ORDER BY source FORMAT JSON"
        ),
        "skills": (
            "SELECT source, skill, any(skill_category) AS category, sum(vacancies_mentioning) AS mentioning "
            "FROM marts.mart_skills GROUP BY source, skill ORDER BY source, skill FORMAT JSON"
        ),
        # mart_skills carries cell sizes only on rows of a mentioned skill; mart_junior counts the same
        # units for every bucket (see radar.site.export.bucket_sizes).
        "skill_base": (
            "SELECT source, sum(vacancies) AS vacancies FROM marts.mart_junior GROUP BY source ORDER BY source "
            "FORMAT JSON"
        ),
        "periods": (
            "SELECT source, toString(min(period_from)) AS period_from, toString(max(period_to)) AS period_to "
            "FROM marts.mart_skills GROUP BY source ORDER BY source FORMAT JSON"
        ),
        "salaries": (
            "SELECT source, role_bucket, grade, region_group, with_from, salary_from_median, with_to, "
            "salary_to_median FROM marts.mart_salaries ORDER BY source, role_bucket, grade, region_group "
            "FORMAT JSON"
        ),
    }


# --- the issue -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    active: int
    new: int
    closed: int | None


@dataclass(frozen=True)
class VacancyWeek:
    days_collected: int
    complete_days: int
    first_collected_week: bool
    total: Counts
    groups: dict[str, Counts]


@dataclass(frozen=True)
class VacancySection:
    source: str
    last_week_with_data: str | None
    current: VacancyWeek | None
    previous: VacancyWeek | None


@dataclass(frozen=True)
class TelegramWeek:
    posts: int
    posts_before_dedup: int
    groups: dict[str, int]


@dataclass(frozen=True)
class TelegramSection:
    newest_post_day: str | None
    current: TelegramWeek | None
    previous: TelegramWeek | None


@dataclass(frozen=True)
class SkillFamily:
    period_from: str
    period_to: str
    base: int
    mentions: dict[str, int]


@dataclass(frozen=True)
class SalaryCell:
    role_bucket: str
    grade: str
    region_group: str
    with_from: int
    from_median: int | None
    with_to: int
    to_median: int | None


@dataclass(frozen=True)
class Issue:
    week: str
    week_start: str
    week_end: str
    generated_at: str
    generated_by: str
    vacancies: list[VacancySection]
    telegram: TelegramSection
    skills: dict[str, SkillFamily]
    salaries: dict[str, list[SalaryCell]]
    compared_with: str | None = None
    skills_previous: dict[str, SkillFamily] = field(default_factory=dict)
    salaries_previous: dict[str, list[SalaryCell]] = field(default_factory=dict)


def as_int(value: Any) -> int | None:
    """ClickHouse JSON quotes 64-bit integers unless the session changes a setting, which the
    read-only account may not; medians come as floats."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value)
    return int(value)


def need_int(value: Any) -> int:
    out = as_int(value)
    if out is None:
        raise ValueError("unexpected NULL in a count column")
    return out


def _sum_closed(values: Iterable[int | None]) -> int | None:
    values = list(values)
    return None if any(v is None for v in values) else sum(v for v in values if v is not None)


def vacancy_week(rows: list[Mapping[str, Any]]) -> VacancyWeek | None:
    if not rows:
        return None
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[ROLE_GROUP[r["role_bucket"]]].append(r)

    def counts(part: list[Mapping[str, Any]]) -> Counts:
        return Counts(
            active=sum(need_int(r["vacancies_active"]) for r in part),
            new=sum(need_int(r["vacancies_new"]) for r in part),
            closed=_sum_closed(as_int(r["vacancies_closed"]) for r in part),
        )

    first = rows[0]
    return VacancyWeek(
        days_collected=need_int(first["days_collected"]),
        complete_days=need_int(first["complete_days"]),
        first_collected_week=bool(as_int(first["is_first_collected_week"])),
        total=counts(rows),
        groups={g: counts(part) for g, part in sorted(by_group.items())},
    )


def telegram_week(rows: list[Mapping[str, Any]]) -> TelegramWeek | None:
    if not rows:
        return None
    groups: dict[str, int] = defaultdict(int)
    for r in rows:
        groups[ROLE_GROUP[r["role_bucket"]]] += need_int(r["telegram_posts"])
    return TelegramWeek(
        posts=sum(need_int(r["telegram_posts"]) for r in rows),
        posts_before_dedup=sum(need_int(r["telegram_posts_before_dedup"]) for r in rows),
        groups=dict(sorted(groups.items())),
    )


def build(
    monday: date,
    results: Mapping[str, list[Mapping[str, Any]]],
    generated_at: datetime,
    previous: Mapping[str, Any] | None = None,
) -> Issue:
    current_day, previous_day = monday.isoformat(), (monday - timedelta(days=7)).isoformat()
    weekly = results["weekly"]
    last_week = {r["source"]: r["last_week"] for r in results["last_week"]}

    def rows(source: str, week: str) -> list[Mapping[str, Any]]:
        return [r for r in weekly if r["source"] == source and r["week"] == week]

    vacancies = [
        VacancySection(
            source=source,
            last_week_with_data=last_week.get(source),
            current=vacancy_week(rows(source, current_day)),
            previous=vacancy_week(rows(source, previous_day)),
        )
        for source in VACANCY_SOURCES
    ]

    periods = {r["source"]: r for r in results["periods"]}
    base = {r["source"]: need_int(r["vacancies"]) for r in results["skill_base"]}
    mentions: dict[str, dict[str, int]] = defaultdict(dict)
    for r in results["skills"]:
        mentions[r["source"]][r["skill"]] = need_int(r["mentioning"])
    skills = {}
    for family in FAMILIES:
        if family not in periods:
            continue
        family_mentions = dict(sorted(mentions[family].items()))
        over = [s for s, n in family_mentions.items() if n > base.get(family, 0)]
        if over:
            raise ValueError(f"{family}: mentions above the base of {base.get(family, 0)} for {over[:5]}")
        skills[family] = SkillFamily(
            period_from=periods[family]["period_from"],
            period_to=periods[family]["period_to"],
            base=base.get(family, 0),
            mentions=family_mentions,
        )

    salaries: dict[str, list[SalaryCell]] = {}
    for family in ("trudvsem", "habr"):
        cells = [
            SalaryCell(
                role_bucket=r["role_bucket"],
                grade=r["grade"],
                region_group=r["region_group"],
                with_from=need_int(r["with_from"]),
                from_median=as_int(r["salary_from_median"]),
                with_to=need_int(r["with_to"]),
                to_median=as_int(r["salary_to_median"]),
            )
            for r in results["salaries"]
            if r["source"] == family and need_int(r["with_from"]) >= SALARY_MIN_N
        ]
        salaries[family] = sorted(cells, key=lambda c: (-c.with_from, c.role_bucket, c.grade, c.region_group))

    issue = Issue(
        week=week_label(monday),
        week_start=current_day,
        week_end=(monday + timedelta(days=6)).isoformat(),
        generated_at=generated_at.astimezone(UTC).isoformat(timespec="seconds"),
        generated_by="python -m radar.site.digest",
        vacancies=vacancies,
        telegram=TelegramSection(
            newest_post_day=periods["telegram"]["period_to"] if "telegram" in periods else None,
            current=telegram_week(rows("telegram", current_day)),
            previous=telegram_week(rows("telegram", previous_day)),
        ),
        skills=skills,
        salaries=salaries,
    )
    if previous is None:
        return issue
    return replace(
        issue,
        compared_with=str(previous["week"]),
        skills_previous={family: SkillFamily(**doc) for family, doc in previous["skills"].items()},
        salaries_previous={family: [SalaryCell(**c) for c in cells] for family, cells in previous["salaries"].items()},
    )


def figures(issue: Issue) -> dict[str, Any]:
    """What data/<week>.json keeps: the issue's own figures, not the previous issue's it was compared with."""
    doc = asdict(issue)
    del doc["skills_previous"], doc["salaries_previous"]
    return doc


# --- text ------------------------------------------------------------------------------------------


def num(n: int) -> str:
    return f"{n:,}".replace(",", NBSP)


def pct(part: float) -> str:
    return f"{part * 100:.1f}".replace(".", ",") + NBSP + "%"


def share(n: int, of: int) -> str:
    return f"{num(n)} из {num(of)} ({pct(n / of)})" if of else f"{num(n)} из 0"


def _sign(delta: float) -> str:
    return "+" if delta > 0 else "-" if delta < 0 else ""


def change(current: int, previous: int) -> str:
    delta = current - previous
    text = f"{_sign(delta)}{num(abs(delta))}"
    if previous >= CHANGE_MIN_BASE and delta:
        text += f" ({_sign(delta)}{pct(abs(delta) / previous)})"
    return text


def points(delta: float) -> str:
    return f"{_sign(delta)}{abs(delta) * 100:.1f}".replace(".", ",") + f"{NBSP}п.{NBSP}п."


def rub(x: int | None) -> str:
    return "нет" if x is None else f"{num(x)}{NBSP}₽"


def day_text(iso: str, year: bool = True) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]}" + (f" {d.year}" if year else "")


def span(start: str, end: str) -> str:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    if a.year != b.year:
        return f"с {day_text(start)} по {day_text(end)}"
    if a.month != b.month:
        return f"с {day_text(start, year=False)} по {day_text(end)}"
    return f"с {a.day} по {day_text(end)}"


def _table(head: list[str], rows: list[list[str]], text_columns: int = 1) -> list[str]:
    align = ["---"] * text_columns + ["--:"] * (len(head) - text_columns)
    return ["| " + " | ".join(r) + " |" for r in [head, align, *rows]]


def render_vacancies(section: VacancySection) -> list[str]:
    lines = [f"## {SOURCE[section.source]}", ""]
    cur, prev = section.current, section.previous
    if cur is None:
        if section.last_week_with_data:
            last = day_text(section.last_week_with_data)
            lines.append(f"За эту неделю данных нет. Последняя неделя с данными началась {last}.")
        else:
            lines.append("Источник за эту неделю ещё не собирался.")
        return [*lines, ""]

    text = (
        f"Открытых IT-вакансий {num(cur.total.active)} (дней сбора {cur.days_collected}, "
        f"из них полных {cur.complete_days})."
    )
    if prev is not None:
        text += (
            f" Неделей раньше {num(prev.total.active)} (дней сбора {prev.days_collected}): "
            f"{change(cur.total.active, prev.total.active)}."
        )
    else:
        text += " Неделей раньше данных нет, сравнения нет."
    text += f" Новых {share(cur.total.new, cur.total.active)}"
    text += "; это первая неделя сбора, вакансии первого дня в новые не входят." if cur.first_collected_week else "."
    if cur.total.closed is not None:
        text += f" Закрыто {num(cur.total.closed)}"
        text += ", среди них могут быть исчезнувшие за неделю без данных." if prev is None else "."
    lines += [text, ""]

    head = ["Группа ролей", "Открыто"]
    if prev is not None:
        head += ["Неделей раньше", "Изменение"]
    head += ["Новые", "Закрыты"]
    rows = []
    for group, c in sorted(cur.groups.items(), key=lambda kv: (-kv[1].active, kv[0])):
        row = [group, num(c.active)]
        if prev is not None:
            before = prev.groups.get(group, Counts(0, 0, 0)).active
            row += [num(before), change(c.active, before)]
        row += [num(c.new), "нет" if c.closed is None else num(c.closed)]
        rows.append(row)
    return [*lines, *_table(head, rows), ""]


def render_telegram(section: TelegramSection, week_end: str) -> list[str]:
    lines = [f"## {SOURCE['telegram']}", ""]
    cur, prev = section.current, section.previous
    if cur is None:
        return [*lines, "Постов за эту неделю в витрине нет.", ""]
    text = (
        f"Постов с вакансиями {num(cur.posts)}: это {share(cur.posts, cur.posts_before_dedup)} "
        "опубликованных, остальные повторяли уже учтённый текст."
    )
    if prev is not None:
        text += f" Неделей раньше {num(prev.posts)}: {change(cur.posts, prev.posts)}."
    text += " Считаются посты, а не вакансии: в одном посте их бывает несколько."
    if section.newest_post_day is not None and section.newest_post_day < week_end:
        text += f" Неделя собрана не полностью: последний пост в витрине от {day_text(section.newest_post_day)}."
    lines += [text, ""]
    head = ["Группа ролей", "Постов"] + (["Неделей раньше", "Изменение"] if prev is not None else [])
    rows = []
    for group, n in sorted(cur.groups.items(), key=lambda kv: (-kv[1], kv[0])):
        row = [group, num(n)]
        if prev is not None:
            before = prev.groups.get(group, 0)
            row += [num(before), change(n, before)]
        rows.append(row)
    return [*lines, *_table(head, rows), ""]


def movers(now: SkillFamily, before: SkillFamily) -> list[tuple[str, float]]:
    """Skills whose share moved since the previous issue, largest moves first, up then down."""
    if not now.base or not before.base:
        return []
    moved = []
    for skill, n in now.mentions.items():
        m = before.mentions.get(skill, 0)
        if n >= MOVER_MIN_MENTIONS and m >= MOVER_MIN_MENTIONS:
            delta = n / now.base - m / before.base
            if delta:
                moved.append((skill, delta))
    up = sorted((x for x in moved if x[1] > 0), key=lambda x: (-x[1], x[0]))[:TOP_MOVERS]
    down = sorted((x for x in moved if x[1] < 0), key=lambda x: (x[1], x[0]))[:TOP_MOVERS]
    return up + down


def render_skills(issue: Issue) -> list[str]:
    lines = [
        "## Навыки",
        "",
        "Доля вакансий, где навык упомянут в заголовке, тексте или тегах, за весь период сбора источника; "
        "у Telegram доля постов за последние 100 дней. По неделям витрина навыков не делится, поэтому "
        "движение навыков считается к прошлому выпуску.",
        "",
    ]
    for family, s in issue.skills.items():
        top = sorted(s.mentions.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_SKILLS]
        listed = ", ".join(f"{skill} {share(n, s.base)}" for skill, n in top)
        unit = "постов" if family == "telegram" else "вакансий"
        lines.append(f"- {SOURCE[family]}, {span(s.period_from, s.period_to)}, {unit} {num(s.base)}: {listed}.")
    lines.append("")
    if issue.compared_with is None:
        return [
            *lines,
            "Сравнения с прошлым выпуском нет: это первый выпуск. Его доли сохранены в "
            f"`site/src/digest/data/{issue.week}.json`, следующий выпуск сравнит с ними.",
            "",
        ]
    lines += [
        f"Движение к выпуску {issue.compared_with}, навыки с {MOVER_MIN_MENTIONS} упоминаниями и больше в обоих:",
        "",
    ]
    for family, s in issue.skills.items():
        before = issue.skills_previous.get(family)
        moved = movers(s, before) if before is not None else []
        if before is None or not moved:
            lines.append(f"- {SOURCE[family]}: заметных сдвигов нет.")
            continue
        parts = [
            f"{skill} {points(delta)} (было {share(before.mentions[skill], before.base)}, "
            f"стало {share(s.mentions[skill], s.base)})"
            for skill, delta in moved
        ]
        lines.append(f"- {SOURCE[family]}: " + "; ".join(parts) + ".")
    return [*lines, ""]


def render_salaries(issue: Issue) -> list[str]:
    lines = [
        "## Зарплаты",
        "",
        "Медианы границ вилки в рублях в месяц, как их указывают работодатели, за весь период сбора. В выпуск "
        f"попадают ячейки (роль, грейд, регион), где нижнюю границу назвали хотя бы {SALARY_MIN_N} вакансий, "
        f"показаны {TOP_SALARY_CELLS} самых больших; n рядом с медианой это число вакансий с этой границей.",
        "",
    ]
    for family, cells in issue.salaries.items():
        if not cells:
            lines += [
                f"{SOURCE[family]}: ни одна ячейка пока не набрала {SALARY_MIN_N} вакансий с нижней границей.",
                "",
            ]
            continue
        before = {(c.role_bucket, c.grade, c.region_group): c for c in issue.salaries_previous.get(family, [])}
        head = ["Роль", "Грейд", "Регион", "Медиана «от»", "n", "Медиана «до»", "n"]
        if issue.compared_with is not None:
            head.append(f"«От» к {issue.compared_with}")
        rows = []
        for c in cells[:TOP_SALARY_CELLS]:
            row = [
                ROLE[c.role_bucket],
                GRADE[c.grade],
                REGION[c.region_group],
                rub(c.from_median),
                num(c.with_from),
                rub(c.to_median),
                num(c.with_to),
            ]
            if issue.compared_with is not None:
                old = before.get((c.role_bucket, c.grade, c.region_group))
                if old is None or old.from_median is None or c.from_median is None:
                    row.append("ячейки не было")
                else:
                    delta = c.from_median - old.from_median
                    row.append(f"{_sign(delta)}{rub(abs(delta))}")
            rows.append(row)
        lines += [f"{SOURCE[family]}:", "", *_table(head, rows, text_columns=3), ""]
    return lines


def render(issue: Issue) -> str:
    generated = datetime.fromisoformat(issue.generated_at).astimezone(MSK).date().isoformat()
    lines = [
        "---",
        f"title: Дайджест {issue.week}",
        "toc: false",
        "---",
        "",
        f"# Неделя {span(issue.week_start, issue.week_end)}",
        "",
        f"Выпуск {issue.week}: что изменилось на рынке IT-вакансий за неделю по сравнению с предыдущей. "
        f"Собран {day_text(generated)} по витринам радара. Как считаются роли, навыки и зарплаты: "
        "[методика](../methodology). API «Работы в России» в сводку не входит: это срез изменённых записей, "
        "а не открытые вакансии.",
        "",
    ]
    for section in issue.vacancies:
        lines += render_vacancies(section)
    lines += render_telegram(issue.telegram, issue.week_end)
    lines += render_skills(issue)
    lines += render_salaries(issue)
    lines += [
        "---",
        "",
        f"Числа выпуска лежат в репозитории в `site/src/digest/data/{issue.week}.json`. Источник: витрины "
        "mart_weekly_dynamics, mart_skills, mart_junior и mart_salaries. Выпуск собирает сценарий n8n "
        f"`{WORKFLOW}` по понедельникам в 09:00 по Москве.",
        "",
    ]
    return "\n".join(lines)


def render_index(issues: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "---",
        "title: Еженедельный дайджест",
        "toc: false",
        "---",
        "",
        "# Еженедельный дайджест рынка",
        "",
        "Каждый понедельник сценарий n8n собирает из витрин радара сводку за прошедшую неделю: открытые "
        "вакансии по группам ролей, новые и закрытые, посты Telegram-каналов, навыки и медианы зарплат. "
        "Каждое число дано вместе с базой, от которой оно считается.",
        "",
        "Выпуски, новые сверху:",
        "",
    ]
    for doc in sorted(issues, key=lambda d: str(d["week"]), reverse=True):
        lines.append(f"- [{doc['week']}, {span(doc['week_start'], doc['week_end'])}](./{doc['week']})")
    return "\n".join([*lines, ""])


# --- files -----------------------------------------------------------------------------------------


def load_issues(out: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "data").glob("*-W*.json"))]


def previous_issue(out: Path, week: str) -> dict[str, Any] | None:
    earlier = [d for d in load_issues(out) if str(d["week"]) < week]
    return max(earlier, key=lambda d: str(d["week"])) if earlier else None


def write(issue: Issue, out: Path) -> tuple[Path, Path, Path]:
    (out / "data").mkdir(parents=True, exist_ok=True)
    page, data, index = out / f"{issue.week}.md", out / "data" / f"{issue.week}.json", out / "index.md"
    page.write_bytes(render(issue).encode("utf-8"))
    data.write_bytes((json.dumps(figures(issue), ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    index.write_bytes(render_index(load_issues(out)).encode("utf-8"))
    return page, data, index


def fetch(sqls: Mapping[str, str]) -> dict[str, list[dict[str, Any]]]:
    """The host path of `run`: the same SELECTs as the read-only account, without n8n."""
    from radar.agent.clickhouse import QueryError, ReadOnlyClickHouse
    from radar.config import load_dotenv

    load_dotenv()
    ch = ReadOnlyClickHouse()
    try:
        results = {}
        for name, sql in sqls.items():
            response = ch.raw(sql)
            if response.status_code != 200:
                raise QueryError(f"{name}: {response.text.strip()[:1500]}", None)
            results[name] = response.json()["data"]
        return results
    finally:
        ch.close()


def _shown(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("queries", help="print the week and its SELECTs as JSON")
    q.add_argument("--week", help="ISO week such as 2026-W40; default the latest complete week in Moscow")
    r = sub.add_parser("render", help="write the issue from a bundle of query results")
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--out", type=Path, default=OUT)
    run = sub.add_parser("run", help="query ClickHouse from the host and write the issue")
    run.add_argument("--week", help="ISO week such as 2026-W40; default the latest complete week in Moscow")
    run.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    now = datetime.now(UTC)
    if args.command == "render":
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        monday, results = parse_week(bundle["week"]), bundle["results"]
    else:
        monday = parse_week(args.week) if args.week else latest_complete_week(now)
        sqls = queries(monday)
        if args.command == "queries":
            listed = [{"name": name, "sql": sql} for name, sql in sqls.items()]
            print(json.dumps({"week": week_label(monday), "queries": listed}, ensure_ascii=False))
            return 0
        results = fetch(sqls)

    out = args.out.resolve()
    issue = build(monday, results, now, previous_issue(out, week_label(monday)))
    page, data, _ = write(issue, out)
    summary = {
        "week": issue.week,
        "page": _shown(page),
        "figures": _shown(data),
        "compared_with": issue.compared_with,
        "issues": len(load_issues(out)),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
