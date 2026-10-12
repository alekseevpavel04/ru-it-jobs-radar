"""Weekly market digest: one short Russian page per ISO week in site/src/digest, built from the marts.

    python -m radar.site.digest queries [--week 2026-W40]
    python -m radar.site.digest render --bundle bundle.json [--out site/src/digest]
    python -m radar.site.digest run [--week 2026-W40] [--out site/src/digest]

The n8n workflow (infra/n8n/workflows/weekly_digest.json) calls `queries` for the SELECTs of the
latest complete week, sends each to ClickHouse as the read-only account, saves the answers as one
bundle and calls `render`. `run` does the same from the host without n8n. Both paths use the standard
library only up to the HTTP call: inside the n8n container there is a bare CPython and no venv.

Only aggregates of marts.* are read and written (DECISIONS 2). Vacancies, Telegram posts and skills
are compared week over week from the marts (mart_weekly_dynamics, mart_skills_weekly). Salaries are a
period aggregate (mart_salaries, role group and source levels), so every issue keeps its own figures in
data/<week>.json and the next issue compares its salaries with them.
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
SKILL_SOURCES = ("trudvsem_snapshot", "habr", "telegram")
SALARY_SOURCES = ("trudvsem", "habr")
TOP_SKILLS = 5
# Below this base a relative change ("+1200 %" from 1 to 13) says nothing; the counts stay.
CHANGE_MIN_BASE = 20
TOP_MOVERS = 3
# A skill moves in the digest only when both weeks count at least this many mentions of it.
MOVER_MIN_MENTIONS = 20
# Complete collection days a vacancy has to be missing before the history closes it, dated to the
# first missing day (dbt macro history_absent_days_to_close; a test keeps the two equal). Closures of
# the last days of the data are not known yet, so an issue made on Monday cannot count the closures
# of the week that just ended.
ABSENT_DAYS_TO_CLOSE = {"trudvsem_snapshot": 7, "habr": 3}
# Same rule as radar.site.export.salary_min_n() (the smallest cell whose 95% interval for the median
# drops the min and the max); a test pins the two together, export itself needs pydantic.
SALARY_MIN_N = 9

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
# Role groups of mart_salaries (seed role_groups), as MART_GROUP on the site.
MART_GROUP = {
    "dev": "Разработка, включая 1С",
    "ml_ds_data": "ML, DS, данные",
    "analyst": "Бизнес- и системный анализ",
    "qa_devops_sys": "QA, DevOps, админы, ИБ",
    "other_it": "Дизайн, управление, прочие IT",
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
# Russian labels of the dictionary's concept skills, as SKILL on the site; technology names stay.
SKILL = {
    "API integration": "Интеграции по API",
    "A/B tests": "A/B-тесты",
    "Algorithms": "Алгоритмы",
    "Antivirus": "Антивирусная защита",
    "Clean architecture": "Чистая архитектура",
    "Communication": "Коммуникация",
    "Computer vision": "Компьютерное зрение",
    "Cryptography": "Криптография",
    "Deep learning": "Глубокое обучение",
    "Design patterns": "Паттерны проектирования",
    "Embedded": "Встраиваемые системы",
    "English": "Английский язык",
    "Firewalls": "Межсетевые экраны",
    "Gradient boosting": "Градиентный бустинг",
    "Information security standards": "Стандарты ИБ",
    "Machine learning": "Машинное обучение",
    "Manual testing": "Ручное тестирование",
    "Microservices": "Микросервисы",
    "Multithreading": "Многопоточность",
    "Networking": "Компьютерные сети",
    "Pentest": "Пентест",
    "Product analytics": "Продуктовая аналитика",
    "Prompt engineering": "Промпт-инжиниринг",
    "Recommender systems": "Рекомендательные системы",
    "Requirements": "Работа с требованиями",
    "Speech recognition": "Распознавание речи",
    "Statistics": "Статистика",
    "System design": "Проектирование систем",
    "Teamwork": "Командная работа",
    "Test automation": "Автоматизация тестирования",
    "Time series": "Временные ряды",
    "Vector databases": "Векторные БД",
    "Virtualization": "Виртуализация",
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
            "telegram_posts_before_dedup, telegram_channels_week FROM marts.mart_weekly_dynamics "
            f"WHERE week IN ('{previous}', '{current}') AND source IN ('trudvsem_snapshot', 'habr', 'telegram') "
            "ORDER BY source, week, role_bucket FORMAT JSON"
        ),
        "last_week": (
            "SELECT source, toString(max(week)) AS last_week FROM marts.mart_weekly_dynamics "
            f"WHERE week <= '{current}' GROUP BY source ORDER BY source FORMAT JSON"
        ),
        # The last collected day of a source (the window of mart_lifetime), taken as its last complete
        # day only when every collected day of its last week was complete; otherwise NULL, unknown.
        "closure_window": (
            "SELECT l.source AS source, if(w.complete_days = w.days_collected, toString(l.period_to), NULL) "
            "AS last_complete_day FROM (SELECT source, max(period_to) AS period_to FROM marts.mart_lifetime "
            "GROUP BY source) AS l INNER JOIN (SELECT source, argMax(complete_days, week) AS complete_days, "
            "argMax(days_collected, week) AS days_collected FROM marts.mart_weekly_dynamics "
            "WHERE source IN ('trudvsem_snapshot', 'habr') GROUP BY source) AS w ON l.source = w.source "
            "ORDER BY source FORMAT JSON"
        ),
        "skills": (
            "SELECT toString(week) AS week, source, skill, vacancies_mentioning, vacancies_in_week "
            f"FROM marts.mart_skills_weekly WHERE week IN ('{previous}', '{current}') "
            "AND source IN ('trudvsem_snapshot', 'habr', 'telegram') ORDER BY source, week, skill FORMAT JSON"
        ),
        # the newest Telegram post day, to say when the week is not fully collected
        "periods": (
            "SELECT source, toString(min(period_from)) AS period_from, toString(max(period_to)) AS period_to "
            "FROM marts.mart_skills GROUP BY source ORDER BY source FORMAT JSON"
        ),
        # one row per role group and one for the whole source; never the cells, which would count every
        # vacancy once per level
        "salaries": (
            "SELECT source, level, role_group, vacancies, with_salary_rub, with_from, salary_from_median, with_to, "
            "salary_to_median FROM marts.mart_salaries WHERE level IN ('group', 'source') "
            "ORDER BY source, level, role_group FORMAT JSON"
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
    last_complete_day: str | None = None
    # the last first-missing day whose closure the data can already confirm
    closures_known_through: str | None = None


@dataclass(frozen=True)
class TelegramWeek:
    posts: int
    posts_before_dedup: int
    channels: int
    groups: dict[str, int]


@dataclass(frozen=True)
class TelegramSection:
    newest_post_day: str | None
    current: TelegramWeek | None
    previous: TelegramWeek | None


@dataclass(frozen=True)
class SkillWeek:
    base: int
    mentions: dict[str, int]


@dataclass(frozen=True)
class SkillSection:
    source: str
    current: SkillWeek | None
    previous: SkillWeek | None


@dataclass(frozen=True)
class SalaryGroup:
    role_group: str
    vacancies: int
    with_salary_rub: int
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
    skills: list[SkillSection]
    salaries: dict[str, list[SalaryGroup]]
    salary_periods: dict[str, dict[str, str]]
    compared_with: str | None = None
    salaries_previous: dict[str, list[SalaryGroup]] = field(default_factory=dict)


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
        # the same channel count on every row of the week: channels with any post, buckets together
        channels=need_int(rows[0]["telegram_channels_week"]),
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
    last_complete = {r["source"]: r["last_complete_day"] for r in results["closure_window"]}

    def rows(source: str, week: str) -> list[Mapping[str, Any]]:
        return [r for r in weekly if r["source"] == source and r["week"] == week]

    def known_through(source: str) -> str | None:
        last = last_complete.get(source)
        if last is None:
            return None
        return (date.fromisoformat(last) - timedelta(days=ABSENT_DAYS_TO_CLOSE[source] - 1)).isoformat()

    vacancies = [
        VacancySection(
            source=source,
            last_week_with_data=last_week.get(source),
            current=vacancy_week(rows(source, current_day)),
            previous=vacancy_week(rows(source, previous_day)),
            last_complete_day=last_complete.get(source),
            closures_known_through=known_through(source),
        )
        for source in VACANCY_SOURCES
    ]

    periods = {r["source"]: r for r in results["periods"]}

    def skill_week(source: str, week: str) -> SkillWeek | None:
        found = [r for r in results["skills"] if r["source"] == source and r["week"] == week]
        if not found:
            return None
        base = need_int(found[0]["vacancies_in_week"])
        mentions = {r["skill"]: need_int(r["vacancies_mentioning"]) for r in found}
        over = sorted(s for s, n in mentions.items() if n > base)
        if over:
            raise ValueError(f"{source} {week}: mentions above the base of {base} for {over[:5]}")
        return SkillWeek(base=base, mentions=dict(sorted(mentions.items())))

    skills = [
        SkillSection(source=s, current=skill_week(s, current_day), previous=skill_week(s, previous_day))
        for s in SKILL_SOURCES
    ]

    salaries: dict[str, list[SalaryGroup]] = {}
    for family in SALARY_SOURCES:
        groups = [
            SalaryGroup(
                role_group=r["role_group"] if r["level"] == "group" else "all",
                vacancies=need_int(r["vacancies"]),
                with_salary_rub=need_int(r["with_salary_rub"]),
                with_from=need_int(r["with_from"]),
                from_median=as_int(r["salary_from_median"]),
                with_to=need_int(r["with_to"]),
                to_median=as_int(r["salary_to_median"]),
            )
            for r in results["salaries"]
            if r["source"] == family and r["level"] in ("group", "source")
        ]
        order = [*MART_GROUP, "all"]
        salaries[family] = sorted(groups, key=lambda g: order.index(g.role_group))

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
        salary_periods={
            family: {"period_from": periods[family]["period_from"], "period_to": periods[family]["period_to"]}
            for family in SALARY_SOURCES
            if family in periods
        },
    )
    if previous is None:
        return issue
    return replace(
        issue,
        compared_with=str(previous["week"]),
        salaries_previous={
            family: [SalaryGroup(**g) for g in groups] for family, groups in previous["salaries"].items()
        },
    )


def figures(issue: Issue) -> dict[str, Any]:
    """What data/<week>.json keeps: the issue's own figures, not the previous issue's it was compared with."""
    doc = asdict(issue)
    del doc["salaries_previous"]
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


def of_channels(n: int) -> str:
    """'из N каналов' agrees with N: из 51 канала, из 47 каналов."""
    return f"{num(n)} {'канала' if n % 10 == 1 and n % 100 != 11 else 'каналов'}"


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


def render_vacancies(section: VacancySection, week_end: str) -> list[str]:
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
    # A closure is dated to the first missing day but known only after the source's run of missing
    # complete days, so the closures of the last week of the data are still partial.
    known = section.closures_known_through
    previous_end = (date.fromisoformat(week_end) - timedelta(days=7)).isoformat()
    settled = cur.total.closed is not None and known is not None and week_end <= known
    previous_settled = (
        not settled
        and prev is not None
        and prev.total.closed is not None
        and known is not None
        and previous_end <= known
    )
    closed_column: str | None = None
    if settled:
        text += f" Закрыто {num(cur.total.closed or 0)}"
        text += ", среди них могут быть исчезнувшие за неделю без данных." if prev is None else "."
        closed_column = "Закрыты"
    elif cur.total.closed is not None:
        text += (
            f" Закрытия этой недели ещё не известны: закрытой вакансия считается после "
            f"{ABSENT_DAYS_TO_CLOSE[section.source]} полных дней сбора без неё"
        )
        if section.last_complete_day is not None:
            text += f", а полные дни сбора есть по {day_text(section.last_complete_day)}"
        text += "."
        if previous_settled and prev is not None:
            text += f" Неделей раньше закрыто {num(prev.total.closed or 0)}."
            closed_column = "Закрыты неделей раньше"
    lines += [text, ""]

    head = ["Группа ролей", "Открыто"]
    if prev is not None:
        head += ["Неделей раньше", "Изменение"]
    head += ["Новые"] + ([closed_column] if closed_column else [])
    rows = []
    for group, c in sorted(cur.groups.items(), key=lambda kv: (-kv[1].active, kv[0])):
        row = [group, num(c.active)]
        before = prev.groups.get(group, Counts(0, 0, 0)) if prev is not None else None
        if before is not None:
            row += [num(before.active), change(c.active, before.active)]
        row.append(num(c.new))
        if closed_column:
            counted = c if settled else before
            closed = counted.closed if counted is not None else None
            row.append("нет" if closed is None else num(closed))
        rows.append(row)
    return [*lines, *_table(head, rows), ""]


def render_telegram(section: TelegramSection, week_end: str) -> list[str]:
    lines = [f"## {SOURCE['telegram']}", ""]
    cur, prev = section.current, section.previous
    if cur is None:
        return [*lines, "Постов за эту неделю в витрине нет.", ""]
    text = (
        f"Постов с вакансиями {num(cur.posts)}: это {share(cur.posts, cur.posts_before_dedup)} "
        f"опубликованных, остальные повторяли уже учтённый текст. Посты пришли из {of_channels(cur.channels)}."
    )
    if prev is not None:
        text += f" Неделей раньше {num(prev.posts)} из {of_channels(prev.channels)}: {change(cur.posts, prev.posts)}."
        if cur.channels != prev.channels:
            text += " Число каналов в неделе изменилось, поэтому изменение постов говорит и об охвате выборки."
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


def movers(now: SkillWeek, before: SkillWeek) -> list[tuple[str, float]]:
    """Skills whose share of the week's vacancies moved since the week before, largest first, up then down."""
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
    days = {v.source: v.current.days_collected for v in issue.vacancies if v.current is not None}
    lines = [
        "## Навыки",
        "",
        "Доля вакансий недели, где навык упомянут в заголовке, тексте или тегах (у Telegram доля постов "
        f"недели), и сдвиг к прошлой неделе для навыков, упомянутых хотя бы {MOVER_MIN_MENTIONS} раз в обе "
        "недели. Неделя из немногих собранных дней даёт меньшую базу, но доли остаются сравнимыми.",
        "",
    ]
    for section in issue.skills:
        cur, prev = section.current, section.previous
        name = SOURCE[section.source]
        if cur is None:
            lines.append(f"- {name}: за эту неделю данных нет.")
            continue
        unit = "постов" if section.source == "telegram" else "вакансий"
        top = sorted(cur.mentions.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_SKILLS]
        listed = ", ".join(f"{SKILL.get(skill, skill)} {share(n, cur.base)}" for skill, n in top)
        collected = f", дней сбора {days[section.source]}" if section.source in days else ""
        text = f"- {name}, {unit} недели {num(cur.base)}{collected}: {listed}."
        if prev is None:
            text += " Прошлой недели в витрине нет, сдвигов не считаем."
        else:
            moved = movers(cur, prev)
            if moved:
                parts = [
                    f"{SKILL.get(skill, skill)} {points(delta)} (было {share(prev.mentions[skill], prev.base)}, "
                    f"стало {share(cur.mentions[skill], cur.base)})"
                    for skill, delta in moved
                ]
                text += " Сдвиги к прошлой неделе: " + "; ".join(parts) + "."
            else:
                text += " Заметных сдвигов к прошлой неделе нет."
        lines.append(text)
    return [*lines, ""]


def render_salaries(issue: Issue) -> list[str]:
    lines = [
        "## Зарплаты",
        "",
        "Медианы границ вилки в рублях в месяц, как их указывают работодатели, по всем вакансиям группы ролей "
        "за весь период сбора; медиана посчитана в витрине по вакансиям группы. Граница показана, если её "
        f"назвали хотя бы {SALARY_MIN_N} вакансий; n рядом с медианой это число вакансий с этой границей.",
        "",
    ]
    for family, groups in issue.salaries.items():
        period = issue.salary_periods.get(family)
        title = f"{SOURCE[family]}, {span(period['period_from'], period['period_to'])}" if period else SOURCE[family]
        before = {g.role_group: g for g in issue.salaries_previous.get(family, [])}
        head = ["Группа ролей", "С суммой", "Медиана «от»", "n", "Медиана «до»", "n"]
        if issue.compared_with is not None:
            head.append(f"«От» к {issue.compared_with}")
        rows = []
        for g in groups:
            row = [
                "Все IT-вакансии" if g.role_group == "all" else MART_GROUP[g.role_group],
                share(g.with_salary_rub, g.vacancies),
                rub(g.from_median) if g.with_from >= SALARY_MIN_N else "мало данных",
                num(g.with_from),
                rub(g.to_median) if g.with_to >= SALARY_MIN_N else "мало данных",
                num(g.with_to),
            ]
            if issue.compared_with is not None:
                old = before.get(g.role_group)
                comparable = old is not None and min(old.with_from, g.with_from) >= SALARY_MIN_N
                if comparable and old is not None and old.from_median is not None and g.from_median is not None:
                    delta = g.from_median - old.from_median
                    row.append(f"{_sign(delta)}{rub(abs(delta))}")
                else:
                    row.append("нет сравнения")
            rows.append(row)
        lines += [f"{title}:", "", *_table(head, rows), ""]
    if issue.compared_with is None:
        lines += [
            "Сравнения зарплат с прошлым выпуском нет: это первый выпуск. Его медианы сохранены в "
            f"`site/src/digest/data/{issue.week}.json`, следующий выпуск сравнит с ними.",
            "",
        ]
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
        lines += render_vacancies(section, issue.week_end)
    lines += render_telegram(issue.telegram, issue.week_end)
    lines += render_skills(issue)
    lines += render_salaries(issue)
    lines += [
        "---",
        "",
        f"Числа выпуска лежат в репозитории в `site/src/digest/data/{issue.week}.json`. Источник: витрины "
        "mart_weekly_dynamics, mart_skills_weekly, mart_skills и mart_salaries. Выпуск собирает модуль "
        f"`radar.site.digest`; по расписанию его вызывает сценарий n8n `{WORKFLOW}` по понедельникам в 09:00 "
        "по Москве.",
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
        "Выпуск это сводка из витрин радара за прошедшую неделю: открытые вакансии по группам ролей, новые и "
        "закрытые, посты Telegram-каналов, навыки и медианы зарплат. По расписанию его собирает сценарий n8n "
        "в понедельник утром. "
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
