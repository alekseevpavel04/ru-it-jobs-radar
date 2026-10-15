"""Quarterly skill report: the year to the end of a finished quarter against the year before it.

    python -m radar.site.quarterly                      the last finished quarter
    python -m radar.site.quarterly --quarter 2026Q3     October 2025 - September 2026 against the year before

Reads the weekly aggregates the autopilot keeps in git (aggregates/telegram/posts/<year>.csv and
skills/<year>.csv: IT units of Telegram by channel after the weekly cross-post dedup) and writes
results/quarterly/<quarter>.json and the release notes docs/quarterly/<quarter>.md. A week belongs to the
period of its Monday. The report comes out every quarter, but compares twelve months with the twelve before:
a quarter against the same quarter a year earlier holds about 6 000 posts, and in the 2026Q3 trial one skill
of thirty in all IT moved beyond noise. Whole years also keep the season out of the comparison.

The change is measured within channels (radar.site.within): a channel that starts or stops posting moves
a pooled share by itself, and the shares shown are the weighted shares of the channels of both years.

Per cell (all IT and each direction of radar.site.skill_trends, all grades) with MIN_PERIOD_POSTS posts or
more in both years in the channels of both: the TOP_SKILLS skills most named in the last year, each named
by MIN_MENTIONS posts or more. A change is called out when its test passes Benjamini-Hochberg at FDR over
the cell's skills; the rest is noise.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from radar.config import ROOT
from radar.results import write_result
from radar.site.market_facts import share
from radar.site.skill_trends import DIRECTIONS, NO_SKILL_CELLS
from radar.site.within import benjamini_hochberg, panel, within_change

AGGREGATES = ROOT / "aggregates" / "telegram"
NOTES = ROOT / "docs" / "quarterly"
SITE = "https://alekseevpavel04.github.io/ru-it-jobs-radar/"
MIN_PERIOD_POSTS = 300
MIN_MENTIONS = 20
TOP_SKILLS = 30
FDR = 0.05
LABEL = {
    "it": "Всё IT",
    "backend": "Backend",
    "frontend": "Frontend",
    "qa": "Тестирование",
    "analytics": "Аналитика",
    "ml": "ML и Data Science",
    "dataeng": "Инженерия данных",
    "devops": "DevOps и администрирование",
    "mobile": "Мобильная разработка",
    "fullstack": "Fullstack",
    "security": "Инфобезопасность",
}
# Short names for the chart and the headline.
SHORT = {**LABEL, "ml": "ML и DS", "devops": "DevOps", "mobile": "Мобильная разработка"}
IN = {
    "it": "во всём IT",
    "backend": "в Backend",
    "frontend": "во Frontend",
    "qa": "в тестировании",
    "analytics": "в аналитике",
    "ml": "в ML и Data Science",
    "dataeng": "в инженерии данных",
    "devops": "в DevOps",
    "mobile": "в мобильной разработке",
    "fullstack": "в Fullstack",
    "security": "в инфобезопасности",
}
RAW = "https://raw.githubusercontent.com/alekseevpavel04/ru-it-jobs-radar/main/"
MONTH = [
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
]
MONTH_OF = [
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
]


def last_finished(today: date) -> str:
    year, q = today.year, (today.month - 1) // 3 + 1
    return f"{year}Q{q - 1}" if q > 1 else f"{year - 1}Q4"


def window(quarter: str) -> tuple[date, date, date]:
    """Start of the year before, start of the last year, and the day after the quarter ends."""
    year, q = int(quarter[:4]), int(quarter[5])
    end = date(year + 1, 1, 1) if q == 4 else date(year, 3 * q + 1, 1)
    return end.replace(year=end.year - 2), end.replace(year=end.year - 1), end


def period_of(week: date, quarter: str) -> str | None:
    first, middle, end = window(quarter)
    return "before" if first <= week < middle else "now" if middle <= week < end else None


def span(start: date, end: date) -> str:
    """The twelve months from start to the day before end, in words: с октября 2025 по сентябрь 2026."""
    last = date(end.year - 1, 12, 1) if end.month == 1 else date(end.year, end.month - 1, 1)
    return f"с {MONTH_OF[start.month - 1]} {start.year} по {MONTH[last.month - 1]} {last.year}"


def plural(n: int, forms: tuple[str, str, str]) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    return forms[1] if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else forms[2]


def directions_of(role: str) -> list[str]:
    if role in NO_SKILL_CELLS:
        return []
    return ["it"] + [d for d, roles in DIRECTIONS.items() if role in roles]


def read_years(folder: Path, years: Iterable[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for year in years:
        path = folder / f"{year}.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                rows += csv.DictReader(f)
    return rows


def report(posts: Iterable[Mapping[str, Any]], skills: Iterable[Mapping[str, Any]], quarter: str) -> dict[str, Any]:
    # cell -> channel -> period -> posts; cell -> skill -> channel -> period -> posts naming the skill
    size: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    channels: dict[str, set[str]] = defaultdict(set)
    for r in posts:
        period = period_of(date.fromisoformat(r["week"]), quarter)
        if period:
            channels[period].add(r["channel"])
            for d in directions_of(r["role"]):
                size[d][r["channel"]][period] += int(r["posts"])
    named: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for r in skills:
        period = period_of(date.fromisoformat(r["week"]), quarter)
        if period:
            for d in directions_of(r["role"]):
                named[d][r["skill"]][r["channel"]][period] += int(r["posts"])

    metrics: dict[str, Any] = {
        "channels.now": {"value": len(channels["now"])},
        "channels.before": {"value": len(channels["before"])},
    }
    cells = []
    for d in LABEL:
        both = {c: size[d][c] for c in panel(size[d], ("before", "now"))}
        n_now = sum(n["now"] for n in size[d].values())
        n_before = sum(n["before"] for n in size[d].values())
        kept_now, kept_before = sum(n["now"] for n in both.values()), sum(n["before"] for n in both.values())
        if min(kept_now, kept_before) < MIN_PERIOD_POSTS:
            continue
        metrics[f"posts.{d}.now"] = {"value": n_now}
        metrics[f"posts.{d}.before"] = {"value": n_before}
        metrics[f"posts.{d}.both_channels_now"] = share(kept_now, n_now)
        metrics[f"posts.{d}.both_channels_before"] = share(kept_before, n_before)
        mentions = Counter({s: sum(c["now"] for c in by_channel.values()) for s, by_channel in named[d].items()})
        by_mentions = sorted(mentions.items(), key=lambda kv: (-kv[1], kv[0]))
        listed = [s for s, n in by_mentions if n >= MIN_MENTIONS][:TOP_SKILLS]
        found = within_change(both, {s: named[d][s] for s in listed}, "before", "now")
        measured = [found[s] for s in listed]
        q_values = benjamini_hochberg([m[2] for m in measured])
        rows = []
        for s, (was, is_, pj), qj in zip(listed, measured, q_values, strict=True):
            metrics[f"share.{d}.{s}.now"] = {"value": round(is_, 4)}
            metrics[f"share.{d}.{s}.before"] = {"value": round(was, 4)}
            metrics[f"change.{d}.{s}"] = {
                "value": round(is_ - was, 4),
                "p": float(f"{pj:.3g}"),
                "q_bh": float(f"{qj:.3g}"),
            }
            rows.append({"skill": s, "now": is_, "before": was, "diff": is_ - was, "q": qj})
        cells.append(
            {"direction": d, "posts_now": n_now, "posts_before": n_before, "channels": len(both), "skills": rows}
        )
    return {"quarter": quarter, "window": window(quarter), "metrics": metrics, "cells": cells}


def _int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def _points(x: float) -> str:
    return f"{x * 100:+.1f}".replace(".", ",").replace("-", "−") + " п. п."


def moves(rep: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(direction, skill row) of every change beyond noise."""
    return [(c["direction"], r) for c in rep["cells"] for r in c["skills"] if r["q"] < FDR]


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} и {items[-1]}"


def leaders(rep: Mapping[str, Any], sign: int, k: int = 3) -> list[tuple[str, dict[str, Any]]]:
    """The k largest changes of one sign, each skill once (HTML falls in Frontend and Fullstack alike) and at
    most two a direction, so that one direction does not take the whole summary."""
    seen: set[str] = set()
    per: Counter[str] = Counter()
    out = []
    for d, r in sorted((m for m in moves(rep) if sign * m[1]["diff"] > 0), key=lambda m: -abs(m[1]["diff"])):
        if r["skill"] not in seen and per[d] < 2:
            seen.add(r["skill"])
            per[d] += 1
            out.append((d, r))
    return out[:k]


def headline(rep: Mapping[str, Any]) -> str:
    def named(part: list[tuple[str, dict[str, Any]]]) -> str:
        groups: dict[str, list[str]] = {}
        for d, r in part:
            groups.setdefault(d, []).append(r["skill"])
        return ", ".join(f"{_and(skills)} {IN[d]}" for d, skills in groups.items())

    up, down = leaders(rep, 1), leaders(rep, -1)
    # "вырос навык", "выросли навыки"
    grew = "вырос" if len(up) == 1 else "выросли"
    fell = "упал" if len(down) == 1 else "упали"
    parts = [f"Сильнее всего за год {grew} {named(up)}" if up else "", f"{fell} {named(down)}" if down else ""]
    text = "; ".join(p for p in parts if p)
    return (text[0].upper() + text[1:] + ".") if text else "Заметных изменений за год нет."


def _signed(x: float) -> str:
    return _points(x).replace(" п. п.", "")


def direction_line(cell: Mapping[str, Any]) -> str:
    """One line a direction: its moves beyond noise, rises first."""
    moved = sorted((r for r in cell["skills"] if r["q"] < FDR), key=lambda r: -r["diff"])
    ups = [f"{r['skill']} {_signed(r['diff'])}" for r in moved if r["diff"] > 0]
    downs = [f"{r['skill']} {_signed(r['diff'])}" for r in moved if r["diff"] < 0]
    body = "; ".join(p for p in (", ".join(ups), ", ".join(downs)) if p) or "заметных изменений нет"
    return f"- **{LABEL[cell['direction']]}:** {body}."


def notes(rep: Mapping[str, Any], images: Mapping[str, str] | None = None) -> str:
    """The release text; images maps "changes" and "directions" to the addresses of the two pictures."""
    first, middle, end = rep["window"]
    m = rep["metrics"]
    images = images or {}
    lines = [
        f"# Навыки в IT за год {span(middle, end)}",
        "",
        f"{headline(rep)} Сравнение с предыдущими двенадцатью месяцами, доля IT-постов с навыком в публичных "
        f"Telegram-каналах. Все навыки по направлениям и грейдам на [сайте]({SITE}).",
        "",
    ]
    if "changes" in images:
        lines += [f"![Что изменилось за год]({images['changes']})", ""]
    if "directions" in images:
        lines += ["## Навыки по направлениям", "", f"![Навыки по направлениям]({images['directions']})", ""]
    moved = [c for c in rep["cells"] if any(r["q"] < FDR for r in c["skills"])]
    still = [LABEL[c["direction"]] for c in rep["cells"] if c not in moved]
    lines += ["## Изменения по направлениям, п. п.", "", *[direction_line(c) for c in moved]]
    if still:
        lines += ["", f"Без заметных изменений за год: {', '.join(still)}."]
    lines += [
        "",
        "<details>",
        "<summary>Как считается</summary>",
        "",
        f"Последние двенадцать месяцев ({span(middle, end)}) против предыдущих ({span(first, middle)}). "
        f"Каналов {m['channels.now']['value']} за последний год и {m['channels.before']['value']} за предыдущий. "
        "Изменение доли считается внутри каналов, которые публиковали вакансии направления в оба года, с одинаковым "
        "весом канала в обоих: так новый или замолчавший канал не сдвигает доли сам по себе. Цветом и числом "
        "отмечены только изменения, которые выходят за случайный разброс и по постам, и по каналам и проходят "
        f"поправку Бенджамини-Хохберга на {round(FDR * 100)} % среди навыков направления. Роль, грейд и навыки из "
        "текста поста читает модель проекта. Числа выпуска лежат в `results/quarterly/`, код в "
        "`src/radar/site/quarterly.py`.",
        "",
        "</details>",
    ]
    return "\n".join(lines) + "\n"


def _style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "Segoe UI" if "Segoe UI" in fonts else "DejaVu Sans", "font.size": 13})
    return plt


UP, DOWN, NOISE, BASE_DOT, MUTED, INK = "#1a7f37", "#cf222e", "#9a9890", "#bdbab2", "#6b6963", "#1a1a1a"


def directions_chart(rep: Mapping[str, Any], subtitle: str, path: Path, per_panel: int = 8) -> None:
    """A panel a direction that moved: first its skills that changed beyond noise, then its most named ones for
    context; a small grey dot a year earlier, a line to the dot of the last year, coloured when the change is
    beyond noise. Directions without a move get no panel."""
    plt = _style()
    cells = [c for c in rep["cells"] if any(r["q"] < FDR for r in c["skills"])]
    cols = 2
    rows_n = -(-len(cells) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 2.75 * rows_n + 1.4), squeeze=False)
    for ax in axes.flat[len(cells) :]:
        ax.set_axis_off()
    for ax, cell in zip(axes.flat, cells, strict=False):
        moved = sorted((r for r in cell["skills"] if r["q"] < FDR), key=lambda r: -abs(r["diff"]))[:per_panel]
        steady = [r for r in sorted(cell["skills"], key=lambda r: -r["now"]) if r not in moved]
        top = sorted(moved + steady[: per_panel - len(moved)], key=lambda r: r["now"])
        reach = max(max(r["now"], r["before"]) for r in top) * 100 * 1.28
        for i, r in enumerate(top):
            color = (UP if r["diff"] > 0 else DOWN) if r["q"] < FDR else NOISE
            was, now = r["before"] * 100, r["now"] * 100
            ax.plot([was, now], [i, i], color=BASE_DOT, lw=2.2, solid_capstyle="round", zorder=1)
            ax.scatter([was], [i], s=22, color=BASE_DOT, zorder=2)
            ax.scatter([now], [i], s=60, color=color, zorder=3, edgecolors="white", linewidths=0.8)
            label = f"{_pct(r['now'])}" + (f"  {_signed(r['diff'])}" if r["q"] < FDR else "")
            ax.text(
                max(was, now) + reach * 0.03,
                i,
                label,
                va="center",
                fontsize=10.5,
                color=color if r["q"] < FDR else MUTED,
                fontweight="bold" if r["q"] < FDR else "normal",
            )
        ax.set_yticks(range(len(top)), [r["skill"] for r in top])
        ax.tick_params(axis="y", length=0, labelsize=11)
        ax.set_xlim(0, reach)
        ax.set_xticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            f"{LABEL[cell['direction']]}  ·  {_int(cell['posts_now'])} "
            f"{plural(cell['posts_now'], ('пост', 'поста', 'постов'))}",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color=INK,
            pad=6,
        )
    fig.subplots_adjust(top=1 - 1.25 / fig.get_figheight(), hspace=0.45, wspace=0.55)
    fig.text(0.02, 0.995, "Навыки по направлениям", fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.995 - 0.45 / fig.get_figheight(), subtitle, fontsize=12.5, color=MUTED, va="top")
    fig.text(0.02, 0.003, f"Радар IT-вакансий · {SITE}", fontsize=9.5, color=MUTED, va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def directions_subtitle(quarter: str) -> str:
    _, middle, end = window(quarter)
    return (
        f"Доля IT-постов направления с навыком: маленькая точка год назад, большая {span(middle, end)}.\n"
        "Зелёным и красным навыки, чья доля сдвинулась сверх случайного разброса, серым самые частые для сравнения"
    )


def chart(rows: list[tuple[str, str, float]], subtitle: str, path: Path) -> None:
    """Bars of the largest changes, each skill once and at most three a direction: (direction, skill, change)."""
    plt = _style()
    up, down, muted, ink = UP, DOWN, MUTED, INK
    picked: list[tuple[str, str, float]] = []
    per: Counter[str] = Counter()
    seen: set[str] = set()
    for d, skill, diff in sorted(rows, key=lambda r: -abs(r[2])):
        if per[d] < 3 and skill not in seen and len(picked) < 14:
            picked.append((d, skill, diff))
            per[d] += 1
            seen.add(skill)
    picked.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(picked) + 1.8))
    ys = range(len(picked))
    ax.barh(list(ys), [r[2] * 100 for r in picked], color=[up if r[2] > 0 else down for r in picked], height=0.62)
    ax.set_yticks(list(ys), [f"{skill}  ·  {SHORT[d]}" for d, skill, _ in picked])
    ax.tick_params(axis="y", length=0, labelsize=12.5)
    ax.axvline(0, color=muted, lw=1)
    for i, (_, _, diff) in enumerate(picked):
        color = up if diff > 0 else down
        ax.text(
            diff * 100 + (0.15 if diff > 0 else -0.15),
            i,
            _points(diff).replace(" п. п.", ""),
            va="center",
            ha="left" if diff > 0 else "right",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
    reach = max(abs(r[2]) for r in picked) * 100 * 1.2
    ax.set_xlim(-reach, reach)
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(top=0.8)
    fig.text(0.02, 0.985, "Что изменилось за год", fontsize=19, fontweight="bold", color=ink, va="top")
    fig.text(0.02, 0.985 - 0.4 / fig.get_figheight(), subtitle, fontsize=12.5, color=muted, va="top")
    fig.text(0.02, -0.02, f"Радар IT-вакансий · {SITE}", fontsize=9.5, color=muted, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def chart_subtitle(quarter: str) -> str:
    _, middle, end = window(quarter)
    return (
        f"Изменение доли IT-постов Telegram с навыком, п. п.: {span(middle, end)} против\n"
        "предыдущих 12 месяцев, внутри одних и тех же каналов; только изменения сверх случайного разброса"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quarter", help="e.g. 2026Q3; default the last finished quarter")
    ap.add_argument("--aggregates", type=Path, default=AGGREGATES)
    args = ap.parse_args()
    quarter = args.quarter or last_finished(date.today())
    first, middle, end = window(quarter)
    years = range(first.year, end.year + 1)
    rep = report(read_years(args.aggregates / "posts", years), read_years(args.aggregates / "skills", years), quarter)
    if not rep["cells"]:
        raise SystemExit(f"{quarter}: no cell has {MIN_PERIOD_POSTS} posts in both years")
    path = write_result(
        f"quarterly/{quarter}.json",
        rep["metrics"],
        source_script="src/radar/site/quarterly.py",
        quarter=quarter,
        last_year=[middle.isoformat(), end.isoformat()],
        year_before=[first.isoformat(), middle.isoformat()],
        min_period_posts=MIN_PERIOD_POSTS,
        min_mentions=MIN_MENTIONS,
        top_skills=TOP_SKILLS,
        fdr=FDR,
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    changes, grid = NOTES / f"{quarter}.png", NOTES / f"{quarter}_directions.png"
    found = [(d, r["skill"], r["diff"]) for d, r in moves(rep)]
    if found:
        chart(found, chart_subtitle(quarter), changes)
    directions_chart(rep, directions_subtitle(quarter), grid)
    # absolute addresses: the same text becomes the release notes, where a relative path does not resolve
    images = {"directions": f"{RAW}{grid.relative_to(ROOT).as_posix()}"}
    if found:
        images["changes"] = f"{RAW}{changes.relative_to(ROOT).as_posix()}"
    out = NOTES / f"{quarter}.md"
    out.write_text(notes(rep, images), encoding="utf-8", newline="\n")
    print(f"{len(rep['metrics'])} metrics -> {path}; notes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
