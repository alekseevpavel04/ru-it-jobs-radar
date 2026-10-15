"""Quarterly skill report: a finished quarter against the same quarter a year earlier.

    python -m radar.site.quarterly                      the last finished quarter
    python -m radar.site.quarterly --quarter 2026Q3

Reads the weekly aggregates the autopilot keeps in git (aggregates/telegram/posts/<year>.csv and
skills/<year>.csv: IT units of Telegram by channel after the weekly cross-post dedup) and writes
results/quarterly/<quarter>.json and the release notes docs/quarterly/<quarter>.md. A week belongs to the
quarter of its Monday. The comparison is with the same quarter a year earlier, so that the season does not
pass for a trend. A month moves a share by about as much as its noise (Python between 16.7 % and 24.7 % of
the months of 2025-2026, about 2 000 posts a month), a quarter holds about 6 000 posts.

The change is measured within channels (radar.site.within): a channel that starts or stops posting moves
a pooled share by itself, and the shares shown are the weighted shares of the channels of both quarters.

Per cell (all IT and each direction of radar.site.skill_trends, all grades) with MIN_QUARTER_POSTS posts or
more in both quarters in the channels of both: the TOP_SKILLS skills most named in the quarter, each named
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
from radar.site.skill_trends import DIRECTIONS
from radar.site.within import benjamini_hochberg, panel, within_change

AGGREGATES = ROOT / "aggregates" / "telegram"
NOTES = ROOT / "docs" / "quarterly"
SITE = "https://alekseevpavel04.github.io/ru-it-jobs-radar/"
MIN_QUARTER_POSTS = 300
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
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}


def quarter_of(day: date) -> str:
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def year_before(quarter: str) -> str:
    return f"{int(quarter[:4]) - 1}{quarter[4:]}"


def last_finished(today: date) -> str:
    year, q = today.year, (today.month - 1) // 3 + 1
    return f"{year}Q{q - 1}" if q > 1 else f"{year - 1}Q4"


def quarter_label(quarter: str, genitive: bool = False) -> str:
    return f"{ROMAN[int(quarter[5])]} {'квартала' if genitive else 'квартал'} {quarter[:4]}"


def plural(n: int, forms: tuple[str, str, str]) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    return forms[1] if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else forms[2]


def directions_of(role: str) -> list[str]:
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
    before = year_before(quarter)
    # cell -> channel -> quarter -> posts; cell -> skill -> channel -> quarter -> posts naming the skill
    size: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    channels: dict[str, set[str]] = defaultdict(set)
    for r in posts:
        q = quarter_of(date.fromisoformat(r["week"]))
        if q in (quarter, before):
            channels[q].add(r["channel"])
            for d in directions_of(r["role"]):
                size[d][r["channel"]][q] += int(r["posts"])
    named: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for r in skills:
        q = quarter_of(date.fromisoformat(r["week"]))
        if q in (quarter, before):
            for d in directions_of(r["role"]):
                named[d][r["skill"]][r["channel"]][q] += int(r["posts"])

    metrics: dict[str, Any] = {
        "channels.now": {"value": len(channels[quarter])},
        "channels.before": {"value": len(channels[before])},
    }
    cells = []
    for d in LABEL:
        both = {c: size[d][c] for c in panel(size[d], (before, quarter))}
        n_now = sum(n[quarter] for n in size[d].values())
        n_before = sum(n[before] for n in size[d].values())
        kept_now, kept_before = sum(n[quarter] for n in both.values()), sum(n[before] for n in both.values())
        if min(kept_now, kept_before) < MIN_QUARTER_POSTS:
            continue
        metrics[f"posts.{d}.now"] = {"value": n_now}
        metrics[f"posts.{d}.before"] = {"value": n_before}
        metrics[f"posts.{d}.both_channels_now"] = share(kept_now, n_now)
        metrics[f"posts.{d}.both_channels_before"] = share(kept_before, n_before)
        mentions = Counter({s: sum(c[quarter] for c in by_channel.values()) for s, by_channel in named[d].items()})
        listed = [s for s, n in mentions.most_common() if n >= MIN_MENTIONS][:TOP_SKILLS]
        found = within_change(both, {s: named[d][s] for s in listed}, before, quarter)
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
    return {"quarter": quarter, "before": before, "metrics": metrics, "cells": cells}


def _int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def _points(x: float) -> str:
    return f"{x * 100:+.1f}".replace(".", ",").replace("-", "−") + " п. п."


def notes(rep: Mapping[str, Any]) -> str:
    q, before = rep["quarter"], rep["before"]
    m = rep["metrics"]
    lines = [
        f"# Навыки в IT-вакансиях Telegram: {quarter_label(q)} против {quarter_label(before, genitive=True)}",
        "",
        f"Доля IT-постов с навыком в публичных каналах ({m['channels.now']['value']} сейчас, "
        f"{m['channels.before']['value']} год назад). Сравнение идёт внутри каналов, которые публиковали "
        "вакансии направления в оба квартала, с одинаковым весом канала в обоих: так новый или замолчавший канал "
        "не сдвигает доли сам по себе. Изменение в процентных пунктах. Навык попадает в «выросли» или «упали», "
        f"если изменение прошло поправку Бенджамини-Хохберга на {round(FDR * 100)} % среди навыков направления; "
        "остальное в пределах случайного разброса. Роль, грейд и навыки из текста читает модель проекта. Все "
        f"навыки и направления на сайте: {SITE}",
    ]
    for cell in rep["cells"]:
        rows = cell["skills"]
        up = sorted((r for r in rows if r["q"] < FDR and r["diff"] > 0), key=lambda r: -r["diff"])
        down = sorted((r for r in rows if r["q"] < FDR and r["diff"] < 0), key=lambda r: r["diff"])
        lines += ["", f"## {LABEL[cell['direction']]}", ""]
        lines.append(
            f"{_int(cell['posts_now'])} {plural(cell['posts_now'], ('пост', 'поста', 'постов'))} против "
            f"{_int(cell['posts_before'])} год назад, {cell['channels']} "
            f"{plural(cell['channels'], ('канал', 'канала', 'каналов'))} в обоих кварталах."
        )
        lines.append("")
        for title, part in (("Выросли", up), ("Упали", down)):
            if part:
                lines.append(
                    f"- **{title}:** "
                    + "; ".join(
                        f"{r['skill']} {_pct(r['before'])} → {_pct(r['now'])} ({_points(r['diff'])})" for r in part
                    )
                    + "."
                )
        rest = len(rows) - len(up) - len(down)
        lines.append(
            f"- Без заметных изменений {rest} из {len(rows)} навыков."
            if up or down
            else f"- Заметных изменений нет ни у одного из {len(rows)} навыков."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quarter", help="e.g. 2026Q3; default the last finished quarter")
    ap.add_argument("--aggregates", type=Path, default=AGGREGATES)
    args = ap.parse_args()
    quarter = args.quarter or last_finished(date.today())
    years = sorted({int(quarter[:4]), int(year_before(quarter)[:4])})
    rep = report(read_years(args.aggregates / "posts", years), read_years(args.aggregates / "skills", years), quarter)
    if not rep["cells"]:
        raise SystemExit(f"{quarter}: no cell has {MIN_QUARTER_POSTS} posts in both quarters")
    path = write_result(
        f"quarterly/{quarter}.json",
        rep["metrics"],
        source_script="src/radar/site/quarterly.py",
        quarter=quarter,
        compared_with=rep["before"],
        min_quarter_posts=MIN_QUARTER_POSTS,
        min_mentions=MIN_MENTIONS,
        top_skills=TOP_SKILLS,
        fdr=FDR,
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    out = NOTES / f"{quarter}.md"
    out.write_text(notes(rep), encoding="utf-8", newline="\n")
    print(f"{len(rep['metrics'])} metrics -> {path}; notes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
