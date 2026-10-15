"""Pictures of README.md, drawn from result files only.

    python scripts/readme_charts.py

Writes docs/img/readme/*.png: the social preview card, the key numbers, what changed in a year
(results/quarterly), the pipeline, the extraction model against rules and teachers on test
(results/extract/final_test.json), and how much the make-up of the channels moved the pooled numbers
(results/market/composition.json). Labels are direct, without legends where a label fits, so a picture reads
on its own when it is shared.
"""

import json
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from radar.config import RESULTS, ROOT
from radar.site import quarterly

OUT = ROOT / "docs" / "img" / "readme"
ACCENT, NAVY, TEAL = "#f2542d", "#1d3557", "#2a9d8f"
LIGHT, PALE, INK, MUTED, GRID = "#d9d6cf", "#efede8", "#1a1a1a", "#6b6963", "#ebe9e4"
UP, DOWN, NOISE = "#1a7f37", "#cf222e", "#9a9890"
FOOTER = "Радар IT-вакансий · github.com/alekseevpavel04/ru-it-jobs-radar"
QUARTER = "2026Q3"

# Segoe UI where it exists (the author's machine), DejaVu Sans elsewhere; both have Cyrillic.
_FONTS = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update(
    {
        "font.family": "Segoe UI" if "Segoe UI" in _FONTS else "DejaVu Sans",
        "font.size": 13,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.facecolor": "white",
    }
)


def metrics(relpath: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((RESULTS / relpath).read_text(encoding="utf-8"))
    return doc["metrics"]


def frame(fig: plt.Figure, title: str, subtitle: str, top: float = 0.86) -> None:
    fig.subplots_adjust(top=top)
    fig.text(0.02, 0.985, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.985 - 0.075 * 5.4 / fig.get_figheight(), subtitle, fontsize=12.5, color=MUTED, va="top")
    fig.text(0.02, -0.02, FOOTER, fontsize=9.5, color=MUTED, va="top")


def result(relpath: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((RESULTS / relpath).read_text(encoding="utf-8"))
    return doc


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(OUT / name)


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def thousands(n: float) -> str:
    return f"{round(n / 1000):,}".replace(",", " ")


def key_numbers() -> None:
    tg = metrics("collect/telegram.json")
    hist = metrics("dbt/history.json")
    test = metrics("extract/final_test.json")
    tiles = [
        (
            f"{thousands(tg['posts']['value'])} тыс.",
            f"постов из {tg['channels_enabled']['value']} Telegram-\nканалов, с 2016 года",
        ),
        (
            f"{thousands(hist['vacancies_tracked_trudvsem_snapshot']['value'])} тыс.",
            "IT-вакансий портала\nс историей каждого дня",
        ),
        (
            f"{num(test['tg_test/sft_1p7b_r16_clean/macro_f1']['value'], 2)}",
            f"F1 своей модели на постах\nTelegram, у правил {num(test['tg_test/regex/macro_f1']['value'], 2)}",
        ),
        ("каждый день", "сайт обновляется сам,\nбез сервера"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.6))
    for ax, (big, small) in zip(axes, tiles, strict=True):
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.04),
                0.96,
                0.92,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc=PALE,
                ec="none",
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.1,
            0.66,
            big,
            fontsize=27 if len(big) < 10 else 22,
            fontweight="bold",
            color=NAVY,
            transform=ax.transAxes,
            va="center",
        )
        ax.text(0.1, 0.3, small, fontsize=11.5, color=MUTED, transform=ax.transAxes, va="center", linespacing=1.35)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "key_numbers.png")


def year_changes() -> None:
    """The quarterly report's own picture (radar.site.quarterly.chart), from its result file."""
    m = metrics(f"quarterly/{QUARTER}.json")
    rows = [
        (key.split(".", 2)[1], key.split(".", 2)[2], v["value"])
        for key, v in m.items()
        if key.startswith("change.") and v["q_bh"] < 0.05
    ]
    out = OUT / "year_changes.png"
    quarterly.chart(rows, quarterly.chart_subtitle(QUARTER), out)
    print(out)


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=color, lw=2))
    ax.text(x + 0.14, y + h - 0.2, title, fontsize=12.5, fontweight="bold", color=color, va="top")
    ax.text(x + 0.14, y + h - 0.58, body, fontsize=10.5, color=INK, va="top", linespacing=1.4)


def _arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=color, lw=1.6, shrinkA=2, shrinkB=2))


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.9)
    ax.set_axis_off()
    top, low, h_top, h_low = 2.55, 0.15, 1.9, 1.6
    _box(ax, 0.1, top, 2.6, h_top, "Источники", "Telegram, 58 каналов\n«Работа в России»\nХабр Карьера, SuperJob", NAVY)
    _box(ax, 3.2, top, 2.5, h_top, "Сбор", "коллекторы на cron\nчистка ПДн до записи\nParquet", NAVY)
    _box(ax, 6.2, top, 2.6, h_top, "Хранилище", "Redpanda\nClickHouse\ndbt: история, сводки", NAVY)
    _box(ax, 9.3, top, 2.3, h_top, "Извлечение", "учителя 8B\nsilver-метки\nQLoRA SFT 1.7B", ACCENT)
    _box(ax, 12.0, top, 1.9, h_top, "Модели", "зарплата\nнавык и вилка\nподбор, дедуп", ACCENT)
    _box(ax, 6.2, low, 3.4, h_low, "Сервисы", "сайт, Superset\nFastAPI, MCP\nагент text-to-SQL", TEAL)
    _box(ax, 10.1, low, 3.8, h_low, "MLOps", "Airflow, MLflow\nPrometheus, Grafana\nEvidently, Kubernetes", TEAL)
    _box(
        ax,
        0.1,
        low,
        5.6,
        h_low,
        "Каждый день на GitHub Actions",
        "новые посты Telegram → студент Q4_K_M\nна CPU раннера → недельные счётчики\n"
        "в репозитории → сайт; раз в квартал релиз",
        UP,
    )
    mid = top + h_top / 2
    for a, b in [
        ((2.7, mid), (3.2, mid)),
        ((5.7, mid), (6.2, mid)),
        ((8.8, mid), (9.3, mid)),
        ((11.6, mid), (12.0, mid)),
    ]:
        _arrow(ax, a, b)
    _arrow(ax, (7.5, top), (7.5, low + h_low))
    _arrow(ax, (12.9, top), (12.9, low + h_low))
    _arrow(ax, (5.7, low + h_low / 2), (6.2, low + h_low / 2), UP)
    ax.text(0.1, 4.85, "Как устроен радар", fontsize=19, fontweight="bold", color=INK, va="top")
    save(fig, "pipeline.png")


def extraction() -> None:
    m = metrics("extract/final_test.json")
    systems = [
        ("regex", "Правила: регулярки и словарь"),
        ("base_1p7b", "Qwen3-1.7B без дообучения"),
        ("teacher_yandex", "Учитель YandexGPT-5-Lite 8B"),
        ("teacher_qwen", "Учитель Qwen3-8B"),
        ("sft_1p7b_r16_clean", "Студент Qwen3-1.7B, QLoRA"),
    ]
    rows = [(label, m[f"tg_test/{key}/macro_f1"]) for key, label in systems]
    diff_teacher = m["tg_test/sft_1p7b_r16_clean_minus_teacher_qwen/macro_f1"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (_, v) in enumerate(rows):
        student = i == len(rows) - 1
        color = ACCENT if student else LIGHT
        ax.barh(i, v["value"], color=color, height=0.6)
        ax.plot([v["ci_low"], v["ci_high"]], [i, i], color=INK if student else MUTED, lw=1.6)
        ax.text(
            v["ci_high"] + 0.012,
            i,
            num(v["value"], 3),
            va="center",
            fontsize=12.5,
            fontweight="bold" if student else "normal",
            color=INK,
        )
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.tick_params(axis="y", length=0, labelsize=12.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,5", "0,75", "1"])
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    beats = diff_teacher["ci_low"] > 0
    tie = diff_teacher["ci_low"] <= 0 <= diff_teacher["ci_high"]
    verdict = "точнее учителя 8B" if beats else "на уровне учителя 8B" if tie else "уступает учителю 8B"
    frame(
        fig,
        f"Дообученная модель 1.7B {verdict} и намного точнее правил",
        f"Macro F1 по восьми полям на {result('extract/tg_test_regex.json')['n_examples']} отложенных постах Telegram, "
        "линия: 95 % интервал бутстрапа",
        top=0.8,
    )
    save(fig, "extraction.png")


def composition() -> None:
    m = metrics("market/composition.json")
    rows: Sequence[tuple[str, float, float]] = [
        ("Python, всё IT", m["pooled.it_all.Python"]["value"], m["within.it_all.Python"]["value"]),
        ("CI/CD, всё IT", m["pooled.it_all.CI/CD"]["value"], m["within.it_all.CI/CD"]["value"]),
        ("React, Frontend", m["pooled.frontend_all.React"]["value"], m["within.frontend_all.React"]["value"]),
        (
            "TypeScript, Frontend",
            m["pooled.frontend_all.TypeScript"]["value"],
            m["within.frontend_all.TypeScript"]["value"],
        ),
        ("CSS, Frontend", m["pooled.frontend_all.CSS"]["value"], m["within.frontend_all.CSS"]["value"]),
        (
            "Доля джунов",
            m["grades.junior.pooled.last"]["value"] - m["grades.junior.pooled.base"]["value"],
            m["grades.junior.fixed.last"]["value"] - m["grades.junior.fixed.base"]["value"],
        ),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, (_, pooled, within) in enumerate(reversed(rows)):
        ax.plot([pooled * 100, within * 100], [i, i], color=LIGHT, lw=3, zorder=1)
        ax.scatter([pooled * 100], [i], s=120, color=NOISE, zorder=2)
        ax.scatter([within * 100], [i], s=140, color=ACCENT, zorder=3)
        for x, color, bold in ((pooled, MUTED, False), (within, ACCENT, True)):
            side = 1 if x >= min(pooled, within) and x == max(pooled, within) else -1
            ax.text(
                x * 100 + side * 0.7,
                i,
                f"{'+' if x > 0 else '−'}{num(abs(x) * 100)}",
                va="center",
                ha="left" if side > 0 else "right",
                fontsize=12,
                color=color,
                fontweight="bold" if bold else "normal",
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.tick_params(axis="y", length=0, labelsize=12.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    span = max(max(abs(p), abs(w)) for _, p, w in rows) * 100 * 1.25
    ax.set_xlim(-span * 0.45, span)
    ax.text(0.99, -0.04, "● весь пул постов", color=NOISE, transform=ax.transAxes, ha="right", fontsize=11.5)
    ax.text(
        0.99, 0.04, "● внутри одних и тех же каналов", color=ACCENT, transform=ax.transAxes, ha="right", fontsize=11.5
    )
    frame(
        fig,
        "Без учёта каналов рост сильно завышался",
        "Изменение доли с 2022 года к 2025-2026, п. п. Новые и выросшие каналы сдвигают общую долю сами по себе;\n"
        "доля джунов среди постов с грейдом, внутри каналов это постоянный вес каждого канала",
        top=0.78,
    )
    save(fig, "composition.png")


def social_preview() -> None:
    """The card GitHub shows when the repository link is shared: 1280 x 640, the size GitHub asks for."""
    tg = metrics("collect/telegram.json")
    test = metrics("extract/final_test.json")
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(NAVY)
    fig.text(0.06, 0.84, "ru-it-jobs-radar", fontsize=40, fontweight="bold", color="white", va="top")
    fig.text(
        0.06,
        0.68,
        "Радар IT-вакансий России: какие навыки просят,\nкак меняется спрос и за что платят больше",
        fontsize=21,
        color="#d9e2ec",
        va="top",
        linespacing=1.35,
    )
    tiles = [
        (f"{thousands(tg['posts']['value'])} тыс.", "постов Telegram"),
        (num(test["tg_test/sft_1p7b_r16_clean/macro_f1"]["value"], 2), "F1 своей Qwen3-1.7B"),
        ("каждый день", "сайт обновляется сам"),
    ]
    for i, (big, small) in enumerate(tiles):
        x = 0.06 + i * 0.31
        fig.text(x, 0.3, big, fontsize=34, fontweight="bold", color=ACCENT, va="center")
        fig.text(x, 0.17, small, fontsize=17, color="#d9e2ec", va="center")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "social_preview.png", dpi=100, facecolor=NAVY)
    plt.close(fig)
    print(OUT / "social_preview.png")


def main() -> None:
    social_preview()
    key_numbers()
    year_changes()
    pipeline()
    extraction()
    composition()


if __name__ == "__main__":
    main()
