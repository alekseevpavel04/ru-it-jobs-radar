"""Field metrics for the extraction schema.

Every metric is a function of column sums of a per-example matrix (n_examples, k). The point
estimate uses all rows; a bootstrap resample re-weights rows, so one matrix serves both, and two
systems scored on the same gold rows can share the same resamples (paired bootstrap).

Conventions (docs/annotation_guide.md, section "Метрики"):
- An invalid or missing prediction scores as all fields null and empty lists, never skipped.
- Categorical fields: accuracy counts null == null as correct; precision, recall and F1 are over
  non-null slots: a value where gold is null is a false positive, null where gold has a value is a
  false negative, a wrong value is both.
- Lists (skills, cities): micro precision, recall and F1 over normalized items.
- Salary: the (from, to) pair is one slot, each bound matches within a relative tolerance. Currency,
  gross/net and period are scored separately on examples where both gold and prediction have a salary.
- F1 with nothing in gold and nothing predicted is 1.0; a share with a zero denominator is undefined.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from radar.extract.schema import VacancyFields

Array = npt.NDArray[np.float64]

MACRO_FIELDS = ("role", "grade", "skills", "salary", "work_format", "cities", "experience_years", "employment_type")
CATEGORICAL = ("role", "grade", "work_format", "employment_type", "lang")
CLASS_MACRO = ("role",)


@dataclass(frozen=True)
class Stat:
    columns: Array
    fn: Callable[[Array], Array]
    share: bool = False

    def value(self, sums: Array | None = None) -> float:
        totals = self.columns.sum(axis=0) if sums is None else sums
        return float(self.fn(totals))

    def numerator_denominator(self) -> tuple[int, int]:
        totals = self.columns.sum(axis=0)
        return round(float(totals[0])), round(float(totals[1]))


def _ratio(sums: Array) -> Array:
    num, den = sums[..., 0], sums[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _f1(sums: Array) -> Array:
    tp, pred, gold = sums[..., 0], sums[..., 1], sums[..., 2]
    denom = pred + gold
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 1.0)


def _precision(sums: Array) -> Array:
    tp, pred = sums[..., 0], sums[..., 1]
    return np.where(pred > 0, tp / np.where(pred > 0, pred, 1), np.where(sums[..., 2] > 0, 0.0, 1.0))


def _recall(sums: Array) -> Array:
    tp, gold = sums[..., 0], sums[..., 2]
    return np.where(gold > 0, tp / np.where(gold > 0, gold, 1), np.where(sums[..., 1] > 0, 0.0, 1.0))


def _class_macro_f1(n_classes: int) -> Callable[[Array], Array]:
    def fn(sums: Array) -> Array:
        tp = sums[..., :n_classes]
        pred = sums[..., n_classes : 2 * n_classes]
        gold = sums[..., 2 * n_classes :]
        denom = pred + gold
        present = denom > 0
        f1 = np.where(present, 2 * tp / np.where(present, denom, 1), 0.0)
        count = present.sum(axis=-1)
        return np.where(count > 0, f1.sum(axis=-1) / np.maximum(count, 1), 1.0)

    return fn


def _slot_stats(prefix: str, columns: Array, with_accuracy: Array | None = None) -> dict[str, Stat]:
    """columns: (n, 3) with tp, predicted non-null, gold non-null."""
    out = {
        f"{prefix}_precision": Stat(columns, _precision),
        f"{prefix}_recall": Stat(columns, _recall),
        f"{prefix}_f1": Stat(columns, _f1),
    }
    if with_accuracy is not None:
        out[f"{prefix}_accuracy"] = Stat(with_accuracy, _ratio, share=True)
    return out


def normalize_item(item: str) -> str:
    return " ".join(item.split()).casefold().replace("ё", "е")


def _close(a: float | None, b: float | None, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


EMPTY = VacancyFields(
    role=None,
    grade=None,
    skills=[],
    salary=None,
    work_format=None,
    cities=[],
    experience_years=None,
    employment_type=None,
    lang="other",
)


def _categorical(prefix: str, gold: Sequence[Any], pred: Sequence[Any], class_macro: bool) -> dict[str, Stat]:
    n = len(gold)
    slot = np.zeros((n, 3))
    acc = np.zeros((n, 2))
    for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
        slot[i] = (p is not None and p == g, p is not None, g is not None)
        acc[i] = (p == g, 1)
    out = _slot_stats(prefix, slot, acc)
    if class_macro:
        classes = sorted({v for v in (*gold, *pred) if v is not None})
        index = {c: j for j, c in enumerate(classes)}
        k = len(classes)
        cols = np.zeros((n, 3 * k))
        for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
            if p is not None:
                cols[i, k + index[p]] = 1
                if p == g:
                    cols[i, index[p]] = 1
            if g is not None:
                cols[i, 2 * k + index[g]] = 1
        out[f"{prefix}_macro_f1"] = Stat(cols, _class_macro_f1(k))
    return out


def _lists(
    prefix: str, gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], norm: Callable[[str], str]
) -> dict[str, Stat]:
    cols = np.zeros((len(gold), 3))
    for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
        gs, ps = {norm(x) for x in g}, {norm(x) for x in p}
        cols[i] = (len(gs & ps), len(ps), len(gs))
    return _slot_stats(prefix, cols)


def _salary(gold: Sequence[VacancyFields], pred: Sequence[VacancyFields], rel_tol: float) -> dict[str, Stat]:
    n = len(gold)
    slot = np.zeros((n, 3))
    acc = np.zeros((n, 2))
    parts = {k: np.zeros((n, 2)) for k in ("from", "to", "currency", "gross_net", "period")}
    for i, (gf, pf) in enumerate(zip(gold, pred, strict=True)):
        g, p = gf.salary, pf.salary
        match = g is not None and p is not None and _close(g.from_, p.from_, rel_tol) and _close(g.to, p.to, rel_tol)
        slot[i] = (match, p is not None, g is not None)
        acc[i] = (match or (g is None and p is None), 1)
        if g is not None and p is not None:
            parts["from"][i] = (_close(g.from_, p.from_, rel_tol), 1)
            parts["to"][i] = (_close(g.to, p.to, rel_tol), 1)
            parts["currency"][i] = (g.currency == p.currency, 1)
            parts["gross_net"][i] = (g.gross_net == p.gross_net, 1)
            parts["period"][i] = (g.period == p.period, 1)
    out = _slot_stats("salary", slot, acc)
    out |= {f"salary_{k}_accuracy": Stat(v, _ratio, share=True) for k, v in parts.items()}
    return out


def _experience(gold: Sequence[VacancyFields], pred: Sequence[VacancyFields]) -> dict[str, Stat]:
    def key(f: VacancyFields) -> tuple[float | None, float | None] | None:
        e = f.experience_years
        return None if e is None else (e.min, e.max)

    return _categorical("experience_years", [key(f) for f in gold], [key(f) for f in pred], class_macro=False)


def _macro(stats: dict[str, Stat], fields: Sequence[str]) -> Stat:
    parts = [stats[f"{f}_f1"] for f in fields]
    widths = [p.columns.shape[1] for p in parts]
    bounds = np.cumsum([0, *widths])

    def fn(sums: Array) -> Array:
        values = [p.fn(sums[..., bounds[j] : bounds[j + 1]]) for j, p in enumerate(parts)]
        mean: Array = np.stack(values, axis=-1).mean(axis=-1)
        return mean

    return Stat(np.hstack([p.columns for p in parts]), fn)


def score(
    gold: Sequence[VacancyFields],
    pred: Sequence[VacancyFields | None],
    salary_rel_tol: float,
    macro_fields: Sequence[str] = MACRO_FIELDS,
    normalize_skill: Callable[[str], str] = normalize_item,
) -> dict[str, Stat]:
    """Per-metric statistics for one system on aligned gold and prediction rows (None = invalid)."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold rows, {len(pred)} predictions")
    valid = np.array([[p is not None, 1] for p in pred], dtype=np.float64).reshape(-1, 2)
    filled = [p if p is not None else EMPTY for p in pred]

    stats: dict[str, Stat] = {"valid_json": Stat(valid, _ratio, share=True)}
    for field in CATEGORICAL:
        g = [getattr(f, field) for f in gold]
        p = [getattr(f, field) if v is not None else None for f, v in zip(filled, pred, strict=True)]
        cat = _categorical(field, g, p, class_macro=field in CLASS_MACRO)
        if field == "lang":
            cat = {"lang_accuracy": cat["lang_accuracy"]}
        stats |= cat
    stats |= _lists("skills", [f.skills for f in gold], [f.skills for f in filled], normalize_skill)
    stats |= _lists("cities", [f.cities for f in gold], [f.cities for f in filled], normalize_item)
    stats |= _salary(gold, filled, salary_rel_tol)
    stats |= _experience(gold, filled)
    stats["macro_f1"] = _macro(stats, macro_fields)
    return stats
