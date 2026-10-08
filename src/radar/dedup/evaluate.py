"""Precision and recall of the dedup rule on the labeled pairs, weighted back to all pairs.

Pairs were drawn through units with a known per-draw probability p (pairs.py). Within a stratum
the Hansen-Hurwitz estimate of a pair total is mean(z / p) over the drawn pairs, so a pair through
a unit with few partners counts for few pairs and one through a template hub for many. Totals add
over strata; precision and recall are ratios of totals. Pairs labeled unsure are left out, and the
stratum mean is taken over the pairs that remain (assumes unsure is unrelated to the label).

Intervals: percentile bootstrap that resamples pairs within each stratum, so every stratum keeps
its size. A stratum with no positive pair contributes zero positives in every resample, and the
bootstrap cannot say how many it misses; `missed_upper` gives the rule-of-three bound for that.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar.dedup.normalize import normalize_title
from radar.dedup.place import place_of, same_place
from radar.dedup.settings import Rule

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Labeled:
    pair_id: str
    stratum: str
    split: str
    p_draw: float
    jaccard: float
    cosine: float
    candidate: bool
    place_ok: bool
    same_title: bool
    same: bool


def labeled_pairs(
    views: dict[str, dict[str, Any]], meta: dict[str, dict[str, Any]], labels: dict[str, str]
) -> list[Labeled]:
    """Join the labeling file, the sample metadata and the final labels; unsure pairs are dropped."""
    out = []
    for pair_id, label in sorted(labels.items()):
        if label == "unsure":
            continue
        m, v = meta[pair_id], views[pair_id]
        a, b = v["a"], v["b"]
        out.append(
            Labeled(
                pair_id=pair_id,
                stratum=m["stratum"],
                split=m["split"],
                p_draw=m["p_draw"],
                jaccard=m["jaccard"],
                cosine=m["cosine"],
                candidate=m["by_minhash"] or m["by_embedding"],
                place_ok=same_place(
                    place_of(a["city"], a["address"], a["remote"], a["source"]),
                    place_of(b["city"], b["address"], b["remote"], b["source"]),
                ),
                same_title=normalize_title(a["title"]) == normalize_title(b["title"]),
                same=label == "same",
            )
        )
    return out


def links(rule: Rule, jaccard: float, cosine: float, place_ok: bool, same_title: bool) -> bool:
    by_score = (rule.jaccard is not None and jaccard >= rule.jaccard) or (
        rule.cosine is not None and cosine >= rule.cosine
    )
    return by_score and (place_ok or not rule.same_place) and (same_title or not rule.same_title)


def decide(rule: Rule, x: Labeled) -> bool:
    """The pipeline only ever links candidate pairs, so a pair outside the candidates is never linked."""
    return x.candidate and links(rule, x.jaccard, x.cosine, x.place_ok, x.same_title)


@dataclass(frozen=True)
class Estimate:
    precision: float
    recall: float
    f1: float
    linked_pairs: float
    positive_pairs: float


class Weighted:
    """Weighted pair totals over strata, for a point estimate and for bootstrap resamples.

    `populations=None` gives Hansen-Hurwitz: a stratum total is mean(z / p). With the stratum
    populations known, the Hajek form divides by the estimated population instead, sum(z / p) /
    sum(1 / p) * N: the same weights, normalized so a stratum always counts for its true size.
    """

    def __init__(self, pairs: Sequence[Labeled], populations: dict[str, int] | None = None) -> None:
        self.strata = sorted({x.stratum for x in pairs})
        self.groups = {s: [x for x in pairs if x.stratum == s] for s in self.strata}
        self.populations = populations

    def _columns(self, s: str, predicted: Callable[[Labeled], bool]) -> Array:
        """Per pair: weighted tp, linked, positive and the bare weight."""
        xs = self.groups[s]
        w = np.array([1 / x.p_draw for x in xs])
        y = np.array([x.same for x in xs], dtype=float)
        d = np.array([predicted(x) for x in xs], dtype=float)
        return np.stack([w * y * d, w * d, w * y, w], axis=1)

    def _scale(self, s: str, sums: Array) -> Array:
        """Stratum totals (tp, linked, positive) from column sums; works on one row or many."""
        if self.populations is None:
            return sums[..., :3] / len(self.groups[s])
        return sums[..., :3] / sums[..., 3:4] * self.populations[s]

    def estimate(self, predicted: Callable[[Labeled], bool]) -> Estimate:
        totals = np.zeros(3)
        for s in self.strata:
            totals += self._scale(s, self._columns(s, predicted).sum(axis=0))
        return _ratios(float(totals[0]), float(totals[1]), float(totals[2]))

    def bootstrap(self, predicted: Callable[[Labeled], bool], n_resamples: int, seed: int) -> tuple[Array, Array]:
        """(precision, recall) per resample; NaN where a resample links nothing or has no positive."""
        rng = np.random.default_rng(seed)
        totals = np.zeros((n_resamples, 3))
        for s in self.strata:
            cols = self._columns(s, predicted)
            n = cols.shape[0]
            counts = rng.multinomial(n, np.full(n, 1 / n), size=n_resamples)
            totals += self._scale(s, counts @ cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = totals[:, 0] / totals[:, 1]
            recall = totals[:, 0] / totals[:, 2]
        return precision, recall


def _ratios(tp: float, linked: float, pos: float) -> Estimate:
    precision = tp / linked if linked else float("nan")
    recall = tp / pos if pos else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return Estimate(precision, recall, f1, linked, pos)


def interval(samples: Array, confidence: float) -> tuple[float, float, int]:
    ok = samples[~np.isnan(samples)]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(ok, [tail, 100 - tail])
    return float(low), float(high), int(ok.size)


def grid(jaccards: Sequence[float], cosines: Sequence[float]) -> list[Rule]:
    """Every threshold pair and each method alone, with and without the place and title conditions."""
    thresholds: list[tuple[float | None, float | None]] = [
        *product(jaccards, cosines),
        *((j, None) for j in jaccards),
        *((None, c) for c in cosines),
    ]
    return [Rule(j, c, place, title) for (j, c), place, title in product(thresholds, (True, False), (False, True))]


def select(
    rules: Sequence[Rule], tuning: Sequence[Labeled], populations: dict[str, int] | None
) -> tuple[Rule, Estimate]:
    """Highest weighted F1 on the tuning half; ties go to higher precision, then to the stricter rule
    (place condition on, title condition off, higher thresholds)."""
    w = Weighted(tuning, populations)
    scored = [(r, w.estimate(partial(decide, r))) for r in rules]

    def key(item: tuple[Rule, Estimate]) -> tuple[float, ...]:
        r, e = item
        p = 0.0 if np.isnan(e.precision) else e.precision
        out_of_reach = 2.0
        return (
            round(e.f1, 9),
            round(p, 9),
            r.same_place,
            not r.same_title,
            out_of_reach if r.jaccard is None else r.jaccard,
            out_of_reach if r.cosine is None else r.cosine,
        )

    return max(scored, key=key)


def missed_upper(pairs: Sequence[Labeled], populations: dict[str, int]) -> dict[str, float]:
    """Per stratum with no positive among its labeled pairs: 3/n of its population, the 95% upper
    bound on positives it may hold (rule of three), as if pairs were drawn uniformly."""
    out = {}
    for s in sorted({x.stratum for x in pairs}):
        xs = [x for x in pairs if x.stratum == s]
        if not any(x.same for x in xs):
            out[s] = 3 / len(xs) * populations[s]
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
