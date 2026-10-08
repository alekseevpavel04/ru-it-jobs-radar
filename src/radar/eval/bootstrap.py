"""Percentile bootstrap over gold examples, single-system and paired.

Paired means both systems are re-scored on the same resampled rows, so the interval is for the
difference on the same vacancies, not for two independent samples. Resamples where a metric is
undefined (a conditional share whose denominator fell to zero) are dropped from that metric's
interval and counted.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from radar.eval.metrics import Array, Stat


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float
    n_resamples: int


def resample_weights(n_examples: int, n_resamples: int, seed: int) -> Array:
    """(n_resamples, n_examples) counts: how many times each example is drawn in each resample."""
    rng = np.random.default_rng(seed)
    probs = np.full(n_examples, 1.0 / n_examples)
    return rng.multinomial(n_examples, probs, size=n_resamples).astype(np.float64)


def _interval(point: float, samples: Array, confidence: float) -> Interval | None:
    if np.isnan(point):
        return None
    ok = samples[~np.isnan(samples)]
    if ok.size == 0:
        return None
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(ok, [tail, 100 - tail])
    return Interval(point, float(low), float(high), int(ok.size))


def bootstrap(stats: Mapping[str, Stat], weights: Array, confidence: float) -> dict[str, Interval | None]:
    return {name: _interval(stat.value(), stat.fn(weights @ stat.columns), confidence) for name, stat in stats.items()}


def paired_bootstrap(
    a: Mapping[str, Stat], b: Mapping[str, Stat], weights: Array, confidence: float
) -> dict[str, Interval | None]:
    """Interval for metric(a) - metric(b), both systems scored on the same resampled gold rows."""
    out: dict[str, Interval | None] = {}
    for name in a.keys() & b.keys():
        if a[name].columns.shape[0] != b[name].columns.shape[0]:
            raise ValueError(f"{name}: systems are scored on different numbers of examples")
        diff = a[name].fn(weights @ a[name].columns) - b[name].fn(weights @ b[name].columns)
        out[name] = _interval(a[name].value() - b[name].value(), diff, confidence)
    return dict(sorted(out.items()))
