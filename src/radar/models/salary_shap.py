"""Exact SHAP values of the salary point model over the inputs of the extraction schema.

The players are the blocks of `salary_features.BLOCKS` (role, grade, region, ..., all skills as one
block), eight of them, so the Shapley value is computed exactly over all 2^8 coalitions instead of
being sampled. The value of a coalition S for a row x is interventional: the mean prediction over a
background sample of training rows whose S columns are replaced by those of x. The values are in the
model's output space, the log of the monthly salary, and add up to f(x) minus the mean prediction
over the background.
"""

from collections.abc import Callable, Sequence
from math import factorial
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def coalition_values(
    predict: Callable[[Array], Array], X: Array, background: Array, blocks: Sequence[Sequence[int]]
) -> Array:
    """(n, 2^M): mean prediction for each row and each coalition, coalition encoded as a bit mask."""
    n, b = len(X), len(background)
    m = len(blocks)
    out = np.empty((n, 2**m))
    tiled = np.tile(background, (n, 1))
    repeated = np.repeat(X, b, axis=0)
    for mask in range(2**m):
        cols = [c for i in range(m) if mask >> i & 1 for c in blocks[i]]
        Z = tiled.copy()
        if cols:
            Z[:, cols] = repeated[:, cols]
        out[:, mask] = predict(Z).reshape(n, b).mean(axis=1)
    return out


def shapley(values: Array) -> Array:
    """(n, M) Shapley values from coalition values (n, 2^M)."""
    m = int(np.log2(values.shape[1]))
    phi = np.zeros((values.shape[0], m))
    weight = [factorial(k) * factorial(m - k - 1) / factorial(m) for k in range(m)]
    for mask in range(2**m):
        size = bin(mask).count("1")
        for i in range(m):
            if not mask >> i & 1:
                phi[:, i] += weight[size] * (values[:, mask | 1 << i] - values[:, mask])
    return phi


def block_shap(
    predict: Callable[[Array], Array], X: Array, background: Array, blocks: Sequence[Sequence[int]]
) -> tuple[Array, Array]:
    """Shapley values (n, M) and the base value of every row (mean prediction over the background)."""
    values = coalition_values(predict, X, background, blocks)
    return shapley(values), values[:, 0]


def summary(
    phi: Array, names: Sequence[str], levels: dict[str, NDArray[Any]], block_of_level: dict[str, str]
) -> dict[str, Any]:
    """Mean |SHAP| per block and the mean SHAP of each level of the categorical inputs.

    `levels` maps an input (role_bucket, grade, ...) to its value in every row, `block_of_level` the
    input to its block. Levels with fewer than 10 rows are left out of the per-level table."""
    blocks = {
        name: {
            "mean_abs_log": round(float(np.mean(np.abs(phi[:, j]))), 4),
            "mean_abs_pct": round(float(np.mean(np.abs(np.expm1(phi[:, j])))) * 100, 1),
        }
        for j, name in enumerate(names)
    }
    per_level: dict[str, dict[str, dict[str, float | int]]] = {}
    for column, values in levels.items():
        j = list(names).index(block_of_level[column])
        table = {}
        for level in sorted(set(values.tolist())):
            mask = values == level
            if mask.sum() < 10:
                continue
            mean = float(np.mean(phi[mask, j]))
            table[str(level)] = {
                "rows": int(mask.sum()),
                "mean_log": round(mean, 4),
                "pct": round(float(np.expm1(mean)) * 100, 1),
            }
        per_level[column] = dict(sorted(table.items(), key=lambda kv: -kv[1]["mean_log"]))
    return {"blocks": dict(sorted(blocks.items(), key=lambda kv: -kv[1]["mean_abs_log"])), "per_level": per_level}
