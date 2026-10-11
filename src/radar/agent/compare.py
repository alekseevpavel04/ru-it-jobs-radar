"""Execution accuracy: does the agent's result hold the reference answer.

A result matches when every reference column has a column of the result with the same values row by
row: order of rows and of columns is free, extra columns are allowed (a share next to its counts),
the number of rows is not. Strings compare case-insensitively. Integers compare exactly; other
numbers within 0.5% or 0.006 (a share rounded to two places), and a column of shares may come as
percent (x100).
"""

import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

REL_TOL = 0.005
ABS_TOL = 0.006
MAX_ASSIGNMENTS = 20_000


def _norm(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def value_match(ref: Any, pred: Any, scale: float = 1.0) -> bool:
    ref, pred = _norm(ref), _norm(pred)
    if ref is None or pred is None:
        return ref is None and pred is None
    ref_num = isinstance(ref, int | float)
    if ref_num != isinstance(pred, int | float):
        return False
    if not ref_num:
        return bool(ref == pred) and scale == 1.0
    target = float(ref) * scale
    diff = abs(float(pred) - target)
    if isinstance(ref, int) and scale == 1.0:
        return diff < 1e-6
    return diff <= max(REL_TOL * max(abs(target), abs(float(pred))), ABS_TOL * scale)


def _is_share_column(values: Sequence[Any]) -> bool:
    nums = [v for v in values if v is not None]
    return bool(nums) and all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in nums)


def _column_candidates(
    ref_rows: Sequence[Sequence[Any]], pred_rows: Sequence[Sequence[Any]], j: int
) -> list[tuple[int, float]]:
    ref_col = [row[j] for row in ref_rows]
    scales = [1.0, 100.0] if _is_share_column(ref_col) else [1.0]
    width = len(pred_rows[0]) if pred_rows else 0
    out = []
    for k in range(width):
        pred_col = [row[k] for row in pred_rows]
        for scale in scales:
            if _multiset_match([[v] for v in ref_col], [[v] for v in pred_col], [(0, scale)]):
                out.append((k, scale))
    return out


def _multiset_match(
    ref_rows: Sequence[Sequence[Any]], pred_rows: Sequence[Sequence[Any]], mapping: Sequence[tuple[int, float]]
) -> bool:
    if len(ref_rows) != len(pred_rows):
        return False
    unused = list(range(len(pred_rows)))
    for ref_row in ref_rows:
        for pos, i in enumerate(unused):
            pred_row = pred_rows[i]
            if all(value_match(ref_row[j], pred_row[k], s) for j, (k, s) in enumerate(mapping)):
                del unused[pos]
                break
        else:
            return False
    return True


def results_match(ref_rows: Sequence[Sequence[Any]], pred_rows: Sequence[Sequence[Any]]) -> bool:
    if len(ref_rows) != len(pred_rows):
        return False
    if not ref_rows:
        return True
    width = len(ref_rows[0])
    candidates = [_column_candidates(ref_rows, pred_rows, j) for j in range(width)]
    if any(not c for c in candidates):
        return False
    tried = 0

    def assign(j: int, chosen: list[tuple[int, float]]) -> bool:
        nonlocal tried
        if j == width:
            tried += 1
            return _multiset_match(ref_rows, pred_rows, chosen)
        for k, scale in candidates[j]:
            if tried >= MAX_ASSIGNMENTS:
                return False
            if any(k == used for used, _ in chosen):
                continue
            if assign(j + 1, [*chosen, (k, scale)]):
                return True
        return False

    return assign(0, [])
