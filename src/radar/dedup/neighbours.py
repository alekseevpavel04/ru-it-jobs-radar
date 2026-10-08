"""Exact cosine neighbours over L2-normalized vectors, in row blocks to bound memory.

Some 25 thousand units fit a brute-force pass in seconds on the CPU, so there is no index to tune
and nothing approximate to account for in recall.
"""

import numpy as np
from numpy.typing import NDArray


def top_k(vectors: NDArray[np.float32], k: int, block: int = 2048) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """For each row, the k most similar other rows and their cosine, most similar first."""
    n = vectors.shape[0]
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        scores = vectors[start:stop] @ vectors.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        idx[start:stop] = np.take_along_axis(part, order, axis=1)
        sim[start:stop] = np.take_along_axis(part_scores, order, axis=1)
    return idx, sim


def knn_pairs(idx: NDArray[np.int64], sim: NDArray[np.float32], min_cosine: float) -> dict[tuple[int, int], float]:
    """Undirected pairs (i < j) where one is among the other's neighbours with cosine at or above the floor."""
    out: dict[tuple[int, int], float] = {}
    rows, cols = np.nonzero(sim >= min_cosine)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        j = int(idx[r, c])
        out[(r, j) if r < j else (j, r)] = float(sim[r, c])
    return out


def cosine(vectors: NDArray[np.float32], i: int, j: int) -> float:
    return float(vectors[i] @ vectors[j])
