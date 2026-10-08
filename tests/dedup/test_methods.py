import numpy as np
import pytest

from radar.dedup.minhash import lsh_pairs
from radar.dedup.neighbours import knn_pairs, top_k
from radar.dedup.normalize import shingles

BASE = (
    "Ищем Python разработчика в команду платежей. Задачи: развитие сервисов на FastAPI, "
    "работа с PostgreSQL и Kafka, ревью кода. Требования: опыт от трёх лет, знание SQL. "
    "Условия: удалённая работа, ДМС, гибкий график, обучение за счёт компании."
)
REPOST = BASE.replace("ДМС, ", "ДМС и спорт, ")
OTHER = (
    "Требуется бухгалтер на первичную документацию. Ведение учёта в 1С, сверка с контрагентами, "
    "подготовка отчётности. Офис в центре города, пятидневка, официальное оформление."
)


def test_lsh_finds_the_repost_and_not_the_other_text() -> None:
    docs = [shingles(t, 3) for t in (BASE, OTHER, REPOST, "")]
    assert set(lsh_pairs(docs, threshold=0.5, num_perm=128, seed=1)) == {(0, 2)}


def test_top_k_is_exact_and_skips_self() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx, sim = top_k(v, k=3, block=7)
    full = v @ v.T
    np.fill_diagonal(full, -np.inf)
    expected = np.argsort(-full, axis=1)[:, :3]
    assert (idx == expected).all()
    assert sim[:, 0] == pytest.approx(full.max(axis=1), abs=1e-6)


def test_knn_pairs_are_undirected_and_above_the_floor() -> None:
    idx = np.array([[1, 2], [0, 2], [1, 0]])
    sim = np.array([[0.95, 0.5], [0.95, 0.9], [0.9, 0.5]], dtype=np.float32)
    assert knn_pairs(idx, sim, min_cosine=0.85) == {(0, 1): pytest.approx(0.95), (1, 2): pytest.approx(0.9)}
