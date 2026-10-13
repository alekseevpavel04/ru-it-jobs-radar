"""Resume-to-vacancy matching: lexical, dense and fused retrieval, a cross-encoder rerank.

Pool, synthetic resumes and relevance labels: `radar.models.matching_data`. Embeddings and reranker
scores are computed in WSL where torch lives: `radar.models.matching_embed`.

Systems, every one scored on the same dev resumes against the whole pool:
- bm25: Okapi BM25 over lowercased word tokens of the vacancy text;
- dense_<model>: cosine of normalized embeddings;
- fusion_<model>@<weight %>: weighted sum of the per-query standardized dense and BM25 scores, the weight
  chosen on dev from the config grid (so its dev number is optimistic by the choice);
- rrf_<model>: reciprocal rank fusion of the two, no parameter fitted;
- rerank: the top `rerank_depth` of the served first stage re-scored by a cross-encoder, on the
  first `rerank_queries` dev resumes only (CPU time), compared paired with its first stage there.
Metrics per resume: nDCG@10 with gains 2^rel - 1, MRR of the first highly relevant vacancy, capped
recall of highly relevant vacancies at 50 (found / min(relevant, 50), R_cap in BEIR). Intervals are
percentile bootstraps over resumes, differences paired on the same resamples.

    python -m radar.models.matching build              pool and resumes from ClickHouse -> data/matching/
    python -m radar.models.matching candidates         first-stage lists for the reranker
    python -m radar.models.matching evaluate           dev -> results/models/matching.json, index -> models/matching/
    python -m radar.models.matching evaluate --final-test   the saved choice on test (once)
"""

import argparse
import json
import logging
import pickle
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import-untyped]

from radar.config import RESULTS, ROOT, load_dotenv
from radar.models.matching_data import (
    CONFIG,
    DEV,
    TEST,
    label_counts,
    load_pool,
    load_query_config,
    make_queries,
    qrels,
)
from radar.models.matching_embed import SPECS, alias, load_encoder, slug, work_dir
from radar.results import git_commit, invocation, write_result

log = logging.getLogger("radar.models.matching")

Array = NDArray[np.float64]
EVAL_CONFIG = ROOT / "configs" / "eval.toml"
MODEL_DIR = ROOT / "models" / "matching"
ARTIFACT = "index.pkl"
RESULT = "models/matching.json"
RESULT_TEST = "models/matching_test.json"
TOKEN = r"(?u)\w[\w#+]*"
METRICS = ("ndcg", "mrr", "recall")


@dataclass(frozen=True)
class Config:
    models: list[str]
    max_length: int
    k1: float
    b: float
    rrf_k: int
    fusion_weights: list[float]
    rerank_depth: int
    reranker: str
    served_models: list[str]
    ndcg_k: int
    recall_k: int
    n_resamples: int
    bootstrap_seed: int
    confidence: float


def load_config(path: Path = CONFIG, eval_path: Path = EVAL_CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    boot = tomllib.loads(eval_path.read_text(encoding="utf-8"))["bootstrap"]
    e, r, m = raw["embed"], raw["retrieval"], raw["metrics"]
    return Config(
        models=list(e["models"]),
        max_length=int(e["max_length"]),
        k1=float(r["bm25_k1"]),
        b=float(r["bm25_b"]),
        rrf_k=int(r["rrf_k"]),
        fusion_weights=[float(w) for w in r["fusion_weights"]],
        rerank_depth=int(r["rerank_depth"]),
        reranker=str(r["reranker"]),
        served_models=list(raw["serving"]["models"]),
        ndcg_k=int(m["ndcg_k"]),
        recall_k=int(m["recall_k"]),
        n_resamples=int(boot["n_resamples"]),
        bootstrap_seed=int(boot["seed"]),
        confidence=float(boot["confidence"]),
    )


# --- retrieval ----------------------------------------------------------------------------------


@dataclass
class BM25:
    k1: float
    b: float
    vectorizer: Any = None
    weights: Any = None  # (n_docs, vocab) sparse BM25 weight of every term in every document

    def fit(self, docs: Sequence[str]) -> "BM25":
        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=TOKEN, dtype=np.float64)
        tf = sparse.csr_matrix(self.vectorizer.fit_transform(docs))
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        length = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * length / length.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape)
        return self

    def scores(self, queries: Sequence[str]) -> Array:
        q = sparse.csr_matrix(self.vectorizer.transform(queries))
        q.data[:] = 1.0  # a query term counts once
        return np.asarray((q @ self.weights.T).toarray(), dtype=np.float64)


def standardize(scores: Array) -> Array:
    sd = scores.std(axis=1, keepdims=True)
    return np.asarray((scores - scores.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0), dtype=np.float64)


def ranks(scores: Array) -> NDArray[np.int64]:
    """Rank of every document per query, 0 for the best; ties broken by document order."""
    order = np.argsort(-scores, axis=1, kind="stable")
    out = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    out[rows, order] = np.arange(scores.shape[1])[None, :]
    return out


def rrf(a: Array, b: Array, k: int) -> Array:
    return np.asarray(1.0 / (k + 1 + ranks(a)) + 1.0 / (k + 1 + ranks(b)), dtype=np.float64)


def reranked(first: Array, rerank: Mapping[int, tuple[list[int], list[float]]], depth: int) -> Array:
    """First-stage scores with the top `depth` of each query replaced by the reranker order above the rest."""
    out = first.copy()
    for q, (cand, scores) in rerank.items():
        top, s = np.asarray(cand[:depth]), np.asarray(scores[:depth], dtype=float)
        out[q, top] = out[q].max() + 1.0 + (s - s.min()) / (np.ptp(s) or 1.0)
    return out


# --- metrics ------------------------------------------------------------------------------------


def per_query(scores: Array, labels: Sequence[NDArray[np.int8]], ndcg_k: int, recall_k: int) -> dict[str, Array]:
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for s, rel in zip(scores, labels, strict=True):
        order = np.argsort(-s, kind="stable")
        gains = (2.0 ** rel[order[:ndcg_k]]) - 1
        ideal = (2.0 ** np.sort(rel)[::-1][:ndcg_k]) - 1
        discount = 1 / np.log2(np.arange(2, ndcg_k + 2))
        out["ndcg"].append(
            float(gains @ discount[: len(gains)] / (ideal @ discount[: len(ideal)])) if ideal.any() else 0.0
        )
        high = rel[order] == 2
        out["mrr"].append(1.0 / (int(np.argmax(high)) + 1) if high.any() else 0.0)
        relevant = int((rel == 2).sum())
        out["recall"].append(float(high[:recall_k].sum()) / min(relevant, recall_k) if relevant else 0.0)
    return {m: np.array(v) for m, v in out.items()}


def resample_weights(n: int, n_resamples: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_resamples).astype(np.float64)


def ci(point: float, samples: Array, confidence: float) -> dict[str, float]:
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return {"value": round(point, 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4)}


def summarize(values: Array, W: Array, confidence: float) -> dict[str, float]:
    return ci(float(values.mean()), (W @ values) / W.sum(axis=1), confidence)


# --- evaluation ---------------------------------------------------------------------------------


@dataclass
class Inputs:
    pool: pd.DataFrame
    queries: pd.DataFrame
    labels: list[NDArray[np.int8]]
    bm25: BM25
    dense_pool: dict[str, Array] = field(default_factory=dict)
    dense_queries: dict[str, Array] = field(default_factory=dict)


def load_inputs(cfg: Config, split: str) -> Inputs:
    wd = work_dir()
    pool = pd.read_parquet(wd / "pool.parquet")
    queries = pd.read_parquet(wd / "queries.parquet")
    keep = (queries["split"] == split).to_numpy()
    q = queries[keep].reset_index(drop=True)
    labels = qrels(q, pool, load_query_config())
    inputs = Inputs(pool, q, labels, BM25(cfg.k1, cfg.b).fit(list(pool["text"])))
    for model in cfg.models:
        target = wd / "emb" / slug(model)
        if not (target / "meta.json").exists():
            log.warning("no embeddings of %s yet, skipped", model)
            continue
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
        if meta["pool_ids_sha256"] != _digest(pool["entity_id"]) or meta["query_ids_sha256"] != _digest(
            queries["query_id"]
        ):
            raise SystemExit(f"embeddings of {model} were computed for another pool or query set; rerun the embed")
        inputs.dense_pool[model] = np.load(target / "pool.npy").astype(np.float64)
        inputs.dense_queries[model] = np.load(target / "queries.npy")[keep].astype(np.float64)
    return inputs


def _digest(ids: pd.Series) -> str:
    import hashlib

    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def system_scores(inputs: Inputs, cfg: Config) -> dict[str, Array]:
    lexical = inputs.bm25.scores(list(inputs.queries["resume"]))
    out = {"bm25": lexical}
    z_lexical = standardize(lexical)
    for model, P in inputs.dense_pool.items():
        dense = inputs.dense_queries[model] @ P.T
        out[f"dense_{alias(model)}"] = dense
        out[f"rrf_{alias(model)}"] = rrf(dense, lexical, cfg.rrf_k)
        z_dense = standardize(dense)
        for w in cfg.fusion_weights:
            out[f"fusion_{alias(model)}@{round(w * 100)}"] = w * z_dense + (1 - w) * z_lexical
    return out


def load_rerank(model: str, query_ids: Sequence[str]) -> dict[int, dict[str, Any]]:
    """Reranker outputs of the queries that have one (the reranker may have run on a subset)."""
    folder = work_dir() / "rerank" / slug(model)
    found = {}
    for i, qid in enumerate(query_ids):
        path = folder / f"{qid}.json"
        if path.exists():
            found[i] = json.loads(path.read_text(encoding="utf-8"))
    return found


def rerank_block(scores: Mapping[str, Array], inputs: Inputs, cfg: Config, split: str) -> dict[str, dict[str, Any]]:
    """The reranker against its own first stage on the resumes it scored, paired."""
    found = load_rerank(cfg.reranker, list(inputs.queries["query_id"]))
    source = rerank_source()
    if not found or source not in scores:
        return {}
    idx = sorted(found)
    labels = [inputs.labels[i] for i in idx]
    first = scores[source][idx]
    top = np.argsort(-first, axis=1, kind="stable")[:, : cfg.rerank_depth]
    for j, i in enumerate(idx):
        if list(found[i]["candidates"][: cfg.rerank_depth]) != top[j].tolist():
            raise SystemExit(f"reranker input of {inputs.queries['query_id'][i]} is not the top of {source}; rerun it")
    pairs = {j: (found[i]["candidates"], found[i]["scores"]) for j, i in enumerate(idx)}
    second = reranked(first, pairs, cfg.rerank_depth)
    a = per_query(first, labels, cfg.ndcg_k, cfg.recall_k)
    b = per_query(second, labels, cfg.ndcg_k, cfg.recall_k)
    W = resample_weights(len(idx), cfg.n_resamples, cfg.bootstrap_seed)
    seconds = [float(found[i]["seconds"]) for i in idx if "seconds" in found[i]]
    out: dict[str, dict[str, Any]] = {f"{split}.rerank.queries": {"value": len(idx)}}
    if seconds:
        out[f"{split}.rerank.seconds_per_query_median"] = {"value": round(float(np.median(seconds)), 1)}
    for m in METRICS:
        out[f"{split}.rerank.first_stage.{m}"] = summarize(a[m], W, cfg.confidence)
        out[f"{split}.rerank.reranked.{m}"] = summarize(b[m], W, cfg.confidence)
        out[f"{split}.rerank.diff.{m}"] = summarize(b[m] - a[m], W, cfg.confidence)
    return out


def model_of(system: str) -> str | None:
    """Encoder of a first-stage system name (`dense_e5-base`, `fusion_e5-base@70`), None for bm25."""
    if system == "bm25":
        return None
    short = system.split("_", 1)[1].split("@")[0]
    return next(m for m in SPECS if alias(m) == short)


def best_fusion(rows: Mapping[str, Mapping[str, Array]], model: str) -> str:
    candidates = [n for n in rows if n.startswith(f"fusion_{alias(model)}@")]
    return max(candidates, key=lambda n: (float(rows[n]["ndcg"].mean()), n))


def evaluate(inputs: Inputs, cfg: Config, split: str, systems: Sequence[str] | None = None) -> dict[str, Any]:
    scores = system_scores(inputs, cfg)
    per = {n: per_query(s, inputs.labels, cfg.ndcg_k, cfg.recall_k) for n, s in scores.items()}
    tuned = {f"fusion_{alias(m)}": best_fusion(per, m) for m in inputs.dense_pool}
    # Fusion is reported at its dev-chosen weight only; the grid stays in `selection`.
    grid = {n: round(float(per[n]["ndcg"].mean()), 4) for n in per if "@" in n}
    keep = {n: v for n, v in per.items() if "@" not in n or n in tuned.values()}
    if systems is not None:
        keep = {n: per[n] for n in systems if n in per}
    W = resample_weights(len(inputs.queries), cfg.n_resamples, cfg.bootstrap_seed)
    metrics: dict[str, dict[str, Any]] = {}
    for name, cols in keep.items():
        for m in METRICS:
            metrics[f"{split}.{name}.{m}"] = summarize(cols[m], W, cfg.confidence)
    best = max(keep, key=lambda n: (float(keep[n]["ndcg"].mean()), n))
    for name, cols in keep.items():
        for other in {"bm25", best} - {name}:
            for m in METRICS:
                metrics[f"{split}.diff.{name}.minus.{other}.{m}"] = summarize(
                    cols[m] - keep[other][m], W, cfg.confidence
                )
    if systems is None:
        metrics |= rerank_block(scores, inputs, cfg, split)
    return {"metrics": metrics, "per_query": keep, "best": best, "fusion_grid_ndcg": grid, "tuned_fusion": tuned}


def rerank_source() -> str:
    path = work_dir() / "rerank_candidates.meta.json"
    if not path.exists():
        return ""
    return str(json.loads(path.read_text(encoding="utf-8"))["system"])


# --- serving ------------------------------------------------------------------------------------


@dataclass
class MatchIndex:
    """What /match loads: the pool with its vectors and the scorer of the served system."""

    system: str
    model: str | None
    fusion_weight: float | None
    bm25: BM25 | None
    vectors: NDArray[np.float32] | None
    max_length: int
    items: list[dict[str, Any]]
    open_now: NDArray[np.bool_]
    version: str
    built_at: str
    rrf_k: int = 60
    _encode: Callable[[Sequence[str]], Any] | None = field(default=None, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_encode"} | {"_encode": None}

    def use_encoder(self, encode: Callable[[Sequence[str]], Any]) -> None:
        """Encode queries with `encode` (the API's ONNX export) instead of loading the model with torch."""
        self._encode = encode

    def query_vector(self, text: str) -> Array:
        if self.model is None:
            raise ValueError("the served system has no dense part")
        if self._encode is None:
            self._encode = load_encoder(self.model, self.max_length)
        vectors, _ = self._encode([SPECS[self.model].query_prefix + text])
        return np.asarray(vectors[0], dtype=np.float64)

    def scores(self, text: str) -> Array:
        parts: dict[str, Array] = {}
        if self.bm25 is not None:
            parts["bm25"] = self.bm25.scores([text])[0]
        if self.vectors is not None:
            parts["dense"] = self.vectors.astype(np.float64) @ self.query_vector(text)
        if self.system == "bm25":
            out = parts["bm25"]
        elif self.system.startswith("dense_"):
            out = parts["dense"]
        elif self.system.startswith("rrf_"):
            out = rrf(parts["dense"][None, :], parts["bm25"][None, :], self.rrf_k)[0]
        else:
            w = float(self.fusion_weight or 0.0)
            out = w * standardize(parts["dense"][None, :])[0] + (1 - w) * standardize(parts["bm25"][None, :])[0]
        return np.asarray(out, dtype=np.float64)


@lru_cache(maxsize=2)
def load_index(path: Path = MODEL_DIR / ARTIFACT) -> MatchIndex:
    with path.open("rb") as f:
        index = pickle.load(f)
    if not isinstance(index, MatchIndex):
        raise TypeError(f"{path} does not hold a MatchIndex")
    return index


def match(resume_text: str, k: int = 10, open_only: bool = True, index: MatchIndex | None = None) -> dict[str, Any]:
    """Top-k vacancies of the pool for a resume text.

    Items carry vacancy_entity_id (the dedup entity of the marts), source family, is_open_now, rank
    (from 1) and score; `open_only` keeps vacancies open on the last collection day. The score is
    comparable within one call only."""
    index = index or load_index()
    if not resume_text.strip():
        raise ValueError("empty resume text")
    s = index.scores(resume_text)
    if open_only:
        s = np.where(index.open_now, s, -np.inf)
    top = np.argsort(-s, kind="stable")[:k]
    kept = [i for i in top if np.isfinite(s[i])]
    items = [index.items[i] | {"rank": r + 1, "score": round(float(s[i]), 4)} for r, i in enumerate(kept)]
    return {"items": items, "system": index.system, "model_version": index.version}


def build_index(inputs: Inputs, system: str, weight: float | None, cfg: Config) -> MatchIndex:
    model = model_of(system)
    pool = inputs.pool
    # Titles, employers and links stay out: the API applies each source's publishing terms itself.
    items = [
        {"vacancy_entity_id": r["entity_id"], "source": r["family"], "is_open_now": bool(r["is_open_now"])}
        for r in pool.to_dict("records")
    ]
    return MatchIndex(
        system=system,
        model=model,
        fusion_weight=weight,
        bm25=None if system.startswith("dense_") else inputs.bm25,
        vectors=None if model is None else inputs.dense_pool[model].astype(np.float32),
        max_length=cfg.max_length,
        items=items,
        open_now=pool["is_open_now"].to_numpy(dtype=bool),
        version=f"match-{system}-{git_commit()}",
        rrf_k=cfg.rrf_k,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# --- commands -----------------------------------------------------------------------------------


def build() -> None:
    raw = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    pool = load_pool(raw["pool"]["extraction_model"])
    queries = make_queries(pool, load_query_config())
    wd = work_dir()
    wd.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(wd / "pool.parquet", index=False)
    queries.to_parquet(wd / "queries.parquet", index=False)
    log.info("pool %d, queries %s", len(pool), queries["split"].value_counts().to_dict())


def candidates(cfg: Config, system: str) -> None:
    """Top `rerank_depth` pool rows of `system` for every dev resume, the reranker's input."""
    inputs = load_inputs(cfg, DEV)
    scores = system_scores(inputs, cfg)[system]
    top = np.argsort(-scores, axis=1, kind="stable")[:, : cfg.rerank_depth]
    lists = {qid: [int(i) for i in row] for qid, row in zip(inputs.queries["query_id"], top, strict=True)}
    wd = work_dir()
    (wd / "rerank_candidates.json").write_text(json.dumps(lists), encoding="utf-8")
    (wd / "rerank_candidates.meta.json").write_text(json.dumps({"system": system}), encoding="utf-8")
    log.info("wrote %d candidate lists of %s", len(lists), system)


def served_choice(result: Mapping[str, Any], cfg: Config) -> str:
    """Best system on dev nDCG@10 whose encoder the API can hold (serving.models)."""
    per = result["per_query"]
    allowed = [n for n in per if n == "bm25" or model_of(n) in cfg.served_models]
    best: str = max(allowed, key=lambda n: (float(per[n]["ndcg"].mean()), n))
    return best


def run_dev(cfg: Config) -> dict[str, Any]:
    inputs = load_inputs(cfg, DEV)
    result = evaluate(inputs, cfg, DEV)
    tuned = result["tuned_fusion"]
    served = served_choice(result, cfg)
    weight = int(served.split("@")[1]) / 100 if "@" in served else None
    index = build_index(inputs, served, weight, cfg)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / ARTIFACT).open("wb") as f:
        pickle.dump(index, f)
    (MODEL_DIR / "choice.json").write_text(
        json.dumps({"systems": list(result["per_query"]), "served": served}, indent=2) + "\n", encoding="utf-8"
    )
    counts = label_counts(inputs.labels)
    metrics = result["metrics"] | {
        "data.pool": {"value": len(inputs.pool)},
        "data.pool_open_now": {"value": int(inputs.pool["is_open_now"].sum())},
        "data.queries_dev": {"value": len(inputs.queries)},
        "data.relevant_median": {"value": float(np.median(np.add(counts["grade1"], counts["grade2"])))},
        "data.highly_relevant_median": {"value": float(np.median(counts["grade2"]))},
    }
    return {
        "metrics": metrics,
        "best_dev": result["best"],
        "served": served,
        "selection": {"tuned_fusion": tuned, "fusion_grid_ndcg": result["fusion_grid_ndcg"]},
        "embeddings": {m: _meta(m) for m in inputs.dense_pool},
        "model_version": index.version,
    }


def _meta(model: str) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((work_dir() / "emb" / slug(model) / "meta.json").read_text(encoding="utf-8"))
    return meta


def run_test(cfg: Config) -> dict[str, Any]:
    choice = json.loads((MODEL_DIR / "choice.json").read_text(encoding="utf-8"))
    path = RESULTS / "test_runs.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\tmatching\t-\t{git_commit()}\t{invocation()}\n")
    inputs = load_inputs(cfg, TEST)
    result = evaluate(inputs, cfg, TEST, systems=choice["systems"])
    return {"metrics": result["metrics"], "served": choice["served"]}


DEFINITIONS = {
    "pool": "IT dedup entities of trudvsem, Habr Career and SuperJob whose latest record has valid fields of the "
    "extraction model with a role other than non_it; text is the extractor input of that record.",
    "queries": "Synthetic resumes rendered by templates from the extracted fields of a seed vacancy (role title, "
    "grade, experience, 60-100% of its skills plus up to 2 common skills of the role, city, remote); "
    "seeds with at least 3 skills, split dev/test by a salted hash of the entity id.",
    "relevance": "2: same role bucket, grade within one step, place fits and the resume covers at least half "
    "of the vacancy's extracted skills; 1: the same without the skill condition; 0 otherwise.",
    "ndcg": "nDCG@10 with gains 2^rel - 1 against the ideal ranking of the whole pool.",
    "mrr": "Reciprocal rank of the first vacancy with relevance 2.",
    "recall": "Vacancies with relevance 2 in the top 50 divided by min(their number, 50).",
    "ci": "Percentile bootstrap over resumes, resamples and seed in configs/eval.toml; differences paired.",
}

LIMITS_RU = [
    "Резюме синтетические: шаблон из полей вакансии, короткий и перечисляющий навыки. Настоящие резюме "
    "длиннее и написаны свободным текстом, на них лексический поиск теряет преимущество совпадения слов.",
    "Релевантность задана правилом по извлечённым полям, а не оценкой человека; ошибки извлечения "
    "становятся ошибками разметки.",
    "Вес слияния выбран на dev, поэтому число fusion на dev оптимистично на величину выбора; тест считается один раз.",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.matching", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    c = sub.add_parser("candidates")
    c.add_argument("--system", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--final-test", action="store_true", help="the one scoring of the test resumes")
    args = ap.parse_args(argv)
    load_dotenv()
    if args.command == "build":
        build()
        return 0
    cfg = load_config()
    if args.command == "candidates":
        candidates(cfg, args.system)
        return 0
    if args.final_test:
        doc = run_test(cfg)
        written = write_result(RESULT_TEST, doc.pop("metrics"), source_script="src/radar/models/matching.py", **doc)
    else:
        doc = run_dev(cfg)
        written = write_result(
            RESULT,
            doc.pop("metrics"),
            source_script="src/radar/models/matching.py",
            definitions=DEFINITIONS,
            limits_ru=LIMITS_RU,
            config=CONFIG.relative_to(ROOT).as_posix(),
            **doc,
        )
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    # Run through the importable module, so the pickled index refers to radar.models.matching.
    from radar.models import matching

    sys.exit(matching.main())
