"""Embeddings and reranker scores for matching, run where torch lives (the training venv in WSL, on CPU).

    wsl -d radar --cd <repo> env HF_HOME=/mnt/d/VScode_projects/assets/huggingface HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/venvs/train/bin/python -m radar.models.matching_embed \
        embed --model intfloat/multilingual-e5-base [--limit 50]

Texts are encoded in chunks of `chunk` (configs/matching.toml) in order of length, each chunk saved
as it is done, so a stopped run continues from the last finished chunk; `--limit` encodes only the
first texts of the pool and the queries and prints the speed (the probe before a long run).

Pooling and prefixes follow each model card: e5 takes `query: ` before a resume and `passage: `
before a vacancy and mean-pools tokens; bge-m3 takes no prefix and uses the [CLS] vector. Vectors are
L2-normalized. The module imports only numpy and pandas at the top, so the API can import
`load_encoder` without the training stack.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar.config import ROOT, data_dir

log = logging.getLogger("radar.models.matching_embed")

CONFIG = ROOT / "configs" / "matching.toml"


@dataclass(frozen=True)
class Spec:
    query_prefix: str
    passage_prefix: str
    pooling: str  # "mean" or "cls"


SPECS = {
    "intfloat/multilingual-e5-small": Spec("query: ", "passage: ", "mean"),
    "intfloat/multilingual-e5-base": Spec("query: ", "passage: ", "mean"),
    "intfloat/multilingual-e5-large": Spec("query: ", "passage: ", "mean"),
    "BAAI/bge-m3": Spec("", "", "cls"),
    "deepvk/USER-bge-m3": Spec("", "", "cls"),
}


def alias(model: str) -> str:
    """Short name in result keys: intfloat/multilingual-e5-base -> e5-base."""
    return model.split("/")[-1].removeprefix("multilingual-")


def work_dir() -> Path:
    return data_dir() / "matching"


def slug(model: str) -> str:
    return model.replace("/", "__")


def load_encoder(model_name: str, max_length: int, threads: int | None = None) -> Callable[[Sequence[str]], Any]:
    """A function from texts (prefix included) to L2-normalized float32 vectors, on CPU."""
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

    if threads:
        torch.set_num_threads(threads)
    spec = SPECS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval()

    def encode(texts: Sequence[str]) -> Any:
        batch = tokenizer(list(texts), max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            if spec.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.numpy().astype(np.float32), int((batch["attention_mask"].sum(1) >= max_length).sum())

    return encode


def encode_chunked(
    texts: Sequence[str], encode: Callable[[Sequence[str]], Any], batch_size: int, chunk: int, out_dir: Path
) -> tuple[np.ndarray, int]:
    """Vectors in the order of `texts`; finished chunks of the same texts are read back, not recomputed."""
    digest = _ids_digest(texts)
    stamp = out_dir / "texts.sha256"
    if stamp.exists() and stamp.read_text(encoding="utf-8") != digest:
        log.info("texts changed since the chunks in %s were written: starting over", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest, encoding="utf-8")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    parts: list[np.ndarray] = []
    truncated = 0
    started = time.perf_counter()
    done_now = 0
    for c, start in enumerate(range(0, len(order), chunk)):
        idx = order[start : start + chunk]
        path = out_dir / f"chunk_{c:05d}.npy"
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            parts.append(np.load(path))
            truncated += json.loads(meta_path.read_text(encoding="utf-8"))["truncated"]
            continue
        vectors, cut = [], 0
        for b in range(0, len(idx), batch_size):
            v, t = encode([texts[i] for i in idx[b : b + batch_size]])
            vectors.append(v)
            cut += t
        block = np.vstack(vectors)
        np.save(path, block)
        meta_path.write_text(json.dumps({"truncated": cut, "rows": len(idx)}), encoding="utf-8")
        parts.append(block)
        truncated += cut
        done_now += len(idx)
        elapsed = time.perf_counter() - started
        left = len(order) - start - len(idx)
        log.info(
            "%d/%d texts, %.0fs, %.2f texts/s, eta %.0fs",
            start + len(idx),
            len(order),
            elapsed,
            done_now / elapsed,
            elapsed / done_now * left,
        )
    stacked = np.vstack(parts)
    out = np.empty_like(stacked)
    out[order] = stacked
    return out, truncated


def embed(model: str, limit: int | None, cfg: dict[str, Any], threads: int | None) -> dict[str, Any]:
    wd = work_dir()
    pool = pd.read_parquet(wd / "pool.parquet", columns=["entity_id", "text"])
    queries = pd.read_parquet(wd / "queries.parquet", columns=["query_id", "resume"])
    spec = SPECS[model]
    encode = load_encoder(model, int(cfg["max_length"]), threads)
    target = wd / "emb" / slug(model)
    meta: dict[str, Any] = {"model": model, "max_length": cfg["max_length"], "device": "cpu", "dtype": "float32"}
    for kind, texts, prefix in (
        ("pool", list(pool["text"]), spec.passage_prefix),
        ("queries", list(queries["resume"]), spec.query_prefix),
    ):
        texts = [prefix + t for t in texts][:limit]
        started = time.perf_counter()
        sub = target / ("probe" if limit else "chunks") / kind
        vectors, truncated = encode_chunked(texts, encode, int(cfg["batch_size"]), int(cfg["chunk"]), sub)
        seconds = time.perf_counter() - started
        meta[kind] = {"texts": len(texts), "truncated": truncated, "seconds": round(seconds, 1), "prefix": prefix}
        log.info(
            "%s %s: %d texts in %.1fs (%.2f/s), %d truncated",
            model,
            kind,
            len(texts),
            seconds,
            len(texts) / seconds,
            truncated,
        )
        if not limit:
            np.save(target / f"{kind}.npy", vectors)
    if not limit:
        meta["pool_ids_sha256"] = _ids_digest(list(pool["entity_id"]))
        meta["query_ids_sha256"] = _ids_digest(list(queries["query_id"]))
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def rerank(model: str, depth: int, candidates: Path, limit: int | None, threads: int | None) -> None:
    """Cross-encoder scores of (resume, vacancy) for the candidate lists in `candidates` (json)."""
    # mypy reports a missing module once, at its first import in load_encoder.
    import torch  # type: ignore[import-not-found, unused-ignore]
    from transformers import (  # type: ignore[import-not-found, unused-ignore]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if threads:
        torch.set_num_threads(threads)
    wd = work_dir()
    pool = pd.read_parquet(wd / "pool.parquet", columns=["text"])
    queries = pd.read_parquet(wd / "queries.parquet", columns=["query_id", "resume"]).set_index("query_id")
    lists = json.loads(candidates.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    scorer = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True).eval()
    out_dir = wd / "rerank" / slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    started, done = time.perf_counter(), 0
    items = list(lists.items())[:limit]
    for n, (qid, cand) in enumerate(items):
        path = out_dir / f"{qid}.json"
        if path.exists():
            continue
        cand = cand[:depth]
        started_query = time.perf_counter()
        pairs = [[queries.loc[qid, "resume"], pool["text"].iloc[i]] for i in cand]
        scores: list[float] = []
        for b in range(0, len(pairs), 8):
            batch = tokenizer(
                pairs[b : b + 8], max_length=512, truncation="only_second", padding=True, return_tensors="pt"
            )
            with torch.inference_mode():
                scores += scorer(**batch).logits.view(-1).float().tolist()
        seconds = time.perf_counter() - started_query
        path.write_text(
            json.dumps({"candidates": cand, "scores": scores, "seconds": round(seconds, 2)}), encoding="utf-8"
        )
        done += 1
        elapsed = time.perf_counter() - started
        log.info(
            "%d/%d queries, %.1fs per query, eta %.0fs",
            n + 1,
            len(items),
            elapsed / done,
            elapsed / done * (len(items) - n - 1),
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.matching_embed")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("embed", help="pool and query embeddings of one model")
    e.add_argument("--model", required=True, choices=sorted(SPECS))
    e.add_argument("--limit", type=int, help="probe: only the first texts, nothing kept")
    e.add_argument("--threads", type=int)
    r = sub.add_parser("rerank", help="cross-encoder scores of the candidate lists")
    r.add_argument("--candidates", type=Path, default=work_dir() / "rerank_candidates.json")
    r.add_argument("--limit", type=int, help="only the first queries (default: rerank_queries of the config)")
    r.add_argument("--threads", type=int)
    args = ap.parse_args(argv)
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    if args.command == "embed":
        embed(args.model, args.limit, cfg["embed"], args.threads)
    else:
        retrieval = cfg["retrieval"]
        limit = args.limit or int(retrieval["rerank_queries"])
        rerank(retrieval["reranker"], int(retrieval["rerank_depth"]), args.candidates, limit, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
