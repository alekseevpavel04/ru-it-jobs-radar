"""The query encoder of /match as ONNX: exported once where torch lives, served with onnxruntime and
tokenizers, so the API image carries no torch.

    wsl -d radar --cd <repo> env HF_HOME=/mnt/d/VScode_projects/assets/huggingface HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/venvs/train/bin/python -m radar.models.matching_onnx export
    python -m radar.models.matching_onnx check      parity with the torch vectors -> results/models/matching_onnx.json

The graph ends at the last hidden state; pooling and L2 normalization stay in numpy here, as
`matching_embed.load_encoder` does them in torch, and the prefix is added by the caller
(`MatchIndex.query_vector`). `check` compares single-resume ONNX vectors, as /match computes them,
with the batched torch vectors the index was evaluated with, and scores the served system on dev
with each.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from radar.config import ROOT
from radar.models.matching_embed import CONFIG, SPECS, slug, work_dir

log = logging.getLogger("radar.models.matching_onnx")

# The Linux wheels of onnxruntime send usage events to Microsoft's collector unless this is set
# before the library is first imported; nothing of this project leaves the machine that way.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

DIRNAME = "onnx"  # next to the index: models/matching/onnx/
MODEL_FILE = "model.onnx"
# The weights sit in a file of their own: onnxruntime then maps it rather than copying the whole graph
# into the heap, and the embedding table (most of e5's weights) is read only for the tokens it sees.
DATA_FILE = "model.onnx.data"
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"
RESULT = "models/matching_onnx.json"
OUTPUT = "last_hidden_state"
# The served export sits next to the index, matching.MODEL_DIR (not imported: export runs without sklearn).
DEFAULT_DIR = ROOT / "models" / "matching" / DIRNAME


# --- serving ------------------------------------------------------------------------------------


def pool(hidden: NDArray[np.float32], mask: NDArray[np.int64], pooling: str) -> NDArray[np.float32]:
    """Mean over the unmasked tokens (or the first token), then L2 norm with torch's epsilon."""
    if pooling == "cls":
        vectors = hidden[:, 0]
    else:
        m = mask[..., None].astype(hidden.dtype)
        vectors = (hidden * m).sum(1) / m.sum(1)
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return np.asarray(vectors / norms, dtype=np.float32)


@dataclass
class OnnxEncoder:
    """Texts (prefix included) to L2-normalized float32 vectors and the number cut at max_length,
    the same contract as the torch encoder of `matching_embed.load_encoder`."""

    tokenizer: Any  # tokenizers.Tokenizer with truncation and padding enabled
    session: Any  # onnxruntime.InferenceSession, or anything with its run()
    pooling: str
    max_length: int

    def __call__(self, texts: Sequence[str]) -> tuple[NDArray[np.float32], int]:
        encodings = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        (hidden,) = self.session.run([OUTPUT], {"input_ids": ids, "attention_mask": mask})
        truncated = int((mask.sum(1) >= self.max_length).sum())
        return pool(np.asarray(hidden, dtype=np.float32), mask, self.pooling), truncated


def read_meta(directory: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    return meta


def load_encoder(directory: Path, model_name: str, max_length: int, threads: int | None = None) -> OnnxEncoder:
    """The exported encoder of `model_name`; an export of another model is refused, not served."""
    import onnxruntime as ort  # type: ignore[import-untyped]
    from tokenizers import Tokenizer

    meta = read_meta(directory)
    if meta["model"] != model_name:
        raise ValueError(f"{directory} holds {meta['model']}, the index needs {model_name}")
    model = directory / MODEL_FILE
    for path in (model, directory / DATA_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
    tokenizer = Tokenizer.from_file(str(directory / TOKENIZER_FILE))
    tokenizer.enable_truncation(max_length)
    tokenizer.enable_padding(pad_id=int(meta["pad_id"]), pad_token=str(meta["pad_token"]))
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    return OnnxEncoder(tokenizer, session, SPECS[model_name].pooling, max_length)


# --- export (torch) -----------------------------------------------------------------------------


def export(model_name: str, out_dir: Path, max_length: int, opset: int) -> dict[str, Any]:
    import onnx  # type: ignore[import-not-found, unused-ignore]
    import torch  # type: ignore[import-not-found, unused-ignore]
    import transformers  # type: ignore[import-not-found, unused-ignore]
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found, unused-ignore]

    # Loaded as matching_embed loads it, so the revision is the one the index vectors came from.
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True, attn_implementation="eager").eval()
    snapshot = Path(str(try_to_load_from_cache(model_name, "config.json"))).parent

    class Hidden(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    texts = ["query: Python", "query: ML-инженер, PyTorch, SQL, Docker. " * 200]
    sample = tokenizer(texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    backend = tokenizer.backend_tokenizer
    backend.enable_truncation(max_length)
    backend.enable_padding(pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)
    if [e.ids for e in backend.encode_batch(texts)] != sample["input_ids"].tolist():
        raise SystemExit("the bare tokenizer does not reproduce AutoTokenizer's ids")
    backend.no_truncation()
    backend.no_padding()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DATA_FILE).unlink(missing_ok=True)  # onnx appends to an existing data file
    started = time.perf_counter()
    torch.onnx.export(
        Hidden(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(out_dir / MODEL_FILE),
        input_names=["input_ids", "attention_mask"],
        output_names=[OUTPUT],
        dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}},
        opset_version=opset,
        dynamo=False,
    )
    graph = onnx.load(str(out_dir / MODEL_FILE))
    onnx.save_model(graph, str(out_dir / MODEL_FILE), save_as_external_data=True, location=DATA_FILE)
    backend.save(str(out_dir / TOKENIZER_FILE))
    meta = {
        "model": model_name,
        "snapshot": snapshot.name,
        "pad_id": tokenizer.pad_token_id,
        "pad_token": tokenizer.pad_token,
        "opset": opset,
        "output": OUTPUT,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "model_sha256": _sha256(out_dir / MODEL_FILE),
        "data_sha256": _sha256(out_dir / DATA_FILE),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- check (onnxruntime) ------------------------------------------------------------------------


def served_scores(index: Any, resumes: Sequence[str], vectors: NDArray[np.float32]) -> NDArray[np.float64]:
    """Scores of the served system over the whole pool, with the query vectors given in advance."""
    spec = SPECS[index.model]
    lookup = {spec.query_prefix + r: v for r, v in zip(resumes, vectors, strict=True)}
    index.use_encoder(lambda texts: (np.stack([lookup[t] for t in texts]), 0))
    return np.vstack([index.scores(r) for r in resumes])


def check(directory: Path, parity_n: int) -> dict[str, Any]:
    from radar.models.matching import (
        load_config,
        load_index,
        per_query,
        resample_weights,
        summarize,
    )
    from radar.models.matching_data import DEV, load_query_config, qrels

    cfg = load_config()
    index = load_index()
    if index.model is None:
        raise SystemExit(f"the served system {index.system} has no encoder")
    wd = work_dir()
    pool_df = pd.read_parquet(wd / "pool.parquet")
    if [i["vacancy_entity_id"] for i in index.items] != pool_df["entity_id"].tolist():
        raise SystemExit("the index was built over another pool than data/matching/pool.parquet")
    queries = pd.read_parquet(wd / "queries.parquet")
    keep = (queries["split"] == DEV).to_numpy()
    dev = queries[keep].reset_index(drop=True)
    torch_vectors = np.load(wd / "emb" / slug(index.model) / "queries.npy")[keep].astype(np.float32)
    resumes = list(dev["resume"])
    prefix = SPECS[index.model].query_prefix

    encode = load_encoder(directory, index.model, index.max_length)
    seconds, onnx_rows = [], []
    for r in resumes:  # one at a time, as /match encodes
        started = time.perf_counter()
        v, _ = encode([prefix + r])
        seconds.append(time.perf_counter() - started)
        onnx_rows.append(v[0])
    onnx_vectors = np.stack(onnx_rows)

    a, b = onnx_vectors[:parity_n], torch_vectors[:parity_n]
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    labels = qrels(dev, pool_df, load_query_config())
    by_onnx = served_scores(index, resumes, onnx_vectors)
    by_torch = served_scores(index, resumes, torch_vectors)
    q_onnx = per_query(by_onnx, labels, cfg.ndcg_k, cfg.recall_k)
    q_torch = per_query(by_torch, labels, cfg.ndcg_k, cfg.recall_k)
    W = resample_weights(len(dev), cfg.n_resamples, cfg.bootstrap_seed)
    top_onnx = np.argsort(-by_onnx, axis=1, kind="stable")[:, : cfg.ndcg_k]
    top_torch = np.argsort(-by_torch, axis=1, kind="stable")[:, : cfg.ndcg_k]
    same_top = int((top_onnx == top_torch).all(axis=1).sum())
    n = len(dev)
    system = index.system
    metrics: dict[str, dict[str, Any]] = {
        "parity.resumes": {"value": len(a)},
        "parity.max_abs_diff": {"value": float(np.abs(a - b).max())},
        "parity.cosine_min": {"value": round(float(cosine.min()), 8)},
        "parity.cosine_mean": {"value": round(float(cosine.mean()), 8)},
        f"dev.{system}.onnx.ndcg": summarize(q_onnx["ndcg"], W, cfg.confidence),
        f"dev.{system}.torch.ndcg": summarize(q_torch["ndcg"], W, cfg.confidence),
        f"dev.{system}.onnx_minus_torch.ndcg": summarize(q_onnx["ndcg"] - q_torch["ndcg"], W, cfg.confidence),
        f"dev.{system}.onnx.mrr": summarize(q_onnx["mrr"], W, cfg.confidence),
        f"dev.{system}.onnx.recall": summarize(q_onnx["recall"], W, cfg.confidence),
        "dev.same_top10": {"value": round(same_top / n, 4), "n": same_top, "denominator": n},
        "encode_ms_median": {"value": round(float(np.median(seconds)) * 1000, 1)},
    }
    return {
        "metrics": metrics,
        "served": system,
        "index_version": index.version,
        "encoder": read_meta(directory),
        "torch_vectors": f"data/matching/emb/{slug(index.model)}/queries.npy, dev rows",
        "comparison": f"ONNX vectors of single resumes against batched torch vectors, first {len(a)} dev resumes; "
        f"nDCG@{cfg.ndcg_k} of the served index over the whole pool on all {n} dev resumes with each",
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.matching_onnx", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="the encoder to ONNX (needs torch)")
    e.add_argument("--model", default="intfloat/multilingual-e5-small", choices=sorted(SPECS))
    e.add_argument("--opset", type=int, default=17)
    c = sub.add_parser("check", help="ONNX against the torch vectors and nDCG of the served system on dev")
    c.add_argument("--parity-n", type=int, default=200)
    for p in (e, c):
        p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    if args.command == "export":
        max_length = int(tomllib.loads(CONFIG.read_text(encoding="utf-8"))["embed"]["max_length"])
        meta = export(args.model, args.dir, max_length, args.opset)
        log.info("exported %s to %s in %.0fs", args.model, args.dir, meta["seconds"])
        return 0
    from radar.results import write_result

    doc = check(args.dir, args.parity_n)
    written = write_result(RESULT, doc.pop("metrics"), source_script="src/radar/models/matching_onnx.py", **doc)
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
