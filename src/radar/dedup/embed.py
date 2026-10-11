"""e5 embeddings of the dedup units, run inside WSL with the training venv (torch lives there).

    wsl -d radar --cd <repo> env HF_HOME=/mnt/d/VScode_projects/assets/huggingface HF_HUB_OFFLINE=1 \
        PYTHONPATH=src ~/venvs/train/bin/python -m radar.dedup.embed

HF_HOME has to be passed: the distro does not set it, and without it the model is looked up in an
empty cache and the run fails.

e5 is asked with the `query: ` prefix on both sides: its model card prescribes it for symmetric
similarity, which comparing two vacancies is. Mean pooling over tokens and L2 norm, as in the card.
Texts are sorted by length before batching so a batch pads to its own longest text, and the order
is restored on write. The model is read from the local HF cache only.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from radar.dedup.settings import load
from radar.dedup.units import inputs_digest, read_units

log = logging.getLogger("radar.dedup.embed")

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "query: "


def encode(texts: list[str], model_name: str, batch_size: int, max_length: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_name, local_files_only=True, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device).eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    truncated = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = tokenizer(
                [PREFIX + texts[i] for i in idx],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            truncated += int((batch["attention_mask"].sum(1) >= max_length).sum())
            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            out[idx] = pooled.cpu().numpy()
            done = start + len(idx)
            if done // batch_size % 50 == 0 or done == len(order):
                elapsed = time.perf_counter() - started
                log.info(
                    "%d/%d texts, %.1fs, eta %.0fs", done, len(order), elapsed, elapsed / done * (len(order) - done)
                )
    seconds = time.perf_counter() - started
    meta = {
        "model": model_name,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dtype": "float16" if device == "cuda" else "float32",
        "batch_size": batch_size,
        "max_length": max_length,
        "prefix": PREFIX,
        "texts": len(texts),
        "truncated_texts": truncated,
        "seconds": round(seconds, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
    }
    return out, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m radar.dedup.embed")
    parser.add_argument("--units", type=Path, default=ROOT / "data" / "dedup" / "units.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "dedup" / "embeddings.npy")
    parser.add_argument("--model", default=load().params.embedding_model)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, help="probe on the first N units; nothing is written")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    units = read_units(args.units)
    texts = [u.document() for u in units[: args.limit]]
    vectors, meta = encode(texts, args.model, args.batch_size, args.max_length)
    log.info("%s", json.dumps(meta))
    if args.limit:
        log.info(
            "probe: %.3fs per text, %.0fs forecast for %d units",
            meta["seconds"] / len(texts),
            meta["seconds"] / len(texts) * len(units),
            len(units),
        )
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vectors)
    meta["inputs_sha"] = inputs_digest(units)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
