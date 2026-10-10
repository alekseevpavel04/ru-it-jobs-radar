"""Embeddings of skill spellings for the offline review of the dictionary. Runs inside WSL
with the training venv, where torch lives:

    wsl -d radar --cd <repo> env PYTHONPATH=src ~/venvs/train/bin/python -m radar.extract.skill_embed \
        --model intfloat/multilingual-e5-base

Reads `data/skills/strings.json` (written by `python -m radar.extract.skill_review candidates`), writes
`data/skills/emb/<model slug>.npy` and a `.json` with the run's facts. Each model is pooled and
prefixed as its card prescribes for symmetric similarity. The model is read from the local HF cache only.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("radar.extract.skill_embed")

ROOT = Path(__file__).resolve().parents[3]
SKILLS_DATA = ROOT / "data" / "skills"

# model: (pooling, prefix)
MODELS: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-base": ("mean", "query: "),
    "intfloat/multilingual-e5-large": ("mean", "query: "),
    "BAAI/bge-m3": ("cls", ""),
    "deepvk/USER-bge-m3": ("cls", ""),
    "ai-forever/FRIDA": ("cls", "paraphrase: "),
}


def slug(model: str) -> str:
    return model.replace("/", "__")


def encode(texts: list[str], model_name: str, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

    pooling, prefix = MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_name, local_files_only=True, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    if model_name == "ai-forever/FRIDA":
        model = model.encoder  # FRIDA is a T5 encoder; the decoder takes no part in embeddings
    model.to(device).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = [prefix + t for t in texts[start : start + batch_size]]
            batch = tokenizer(chunk, max_length=64, truncation=True, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            hidden = model(**batch).last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            out[start : start + len(chunk)] = pooled.cpu().numpy()
    meta = {
        "model": model_name,
        "pooling": pooling,
        "prefix": prefix,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "texts": len(texts),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return out, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m radar.extract.skill_embed")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--strings", type=Path, default=SKILLS_DATA / "strings.json")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    doc = json.loads(args.strings.read_text(encoding="utf-8"))
    vectors, meta = encode(doc["strings"], args.model, args.batch_size)
    meta["strings_sha256"] = doc["sha256"]
    out = SKILLS_DATA / "emb" / f"{slug(args.model)}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info("%s", json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
