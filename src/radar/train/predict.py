"""Predictions of a student on a dev split, for `python -m radar.eval`.

    bash scripts/train_run.sh predict --adapter models/sft/qwen3_1p7b_r16/adapter --split dev --system sft_1p7b_r16
    bash scripts/train_run.sh predict --adapter models/sft/qwen3_1p7b_r16/epoch-1 --split tg_dev --system sft_1p7b_e1
    bash scripts/train_run.sh predict --model Qwen/Qwen3-4B --split dev --system base_4b_student_prompt

The input is the record's extractor input from data/gold/<split>_inputs.jsonl, cut to the
max_input_tokens the adapter was trained with and wrapped in the same system prompt and chat
template (dataset_meta.json next to the adapter; data/sft/meta.json for a base model). Greedy
decoding, batches sorted by length, left padding.

Decoding is not constrained by the JSON schema. The training venv's transformers stack has no
grammar backend (xgrammar sits in the vLLM venv), and the share of valid JSON without a grammar is
the measure EXPERIMENTS uses to decide on DPO, which constrained decoding would hide. Constrained
serving is measured in the inference benchmark. Each answer is checked against the schema here and the invalid
count goes to the sidecar; radar.eval scores an invalid answer as all fields empty.

Writes data/predictions/<split>/<system>.jsonl ({"id", "output", "valid", "finish", "truncated_input",
"prompt_tokens", "completion_tokens", "batch", "batch_size", "batch_seconds"}) and <system>.meta.json.
Test splits are not served.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar.config import ROOT, data_dir
from radar.eval.cli import SYSTEM_NAME
from radar.extract.schema import try_parse
from radar.results import git_commit
from radar.train.config import BACKENDS
from radar.train.dataset import cap_input, read_jsonl, render_prompt, token_spans, write_jsonl

log = logging.getLogger("radar.train.predict")

SPLITS = ("dev", "tg_dev")
MAX_NEW_TOKENS = 512  # the longest silver answer is about 210 tokens


def dataset_meta(adapter: Path | None) -> dict[str, Any]:
    path = adapter / "dataset_meta.json" if adapter else data_dir() / "sft" / "meta.json"
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def adapter_sha256(adapter: Path) -> str:
    return hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()


def batches(order: Sequence[int], size: int) -> list[list[int]]:
    return [list(order[i : i + size]) for i in range(0, len(order), size)]


def generate(
    llm: Any, tok: Any, prompts: Sequence[str], batch_size: int, max_new_tokens: int, cpu: bool
) -> list[dict[str, Any]]:
    import torch  # type: ignore[import-not-found]

    lengths = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    device = "cpu" if cpu else "cuda"
    out: list[dict[str, Any]] = [{} for _ in prompts]
    for b, idx in enumerate(batches(order, batch_size)):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        started = time.perf_counter()
        with torch.no_grad():
            gen = llm.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        seconds = time.perf_counter() - started
        new = gen[:, enc.input_ids.shape[1] :]
        for row, i in enumerate(idx):
            ids = new[row].tolist()
            ended = tok.eos_token_id in ids
            if ended:
                ids = ids[: ids.index(tok.eos_token_id)]
            ids = [t for t in ids if t != tok.pad_token_id]
            out[i] = {
                "output": tok.decode(ids, skip_special_tokens=True).strip(),
                "finish": "stop" if ended else "length",
                "prompt_tokens": lengths[i],
                "completion_tokens": len(ids) + int(ended),
                "batch": b,
                "batch_size": len(idx),
                "batch_seconds": round(seconds, 3),
            }
        log.info("batch %d: %d records in %.1f s", b, len(idx), seconds)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.predict", description=__doc__.splitlines()[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--adapter", type=Path, help="adapter directory (adapter/ or epoch-<k>/ of a run)")
    who.add_argument("--model", help="a base model alone, with the student prompt of data/sft/meta.json")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--system", required=True, help="predictions file name, e.g. qwen3_1p7b_sft_r16")
    ap.add_argument("--backend", choices=BACKENDS, default="unsloth")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, help="first N records only, for a probe (with --out)")
    ap.add_argument("--cpu", action="store_true", help="float32 on the CPU, no quantization (dry run)")
    ap.add_argument("--out", type=Path, help="default data/predictions/<split>/<system>.jsonl; set it for probes")
    args = ap.parse_args(argv)
    if not SYSTEM_NAME.match(args.system):
        raise SystemExit(f"system name {args.system!r}: lowercase letters, digits, _ and -")
    if args.limit and not args.out:
        raise SystemExit("--limit writes a partial file: give it --out, outside data/predictions")

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    adapter = args.adapter.resolve() if args.adapter else None
    meta = dataset_meta(adapter)
    records = read_jsonl(data_dir() / "gold" / f"{args.split}_inputs.jsonl")[: args.limit]

    from radar.train.hf import load_for_inference

    llm, tok = load_for_inference(adapter, args.model, args.backend, meta["max_seq_len"], args.cpu)
    prompts, truncated = [], []
    for r in records:
        user, cut = cap_input(r["input"], token_spans(tok, r["input"]), meta["max_input_tokens"])
        prompts.append(render_prompt(tok, meta["system_prompt"], user))
        truncated.append(cut)

    started = time.perf_counter()
    answers = generate(llm, tok, prompts, args.batch, args.max_new_tokens, args.cpu)
    wall = time.perf_counter() - started
    rows = []
    for r, a, cut in zip(records, answers, truncated, strict=True):
        rows.append({"id": r["id"], **a, "valid": try_parse(a["output"]) is not None, "truncated_input": cut})

    out = args.out or data_dir() / "predictions" / args.split / f"{args.system}.jsonl"
    write_jsonl(out, rows)
    n_valid = sum(r["valid"] for r in rows)
    sidecar = {
        "system": args.system,
        "split": args.split,
        "adapter": adapter.relative_to(ROOT).as_posix() if adapter and adapter.is_relative_to(ROOT) else None,
        "adapter_sha256": adapter_sha256(adapter) if adapter else None,
        "model": args.model,
        "backend": "transformers+peft (cpu)" if args.cpu else args.backend,
        "prompt_sha256": meta["prompt_sha256"],
        "max_input_tokens": meta["max_input_tokens"],
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens, "json_schema_constrained": False},
        "batch": args.batch,
        "records": len(rows),
        "limit": args.limit,
        "valid_json": {"n": n_valid, "denominator": len(rows)},
        "finish_length": sum(r["finish"] == "length" for r in rows),
        "truncated_inputs": sum(truncated),
        "wall_seconds": round(wall, 1),
        "completion_tokens_per_s": round(sum(r["completion_tokens"] for r in rows) / wall, 1),
        "commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    log.info("%s on %s: valid JSON %d of %d, %.0f s -> %s", args.system, args.split, n_valid, len(rows), wall, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
