"""AWQ 4-bit checkpoint of the merged student for vLLM, in its own WSL venv ~/venvs/awq.

    MSYS_NO_PATHCONV=1 wsl.exe -d radar --cd /mnt/d/VScode_projects/ru-it-jobs-radar --exec env \
        PYTHONPATH=src HF_HUB_OFFLINE=1 /home/radar/venvs/awq/bin/python -m radar.train.awq --name qwen3_1p7b_r16_clean

llm-compressor's AWQ (weights 4 bit, asymmetric, group 128, activations 16 bit; lm_head left in
bf16) on models/merged/<name>, written as a compressed-tensors checkpoint to models/awq/<name>/,
which vLLM serves with its Marlin kernels. The venv is separate: no llm-compressor release fits
the pins of the vLLM venv (compressed-tensors 0.17.0, torch 2.13), and in the training venv it
would move torch.

Calibration examples are drawn from data/sft/train.jsonl, the student's own training rows (the
system prompt, the capped input and the silver answer in the training format), never from a dev
or test split, with the seed of the run config.
"""

import argparse
import json
import logging
import random
import sys
from datetime import UTC, datetime
from typing import Any

from radar.config import ROOT, data_dir
from radar.results import git_commit
from radar.train.dataset import read_jsonl, render_prompt

log = logging.getLogger("radar.train.awq")

MERGED = ROOT / "models" / "merged"
AWQ = ROOT / "models" / "awq"
SEED = 20261008
SCHEME = "W4A16_ASYM"


def calibration_texts(rows: list[dict[str, Any]], tokenizer: Any, n: int, seed: int) -> list[str]:
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    texts = []
    for row in picked:
        system, user, answer = (m["content"] for m in row["messages"])
        texts.append(render_prompt(tokenizer, system, user) + answer + tokenizer.eos_token)
    return texts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.awq", description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="merged model, models/merged/<name>")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    import torch  # type: ignore[import-not-found]
    from datasets import Dataset  # type: ignore[import-not-found]
    from llmcompressor import oneshot  # type: ignore[import-not-found]
    from llmcompressor.modifiers.quantization import QuantizationModifier  # type: ignore[import-not-found]
    from llmcompressor.modifiers.transform.awq import AWQModifier  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    merged = MERGED / args.name
    out = AWQ / args.name
    tok = AutoTokenizer.from_pretrained(str(merged))
    model = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.bfloat16)
    texts = calibration_texts(read_jsonl(data_dir() / "sft" / "train.jsonl"), tok, args.samples, args.seed)
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=args.max_seq_len)
    ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    # cached layer inputs on the CPU: on the GPU they filled the 8 GB card, the driver spilled into
    # shared memory and one grid point took 5 s (about 3 h for the model, logs/quant_awq_try1.log)
    awq = AWQModifier(offload_device=torch.device("cpu"))
    recipe = [awq, QuantizationModifier(targets=["Linear"], scheme=SCHEME, ignore=["lm_head"])]
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.max_seq_len, num_calibration_samples=len(ds))
    model.save_pretrained(str(out), save_compressed=True)
    tok.save_pretrained(str(out))
    (out / "awq.json").write_text(
        json.dumps(
            {
                "merged": merged.relative_to(ROOT).as_posix(),
                "scheme": SCHEME,
                "ignore": ["lm_head"],
                "calibration": {
                    "file": "data/sft/train.jsonl",
                    "samples": len(ds),
                    "seed": args.seed,
                    "max_seq_len": args.max_seq_len,
                    "tokens": sum(len(x) for x in enc["input_ids"]),
                },
                "commit": git_commit(),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("AWQ checkpoint -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
