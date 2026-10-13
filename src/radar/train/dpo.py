"""One DPO iteration on top of the SFT student, in the WSL training venv.

The base loads in 4 bits as in QLoRA SFT (bitsandbytes NF4 through PEFT; Unsloth patches TRL's
trainers and is left out here) and the SFT adapter loads trainable on it. TRL's DPOTrainer copies
that adapter into a frozen "ref" adapter, so the reference policy is the SFT student itself, not the
bare base. The loss is sigmoid DPO plus, with train.sft_weight above 0, the NLL of the chosen answer
(TRL loss "sft"): chosen and rejected answers often differ by one token, and plain DPO then tends to
lower the likelihood of both.

Pairs come from radar.train.dpo_pairs (data/dpo/train.jsonl, val.jsonl): the prompt is rendered
here with the adapter's tokenizer and the system prompt of its dataset_meta.json, as in SFT and in
radar.train.predict, and both answers end with the end-of-turn token. Writes to models/dpo/<name>/:
trainer checkpoints (resumable), epoch-<k>/ adapters, adapter/ at the end with the tokenizer and the
SFT adapter's dataset_meta.json, so radar.train.predict serves it unchanged; run.json and the local
tracking copy. MLflow experiment radar-dpo (probes and dry runs in radar-dpo-probe, -dryrun). The
summary goes to results/dpo/train_<name>.json, a probe's to results/dpo/probe_<name>.json.

The VRAM and step-time guard (radar.train.guard) stops the run with a checkpoint, exit
code 3; --resume continues.

    bash scripts/train_run.sh dpo --config configs/dpo/qwen3_1p7b_clean.toml --probe 50   # speed, VRAM, forecast
    bash scripts/train_run.sh dpo --config configs/dpo/qwen3_1p7b_clean.toml
    bash scripts/train_run.sh dpo --config configs/dpo/qwen3_1p7b_clean.toml --resume
    bash scripts/train_run.sh dpo --config configs/dpo/qwen3_1p7b_clean.toml --dry-run --no-mlflow   # CPU, tiny model
"""

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar.config import ROOT, load_dotenv, optional
from radar.results import git_commit, write_result
from radar.train.config import DpoConfig, load_dpo
from radar.train.dataset import read_jsonl, render_prompt, sha256_file
from radar.train.guard import StepGuard, vram_limit_mib
from radar.train.sft import GUARD_EXIT, has_checkpoint, make_callback, versions
from radar.train.tracking import open_tracker

log = logging.getLogger("radar.train.dpo")

EXPERIMENT = "radar-dpo"
DRY_RUN_STEPS = 2
DRY_RUN_PAIRS = 8


def rows_for_trainer(pairs: Sequence[Mapping[str, Any]], tokenizer: Any, system: str) -> list[dict[str, str]]:
    """Text rows: the chat prompt up to the answer, and each answer with the end-of-turn token."""
    eos = tokenizer.eos_token
    return [
        {
            "prompt": render_prompt(tokenizer, system, p["user"]),
            "chosen": p["chosen"] + eos,
            "rejected": p["rejected"] + eos,
        }
        for p in pairs
    ]


def pair_length(p: Mapping[str, Any]) -> int:
    return len(p["user"]) + max(len(p["chosen"]), len(p["rejected"]))


def probe_pairs(pairs: list[dict[str, Any]], n: int, batch: int) -> list[dict[str, Any]]:
    """`n` pairs for the probe: the 2 * batch longest first, so the peak VRAM is near the worst case."""
    longest = sorted(pairs, key=lambda p: -pair_length(p))[: 2 * batch]
    ids = {p["id"] for p in longest}
    return longest + [p for p in pairs if p["id"] not in ids][: max(n - len(longest), 0)]


def loss_settings(sft_weight: float) -> tuple[list[str], list[float] | None]:
    if sft_weight > 0:
        return ["sigmoid", "sft"], [1.0, sft_weight]
    return ["sigmoid"], None


def check_pairs(pairs_meta: Mapping[str, Any], adapter_meta: Mapping[str, Any]) -> None:
    if pairs_meta["prompt_sha256"] != adapter_meta["prompt_sha256"]:
        raise SystemExit("the pairs were built on another student prompt than the adapter's: rebuild them")


def save_policy(model: Any, tokenizer: Any, path: Path, meta: Mapping[str, Any]) -> None:
    """The trained adapter only; the frozen "ref" copy stays out of the saved directory."""
    model.save_pretrained(path, selected_adapters=["default"])
    tokenizer.save_pretrained(path)
    (path / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_trained_adapter(model: Any, checkpoint: Path) -> None:
    """The trained adapter of a checkpoint back into "default". With a second adapter in the model the
    trainer saves "ref" in a subdirectory, and on resume transformers 5 loads only subdirectory adapters,
    so "default" would restart from the SFT weights (found in the CPU dry run)."""
    from peft import set_peft_model_state_dict  # type: ignore[import-not-found]
    from safetensors.torch import load_file  # type: ignore[import-not-found]

    loaded = set_peft_model_state_dict(model, load_file(str(checkpoint / "adapter_model.safetensors")), "default")
    if loaded.unexpected_keys:
        raise SystemExit(f"{checkpoint}: adapter keys the model lacks, e.g. {loaded.unexpected_keys[:2]}")


def forecast_from_probe(seconds: Sequence[float], batch: int, train_pairs: int, epochs: float) -> dict[str, float]:
    """The probe runs one micro-batch per step: time per pair from the steps after the first."""
    steps = list(seconds[1:])
    if not steps:
        return {}
    per_pair = statistics.median(steps) / batch
    return {"probe_seconds_per_pair": per_pair, "forecast_run_h": per_pair * train_pairs * epochs / 3600}


def load_policy(cfg: DpoConfig, out: Path, cpu: bool) -> tuple[Any, Any]:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    from radar.train.hf import adapter_base, bnb_config

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.sft_adapter))
    base = adapter_base(cfg.sft_adapter)
    if cpu:
        import torch  # type: ignore[import-not-found]
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        from radar.train.hf import tiny_base

        # the SFT adapter's shapes do not fit the tiny model: a fresh LoRA of the same rank stands in
        adapter_config = json.loads((cfg.sft_adapter / "adapter_config.json").read_text(encoding="utf-8"))
        tiny = tiny_base(base, out / "tiny-base", cfg.train.seed)
        model = AutoModelForCausalLM.from_pretrained(str(tiny), dtype=torch.float32)
        lora = LoraConfig(
            r=adapter_config["r"],
            lora_alpha=adapter_config["lora_alpha"],
            target_modules=adapter_config["target_modules"],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora), tokenizer

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb_config(), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return PeftModel.from_pretrained(model, str(cfg.sft_adapter), is_trainable=True), tokenizer


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.dpo", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in the run directory")
    ap.add_argument("--probe", type=int, metavar="N", help="one micro-batch per step over N pairs, forecast the run")
    ap.add_argument("--dry-run", action="store_true", help="CPU, tiny random model, 2 steps")
    ap.add_argument("--no-mlflow", action="store_true", help="track locally only")
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (dry run default 2)")
    args = ap.parse_args(argv)

    cfg = load_dpo(args.config)
    cpu = args.dry_run
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = cfg.output_dir
    if args.dry_run:
        out = ROOT / "models" / "dpo" / f"dryrun_{cfg.name}"
    elif args.probe:
        out = ROOT / "models" / "dpo" / f"probe_{cfg.name}"
    if (args.dry_run or args.probe) and not args.resume and out.exists():
        shutil.rmtree(out)
    if has_checkpoint(out) and not args.resume:
        raise SystemExit(f"{out} has checkpoints: pass --resume, or move the directory away to start over")
    if args.resume and not has_checkpoint(out):
        raise SystemExit(f"--resume: no checkpoint in {out}")
    out.mkdir(parents=True, exist_ok=True)

    adapter_meta = json.loads((cfg.sft_adapter / "dataset_meta.json").read_text(encoding="utf-8"))
    pairs_meta = json.loads((cfg.pairs_dir / "meta.json").read_text(encoding="utf-8"))
    check_pairs(pairs_meta, adapter_meta)
    train = read_jsonl(cfg.pairs_dir / "train.jsonl")
    val = read_jsonl(cfg.pairs_dir / "val.jsonl")
    n_train_pairs = len(train)
    t = cfg.train
    if args.dry_run:
        train, val = train[:DRY_RUN_PAIRS], val[: DRY_RUN_PAIRS // 2]
    elif args.probe:
        train, val = probe_pairs(train, args.probe, t.batch), []

    load_dotenv()
    uri = None if args.no_mlflow else optional("MLFLOW_TRACKING_URI")
    if uri is None and not args.no_mlflow:
        raise SystemExit("MLFLOW_TRACKING_URI is not set: start the ml profile and set it, or pass --no-mlflow")
    mode = "dryrun" if args.dry_run else "probe" if args.probe else "train"
    tracker = open_tracker(out, cfg.name, uri, args.resume, EXPERIMENT if mode == "train" else f"{EXPERIMENT}-{mode}")

    import torch

    limit = None
    if not cpu:
        free, total = torch.cuda.mem_get_info()
        limit = vram_limit_mib(total / 2**20, (total - free) / 2**20, cfg.guard.vram_margin_mib)
        log.info("VRAM: %.0f MiB card, %.0f MiB in use before loading", total / 2**20, (total - free) / 2**20)

    t_load = time.perf_counter()
    model, tokenizer = load_policy(cfg, out, cpu)
    load_s = time.perf_counter() - t_load

    from datasets import Dataset  # type: ignore[import-not-found]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found]

    loss_type, loss_weights = loss_settings(t.sft_weight)
    dpo_args = DPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=t.batch,
        per_device_eval_batch_size=1,
        # the probe takes one micro-batch per step to time a pair; peak memory does not depend on accumulation
        gradient_accumulation_steps=1 if args.probe else t.grad_accum,
        num_train_epochs=1 if args.probe else t.epochs,
        max_steps=args.max_steps or (DRY_RUN_STEPS if args.dry_run else -1),
        learning_rate=t.lr,
        warmup_steps=t.warmup_ratio,  # a float below 1 is a share of all steps in transformers 5
        lr_scheduler_type=t.scheduler,
        weight_decay=t.weight_decay,
        optim="adamw_torch" if cpu else t.optim,
        bf16=not cpu,
        use_cpu=cpu,
        max_length=t.max_seq_len,
        beta=t.beta,
        loss_type=loss_type,
        loss_weights=loss_weights,
        gradient_checkpointing=not cpu,
        logging_steps=1 if (args.dry_run or args.probe) else t.logging_steps,
        eval_strategy="steps" if val else "no",
        eval_steps=1 if args.dry_run else t.eval_steps,
        save_strategy="no" if args.probe else "steps",
        save_steps=1 if args.dry_run else t.save_steps,
        save_total_limit=t.save_total_limit,
        include_num_input_tokens_seen="non_padding",
        report_to="none",
        seed=t.seed,
        dataset_num_proc=1,
    )
    system = adapter_meta["system_prompt"]
    guard = StepGuard(limit, cfg.guard.step_time_factor, cfg.guard.reference_steps, cfg.guard.patience)
    watch = make_callback(guard, tracker, out, adapter_meta, tokenizer, cpu, save=save_policy)
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=Dataset.from_list(rows_for_trainer(train, tokenizer, system)),
        eval_dataset=Dataset.from_list(rows_for_trainer(val, tokenizer, system)) if val else None,
        processing_class=tokenizer,
        callbacks=[watch],
    )
    params = cfg.params() | {
        "train_pairs": len(train),
        "eval_pairs": len(val),
        "pairs_train_sha256": pairs_meta["train_sha256"],
        "pairs_val_sha256": pairs_meta["val_sha256"],
        "prompt_sha256": adapter_meta["prompt_sha256"],
        "loss_type": ",".join(loss_type),
        "commit": git_commit(),
        "mode": mode,
    }
    tracker.log_params(params)
    tracker.set_tags({"base": cfg.sft_adapter.name, "backend": "peft"} | versions("peft"))

    checkpoint = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint  # type: ignore[import-not-found]

        checkpoint = get_last_checkpoint(str(out))
        load_trained_adapter(model, Path(checkpoint))
        log.info("resuming from %s", checkpoint)

    if not cpu:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = "failed"
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        status = "stopped_by_guard" if watch.reason else "finished"
    finally:
        summary: dict[str, Any] = {
            "load_seconds": load_s,
            "train_seconds": time.perf_counter() - started,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch or 0.0,
            "step_seconds_median": statistics.median(watch.seconds[1:] or watch.seconds or [0.0]),
            "guard_peak_reserved_mib": guard.peak_reserved_mib,
        }
        if not cpu:
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            summary["vram_limit_mib"] = limit or 0.0
        if args.probe:
            summary |= forecast_from_probe(watch.seconds, t.batch, n_train_pairs, t.epochs)
        tracker.log_metrics(summary, trainer.state.global_step)
        tracker.set_tags({"guard_reason": watch.reason or ""})
        tracker.end(status)

    summary |= last_logged(trainer.state.log_history)
    if status == "finished" and not args.probe:
        save_policy(model, tokenizer, out / "adapter", adapter_meta)
    run = {
        "name": cfg.name,
        "config_file": args.config.as_posix(),
        "config_sha256": sha256_file(args.config),
        "status": status,
        "guard_reason": watch.reason,
        "params": params,
        "summary": summary,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    sessions_file = out / "run.json"
    previous = json.loads(sessions_file.read_text(encoding="utf-8"))["sessions"] if sessions_file.exists() else []
    sessions_file.write_text(json.dumps({"sessions": [*previous, run]}, indent=2) + "\n", encoding="utf-8")
    log.info("%s: %s, %s", cfg.name, status, json.dumps(summary))
    if status == "finished" and not args.dry_run:
        print(write_run_result(cfg, out, args.config, run, probe=bool(args.probe)))
    return GUARD_EXIT if watch.reason else 0


LOGGED = {
    "train_loss_last": "loss",
    "eval_loss_last": "eval_loss",
    "train_reward_accuracy_last": "rewards/accuracies",
    "eval_reward_accuracy_last": "eval_rewards/accuracies",
    "eval_reward_margin_last": "eval_rewards/margins",
    "eval_logps_chosen_last": "eval_logps/chosen",
    "eval_logps_rejected_last": "eval_logps/rejected",
}


def last_logged(history: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, key in LOGGED.items():
        values = [h[key] for h in history if key in h]
        out[name] = values[-1] if values else None
    return out


RESULT_KEYS = (
    "train_seconds",
    "global_step",
    "epoch",
    "step_seconds_median",
    "peak_reserved_mib",
    "peak_allocated_mib",
    "vram_limit_mib",
    *LOGGED,
    "probe_seconds_per_pair",
    "forecast_run_h",
)


def write_run_result(cfg: DpoConfig, out: Path, config: Path, run: Mapping[str, Any], probe: bool) -> Path:
    s = run["summary"]
    metrics = {k: {"value": round(s[k], 4)} for k in RESULT_KEYS if s.get(k) is not None}
    extra: dict[str, Any] = {}
    if probe:
        extra["forecast"] = (
            "forecast_run_h: the median time of a one-pair step after the first, times the training pairs and "
            "epochs; evaluation and checkpoints not included"
        )
    else:
        run_id_file = out / "mlflow_run_id.txt"
        extra["adapter_dir"] = (out / "adapter").relative_to(ROOT).as_posix()
        extra["mlflow_run_id"] = run_id_file.read_text(encoding="utf-8").strip() if run_id_file.exists() else None
        extra["note"] = "this session only; every session is in models/dpo/<name>/run.json"
    return write_result(
        f"dpo/{'probe' if probe else 'train'}_{cfg.name}.json",
        metrics,
        source_script="src/radar/train/dpo.py",
        config_file=config.as_posix(),
        config_sha256=run["config_sha256"],
        params=run["params"],
        **extra,
    )


if __name__ == "__main__":
    sys.exit(main())
