"""QLoRA SFT of a Qwen3 student on the silver dataset, in the WSL training venv.

Reads data/sft/{train,val}.jsonl (radar.train.dataset) and a run config, configs/sft/<run>.toml.
The loss covers the answer only (prompt-completion rows, completion_only_loss). Writes to
models/sft/<name>/: trainer checkpoints (resumable), epoch-<k>/ adapters, adapter/ at the end with
the tokenizer and dataset_meta.json (prompt and input cap the predictor must reuse), run.json, and the
local copy of the tracking. Params, losses, step time, tokens per second and VRAM go to MLflow
(MLFLOW_TRACKING_URI, compose profile ml), experiment radar-sft; probes and dry runs to
radar-sft-probe and radar-sft-dryrun. The run summary also goes to results/sft/train_<name>.json, a
probe's to results/sft/probe_<name>.json with the forecast of the full run.

The guard (radar.train.guard) stops the run, saving a checkpoint, when torch's reserved
memory passes the card minus a margin or the time per token grows many times over the first steps;
the exit code is then 3 and --resume continues from that checkpoint.

    bash scripts/train_run.sh sft --config configs/sft/qwen3_1p7b.toml --probe 50   # speed, VRAM, forecast
    bash scripts/train_run.sh sft --config configs/sft/qwen3_1p7b.toml
    bash scripts/train_run.sh sft --config configs/sft/qwen3_1p7b.toml --resume
    bash scripts/train_run.sh sft --config configs/sft/qwen3_1p7b.toml --backend peft
    bash scripts/train_run.sh sft --config configs/sft/qwen3_1p7b.toml --dry-run --no-mlflow   # CPU, tiny model
"""

import argparse
import dataclasses
import json
import logging
import os
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar.config import ROOT, load_dotenv, optional
from radar.results import git_commit, write_result
from radar.train.config import BACKENDS, RunConfig, load_run
from radar.train.dataset import read_jsonl, render_prompt, sha256_file
from radar.train.guard import StepGuard, vram_limit_mib
from radar.train.tracking import EXPERIMENT, Tracker, open_tracker

log = logging.getLogger("radar.train.sft")

GUARD_EXIT = 3
DRY_RUN_STEPS = 2
DRY_RUN_EXAMPLES = 8


def rows_for_trainer(examples: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    """Prompt-completion rows: the chat prompt up to the answer, and the answer with end of turn."""
    rows = []
    for ex in examples:
        system, user, answer = (m["content"] for m in ex["messages"])
        rows.append({"prompt": render_prompt(tokenizer, system, user), "completion": answer + tokenizer.eos_token})
    return rows


def check_dataset(examples: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], cfg: RunConfig) -> None:
    if meta["max_seq_len"] != cfg.train.max_seq_len:
        raise SystemExit(
            f"dataset was built for max_seq_len {meta['max_seq_len']}, the run config says {cfg.train.max_seq_len}"
        )
    prompts = {ex["messages"][0]["content"] for ex in examples}
    if prompts != {meta["system_prompt"]}:
        raise SystemExit("the examples' system prompt differs from data/sft/meta.json: rebuild the dataset")


def probe_rows(examples: list[dict[str, Any]], n: int, batch: int) -> list[dict[str, Any]]:
    """`n` examples for the probe: the 2 * batch longest, so the peak VRAM is near the worst case,
    and the rest from the start of the (already shuffled by group hash) training file."""
    longest = sorted(examples, key=lambda e: -(e["prompt_tokens"] + e["completion_tokens"]))[: 2 * batch]
    ids = {e["id"] for e in longest}
    return longest + [e for e in examples if e["id"] not in ids][: max(n - len(longest), 0)]


def has_checkpoint(out: Path) -> bool:
    return any(out.glob("checkpoint-*"))


def save_adapter(model: Any, tokenizer: Any, path: Path, meta: Mapping[str, Any]) -> None:
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_callback(
    guard: StepGuard,
    tracker: Tracker,
    out: Path,
    meta: Mapping[str, Any],
    tokenizer: Any,
    cpu: bool,
    save: Callable[[Any, Any, Path, Mapping[str, Any]], None] = save_adapter,
) -> Any:
    import torch  # type: ignore[import-not-found]
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    class Watch(TrainerCallback):  # type: ignore[misc]
        """Times each optimizer step, feeds the guard, mirrors trainer logs to the tracker and keeps
        an adapter per epoch, so an earlier epoch can be scored on dev without retraining."""

        def __init__(self) -> None:
            self.started = 0.0
            self.tokens_before = 0
            self.reason: str | None = None
            self.seconds: list[float] = []
            self.tokens: list[int] = []
            self.last: dict[str, float] = {}

        def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            self.tokens_before = state.num_input_tokens_seen

        def on_step_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            self.started = time.perf_counter()

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            if not cpu:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - self.started
            # tokens counted between step ends, wherever the trainer counts them within a step
            tokens = int(state.num_input_tokens_seen - self.tokens_before)
            self.tokens_before = state.num_input_tokens_seen
            reserved = 0.0 if cpu else torch.cuda.memory_reserved() / 2**20
            self.seconds.append(seconds)
            self.tokens.append(tokens)
            self.last = {"step_seconds": seconds, "vram_reserved_mib": reserved}
            if tokens > 0:
                self.last["step_tokens_per_s"] = tokens / seconds
            reason = guard.check(seconds, tokens, reserved)
            if reason:
                log.error("guard: %s; saving a checkpoint and stopping", reason)
                self.reason = reason
                control.should_save = True
                control.should_training_stop = True

        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> None:
            logs = dict(logs or {})
            tracker.log_metrics(logs | self.last if "loss" in logs else logs, state.global_step)

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.reason is None:
                save(kwargs["model"], tokenizer, out / f"epoch-{round(state.epoch)}", meta)

    return Watch()


def versions(backend: str) -> dict[str, str]:
    import peft  # type: ignore[import-not-found]
    import torch
    import transformers
    import trl  # type: ignore[import-not-found]

    out = {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__}
    out["peft"] = peft.__version__
    if backend == "unsloth":
        import unsloth  # type: ignore[import-not-found]

        out["unsloth"] = unsloth.__version__
    return out


def forecast(watch: Any, train_tokens: int, epochs: float) -> dict[str, float]:
    """Time of the full run from the probe's speed, steps after the first (warm-up) only."""
    pairs = [(s, t) for s, t in zip(watch.seconds[1:], watch.tokens[1:], strict=True) if t > 0]
    if not pairs:
        return {}
    tokens_per_s = sum(t for _, t in pairs) / sum(s for s, _ in pairs)
    epoch_h = train_tokens / tokens_per_s / 3600
    return {"probe_tokens_per_s": tokens_per_s, "forecast_epoch_h": epoch_h, "forecast_run_h": epoch_h * epochs}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.sft", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in the run directory")
    ap.add_argument("--backend", choices=BACKENDS, help="override the config (peft is the fallback)")
    ap.add_argument("--probe", type=int, metavar="N", help="train one pass over N examples, forecast the full run")
    ap.add_argument("--dry-run", action="store_true", help="CPU, tiny random model, 2 steps")
    ap.add_argument("--no-mlflow", action="store_true", help="track locally only")
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (dry run default 2)")
    args = ap.parse_args(argv)

    cfg = load_run(args.config)
    if args.backend:
        cfg = dataclasses.replace(cfg, backend=args.backend)
    cpu = args.dry_run
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the dry run must not touch the card, even to probe it
    if args.dry_run:
        cfg = dataclasses.replace(cfg, output_dir=ROOT / "models" / "sft" / f"dryrun_{cfg.name}")
    elif args.probe:
        cfg = dataclasses.replace(cfg, output_dir=ROOT / "models" / "sft" / f"probe_{cfg.name}")
    if (args.dry_run or args.probe) and not args.resume and cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    out = cfg.output_dir
    if has_checkpoint(out) and not args.resume:
        raise SystemExit(f"{out} has checkpoints: pass --resume, or move the directory away to start over")
    if args.resume and not has_checkpoint(out):
        raise SystemExit(f"--resume: no checkpoint in {out}")

    meta = json.loads(cfg.dataset_meta.read_text(encoding="utf-8"))
    train = read_jsonl(cfg.train_file)
    val = read_jsonl(cfg.val_file)
    check_dataset(train + val, meta, cfg)
    train_tokens = sum(e["prompt_tokens"] + e["completion_tokens"] for e in train)
    if args.dry_run:
        train, val = train[:DRY_RUN_EXAMPLES], val[: DRY_RUN_EXAMPLES // 2]
    elif args.probe:
        train, val = probe_rows(train, args.probe, cfg.train.batch), []

    load_dotenv()
    uri = None if args.no_mlflow else optional("MLFLOW_TRACKING_URI")
    if uri is None and not args.no_mlflow:
        raise SystemExit("MLFLOW_TRACKING_URI is not set: start the ml profile and set it, or pass --no-mlflow")
    mode = "dryrun" if args.dry_run else "probe" if args.probe else "train"
    # dry runs and probes go to their own experiments, so the training runs stay comparable
    experiment = EXPERIMENT if mode == "train" else f"{EXPERIMENT}-{mode}"
    tracker = open_tracker(out, cfg.name, uri, args.resume, experiment)

    import torch

    limit = None
    if not cpu:
        free, total = torch.cuda.mem_get_info()
        limit = vram_limit_mib(total / 2**20, (total - free) / 2**20, cfg.guard.vram_margin_mib)
        log.info(
            "VRAM: %.0f MiB card, %.0f MiB in use before loading, torch may reserve %.0f MiB",
            total / 2**20,
            (total - free) / 2**20,
            limit,
        )

    model_path = cfg.model
    if args.dry_run:
        from radar.train.hf import tiny_base

        model_path = str(tiny_base(cfg.model, out / "tiny-base", cfg.train.seed))
    from radar.train.hf import load_for_training

    t_load = time.perf_counter()
    model, tokenizer = load_for_training(cfg, model_path, cpu)
    load_s = time.perf_counter() - t_load

    # after the model: Unsloth patches transformers and trl when it is imported first
    from datasets import Dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer

    t = cfg.train
    sft_args = SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=t.batch,
        # eval keeps full-vocabulary logits (about 152k per token): at the train micro-batch of 4 the
        # 1.7B run reserved 8 786 MiB there, past the card, while training steps stayed at 3 344 MiB
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t.grad_accum,
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
        packing=False,
        completion_only_loss=True,
        gradient_checkpointing=cfg.backend == "peft" and not cpu,
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
    guard = StepGuard(limit, cfg.guard.step_time_factor, cfg.guard.reference_steps, cfg.guard.patience)
    watch = make_callback(guard, tracker, out, meta, tokenizer, cpu)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(rows_for_trainer(train, tokenizer)),
        eval_dataset=Dataset.from_list(rows_for_trainer(val, tokenizer)) if val else None,
        processing_class=tokenizer,
        callbacks=[watch],
    )
    params = cfg.params() | {
        "train_examples": len(train),
        "val_examples": len(val),
        "dataset_train_sha256": meta["train_sha256"],
        "dataset_val_sha256": meta["val_sha256"],
        "prompt_sha256": meta["prompt_sha256"],
        "max_input_tokens": meta["max_input_tokens"],
        "commit": git_commit(),
        "mode": mode,
    }
    tracker.log_params(params)
    tracker.set_tags({"model": cfg.model, "backend": cfg.backend} | versions(cfg.backend if not cpu else "peft"))

    if not cpu:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = "failed"
    try:
        trainer.train(resume_from_checkpoint=True if args.resume else None)
        status = "stopped_by_guard" if watch.reason else "finished"
    finally:
        train_s = time.perf_counter() - started
        summary: dict[str, Any] = {
            "load_seconds": load_s,
            "train_seconds": train_s,
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
            summary |= forecast(watch, train_tokens, t.epochs)
        tracker.log_metrics(summary, trainer.state.global_step)
        tracker.set_tags({"guard_reason": watch.reason or ""})
        tracker.end(status)

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    evals = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    summary |= {"train_loss_last": losses[-1] if losses else None, "eval_loss_last": evals[-1] if evals else None}
    if status == "finished" and not args.probe:
        save_adapter(model, tokenizer, out / "adapter", meta)
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
        print(write_run_result(cfg, args.config, run, meta, probe=bool(args.probe)))
    if watch.reason:
        return GUARD_EXIT
    return 0


RESULT_KEYS = (
    "train_seconds",
    "global_step",
    "epoch",
    "step_seconds_median",
    "peak_reserved_mib",
    "peak_allocated_mib",
    "vram_limit_mib",
    "train_loss_last",
    "eval_loss_last",
    "probe_tokens_per_s",
    "forecast_epoch_h",
    "forecast_run_h",
)


def write_run_result(
    cfg: RunConfig, config: Path, run: Mapping[str, Any], meta: Mapping[str, Any], probe: bool
) -> Path:
    """results/sft/train_<name>.json for a training run, results/sft/probe_<name>.json for a probe."""
    s = run["summary"]
    metrics = {k: {"value": round(s[k], 4)} for k in RESULT_KEYS if s.get(k) is not None}
    extra: dict[str, Any] = {}
    if probe:
        extra["forecast"] = (
            "forecast_*_h: the full training file's prompt and answer tokens over the probe's tokens per "
            "second (steps after the first), times epochs; evaluation and checkpoint time not included"
        )
    else:
        run_id_file = cfg.output_dir / "mlflow_run_id.txt"
        extra["adapter_dir"] = (cfg.output_dir / "adapter").relative_to(ROOT).as_posix()
        extra["mlflow_run_id"] = run_id_file.read_text(encoding="utf-8").strip() if run_id_file.exists() else None
        extra["note"] = (
            "this session only: a resumed run reports the time and peaks of its last session; "
            "every session is in models/sft/<name>/run.json"
        )
    return write_result(
        f"sft/{'probe' if probe else 'train'}_{cfg.name}.json",
        metrics,
        source_script="src/radar/train/sft.py",
        config_file=config.as_posix(),
        config_sha256=run["config_sha256"],
        params=run["params"],
        dataset_meta={k: meta[k] for k in ("train_sha256", "val_sha256", "prompt_sha256", "max_input_tokens")},
        **extra,
    )


if __name__ == "__main__":
    sys.exit(main())
