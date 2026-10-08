"""QLoRA SFT smoke test: does a Qwen3 student train in 8 GB at sequence length 2048, and how fast.

Runs inside WSL from the training venv, for example:
    python training/smoke_qlora.py --model Qwen/Qwen3-4B --backend unsloth --batch 2 --grad-accum 8

The 48 synthetic examples are seen exactly steps * batch * grad_accum / 48 times, so tokens per second is
counted from the known token lengths. After training, one extra forward and backward pass on a full batch of
max-length sequences gives the worst-case peak memory that the real corpus can hit.
"""

import argparse
import statistics
import time
from typing import Any

import torch
from smoke_data import build_examples, messages, target
from smoke_gpu import VramPoller, used_mib, write_part
from smoke_schema import is_valid

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_unsloth(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_len,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        use_exact_model_name=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0,
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tok


def load_peft(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora), AutoTokenizer.from_pretrained(args.model)


def render(tok: Any, text: str) -> str:
    prompt: str = tok.apply_chat_template(
        messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt


def max_length_row(tok: Any, example: dict[str, Any], max_len: int) -> dict[str, str]:
    """Repeat the vacancy text and cut it so that prompt plus completion lands just under max_len tokens."""
    completion = target(example["fields"]) + tok.eos_token
    overhead = len(tok(render(tok, "") + completion).input_ids)
    budget = max_len - overhead - 8
    ids = tok(example["text"]).input_ids
    ids = (ids * (budget // len(ids) + 1))[:budget]
    return {"prompt": render(tok, tok.decode(ids)), "completion": completion}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--backend", choices=["unsloth", "peft"], default="unsloth")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", required=True, help="part file name, e.g. qlora_4b_unsloth")
    args = ap.parse_args()

    baseline = used_mib()
    t_load = time.perf_counter()
    model, tok = load_unsloth(args) if args.backend == "unsloth" else load_peft(args)
    load_s = time.perf_counter() - t_load

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    examples = build_examples(n=48, seed=args.seed)
    rows = [{"prompt": render(tok, ex["text"]), "completion": target(ex["fields"]) + tok.eos_token} for ex in examples]
    lengths = [min(len(tok(r["prompt"] + r["completion"]).input_ids), args.max_len) for r in rows]
    passes = args.steps * args.batch * args.grad_accum / len(rows)

    step_ends: list[float] = []

    class StepTimer(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, *_: Any, **__: Any) -> None:
            torch.cuda.synchronize()
            step_ends.append(time.perf_counter())

    def run(train_rows: list[dict[str, str]], steps: int, grad_accum: int) -> tuple[Any, float, VramPoller]:
        config = SFTConfig(
            output_dir="/tmp/radar_smoke",
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=grad_accum,
            max_steps=steps,
            learning_rate=args.lr,
            warmup_steps=3,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            bf16=True,
            max_length=args.max_len,
            packing=False,
            completion_only_loss=True,
            gradient_checkpointing=args.backend == "peft",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            dataset_num_proc=1,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=Dataset.from_list(train_rows),
            processing_class=tok,
            callbacks=[StepTimer()],
        )
        step_ends.clear()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with VramPoller() as poll:
            trainer.train()
        return trainer, start, poll

    trainer, t_train, train_poll = run(rows, args.steps, args.grad_accum)
    train_s = time.perf_counter() - t_train
    train_peak_alloc = torch.cuda.max_memory_allocated() // 2**20
    train_peak_reserved = torch.cuda.max_memory_reserved() // 2**20
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    durations = [b - a for a, b in zip([t_train, *step_ends[:-1]], step_ends, strict=True)]
    del trainer

    # Worst case through the same trainer path: every sequence in the micro-batch sits at the length limit.
    stress_rows = [max_length_row(tok, examples[i], args.max_len) for i in range(args.batch)]
    stress_lengths = [len(tok(r["prompt"] + r["completion"]).input_ids) for r in stress_rows]
    _, t_stress, stress_poll = run(stress_rows, 3, 1)
    stress_durations = [b - a for a, b in zip([t_stress, *step_ends[:-1]], step_ends, strict=True)]
    stress_alloc = torch.cuda.max_memory_allocated() // 2**20
    stress_reserved = torch.cuda.max_memory_reserved() // 2**20
    torch.cuda.empty_cache()

    held_out = build_examples(n=3, seed=202)
    valid = 0
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    model.eval()
    for ex in held_out:
        enc = tok(render(tok, ex["text"]), return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=400, do_sample=False)
        valid += is_valid(tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True))

    import peft
    import transformers
    import trl

    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__}
    versions["peft"] = peft.__version__
    if args.backend == "unsloth":
        import unsloth

        versions["unsloth"] = unsloth.__version__

    payload = {
        "model": args.model,
        "backend": args.backend,
        "config": {
            "quant": "bnb nf4, double quant, bf16 compute",
            "lora_rank": args.rank,
            "lora_alpha": 2 * args.rank,
            "lora_targets": LORA_TARGETS,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "steps": args.steps,
            "max_len": args.max_len,
            "lr": args.lr,
            "optim": "adamw_8bit",
            "gradient_checkpointing": "unsloth" if args.backend == "unsloth" else "torch",
            "examples": len(rows),
            "passes_over_data": passes,
            "seq_len_min_median_max": [min(lengths), int(statistics.median(lengths)), max(lengths)],
        },
        "versions": versions,
        "load_s": round(load_s, 1),
        "train_s": round(train_s, 1),
        "sec_per_step_median": round(statistics.median(durations[1:]), 2),
        "first_step_s": round(durations[0], 2),
        "tokens_per_s": round(passes * sum(lengths) / train_s, 1),
        "loss_first": round(losses[0], 4),
        "loss_last3_mean": round(statistics.mean(losses[-3:]), 4),
        "vram_baseline_mib": baseline,
        "train_peak_alloc_mib": train_peak_alloc,
        "train_peak_reserved_mib": train_peak_reserved,
        "train_peak_nvidia_smi_mib": train_poll.peak,
        "stress_seq_lens": stress_lengths,
        "stress_sec_per_step_median": round(statistics.median(stress_durations), 2),
        "stress_peak_alloc_mib": stress_alloc,
        "stress_peak_reserved_mib": stress_reserved,
        "stress_peak_nvidia_smi_mib": stress_poll.peak,
        "vram_total_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "held_out_valid_json": valid,
        "held_out_n": len(held_out),
    }
    path = write_part(args.tag, payload)
    print(
        f"{args.tag}: {payload['sec_per_step_median']} s/step, {payload['tokens_per_s']} tok/s, "
        f"alloc peak {max(train_peak_alloc, stress_alloc)} MiB, smi peak {max(train_poll.peak, stress_poll.peak)} MiB, "
        f"loss {payload['loss_first']} -> {payload['loss_last3_mean']} -> {path}"
    )


if __name__ == "__main__":
    main()
