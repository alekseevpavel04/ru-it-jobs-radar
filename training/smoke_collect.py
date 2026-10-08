"""Collect the per-step smoke files from logs/gpu_smoke/ into results/env/gpu_smoke.json.

Run on Windows from the project venv after training/gpu_smoke.sh has produced the parts:
    .venv/Scripts/python training/smoke_collect.py
"""

import json
import subprocess
from typing import Any

from smoke_gpu import PARTS_DIR

from radar.results import write_result

QLORA = {
    "qlora_1p7b": "qlora_1p7b_unsloth",
    "qlora_1p7b_peft": "qlora_1p7b_peft",
    "qlora_4b": "qlora_4b_unsloth",
    "qlora_4b_peft": "qlora_4b_peft",
}
VLLM = {"vllm_1p7b": "vllm_1p7b", "vllm_4b_awq": "vllm_4b_awq", "vllm_4b_bnb": "vllm_4b_bnb"}
LLAMA = {"llama_qwen8b": "llama_qwen", "llama_yandex8b": "llama_yandex"}


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4), "n": n, "denominator": d}


def qlora_metrics(prefix: str, p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{prefix}_peak_vram_mib": {
            "value": max(p["train_peak_nvidia_smi_mib"], p["stress_peak_nvidia_smi_mib"]),
            "what": "nvidia-smi, whole card incl. desktop baseline, max of training and full-length stress",
        },
        f"{prefix}_torch_peak_alloc_mib": {
            "value": max(p["train_peak_alloc_mib"], p["stress_peak_alloc_mib"]),
            "what": "torch.cuda.max_memory_allocated, max of training and full-length stress",
        },
        f"{prefix}_sec_per_step": {"value": p["sec_per_step_median"], "what": "median over steps 2..N"},
        f"{prefix}_tokens_per_s": {"value": p["tokens_per_s"], "what": "prompt+completion tokens / train time"},
        f"{prefix}_loss_first": {"value": p["loss_first"]},
        f"{prefix}_loss_last3": {"value": p["loss_last3_mean"]},
        f"{prefix}_held_out_valid_json": share(p["held_out_valid_json"], p["held_out_n"]),
    }


def vllm_metrics(prefix: str, p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    s, c = p["sequential_schema"], p["concurrent_schema"]
    return {
        f"{prefix}_peak_vram_mib": {"value": p["vram_peak_mib"], "what": "nvidia-smi, whole card"},
        f"{prefix}_gpu_memory_utilization": {"value": p["config"]["gpu_memory_utilization"]},
        f"{prefix}_tokens_per_s": {"value": s["tok_per_s"], "what": "one request at a time, JSON schema"},
        f"{prefix}_tokens_per_s_concurrent": {
            "value": c["tok_per_s"],
            "what": f"{c['requests']} requests in parallel, JSON schema",
        },
        f"{prefix}_valid_json": share(s["valid_json"] + c["valid_json"], s["requests"] + c["requests"]),
        f"{prefix}_valid_json_no_schema": share(
            p["sequential_no_schema"]["valid_json"], p["sequential_no_schema"]["requests"]
        ),
    }


def llama_metrics(prefix: str, p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{prefix}_valid_json": share(p["valid_json"], p["requests"]),
        f"{prefix}_decode_tokens_per_s": {"value": p["decode_tok_per_s_median"], "what": "server timings, median"},
        f"{prefix}_end_to_end_tokens_per_s": {"value": p["end_to_end_tok_per_s"]},
        f"{prefix}_peak_vram_mib": {"value": p["vram_peak_mib"], "what": "nvidia-smi, whole card"},
    }


def gpu_info() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    name, driver, total = (x.strip() for x in out.split(","))
    return {"name": name, "driver": driver, "memory_total": total}


def main() -> None:
    parts = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(PARTS_DIR.glob("*.json"))}
    metrics: dict[str, dict[str, Any]] = {}
    failed: dict[str, Any] = {}
    missing: list[str] = []
    for group, build in ((QLORA, qlora_metrics), (VLLM, vllm_metrics), (LLAMA, llama_metrics)):
        for prefix, name in group.items():
            part = parts.get(name)
            if part is None:
                missing.append(name)
            elif part.get("failed"):
                failed[name] = part
            else:
                metrics.update(build(prefix, part))

    path = write_result(
        "env/gpu_smoke.json",
        metrics,
        source_script="training/smoke_collect.py",
        repro_command="bash training/gpu_smoke.sh",
        gpu=gpu_info(),
        failed=failed,
        missing=missing,
        parts=parts,
    )
    print(f"{len(metrics)} metrics, failed {sorted(failed)}, missing {missing} -> {path}")


if __name__ == "__main__":
    main()
