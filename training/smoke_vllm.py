"""vLLM smoke test: the OpenAI-compatible server fits in 8 GB, answers under a JSON schema, and how fast.

Runs inside WSL from the vLLM venv, for example:
    python training/smoke_vllm.py --model Qwen/Qwen3-1.7B --tag vllm_1p7b --gpu-util 0.85
The script starts `vllm serve` itself and stops it by PID when done.
"""

import argparse
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from smoke_data import build_examples, messages
from smoke_gpu import VramPoller, used_mib, write_part
from smoke_schema import is_valid, json_schema

ROOT = Path(__file__).resolve().parents[1]
LOG_PATTERNS = {
    "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "model_weights_gib": r"Model loading took ([\d.]+) GiB",
}


def wait_ready(base: str, proc: subprocess.Popen[bytes], timeout_s: float = 900) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {proc.returncode}, see its log")
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                return time.perf_counter() - start
        except httpx.TransportError:
            pass
        time.sleep(2)
    raise TimeoutError("vllm serve did not become ready")


def ask(base: str, model: str, text: str, schema: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages(text),
        "temperature": 0,
        "max_tokens": 600,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "vacancy", "schema": schema}}
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=600)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"] or ""
    return {
        "wall_s": round(wall, 3),
        "completion_tokens": data["usage"]["completion_tokens"],
        "valid": is_valid(content),
        "finish_reason": data["choices"][0]["finish_reason"],
    }


def summarize(answers: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    tokens = sum(a["completion_tokens"] for a in answers)
    return {
        "requests": len(answers),
        "valid_json": sum(a["valid"] for a in answers),
        "completion_tokens": tokens,
        "wall_s": round(wall, 2),
        "tok_per_s": round(tokens / wall, 1),
        "per_request_tok_per_s_median": round(
            statistics.median(a["completion_tokens"] / a["wall_s"] for a in answers), 1
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--quantization", default=None, help="e.g. bitsandbytes; AWQ checkpoints are detected by vLLM")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    cmd = [
        str(Path(sys.executable).parent / "vllm"),
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_util),
        "--max-model-len",
        str(args.max_model_len),
        "--served-model-name",
        "student",
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]

    baseline = used_mib()
    log_path = ROOT / "logs" / f"{args.tag}_server.log"
    with log_path.open("wb") as log:
        # FlashInfer's top-p/top-k sampler JIT-compiles with nvcc, which the distro does not have.
        env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            try:
                load_s = wait_ready(base, proc)
            except RuntimeError:
                errors = [ln for ln in log_path.read_text(errors="replace").splitlines() if "rror" in ln]
                write_part(
                    args.tag,
                    {
                        "model": args.model,
                        "failed": True,
                        "quantization": args.quantization,
                        "error": errors[-3:],
                        "server_log": log_path.name,
                    },
                )
                raise
            loaded = used_mib()
            schema = json_schema()
            texts = [ex["text"] for ex in build_examples(n=args.requests, seed=101)]
            ask(base, "student", texts[0], schema)  # warm-up: grammar compilation and first CUDA graphs
            with VramPoller() as poller:
                t0 = time.perf_counter()
                sequential = [ask(base, "student", t, schema) for t in texts]
                seq_wall = time.perf_counter() - t0
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=len(texts)) as pool:
                    concurrent = list(pool.map(lambda t: ask(base, "student", t, schema), texts))
                conc_wall = time.perf_counter() - t0
                t0 = time.perf_counter()
                free = [ask(base, "student", t, None) for t in texts[:5]]
                free_wall = time.perf_counter() - t0
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=120)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    from_log = {}
    for key, pattern in LOG_PATTERNS.items():
        found = re.findall(pattern, log_text)
        from_log[key] = float(found[-1].replace(",", "")) if found else None

    import vllm

    payload = {
        "model": args.model,
        "config": {
            "gpu_memory_utilization": args.gpu_util,
            "max_model_len": args.max_model_len,
            "quantization": args.quantization or "from checkpoint",
            "structured_output": "response_format json_schema",
        },
        "versions": {"vllm": vllm.__version__},
        "load_s": round(load_s, 1),
        "vram_baseline_mib": baseline,
        "vram_loaded_mib": loaded,
        "vram_peak_mib": poller.peak,
        "from_server_log": from_log,
        "sequential_schema": summarize(sequential, seq_wall),
        "concurrent_schema": summarize(concurrent, conc_wall),
        "sequential_no_schema": summarize(free, free_wall),
    }
    path = write_part(args.tag, payload)
    s, c = payload["sequential_schema"], payload["concurrent_schema"]
    print(
        f"{args.tag}: valid {s['valid_json']}/{s['requests']} seq, {c['valid_json']}/{c['requests']} conc; "
        f"{s['tok_per_s']} tok/s seq, {c['tok_per_s']} tok/s conc; smi peak {poller.peak} MiB -> {path}"
    )


if __name__ == "__main__":
    main()
