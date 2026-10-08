"""Teacher smoke test: llama-server answers under a JSON schema, how often the JSON is valid, how fast.

Run on Windows from the project venv: .venv/Scripts/python training/smoke_llama.py qwen --requests 10
The script starts scripts/llm_server.sh itself and stops the process tree by PID when done.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from smoke_data import build_examples, messages
from smoke_gpu import VramPoller, used_mib, write_part
from smoke_schema import is_valid, json_schema

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"qwen": "Qwen3-8B-Q4_K_M", "yandex": "YandexGPT-5-Lite-8B-instruct-Q4_K_M"}


def wait_ready(base: str, proc: subprocess.Popen[bytes], timeout_s: float = 300) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}, see its log")
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                return time.perf_counter() - start
        except httpx.TransportError:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become ready")


def stop_tree(proc: subprocess.Popen[bytes]) -> None:
    # bash execs llama-server.exe as a child Windows process; stop the whole tree by our PID.
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    proc.wait(timeout=30)


def ask(base: str, model: str, text: str, schema: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages(text),
        "temperature": 0,
        "max_tokens": 600,
        "response_format": {"type": "json_schema", "json_schema": {"name": "vacancy", "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=600)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    timings = data.get("timings", {})
    return {
        "wall_s": round(wall, 3),
        "completion_tokens": data["usage"]["completion_tokens"],
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "decode_tok_per_s": timings.get("predicted_per_second"),
        "prompt_tok_per_s": timings.get("prompt_per_second"),
        "valid": is_valid(content),
        "has_think_tag": "<think>" in content,
        "finish_reason": data["choices"][0]["finish_reason"],
        "content_head": content[:200],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(ALIASES))
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    baseline = used_mib()
    log_path = ROOT / "logs" / f"llama_server_{args.model}.log"
    env = {"LLAMA_PORT": str(args.port)}
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [
                shutil.which("bash") or "bash",
                (ROOT / "scripts" / "llm_server.sh").as_posix(),
                args.model,
                str(args.slots),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
        )
        try:
            load_s = wait_ready(base, proc)
            loaded = used_mib()
            schema = json_schema()
            texts = [ex["text"] for ex in build_examples(n=args.requests, seed=101)]
            with VramPoller() as poller:
                answers = [ask(base, ALIASES[args.model], t, schema) for t in texts]
        finally:
            stop_tree(proc)

    valid = sum(a["valid"] for a in answers)
    decode = [a["decode_tok_per_s"] for a in answers if a["decode_tok_per_s"]]
    total_tokens = sum(a["completion_tokens"] for a in answers)
    total_wall = sum(a["wall_s"] for a in answers)
    payload = {
        "model": ALIASES[args.model],
        "server_args": {"slots": args.slots, "ctx_per_slot": 4096, "ngl": 99, "flash_attn": "on", "kv_cache": "q8_0"},
        "requests": len(answers),
        "valid_json": valid,
        "think_tags": sum(a["has_think_tag"] for a in answers),
        "load_s": round(load_s, 1),
        "vram_baseline_mib": baseline,
        "vram_loaded_mib": loaded,
        "vram_peak_mib": poller.peak,
        "decode_tok_per_s_median": round(statistics.median(decode), 1) if decode else None,
        "end_to_end_tok_per_s": round(total_tokens / total_wall, 1),
        "answers": answers,
    }
    path = write_part(f"llama_{args.model}", payload)
    print(f"{args.model}: valid {valid}/{len(answers)}, decode {payload['decode_tok_per_s_median']} tok/s -> {path}")


if __name__ == "__main__":
    main()
