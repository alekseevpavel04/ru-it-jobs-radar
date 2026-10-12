"""Predictions of a quantized student behind an inference server, for `python -m radar.eval`.

    STUDENT_GGUF=models/gguf/qwen3_1p7b_r16_clean/qwen3_1p7b_r16_clean-Q4_K_M.gguf \
        LLAMA_PORT=8093 bash scripts/llm_server.sh student 4
    python -m radar.extract.student_server --name qwen3_1p7b_r16_clean --split dev \
        --system sft_1p7b_clean_q4_k_m --workers 4 --base-url http://127.0.0.1:8093
    ... --schema                       decoding constrained by configs/extraction_schema.json
    ... --workers 1 --out data/inference/bench/sft_1p7b_clean_q4_k_m_w1.jsonl      a speed run
    ... --backend vllm --base-url http://127.0.0.1:8095                           the AWQ checkpoint in vLLM

Port 8081 is taken by the Redpanda console of the compose stack. vLLM (radar.train.awq) runs in WSL:
    VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/vllm serve models/awq/<name> --host 127.0.0.1 \
        --port 8095 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 8 \
        --served-model-name <name>-awq
The FlashInfer sampler compiles itself at start and the distro has no nvcc; greedy decoding does
not need it.

Prompts. Each record's prompt is the text radar.train.quantize rendered with the adapter's tokenizer
(models/merged/<name>/prompts/<split>.jsonl): the student's system prompt, the chat template with
Qwen3 thinking off and the input cut to max_input_tokens, exactly as radar.train.predict builds it.
It goes to the server as text (llama-server /completion, vLLM /v1/completions), so no server-side
chat template touches the scores. For llama-server what a chat client would get is checked on every
record: /apply-template with the same messages and enable_thinking false must give the same string
(template_match in the sidecar). vLLM reads the adapter's own tokenizer and chat template, the ones
that rendered the prompt, so there is nothing to check there. The server's prompt token count must
equal the tokenizer's (prompt_token_mismatch).

Decoding is greedy (temperature 0), at most 512 new tokens, as in radar.train.predict. Without
--schema it is plain, as the student was trained; with --schema the server compiles the extraction
schema into a grammar, as for the teachers. Each answer is checked against the schema and an invalid
one counts as all fields empty in radar.eval.

Speed. `workers` requests are in flight at once; start llama-server with as many slots. records_per_s
is records over the wall time of the run, latency is per record as the client saw it, and the peak
memory of the card is sampled with nvidia-smi during the run (the whole card, other processes too;
vLLM reserves its gpu_memory_utilization share up front, so its peak says little about need).

Writes data/predictions/<split>/<system>.jsonl ({"id", "output", "valid", "finish", "truncated_input",
"prompt_tokens", "completion_tokens", "seconds", "template_match"}) and <system>.meta.json. Test splits
are not served.

Corpus extraction. `backend` is the factory radar.extract.corpus_run loads: each input is
rendered by radar.train.quantize.prompt_rows with the merged model's tokenizer and dataset_meta.json,
the function that rendered the dev prompts every served variant was scored on, and goes to the same
LlamaServer or Vllm client. It needs transformers, so the runner runs in the WSL training venv:
    PYTHONPATH=src HF_HUB_OFFLINE=1 ~/venvs/train/bin/python -m radar.extract.corpus_run run \
        --backend radar.extract.student_server:backend --backend-opt base_url=http://127.0.0.1:8095 --workers 8
Options: name (models/merged/<name>), server (vllm or llama), quant (awq, or the GGUF type llama-server
serves), schema (on or off), base_url. model_id is <name>-<quant>, with -noschema when decoding is
free: the AWQ checkpoint without the schema broke one tg_dev answer of 54 (E14), so it runs with it.
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import httpx

from radar.config import ROOT, data_dir
from radar.eval.cli import SYSTEM_NAME
from radar.extract.corpus_run import Answer
from radar.extract.schema import json_schema, try_parse
from radar.results import git_commit
from radar.train.predict import dataset_meta
from radar.train.quantize import prompt_rows

log = logging.getLogger("radar.extract.student_server")

SPLITS = ("dev", "tg_dev")
MAX_NEW_TOKENS = 512
MERGED = ROOT / "models" / "merged"
DEFAULT_URL = {"llama": "http://127.0.0.1:8080", "vllm": "http://127.0.0.1:8000"}
THINKING_OFF = {"enable_thinking": False}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shown(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def answer_row(
    row: Mapping[str, Any], text: str, finish: str, tokens: tuple[int, int], seconds: float, template: bool | None
) -> dict[str, Any]:
    output = text.strip()
    return {
        "id": row["id"],
        "output": output,
        "valid": try_parse(output) is not None,
        "finish": finish,
        "truncated_input": row["truncated_input"],
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "seconds": round(seconds, 3),
        "template_match": template,
    }


class Backend(Protocol):
    def info(self) -> dict[str, Any]: ...

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]: ...


class LlamaServer:
    """llama-server's native API: /props, /apply-template, /completion."""

    def __init__(self, client: httpx.Client, max_new_tokens: int, schema: Mapping[str, Any] | None) -> None:
        self.client, self.max_new_tokens, self.schema = client, max_new_tokens, schema

    def body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": self.max_new_tokens,
            "temperature": 0,
            "cache_prompt": True,
        }
        if self.schema is not None:
            body["json_schema"] = self.schema
        return body

    def info(self) -> dict[str, Any]:
        r = self.client.get("/props")
        r.raise_for_status()
        props = r.json()
        return {
            "model_path": props["model_path"],
            "model_alias": props.get("model_alias"),
            "model_ftype": props.get("model_ftype"),
            "build_info": props.get("build_info"),
            "total_slots": props["total_slots"],
            "n_ctx_per_slot": props["default_generation_settings"]["n_ctx"],
        }

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
        template = self.client.post(
            "/apply-template", json={"messages": messages, "chat_template_kwargs": THINKING_OFF}
        )
        template.raise_for_status()
        started = time.perf_counter()
        r = self.client.post("/completion", json=self.body(row["prompt"]))
        seconds = time.perf_counter() - started
        r.raise_for_status()
        data = r.json()
        # stop_type: eos and word end the answer, limit cuts it at n_predict
        finish = "length" if data["stop_type"] == "limit" else "stop"
        tokens = (data["tokens_evaluated"], data["tokens_predicted"])
        return answer_row(row, data["content"], finish, tokens, seconds, template.json()["prompt"] == row["prompt"])


class Vllm:
    """vLLM's OpenAI-compatible /v1/completions with a text prompt."""

    def __init__(self, client: httpx.Client, max_new_tokens: int, schema: Mapping[str, Any] | None) -> None:
        self.client, self.max_new_tokens, self.schema = client, max_new_tokens, schema
        self.model = ""

    def body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        if self.schema is not None:
            body["structured_outputs"] = {"json": self.schema}
        return body

    def info(self) -> dict[str, Any]:
        r = self.client.get("/v1/models")
        r.raise_for_status()
        model = r.json()["data"][0]
        self.model = model["id"]
        return {
            "model_path": model.get("root"),
            "model_alias": model["id"],
            "max_model_len": model.get("max_model_len"),
        }

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        r = self.client.post("/v1/completions", json=self.body(row["prompt"]))
        seconds = time.perf_counter() - started
        r.raise_for_status()
        data = r.json()
        choice, usage = data["choices"][0], data["usage"]
        finish = "length" if choice["finish_reason"] == "length" else "stop"
        tokens = (usage["prompt_tokens"], usage["completion_tokens"])
        return answer_row(row, choice["text"], finish, tokens, seconds, None)


BACKENDS: dict[str, Callable[[httpx.Client, int, Mapping[str, Any] | None], Backend]] = {
    "llama": LlamaServer,
    "vllm": Vllm,
}


def quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile."""
    xs = sorted(values)
    return xs[min(len(xs) - 1, max(0, -(-round(q * 100) * len(xs) // 100) - 1))]


def nvidia_smi_used() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.split()[0])


class PeakSampler:
    """The largest value `sample` returned while the context was open, polled every `every` seconds."""

    def __init__(self, sample: Callable[[], int | None], every: float = 0.5) -> None:
        self.sample, self.every = sample, every
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _take(self) -> None:
        value = self.sample()
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            self._take()

    def __enter__(self) -> "PeakSampler":
        self._take()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._take()


def local_path(served: str) -> Path:
    """A path as the server saw it; vLLM runs in WSL and reports /mnt/<drive>/..."""
    parts = served.split("/")
    if served.startswith("/mnt/") and len(parts) > 3 and len(parts[2]) == 1:
        return Path(f"{parts[2].upper()}:/", *parts[3:])
    return Path(served)


def weights_info(model_path: str | None) -> dict[str, Any]:
    """The served weights: a GGUF file with its type and hash from the manifest radar.train.quantize
    wrote next to it, or a checkpoint directory with the size of its safetensors."""
    if model_path is None:
        return {}
    path = local_path(model_path)
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        quant = config.get("quantization_config") or {}
        return {
            "dir": shown(path),
            "size_mib": round(sum(f.stat().st_size for f in files) / 2**20, 1),
            "quantization": quant.get("quant_method"),
        }
    info: dict[str, Any] = {"file": shown(path), "size_mib": round(path.stat().st_size / 2**20, 1)}
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            if Path(entry["file"]).name == path.name:
                info |= {"type": entry["type"], "sha256": entry["sha256"]}
    return info


def share(flags: Sequence[bool | None]) -> dict[str, int] | None:
    known = [f for f in flags if f is not None]
    return {"n": sum(known), "denominator": len(known)} if known else None


def summary(rows: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]], wall: float) -> dict[str, Any]:
    n = len(rows)
    by_id = {p["id"]: p for p in prompts}
    seconds = [r["seconds"] for r in rows]
    return {
        "records": n,
        "valid_json": {"n": sum(r["valid"] for r in rows), "denominator": n},
        "finish_length": sum(r["finish"] == "length" for r in rows),
        "truncated_inputs": sum(r["truncated_input"] for r in rows),
        "template_match": share([r["template_match"] for r in rows]),
        "prompt_token_mismatch": sum(r["prompt_tokens"] != by_id[r["id"]]["prompt_tokens"] for r in rows),
        "wall_seconds": round(wall, 1),
        "records_per_s": round(n / wall, 3),
        "completion_tokens_per_s": round(sum(r["completion_tokens"] for r in rows) / wall, 1),
        "latency_s": {
            "p50": round(quantile(seconds, 0.5), 3),
            "p95": round(quantile(seconds, 0.95), 3),
            "max": round(max(seconds), 3),
        },
    }


# --- corpus extraction --------------------------------------------------------------------------


class CorpusBackend:
    """The served student as a radar.extract.corpus_run backend, one input per request."""

    def __init__(self, server: Backend, tokenizer: Any, meta: Mapping[str, Any], model_id: str) -> None:
        self.server, self.tokenizer, self.meta = server, tokenizer, meta
        self.model_id = model_id
        self.prompt_sha256: str | None = meta["prompt_sha256"]
        # A fast tokenizer is not safe to call from the runner's worker threads at once; rendering
        # takes milliseconds against seconds of generation.
        self._render = threading.Lock()

    def answer(self, text: str) -> Answer:
        with self._render:
            (row,) = prompt_rows([{"id": "", "input": text}], self.tokenizer, self.meta)
        cut = row["truncated_input"]
        try:
            out = self.server.predict(row)
        except httpx.HTTPStatusError as e:
            return Answer(None, error=f"HTTP {e.response.status_code}: {e.response.text[:300]}", input_truncated=cut)
        except httpx.TimeoutException as e:
            return Answer(None, error=f"timeout: {e!r}", input_truncated=cut)
        # Another token count or template means the server reads another prompt than the scored one.
        if out["prompt_tokens"] != row["prompt_tokens"] or out["template_match"] is False:
            raise RuntimeError(
                f"the server read {out['prompt_tokens']} prompt tokens, the tokenizer rendered "
                f"{row['prompt_tokens']}; template match {out['template_match']}"
            )
        return Answer(out["output"], out["finish"], input_truncated=cut, seconds=out["seconds"])

    def extract(self, inputs: Sequence[str]) -> list[Answer]:
        return [self.answer(text) for text in inputs]


def served_weights(model_path: str) -> tuple[str, str]:
    """(directory, name) of a served checkpoint directory, (directory, stem) of a GGUF file."""
    path = PurePosixPath(model_path.replace("\\", "/"))
    return path.parent.name, path.stem if path.suffix == ".gguf" else path.name


def connect(
    name: str, tokenizer: Any, client: httpx.Client, server: str, quant: str, schema: bool, max_new_tokens: int
) -> CorpusBackend:
    """A corpus backend after the server confirmed it serves the weights `name` and `quant` stand for."""
    if server not in BACKENDS:
        raise SystemExit(f"server {server!r}: one of {', '.join(sorted(BACKENDS))}")
    client_backend = BACKENDS[server](client, max_new_tokens, json_schema() if schema else None)
    info = client_backend.info()
    # models/awq/<name> in vLLM, models/gguf/<name>/<name>-<type>.gguf in llama-server
    expected = (quant, name) if server == "vllm" else (name, f"{name}-{quant}")
    if served_weights(info["model_path"]) != expected:
        raise SystemExit(f"the server serves {info['model_path']}, not {name} {quant}")
    model_id = f"{name}-{quant.lower()}" + ("" if schema else "-noschema")
    return CorpusBackend(client_backend, tokenizer, dataset_meta(MERGED / name), model_id)


def backend(
    name: str = "qwen3_1p7b_r16_clean",
    server: str = "vllm",
    quant: str = "awq",
    schema: str = "on",
    base_url: str | None = None,
    max_new_tokens: str = str(MAX_NEW_TOKENS),
) -> CorpusBackend:
    """Factory for `python -m radar.extract.corpus_run run --backend radar.extract.student_server:backend`."""
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    if schema not in ("on", "off"):
        raise SystemExit(f"schema={schema!r}: on or off")
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED / name))
    client = httpx.Client(base_url=base_url or DEFAULT_URL[server], timeout=httpx.Timeout(600.0, connect=10.0))
    return connect(name, tokenizer, client, server, quant, schema == "on", int(max_new_tokens))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.extract.student_server", description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="merged model, models/merged/<name> (its prompts/ are read)")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--system", required=True, help="predictions file name, e.g. sft_1p7b_clean_q4_k_m")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="llama")
    ap.add_argument("--workers", type=int, default=4, help="requests in flight; llama-server needs as many slots")
    ap.add_argument("--schema", action="store_true", help="constrain decoding by the extraction JSON schema")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--base-url", help="server root, without /v1; default per backend")
    ap.add_argument("--limit", type=int, help="first N records only, for a probe (with --out)")
    ap.add_argument("--out", type=Path, help="default data/predictions/<split>/<system>.jsonl; set it for probes")
    args = ap.parse_args(argv)
    if not SYSTEM_NAME.match(args.system):
        raise SystemExit(f"system name {args.system!r}: lowercase letters, digits, _ and -")
    if args.limit and not args.out:
        raise SystemExit("--limit writes a partial file: give it --out, outside data/predictions")

    prompts_file = MERGED / args.name / "prompts" / f"{args.split}.jsonl"
    prompts = read_jsonl(prompts_file)[: args.limit]
    meta = json.loads((MERGED / args.name / "dataset_meta.json").read_text(encoding="utf-8"))
    client = httpx.Client(
        base_url=args.base_url or DEFAULT_URL[args.backend], timeout=httpx.Timeout(600.0, connect=10.0)
    )
    backend = BACKENDS[args.backend](client, args.max_new_tokens, json_schema() if args.schema else None)
    server = backend.info()
    if server.get("total_slots", args.workers) < args.workers:
        raise SystemExit(f"{args.workers} workers against {server['total_slots']} slots: start the server with more")

    with PeakSampler(nvidia_smi_used) as vram, ThreadPoolExecutor(max_workers=args.workers) as pool:
        started = time.perf_counter()
        rows = list(pool.map(backend.predict, prompts))
        wall = time.perf_counter() - started

    out = args.out or data_dir() / "predictions" / args.split / f"{args.system}.jsonl"
    write_jsonl(out, rows)
    stats = summary(rows, prompts, wall)
    sidecar = {
        "system": args.system,
        "split": args.split,
        "backend": args.backend,
        "server": server,
        "weights": weights_info(server["model_path"]),
        "prompts_file": shown(prompts_file),
        "prompts_sha256": hashlib.sha256(prompts_file.read_bytes()).hexdigest(),
        "prompt_sha256": meta["prompt_sha256"],
        "max_input_tokens": meta["max_input_tokens"],
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens, "json_schema_constrained": args.schema},
        "workers": args.workers,
        "limit": args.limit,
        **stats,
        "vram_peak_mib": vram.peak,
        "commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    log.info(
        "%s on %s: valid JSON %d of %d, %.2f records/s, latency p50 %.2f s -> %s",
        args.system,
        args.split,
        stats["valid_json"]["n"],
        len(rows),
        stats["records_per_s"],
        stats["latency_s"]["p50"],
        out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
