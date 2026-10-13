"""Final test of the extractors: every system on test and tg_test, scored once in one pass.

    python -m radar.eval.final --status                  # what is predicted and what is left, no gold read
    python -m radar.eval.final --final-test              # predict what is missing, then score everything
    python -m radar.eval.final --final-test --with-dpo   # with the DPO student, if E15 kept it

Prediction. Each system answers the inputs its dev run answered, with the dev run's settings, read
from the dev sidecar (data/predictions/dev/<system>.meta.json, the teacher's data/teacher/<t>/dev.meta.json):
batch and backend of radar.train.predict, workers and schema of radar.extract.student_server, prompt
version and workers of radar.extract.teacher. A system without a dev run is refused. Test inputs
are data/gold/test_inputs.jsonl (440) and tg_test_inputs.jsonl (140, the gold set keeps 130 after
its exclusions).

    regex                    radar.extract.baseline, in this process
    base_*, sft_*, dpo_*     radar.train.predict in the WSL training venv (scripts/train_run.sh)
    sft_1p7b_clean_<gguf>    llama-server (scripts/llm_server.sh student) on the merged student's GGUF,
                             prompts rendered in WSL by `radar.train.quantize prompts`
    sft_1p7b_clean_awq_schema  vLLM in WSL on the AWQ checkpoint, decoding under the schema, as the
                             corpus is labeled (E14)
    teacher_qwen, _yandex    llama-server (scripts/llm_server.sh qwen|yandex) and radar.extract.teacher,
                             with the prompt file of the dev run taken from git by its sha256

Resuming. A system and split is done when its predictions file holds exactly the input ids and its
sidecar is written; a done one is skipped, so after a stop the same command goes on. Servers this
command starts are stopped by PID when it ends (Ctrl+C included). A GPU step starts only on a card
with at most GPU_FREE_MIB in use: the corpus runner and its vLLM must be stopped first.

Scoring runs once every system is done on both splits: `python -m radar.eval` with --final-test for
each system and each comparison (results/extract/<split>_<system>.json, <split>_<a>_vs_<b>.json, paired
bootstrap on the same gold rows), then results/extract/final_test.json collects macro F1, valid JSON
and the field F1 with intervals from those files. Every prediction step and every scoring call is a
line in results/test_runs.log. A second scoring needs --rescore with the reason, which goes to the log
and must go to docs/EXPERIMENTS.md.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from radar import config, results
from radar.config import ROOT, data_dir
from radar.eval import cli
from radar.eval.cli import TEST_SPLITS

log = logging.getLogger("radar.eval.final")

STUDENT = "sft_1p7b_r16_clean"
MERGED_NAME = "qwen3_1p7b_r16_clean"
DEV_OF = {"test": "dev", "tg_test": "tg_dev"}
GPU_FREE_MIB = 1500
WSL_DISTRO = os.environ.get("RADAR_WSL_DISTRO", "radar")
PORTS = {"llama": 8093, "vllm": 8095, "teacher": 8096}  # 8090 is the Airflow API of the stack
TEACHER_CTX_PER_SLOT = "8192"  # the teacher prompt is about 6 500 tokens (scripts/llm_server.sh)
PROMPT_FILE = "configs/teacher_prompt.md"

FIELD_METRICS = (
    "macro_f1",
    "valid_json",
    "role_f1",
    "role_macro_f1",
    "grade_f1",
    "skills_f1",
    "salary_f1",
    "work_format_f1",
    "cities_f1",
    "experience_years_f1",
    "employment_type_f1",
)


@dataclass(frozen=True)
class System:
    name: str
    kind: str
    opts: Mapping[str, str] = field(default_factory=dict)
    gpu: bool = True


SYSTEMS: tuple[System, ...] = (
    System("regex", "regex", gpu=False),
    System(STUDENT, "hf", {"adapter": "models/sft/qwen3_1p7b_r16_clean/adapter"}),
    System("sft_4b_r16", "hf", {"adapter": "models/sft/qwen3_4b_r16/adapter"}),
    System("base_1p7b", "hf", {"model": "Qwen/Qwen3-1.7B"}),
    System("base_4b", "hf", {"model": "Qwen/Qwen3-4B"}),
    System("sft_1p7b_clean_f16", "llama", {"quant": "F16"}),
    System("sft_1p7b_clean_q8_0", "llama", {"quant": "Q8_0"}),
    System("sft_1p7b_clean_q4_k_m", "llama", {"quant": "Q4_K_M"}),
    System("sft_1p7b_clean_awq_schema", "vllm"),
    System("teacher_qwen", "teacher", {"teacher": "qwen"}),
    System("teacher_yandex", "teacher", {"teacher": "yandex"}),
)
DPO = System("dpo_1p7b_clean", "hf", {"adapter": "models/dpo/qwen3_1p7b_r16_clean_dpo/adapter"})

# a - b on the same gold rows
COMPARISONS: tuple[tuple[str, str], ...] = (
    (STUDENT, "regex"),
    (STUDENT, "teacher_qwen"),
    (STUDENT, "teacher_yandex"),
    (STUDENT, "base_1p7b"),
    (STUDENT, "sft_4b_r16"),
    ("sft_4b_r16", "base_4b"),
    ("sft_1p7b_clean_f16", STUDENT),
    ("sft_1p7b_clean_q8_0", STUDENT),
    ("sft_1p7b_clean_q4_k_m", STUDENT),
    ("sft_1p7b_clean_awq_schema", STUDENT),
    ("sft_1p7b_clean_q8_0", "sft_1p7b_clean_f16"),
    ("sft_1p7b_clean_q4_k_m", "sft_1p7b_clean_f16"),
    ("sft_1p7b_clean_awq_schema", "sft_1p7b_clean_f16"),
    ("teacher_qwen", "regex"),
    ("teacher_yandex", "regex"),
    ("teacher_qwen", "teacher_yandex"),
    (DPO.name, STUDENT),
)


def systems(with_dpo: bool) -> tuple[System, ...]:
    return (*SYSTEMS, DPO) if with_dpo else SYSTEMS


def comparisons(names: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in COMPARISONS if a in names and b in names]


# --- files --------------------------------------------------------------------------------------


def inputs_path(split: str) -> Path:
    return data_dir() / "gold" / f"{split}_inputs.jsonl"


def prediction_path(split: str, system: str) -> Path:
    return data_dir() / "predictions" / split / f"{system}.jsonl"


def sidecar_path(split: str, system: str) -> Path:
    return prediction_path(split, system).with_suffix(".meta.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def input_ids(split: str) -> list[str]:
    return [r["id"] for r in read_jsonl(inputs_path(split))]


def is_complete(split: str, system: str) -> bool:
    path, sidecar = prediction_path(split, system), sidecar_path(split, system)
    if not (path.exists() and sidecar.exists()):
        return False
    ids = [r["id"] for r in read_jsonl(path)]
    return len(ids) == len(set(ids)) and set(ids) == set(input_ids(split))


def todo(system: System) -> list[str]:
    return [s for s in TEST_SPLITS if not is_complete(s, system.name)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log_run(system: str, what: str) -> None:
    """A line in results/test_runs.log, in the columns radar.eval writes: time, system, what, commit, command."""
    path = config.RESULTS / "test_runs.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now()}\t{system}\t{what}\t{results.git_commit()}\t{results.invocation()}\n")


def dev_sidecar(system: str, split: str) -> dict[str, Any]:
    path = sidecar_path(DEV_OF[split], system)
    if not path.exists():
        raise SystemExit(f"{system} has no {DEV_OF[split]} run ({path}): a system goes to test only after dev")
    return read_json(path)


# --- processes ----------------------------------------------------------------------------------


def git_bash() -> str:
    """Git for Windows' bash: a bare `bash` on Windows may resolve to WSL's."""
    if os.name != "nt":
        return shutil.which("bash") or "bash"
    git = shutil.which("git")
    if git:
        for parent in Path(git).resolve().parents:
            if (parent / "bin" / "bash.exe").exists():
                return str(parent / "bin" / "bash.exe")
    raise SystemExit("Git Bash not found next to git: the servers start through scripts/llm_server.sh")


def stop_tree(proc: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    """Terminate a process and its children by PID, then kill what is left after `timeout`."""
    if proc.poll() is not None:
        return
    parent = psutil.Process(proc.pid)
    procs = [*parent.children(recursive=True), parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        p.kill()
    proc.wait(timeout=timeout)


def wait_ready(url: str, proc: subprocess.Popen[bytes], timeout: float, log_file: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the server exited with {proc.returncode} before it was ready, see {log_file}")
        try:
            if httpx.get(url, timeout=5.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(3)
    raise SystemExit(f"{url} not ready after {timeout:.0f} s, see {log_file}")


def port_free(port: int) -> bool:
    try:
        httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0)
    except httpx.TransportError:
        return True
    return False


def to_wsl(path: Path) -> str:
    drive, rest = path.resolve().drive, path.resolve().as_posix().split(":", 1)[-1]
    return f"/mnt/{drive[0].lower()}{rest}" if drive else path.as_posix()


@dataclass
class Context:
    """What touches the machine: subprocesses, the card, the clock. Tests replace it."""

    logs: Path = ROOT / "logs" / "final_test"
    distro: str = WSL_DISTRO

    def log_file(self, name: str) -> Path:
        self.logs.mkdir(parents=True, exist_ok=True)
        return self.logs / f"{name}.log"

    def wsl(self, args: Sequence[str], log_name: str) -> None:
        """A command inside the distro from the repo root, output appended to logs/final_test/<log_name>.log."""
        cmd = ["wsl.exe", "-d", self.distro, "--cd", to_wsl(ROOT), "--exec", *args]
        log_file = self.log_file(log_name)
        log.info("WSL: %s (log %s)", " ".join(args), log_file)
        with log_file.open("ab") as out:
            done = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=False)
        if done.returncode != 0:
            raise SystemExit(f"{' '.join(args)} failed with {done.returncode}, see {log_file}")

    def gpu_used_mib(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.split()[0])

    def require_free_gpu(self, wait_s: float = 90.0) -> None:
        """The card is free when at most GPU_FREE_MIB is in use; a server just stopped gets `wait_s`."""
        deadline = time.monotonic() + wait_s
        used = self.gpu_used_mib()
        while used > GPU_FREE_MIB and time.monotonic() < deadline:
            time.sleep(5)
            used = self.gpu_used_mib()
        if used > GPU_FREE_MIB:
            raise SystemExit(
                f"{used} MiB in use on the card (free means at most {GPU_FREE_MIB}): stop the corpus runner and "
                "vLLM first, then run the same command again"
            )

    @contextmanager
    def llama_server(self, kind: str, slots: int, env: Mapping[str, str], port: int, name: str) -> Iterator[str]:
        if not port_free(port):
            raise SystemExit(f"port {port} is taken: stop what listens there")
        log_file = self.log_file(f"{name}_server")
        with log_file.open("ab") as out:
            proc = subprocess.Popen(
                [git_bash(), "scripts/llm_server.sh", kind, str(slots)],
                cwd=ROOT,
                env={**os.environ, **env, "LLAMA_PORT": str(port)},
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        log.info("llama-server %s, PID %d, port %d (log %s)", kind, proc.pid, port, log_file)
        try:
            wait_ready(f"http://127.0.0.1:{port}/health", proc, 600, log_file)
            yield f"http://127.0.0.1:{port}"
        finally:
            stop_tree(proc)
            log.info("llama-server PID %d stopped", proc.pid)

    @contextmanager
    def vllm_server(self, model_dir: Path, served_name: str, port: int, name: str) -> Iterator[str]:
        """vLLM in the distro with the settings of the dev run (E14), stopped by its PID inside WSL."""
        if not port_free(port):
            raise SystemExit(f"port {port} is taken: stop what listens there")
        log_file = self.log_file(f"{name}_server")
        pid_file = self.logs / f"{name}_vllm.pid"
        pid_file.unlink(missing_ok=True)
        script = (
            f"echo $$ > {to_wsl(pid_file)}; exec env VLLM_USE_FLASHINFER_SAMPLER=0 "
            "HF_HOME=/mnt/d/VScode_projects/assets/huggingface HF_HUB_OFFLINE=1 "
            f"$HOME/venvs/vllm/bin/vllm serve {to_wsl(model_dir)} --host 127.0.0.1 --port {port} "
            "--gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 8 "
            f"--served-model-name {served_name}"
        )
        with log_file.open("ab") as out:
            proc = subprocess.Popen(
                ["wsl.exe", "-d", self.distro, "--cd", to_wsl(ROOT), "--exec", "bash", "-c", script],
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        log.info("vLLM %s, port %d (log %s)", served_name, port, log_file)
        try:
            wait_ready(f"http://127.0.0.1:{port}/v1/models", proc, 900, log_file)
            yield f"http://127.0.0.1:{port}"
        finally:
            if pid_file.exists():
                pid = pid_file.read_text(encoding="utf-8").strip()
                subprocess.run(["wsl.exe", "-d", self.distro, "--exec", "kill", "-TERM", pid], check=False)
                log.info("vLLM PID %s (WSL) stopped", pid)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                stop_tree(proc)


# --- prediction ---------------------------------------------------------------------------------


Predictor = Callable[[System, Sequence[str], Context], None]


def predict_regex(system: System, splits: Sequence[str], ctx: Context) -> None:
    from radar.extract.baseline import read_inputs, write_predictions

    for split in splits:
        started = time.perf_counter()
        n = write_predictions(read_inputs(inputs_path(split)), prediction_path(split, system.name))
        write_json(
            sidecar_path(split, system.name),
            {
                "system": system.name,
                "split": split,
                "backend": "rules (radar.extract.baseline)",
                "records": n,
                "wall_seconds": round(time.perf_counter() - started, 1),
                "commit": results.git_commit(),
                "created_at": now(),
            },
        )


def predict_hf(system: System, splits: Sequence[str], ctx: Context) -> None:
    who = ["--adapter", system.opts["adapter"]] if "adapter" in system.opts else ["--model", system.opts["model"]]
    for split in splits:
        dev = dev_sidecar(system.name, split)
        ctx.require_free_gpu()
        args = ["bash", "scripts/train_run.sh", "predict", *who, "--split", split, "--system", system.name]
        args += ["--batch", str(dev["batch"]), "--backend", dev["backend"], "--final-test"]
        ctx.wsl(args, f"{system.name}_{split}")


def ensure_prompts(ctx: Context) -> None:
    """Test prompts of the merged student, rendered in WSL with its tokenizer, as for dev."""
    folder = ROOT / "models" / "merged" / MERGED_NAME / "prompts"
    missing = [
        s
        for s in TEST_SPLITS
        if not (folder / f"{s}.jsonl").exists() or [r["id"] for r in read_jsonl(folder / f"{s}.jsonl")] != input_ids(s)
    ]
    if missing:
        args = ["bash", "scripts/train_run.sh", "quantize", "prompts", "--name", MERGED_NAME, "--splits", *missing]
        ctx.wsl([*args, "--final-test"], "merged_prompts")


def served_args(system: System, split: str, base_url: str, backend: str) -> list[str]:
    dev = dev_sidecar(system.name, split)
    args = ["--name", MERGED_NAME, "--split", split, "--system", system.name, "--backend", backend]
    args += ["--workers", str(dev["workers"]), "--base-url", base_url, "--final-test"]
    if dev["decoding"]["json_schema_constrained"]:
        args.append("--schema")
    return args


def predict_llama(system: System, splits: Sequence[str], ctx: Context) -> None:
    from radar.extract import student_server

    ensure_prompts(ctx)
    quant = system.opts["quant"]
    gguf = Path("models") / "gguf" / MERGED_NAME / f"{MERGED_NAME}-{quant}.gguf"
    slots = max(int(dev_sidecar(system.name, s)["workers"]) for s in splits)
    ctx.require_free_gpu()
    env = {"STUDENT_GGUF": gguf.as_posix()}
    with ctx.llama_server("student", slots, env, PORTS["llama"], system.name) as url:
        for split in splits:
            student_server.main(served_args(system, split, url, "llama"))


def predict_vllm(system: System, splits: Sequence[str], ctx: Context) -> None:
    from radar.extract import student_server

    ensure_prompts(ctx)
    ctx.require_free_gpu()
    model = ROOT / "models" / "awq" / MERGED_NAME
    with ctx.vllm_server(model, f"{MERGED_NAME}-awq", PORTS["vllm"], system.name) as url:
        for split in splits:
            student_server.main(served_args(system, split, url, "vllm"))


def prompt_by_sha(sha: str, versions: Sequence[bytes]) -> bytes:
    for text in versions:
        if hashlib.sha256(text).hexdigest() == sha:
            return text
    raise SystemExit(f"no version of {PROMPT_FILE} in git has sha256 {sha}")


def git_versions(path: str) -> list[bytes]:
    commits = subprocess.run(
        ["git", "log", "--format=%H", "--", path], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    current = (ROOT / path).read_bytes()
    shown = [
        subprocess.run(["git", "show", f"{c}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        for c in commits
    ]
    return [current, *shown]


def teacher_prompt(sha: str) -> Path:
    """The teacher prompt file whose sha256 the dev run recorded, written next to the teacher outputs."""
    out = data_dir() / "teacher" / "prompts" / f"teacher_prompt_{sha[:12]}.md"
    if not out.exists() or sha256_file(out) != sha:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(prompt_by_sha(sha, git_versions(PROMPT_FILE)))
    return out


def export_teacher(system: System, split: str, teacher_meta: Mapping[str, Any]) -> None:
    """The teacher's rows become the predictions file, as on dev (the same bytes)."""
    from radar.extract.teacher import output_path

    src = output_path(system.opts["teacher"], split)
    rows = read_jsonl(src)
    dst = prediction_path(split, system.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    write_json(
        sidecar_path(split, system.name),
        {
            "system": system.name,
            "split": split,
            "backend": "llama-server",
            "teacher_file": src.relative_to(ROOT).as_posix() if src.is_relative_to(ROOT) else str(src),
            "teacher_file_sha256": sha256_file(src),
            "model": teacher_meta["model"],
            "prompt_sha256": teacher_meta["prompt_sha256"],
            "request": teacher_meta["request"],
            "records": len(rows),
            "valid_json": {"n": sum(bool(r["valid"]) for r in rows), "denominator": len(rows)},
            "request_errors": sum(r["error"] is not None for r in rows),
            "commit": results.git_commit(),
            "created_at": now(),
        },
    )


def predict_teacher(system: System, splits: Sequence[str], ctx: Context) -> None:
    from radar.extract import teacher
    from radar.extract.teacher import meta_path, output_path

    name = system.opts["teacher"]
    dev_meta = {s: read_json(meta_path(name, DEV_OF[s])) for s in splits}
    workers = {s: int(dev_meta[s]["runs"][-1]["workers"]) for s in splits}
    ctx.require_free_gpu()
    env = {"LLAMA_CTX_PER_SLOT": TEACHER_CTX_PER_SLOT}
    with ctx.llama_server(name, max(workers.values()), env, PORTS["teacher"], system.name) as url:
        os.environ["LLM_BASE_URL"] = f"{url}/v1"
        for split in splits:
            prompt = teacher_prompt(dev_meta[split]["prompt_sha256"])
            args = ["--teacher", name, "--split", split, "--workers", str(workers[split])]
            args += ["--prompt", str(prompt), "--final-test"]
            # a second pass with --resume asks again for requests that failed in the first
            for attempt in range(2):
                resume = ["--resume"] if output_path(name, split).exists() else []
                teacher.main([*args, *resume])
                if attempt == 0 and not any(r["error"] for r in read_jsonl(output_path(name, split))):
                    break
            export_teacher(system, split, read_json(meta_path(name, split)))


KINDS: dict[str, Predictor] = {
    "regex": predict_regex,
    "hf": predict_hf,
    "llama": predict_llama,
    "vllm": predict_vllm,
    "teacher": predict_teacher,
}


def predict_all(chosen: Sequence[System], ctx: Context) -> None:
    for system in chosen:
        splits = todo(system)
        if not splits:
            log.info("%s: done on %s, skipped", system.name, ", ".join(TEST_SPLITS))
            continue
        log_run(system.name, f"predict:{','.join(splits)}")
        log.info("%s: predicting %s", system.name, ", ".join(splits))
        started = time.perf_counter()
        KINDS[system.kind](system, splits, ctx)
        left = todo(system)
        if left:
            raise SystemExit(f"{system.name}: predictions on {', '.join(left)} are incomplete after the run")
        log.info("%s: done in %.0f s", system.name, time.perf_counter() - started)


# --- scoring ------------------------------------------------------------------------------------


def score_all(names: Sequence[str]) -> None:
    pairs = comparisons(names)
    for split in TEST_SPLITS:
        for name in names:
            others: list[str | None] = [b for a, b in pairs if a == name]
            for other in others or [None]:
                argv = ["--system", name, "--split", split, "--final-test"]
                if other:
                    argv += ["--compare", other]
                if cli.main(argv) != 0:
                    raise SystemExit(f"scoring {name} on {split} failed")


def pick(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {k: metric[k] for k in ("value", "ci_low", "ci_high", "n", "denominator") if k in metric}


def summarize(chosen: Sequence[System]) -> dict[str, dict[str, Any]]:
    names = [s.name for s in chosen]
    out: dict[str, dict[str, Any]] = {}
    for split in TEST_SPLITS:
        for name in names:
            scored = read_json(config.RESULTS / "extract" / f"{split}_{name}.json")["metrics"]
            for m in FIELD_METRICS:
                if m in scored:
                    out[f"{split}/{name}/{m}"] = pick(scored[m])
        for a, b in comparisons(names):
            diff = read_json(config.RESULTS / "extract" / f"{split}_{a}_vs_{b}.json")["metrics"]
            for m in FIELD_METRICS:
                if m in diff:
                    out[f"{split}/{a}_minus_{b}/{m}"] = pick(diff[m])
    return out


def write_summary(chosen: Sequence[System], rescore: str | None) -> Path:
    names = [s.name for s in chosen]
    served = {
        s.name: {
            split: {
                k: v
                for k, v in read_json(sidecar_path(split, s.name)).items()
                if k
                in (
                    "backend",
                    "adapter",
                    "adapter_sha256",
                    "model",
                    "weights",
                    "decoding",
                    "prompt_sha256",
                    "batch",
                    "workers",
                )
            }
            for split in TEST_SPLITS
        }
        for s in chosen
    }
    return results.write_result(
        "extract/final_test.json",
        summarize(chosen),
        source_script="src/radar/eval/final.py",
        systems={s.name: {"kind": s.kind, **s.opts} for s in chosen},
        comparisons=[f"{a} - {b}" for a, b in comparisons(names)],
        gold_sha256={split: sha256_file(data_dir() / "gold" / f"{split}.jsonl") for split in TEST_SPLITS},
        predictions_sha256={split: {n: sha256_file(prediction_path(split, n)) for n in names} for split in TEST_SPLITS},
        served=served,
        rescore_reason=rescore,
        method="each system and comparison scored by `python -m radar.eval --final-test` (results/extract/"
        "<split>_<system>.json, <split>_<a>_vs_<b>.json: 95% percentile bootstrap, paired on the same gold rows, "
        "settings of configs/eval.toml); this file copies their values; <a>_minus_<b> is a - b",
    )


# --- status and main ----------------------------------------------------------------------------


def status(chosen: Sequence[System]) -> int:
    print(f"{'system':28} {'kind':8} " + " ".join(f"{s:>12}" for s in TEST_SPLITS))
    for s in chosen:
        cells = []
        for split in TEST_SPLITS:
            path = prediction_path(split, s.name)
            n = len(read_jsonl(path)) if path.exists() else 0
            mark = "done" if is_complete(split, s.name) else f"{n}/{len(input_ids(split))}"
            cells.append(f"{mark:>12}")
        print(f"{s.name:28} {s.kind:8} " + " ".join(cells))
    scored = config.RESULTS / "extract" / "final_test.json"
    print(f"scored: {'yes, ' + scored.as_posix() if scored.exists() else 'no'}")
    return 0


def run(chosen: Sequence[System], ctx: Context, only: Sequence[str] = (), rescore: str | None = None) -> int:
    names = [s.name for s in chosen]
    unknown = sorted(set(only) - set(names))
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; known: {', '.join(names)}")
    log_run("final", "start")
    predict_all([s for s in chosen if not only or s.name in only], ctx)
    missing = [f"{s.name}/{split}" for s in chosen for split in todo(s)]
    if missing:
        log.info("not scored: %s still to predict", ", ".join(missing))
        return 0
    summary = config.RESULTS / "extract" / "final_test.json"
    if summary.exists() and not rescore:
        log.info("%s exists: the test is scored once; --rescore REASON scores it again", summary)
        return 0
    if rescore:
        log_run("final", f"rescore: {rescore}")
    score_all(names)
    path = write_summary(chosen, rescore)
    log_run("final", "scored")
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.eval.final", description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true", help="what is predicted and what is left")
    mode.add_argument("--final-test", action="store_true", help="predict what is missing, then score once")
    ap.add_argument("--with-dpo", action="store_true", help=f"add {DPO.name} (only if E15 kept DPO)")
    ap.add_argument("--only", nargs="+", default=[], metavar="SYSTEM", help="predict these systems only")
    ap.add_argument("--rescore", metavar="REASON", help="score again; the reason goes to results/test_runs.log")
    ap.add_argument(
        "--skip",
        nargs=2,
        action="append",
        default=[],
        metavar=("SYSTEM", "REASON"),
        help="leave a system out of the test, before any scoring; the reason goes to results/test_runs.log",
    )
    args = ap.parse_args(argv)
    chosen = systems(args.with_dpo)
    skipped = {name: reason for name, reason in args.skip}
    unknown = sorted(set(skipped) - {s.name for s in chosen})
    if unknown:
        raise SystemExit(f"unknown systems to skip {unknown}")
    chosen = tuple(s for s in chosen if s.name not in skipped)
    if args.status:
        return status(chosen)
    for name, reason in skipped.items():
        log_run(name, f"skipped: {reason}")
    return run(chosen, Context(), args.only, args.rescore)


if __name__ == "__main__":
    sys.exit(main())
