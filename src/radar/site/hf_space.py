"""The Space: assemble its files, pick texts for a local run, run it in Docker with 2 CPUs.

    python -m radar.site.hf_space pick        20 trudvsem texts outside the gold set -> data/hf/space_bench.jsonl,
                                               3 short ones as the page's examples -> data/hf/space_examples.json
    python -m radar.site.hf_space assemble    storage/hf/space: hf/space, the vendored radar modules, the
                                               tokenizer, the salary model, assets/space.json, the card
    python -m radar.site.hf_space bench       build the image in WSL, start it with --cpus 2, check the prompt
                                               rendering on dev, time 20 requests -> results/public/space_local.json

The image runs the GGUF with llama-server and renders prompts with radar.train.dataset, so a few
modules of this package travel with it: the entry modules and everything they import from radar at
module level (`vendored`), so a new import in the salary model reaches the image without an edit here.
The assembled tree is the Space repo as an upload would send it.
On Hugging Face the GGUF is downloaded from the model repo at start; locally it is mounted.

Texts. Only trudvsem texts are publishable (DECISIONS 2), and only those not reposted from other
sites (source_type "Вакансия интернет ресурса"). The local run uses IT candidates of the latest
snapshot, in a fixed hash order, whose vacancy id is not in any gold split, and that the corpus
extraction (6.7) already answered and read as IT: the Space's answer is compared with it.

The host is shared (the corpus extraction and other jobs run in the same WSL distro), so the run
records the load average of the distro before and after the requests next to the timings.
"""

import argparse
import ast
import json
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from radar.config import ROOT, data_dir, load_dotenv
from radar.extract.inputs import input_for_record
from radar.extract.schema import dump, try_parse
from radar.models.salary import load_model
from radar.results import write_result
from radar.site.hf_common import HF, STORAGE, load_config, render_card, results_doc, shown, write_json
from radar.stream.clickhouse import ClickHouse

log = logging.getLogger("radar.site.hf_space")

SPACE_SRC = HF / "space"
OUT = STORAGE / "space"
BENCH = data_dir() / "hf" / "space_bench.jsonl"
EXAMPLES = data_dir() / "hf" / "space_examples.json"
RESULT = "public/space_local.json"
# What the app imports from this package; their module-level radar imports follow (vendored).
ENTRY_MODULES = ("radar.models.salary", "radar.extract.schema", "radar.train.dataset")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "dataset_meta.json")
REPOSTED = "Вакансия интернет ресурса"
N_BENCH = 20
N_EXAMPLES = 3
EXAMPLE_MAX_CHARS = 1800
DISTRO = "radar"
IMAGE = "radar/space:local"
CONTAINER = "radar-space-bench"
PORT = 7860
CPUS = "2"
MEMORY = "16g"


def module_file(name: str, src: Path) -> Path:
    base = src.joinpath(*name.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def module_imports(path: Path, src: Path) -> set[str]:
    """radar modules imported at module level of `path` (imports inside functions run only when called)."""
    found: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "radar"}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith("radar"):
            found.add(node.module)
            # `from radar.models import salary` imports a module, not a name
            names = (f"{node.module}.{a.name}" for a in node.names)
            found |= {name for name in names if module_file(name, src).exists()}
    return found


def vendored(entries: Sequence[str] = ENTRY_MODULES, src: Path = ROOT / "src") -> list[str]:
    """Files under src/ the entry modules need: their radar imports, recursively, with every package __init__."""
    seen: set[str] = set()
    todo = list(entries)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        todo += sorted(module_imports(module_file(name, src), src))
    return sorted(module_file(n, src).relative_to(src).as_posix() for n in seen)


def gguf_repo_path(name: str, quant: str) -> str:
    """Where a GGUF lies in the model repo (radar.site.hf_model puts it there)."""
    return f"gguf/{name}-{quant}.gguf"


def gguf_entry(name: str, quant: str) -> dict[str, Any]:
    manifest = json.loads((ROOT / "models" / "gguf" / name / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["type"] == quant:
            return {**entry, "manifest_created_at": manifest["created_at"]}
    raise KeyError(f"models/gguf/{name}/manifest.json has no {quant}")


# --- pick ---------------------------------------------------------------------------------------

_CANDIDATES_SQL = """
SELECT s.source_id AS source_id, toString(s.dt) AS day, s.content_hash AS content_hash
FROM staging.stg_trudvsem_snapshot AS s
WHERE s.is_it_candidate AND s.has_text AND coalesce(s.source_type, '') != {reposted:String}
  AND s.dt = (SELECT max(dt) FROM staging.stg_trudvsem_snapshot)
ORDER BY cityHash64(s.source_id)
LIMIT 400
FORMAT JSONEachRow
"""


def _q(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def gold_vacancy_ids() -> set[str]:
    ids = set()
    for path in (data_dir() / "gold").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"].split(":", 1)[1])
    return ids


def corpus_fields(ch: Any, model_id: str, keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    """The corpus extraction's fields of these (source_id, content_hash), as the schema dumps them; only
    vacancies it read as IT, the kind of text the Space is for."""
    literal = ", ".join(f"({_q(i)}, {_q(h)})" for i, h in keys)
    body = ch.query(
        "SELECT source_id, content_hash, role, grade, skills, salary_from, salary_to, salary_currency, "
        "salary_gross_net, salary_period, work_format, cities, experience_min, experience_max, employment_type, lang "
        "FROM extract.vacancy_fields FINAL "
        "WHERE source = 'trudvsem_snapshot' AND model_id = {model:String} AND valid AND role != 'non_it' "
        f"AND (source_id, content_hash) IN ({literal}) FORMAT JSONEachRow",
        model=model_id,
    )
    out = {}
    for line in body.splitlines():
        r = json.loads(line)
        out[(r["source_id"], r["content_hash"])] = fields_of_row(r)
    return out


def fields_of_row(r: dict[str, Any]) -> dict[str, Any]:
    """A row of extract.vacancy_fields back in the shape of the extraction schema."""
    salary = None
    if any(r[f"salary_{k}"] is not None for k in ("from", "to", "currency", "gross_net", "period")):
        salary = {
            "from": r["salary_from"],
            "to": r["salary_to"],
            "currency": r["salary_currency"],
            "gross_net": r["salary_gross_net"],
            "period": r["salary_period"],
        }
    experience = None
    if r["experience_min"] is not None or r["experience_max"] is not None:
        experience = {"min": r["experience_min"], "max": r["experience_max"]}
    raw = {
        "role": r["role"],
        "grade": r["grade"],
        "skills": r["skills"],
        "salary": salary,
        "work_format": r["work_format"],
        "cities": r["cities"],
        "experience_years": experience,
        "employment_type": r["employment_type"],
        "lang": r["lang"],
    }
    parsed = try_parse(json.dumps(raw, ensure_ascii=False))
    if parsed is None:
        raise ValueError(f"a valid row of extract.vacancy_fields does not parse: {r['source_id']}")
    return dump(parsed)


def cmd_pick(model_id: str) -> int:
    load_dotenv()
    ch = ClickHouse()
    candidates = [json.loads(line) for line in ch.query(_CANDIDATES_SQL, reposted=REPOSTED).splitlines()]
    gold = gold_vacancy_ids()
    candidates = [c for c in candidates if c["source_id"] not in gold]
    keys = [(c["source_id"], c["content_hash"]) for c in candidates]
    answered = corpus_fields(ch, model_id, keys)
    literal = ", ".join(f"({_q(c['source_id'])}, toDate({_q(c['day'])}), {_q(c['content_hash'])})" for c in candidates)
    body = ch.query(
        "SELECT source_id, toString(dt) AS day, content_hash, title, text, payload, url FROM raw.vacancies "
        f"WHERE source = 'trudvsem_snapshot' AND (source_id, dt, content_hash) IN ({literal}) "
        "ORDER BY fetched_at DESC LIMIT 1 BY source_id, dt, content_hash FORMAT JSONEachRow"
    )
    raw = {(r["source_id"], r["content_hash"]): r for r in map(json.loads, body.splitlines())}
    rows = []
    for c in candidates:
        key = (c["source_id"], c["content_hash"])
        if key not in answered or key not in raw:
            continue
        r = raw[key]
        text = input_for_record(
            {"source": "trudvsem_snapshot", "title": r["title"], "text": r["text"], "payload": r["payload"]}
        )
        rows.append(
            {
                "id": f"trudvsem_snapshot:{c['source_id']}",
                "dt": c["day"],
                "content_hash": c["content_hash"],
                "url": r["url"],
                "input": text,
                "corpus_fields": answered[key],
            }
        )
    bench = rows[:N_BENCH]
    if len(bench) < N_BENCH:
        raise SystemExit(f"only {len(bench)} texts qualify, {N_BENCH} needed")
    examples = [r for r in rows[N_BENCH:] if len(r["input"]) <= EXAMPLE_MAX_CHARS][:N_EXAMPLES]
    BENCH.parent.mkdir(parents=True, exist_ok=True)
    BENCH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bench), encoding="utf-8")
    write_json(EXAMPLES, {"examples": [{"id": e["id"], "url": e["url"], "text": e["input"]} for e in examples]})
    log.info("%d bench texts -> %s, %d examples -> %s", len(bench), shown(BENCH), len(examples), shown(EXAMPLES))
    return 0


# --- assemble -----------------------------------------------------------------------------------


def space_info(cfg: dict[str, Any]) -> dict[str, Any]:
    name, quant = cfg["model"]["name"], cfg["model"]["space_gguf"]
    system = cfg["model"]["space_system"]
    quant_summary = results_doc("extract/quant_summary.json")["metrics"]
    dev = results_doc(f"extract/dev_{system}.json")
    f1 = quant_summary[f"dev/{system}/macro_f1"]
    valid = quant_summary[f"dev/{system}/valid_json"]
    salary = load_model(ROOT / cfg["salary"]["artifact"])
    salary_result = results_doc(cfg["salary"]["result"])
    if salary_result["model_version"] != salary.version:
        raise SystemExit(
            f"{cfg['salary']['result']} scores {salary_result['model_version']}, "
            f"{cfg['salary']['artifact']} is {salary.version}: rerun the salary model first"
        )
    sm = salary_result["metrics"]
    meta = json.loads((ROOT / "models" / "merged" / name / "dataset_meta.json").read_text(encoding="utf-8"))
    entry = gguf_entry(name, quant)
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"]
    repos = cfg["repos"]
    return {
        "model": {
            "name": name,
            "base": cfg["model"]["base"],
            "gguf_type": quant,
            "gguf_sha256": entry["sha256"],
            "gguf_size_mib": entry["size_mib"],
            "built_at": entry["manifest_created_at"][:10],
            "max_input_tokens": meta["max_input_tokens"],
            "prompt_sha256": meta["prompt_sha256"],
        },
        "gguf": {"repo": repos["model"], "file": gguf_repo_path(name, quant), "revision": None},
        "dev": {
            "system": system,
            "n_examples": dev["n_examples"],
            "macro_f1": f1["value"],
            "ci_low": f1["ci_low"],
            "ci_high": f1["ci_high"],
            "valid_json": {"n": valid["n"], "denominator": valid["denominator"]},
            "source": "results/extract/quant_summary.json",
        },
        "salary_source": cfg["salary"]["source"],
        "salary": {
            "version": salary.version,
            "trained_at": salary.trained_at[:10],
            "level": salary.level,
            "rows_train": sm["data_rows_train"]["value"],
            "rows_calib": sm["data_rows_calib"]["value"],
            "dev_coverage": {
                "n": sm["dev_gbm_cqr_coverage"]["n"],
                "denominator": sm["dev_gbm_cqr_coverage"]["denominator"],
            },
            "source": f"results/{cfg['salary']['result']}",
        },
        "links": {
            "site": repos["site"],
            "github": repos["github"],
            "model": f"https://huggingface.co/{repos['model']}",
            "dataset": f"https://huggingface.co/datasets/{repos['dataset']}",
        },
        "examples": examples,
        "speed": speed(),
    }


def speed() -> dict[str, Any] | None:
    """The local run's typical answer time, for the page to say how long to wait."""
    path = ROOT / "results" / RESULT
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        "p50_s": doc["metrics"]["latency_p50_s"]["value"],
        "p95_s": doc["metrics"]["latency_p95_s"]["value"],
        "measured_at": doc["measured_at"][:10],
        "source": f"results/{RESULT}",
    }


def cmd_assemble() -> int:
    cfg = load_config()
    name = cfg["model"]["name"]
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SPACE_SRC, OUT, ignore=shutil.ignore_patterns("__pycache__", "README.md.in"))
    for rel in vendored():
        dst = OUT / "vendor" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / rel, dst)
    for file in TOKENIZER_FILES:
        dst = OUT / "assets" / "tokenizer" / file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "models" / "merged" / name / file, dst)
    (OUT / "assets" / "salary").mkdir(parents=True)
    shutil.copy2(ROOT / cfg["salary"]["artifact"], OUT / "assets" / "salary" / "salary.pkl")
    write_json(OUT / "assets" / "space.json", space_info(cfg))
    card = render_card((SPACE_SRC / "README.md.in").read_text(encoding="utf-8"))
    (OUT / "README.md").write_text(card.published, encoding="utf-8", newline="\n")
    log.info("space -> %s (%d numbers in the card, pending %s)", shown(OUT), card.numbers, card.pending or "none")
    return 0


# --- bench --------------------------------------------------------------------------------------


def wsl_path(path: Path) -> str:
    drive, rest = path.resolve().drive, path.resolve().as_posix()[2:]
    return str(PurePosixPath("/mnt", drive.rstrip(":").lower()) / rest.lstrip("/"))


def wsl(*args: str, check: bool = True, capture: bool = True) -> str:
    out = subprocess.run(["wsl", "-d", DISTRO, "--", *args], capture_output=capture, text=True, encoding="utf-8")
    if check and out.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}: exit {out.returncode}\n{out.stderr}")
    return out.stdout or ""


def quantile(values: Sequence[float], q: float) -> float:
    from radar.extract.student_server import quantile as nearest_rank

    return nearest_rank(values, q)


def wait_ready(client: httpx.Client, timeout_s: float = 600.0) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        # wsl.exe itself now and then exits with -1 and no output; only a docker answer counts.
        state = subprocess.run(
            ["wsl", "-d", DISTRO, "--", "docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if state.returncode == 0 and state.stdout.strip() != "true":
            logs = wsl("docker", "logs", "--tail", "40", CONTAINER)
            raise RuntimeError(f"the container stopped:\n{logs}")
        try:
            if client.get("/health").status_code == 200:
                return time.perf_counter() - started
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"the Space did not answer /health in {timeout_s:.0f} s")


def container_memory_mib() -> float:
    usage = wsl("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER).split("/")[0].strip()
    number, unit = float(usage[:-3]), usage[-3:]
    return round(number * {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}[unit], 1)


def cmd_bench(build: bool) -> int:
    cfg = load_config()
    name, quant = cfg["model"]["name"], cfg["model"]["space_gguf"]
    gguf = ROOT / "models" / "gguf" / name / f"{name}-{quant}.gguf"
    rows = [json.loads(line) for line in BENCH.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"]
    # The GGUF is read from the distro's own disk: a model mapped over /mnt/d would time the 9p share.
    cache = "/var/tmp/radar-space"
    wsl("mkdir", "-p", cache)
    wsl("cp", "-u", wsl_path(gguf), f"{cache}/model.gguf")
    if build:
        started = time.perf_counter()
        wsl("docker", "build", "-t", IMAGE, wsl_path(OUT), capture=False)
        log.info("image built in %.0f s", time.perf_counter() - started)
    image_bytes = int(wsl("docker", "image", "inspect", IMAGE, "--format", "{{.Size}}").strip())
    wsl("docker", "rm", "-f", CONTAINER, check=False)
    prompts = ROOT / "models" / "merged" / name / "prompts" / "dev.jsonl"
    inputs = data_dir() / "gold" / "dev_inputs.jsonl"
    started = time.perf_counter()
    wsl(
        "docker", "run", "-d", "--name", CONTAINER, "--cpus", CPUS, "--memory", MEMORY, "-p", f"{PORT}:{PORT}",
        "-v", f"{cache}/model.gguf:/models/model.gguf:ro",
        "-v", f"{wsl_path(prompts)}:/check/prompts.jsonl:ro",
        "-v", f"{wsl_path(inputs)}:/check/inputs.jsonl:ro",
        "-e", "GGUF_PATH=/models/model.gguf", IMAGE,
    )  # fmt: skip
    client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=httpx.Timeout(600.0, connect=5.0))
    try:
        ready = wait_ready(client)
        cold_start = time.perf_counter() - started
        log.info("ready in %.1f s (health polled for %.1f s)", cold_start, ready)
        render = json.loads(
            wsl("docker", "exec", CONTAINER, "python", "-m", "app.prompt", "check",
                "/check/inputs.jsonl", "/check/prompts.jsonl").strip().splitlines()[-1]
        )  # fmt: skip
        log.info("prompt rendering on dev: %s", render)
        load_before = float(wsl("cat", "/proc/loadavg").split()[0])
        t0 = time.perf_counter()
        first = client.post("/api/extract", json={"text": examples[0]["text"]})
        first.raise_for_status()
        first_s = time.perf_counter() - t0
        answers = []
        for row in rows:
            t0 = time.perf_counter()
            r = client.post("/api/extract", json={"text": row["input"]})
            r.raise_for_status()
            answers.append({"id": row["id"], "wall_s": round(time.perf_counter() - t0, 3), **r.json()})
            log.info("%s %.2f s", row["id"], answers[-1]["wall_s"])
        memory = container_memory_mib()
        load_after = float(wsl("cat", "/proc/loadavg").split()[0])
        cpu = wsl("bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip()
        cpus = int(wsl("nproc").strip())
    finally:
        wsl("docker", "rm", "-f", CONTAINER, check=False)

    n = len(answers)
    wall = [a["wall_s"] for a in answers]
    model_s = [a["seconds"]["model"] for a in answers]
    valid = sum(a["valid"] for a in answers)
    same_tokens = sum(a["prompt_tokens"] == a["prompt_tokens_tokenizer"] for a in answers)
    by_id = {r["id"]: r for r in rows}
    same_fields = sum(a["valid"] and a["fields"] == by_id[a["id"]]["corpus_fields"] for a in answers)
    with_salary = sum(a["salary"] is not None for a in answers)
    metrics: dict[str, dict[str, Any]] = {
        "latency_p50_s": {"value": round(quantile(wall, 0.5), 2)},
        "latency_p95_s": {"value": round(quantile(wall, 0.95), 2)},
        "latency_max_s": {"value": round(max(wall), 2)},
        "model_p50_s": {"value": round(quantile(model_s, 0.5), 2)},
        "model_p95_s": {"value": round(quantile(model_s, 0.95), 2)},
        "prompt_p50_s": {"value": round(quantile([a["seconds"]["prompt"] for a in answers], 0.5), 2)},
        "generate_p50_s": {"value": round(quantile([a["seconds"]["generate"] for a in answers], 0.5), 2)},
        "host_load_1min": {"value": round(max(load_before, load_after), 1), "before": load_before, "after": load_after},
        "host_cpus": {"value": cpus},
        "first_request_s": {"value": round(first_s, 2)},
        "cold_start_s": {"value": round(cold_start, 1)},
        "valid_json": {"value": round(valid / n, 4), "n": valid, "denominator": n},
        "salary_estimates": {"value": round(with_salary / n, 4), "n": with_salary, "denominator": n},
        "same_fields_as_corpus_awq": {"value": round(same_fields / n, 4), "n": same_fields, "denominator": n},
        "prompt_tokens_match": {"value": round(same_tokens / n, 4), "n": same_tokens, "denominator": n},
        "dev_prompt_render_match": {
            "value": round(render["prompt_match"] / render["records"], 4),
            "n": render["prompt_match"],
            "denominator": render["records"],
        },
        "dev_prompt_tokens_match": {
            "value": round(render["prompt_tokens_match"] / render["records"], 4),
            "n": render["prompt_tokens_match"],
            "denominator": render["records"],
        },
        "input_truncated": {"value": sum(a["input_truncated"] for a in answers)},
        "container_memory_mib": {"value": memory},
        "image_size_mib": {"value": round(image_bytes / 2**20)},
    }
    write_result(
        RESULT,
        metrics,
        source_script="src/radar/site/hf_space.py",
        setup={
            "docker": f"--cpus {CPUS} --memory {MEMORY}, WSL distro {DISTRO}",
            "host_cpu": cpu,
            "gguf": f"models/gguf/{name}/{name}-{quant}.gguf",
            "requests": "sequential, one in flight; wall time as the client saw it, model time inside the app",
            "first_request": "one example text before the measured ones, not counted in the quantiles",
            "cold_start": "docker run to the first 200 of /health (llama-server loaded the model)",
            "texts": "trudvsem snapshot IT candidates outside the gold set, fixed hash order (python -m "
            "radar.site.hf_space pick); same_fields_as_corpus_awq compares with the 6.7 corpus answer of the "
            "AWQ checkpoint of the same model",
            "dev_prompt_render": "app.prompt check inside the image against models/merged/<name>/prompts/dev.jsonl",
        },
        per_request=[
            {
                "id": a["id"],
                "wall_s": a["wall_s"],
                "model_s": a["seconds"]["model"],
                "prompt_s": a["seconds"]["prompt"],
                "generate_s": a["seconds"]["generate"],
                "prompt_tokens": a["prompt_tokens"],
                "completion_tokens": a["completion_tokens"],
                "valid": a["valid"],
                "same_fields_as_corpus": a["valid"] and a["fields"] == by_id[a["id"]]["corpus_fields"],
            }
            for a in answers
        ],
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    log.info("p50 %.2f s, p95 %.2f s, valid %d of %d -> results/%s", quantile(wall, 0.5), quantile(wall, 0.95),
             valid, n, RESULT)  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.site.hf_space", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick", help="texts for the local run and the page's examples (ClickHouse)")
    pick.add_argument("--model-id", help="corpus extraction to compare with; default [dataset] model_id of hf.toml")
    sub.add_parser("assemble", help="lay out storage/hf/space")
    bench = sub.add_parser("bench", help="build and run the image in WSL with 2 CPUs, time 20 requests")
    bench.add_argument("--no-build", action="store_true", help="reuse the image built before")
    args = ap.parse_args(argv)
    if args.command == "pick":
        return cmd_pick(args.model_id or load_config()["dataset"]["model_id"])
    if args.command == "assemble":
        return cmd_assemble()
    return cmd_bench(build=not args.no_build)


if __name__ == "__main__":
    sys.exit(main())
