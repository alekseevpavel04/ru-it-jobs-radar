"""Teacher labeling: a local 8B model under the extraction JSON schema labels the corpus.

The teacher runs in llama-server (scripts/llm_server.sh qwen|yandex), reached through the
OpenAI-compatible API at LLM_BASE_URL. Decoding is grammar-constrained by `response_format`
json_schema (configs/extraction_schema.json), temperature 0; Qwen3 thinking is off. The system prompt
is configs/teacher_prompt.md, the user message is the extractor input of radar.extract.inputs, the
same string every other system reads.

Records come from the corpus joined with the split: train_pool is every corpus record of
a train_pool group, minus any record listed under a test, dev or withheld group. dev is read from
data/gold/dev_inputs.jsonl, the inputs the gold set was labeled on (gold labeling only). Test is never
served here.

Output, data/teacher/<teacher>/<split>.jsonl, one line per record:
    {"id", "output", "raw", "valid", "repaired", "seconds", "prompt_tokens", "completion_tokens",
     "finish_reason", "input_sha256", "error"}
`output` is the validated answer re-serialized (null when the answer broke the schema), `raw` the
model's text as returned, `repaired` marks a salary or experience object without bounds that became
null (see `validate`). The sidecar <split>.meta.json holds the prompt and schema sha256, the
model alias, the request settings and every run with its wall time; --resume refuses to mix outputs
of a different prompt, schema or model. A record whose input changed since it was labeled, or whose
request failed, is labeled again on resume.

    LLAMA_CTX_PER_SLOT=8192 bash scripts/llm_server.sh qwen 3      # the server, three slots
    python -m radar.extract.teacher --teacher qwen --split dev --workers 3
    python -m radar.extract.teacher --teacher qwen --split train_pool --limit 50 --workers 3 --resume
    python -m radar.extract.teacher --teacher qwen --split train_pool --workers 3 --resume
    python -m radar.extract.teacher --teacher qwen --split train_pool --limit 3000 --keep-all ml_ds_data --resume
    python -m radar.extract.teacher --probe-report 50
"""

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from pydantic import ValidationError

from radar.config import ROOT, data_dir, load_dotenv, require
from radar.eval.agreement import compare
from radar.eval.cli import load_config
from radar.extract.corpus import largest_remainder, permanent_random
from radar.extract.schema import SCHEMA_VERSION, dump, json_schema, parse
from radar.results import write_result

log = logging.getLogger("radar.extract.teacher")

PROMPT = ROOT / "configs" / "teacher_prompt.md"
SPLITS = ("train_pool", "dev")
MAX_TOKENS = 1024
RETRIES = 3
RETRY_PAUSE_S = 5.0
MAX_FAILURES_IN_A_ROW = 10


@dataclass(frozen=True)
class Teacher:
    name: str
    alias: str
    extra_body: Mapping[str, Any]


TEACHERS = {
    "qwen": Teacher("qwen", "Qwen3-8B-Q4_K_M", {"chat_template_kwargs": {"enable_thinking": False}}),
    "yandex": Teacher("yandex", "YandexGPT-5-Lite-8B-instruct-Q4_K_M", {}),
}


@dataclass(frozen=True)
class Item:
    id: str
    stratum: str
    input: str


class ServerError(RuntimeError):
    """The server is down or failing on every request: stop the run, the progress is on disk."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def teacher_dir(teacher: str) -> Path:
    return data_dir() / "teacher" / teacher


def output_path(teacher: str, split: str) -> Path:
    return teacher_dir(teacher) / f"{split}.jsonl"


def meta_path(teacher: str, split: str) -> Path:
    return teacher_dir(teacher) / f"{split}.meta.json"


# --- records ------------------------------------------------------------------------------------


def train_pool_items(corpus: Path, seed: int) -> list[Item]:
    split = pq.read_table(corpus / "split.parquet", columns=["split", "member_ids"]).to_pylist()
    pool = {rid for row in split if row["split"] == "train_pool" for rid in row["member_ids"]}
    held = {rid for row in split if row["split"] != "train_pool" for rid in row["member_ids"]}
    records = pq.read_table(corpus / "corpus.parquet", columns=["id", "source", "role_proxy", "input"]).to_pylist()
    items = [
        Item(r["id"], f"{r['source']}|{r['role_proxy']}", r["input"])
        for r in records
        if r["id"] in pool and r["id"] not in held
    ]
    return sorted(items, key=lambda i: (permanent_random(seed, i.id), i.id))


def dev_items(path: Path, seed: int) -> list[Item]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [Item(str(r["id"]), str(r["source"]), str(r["input"])) for r in rows]
    return sorted(items, key=lambda i: (permanent_random(seed, i.id), i.id))


def stratified(items: Sequence[Item], n: int, keep_all: Iterable[str] = ()) -> list[Item]:
    """Every item whose role proxy is in `keep_all`, then a sample of the rest proportional to the
    source x role proxy strata. `items` come in a fixed random order, so a stratum gives its first
    items, and the result keeps that order with the kept items first."""
    keep = set(keep_all)
    kept = [i for i in items if i.stratum.split("|")[-1] in keep]
    rest = [i for i in items if i.stratum.split("|")[-1] not in keep]
    budget = max(n - len(kept), 0)
    sizes = Counter(i.stratum for i in rest)
    total = sum(sizes.values())
    quota = largest_remainder({h: budget * s / total for h, s in sizes.items()}, min(budget, total)) if total else {}
    taken: Counter[str] = Counter()
    chosen = []
    for item in rest:
        if taken[item.stratum] < quota.get(item.stratum, 0):
            taken[item.stratum] += 1
            chosen.append(item)
    return kept + chosen


def select(split: str, limit: int | None, keep_all: Sequence[str], seed: int) -> list[Item]:
    if split == "train_pool":
        items = train_pool_items(data_dir() / "corpus", seed)
        return items if limit is None and not keep_all else stratified(items, limit or len(items), keep_all)
    if split == "dev":
        items = dev_items(data_dir() / "gold" / "dev_inputs.jsonl", seed)
        return items[:limit] if limit is not None else items
    raise ValueError(f"split {split!r} is not served by the teacher script")


# --- requests -----------------------------------------------------------------------------------


def request_body(teacher: Teacher, system: str, text: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": teacher.alias,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": {"name": "vacancy", "schema": schema}},
        **teacher.extra_body,
    }


# An object without a single bound is how a model writes "not stated"; the guide calls that null.
EMPTY_OBJECTS = (("salary", ("from", "to")), ("experience_years", ("min", "max")))


def validate(raw: str | None) -> tuple[str | None, bool]:
    """The answer re-serialized after the schema check (None if it broke the schema), and whether a
    salary or experience object without bounds was turned into null on the way.

    The grammar cannot forbid such an object: a variant "first bound null, second a number" was tried
    and made the model invent a number (salary to 300 000 on vacancies without a salary)."""
    if raw is None:
        return None, False
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    repaired = False
    if isinstance(obj, dict):
        for key, bounds in EMPTY_OBJECTS:
            value = obj.get(key)
            if isinstance(value, dict) and all(value.get(b) is None for b in bounds):
                obj[key] = None
                repaired = True
    try:
        return json.dumps(dump(parse(json.dumps(obj, ensure_ascii=False))), ensure_ascii=False), repaired
    except ValidationError:
        return None, repaired


def failed_row(item: Item, seconds: float | None, error: str) -> dict[str, Any]:
    return {
        "id": item.id,
        "output": None,
        "raw": None,
        "valid": False,
        "repaired": False,
        "seconds": seconds,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "input_sha256": sha256_text(item.input),
        "error": error,
    }


def label_one(client: httpx.Client, body: dict[str, Any], item: Item, pause_s: float = RETRY_PAUSE_S) -> dict[str, Any]:
    """One record. A 4xx answer (the request itself is wrong for this server) is a failed row at
    once; a 5xx is retried, then written as a failed row. A resumed run tries failed rows again. A
    server that does not answer at all stops the run."""
    for attempt in range(1, RETRIES + 1):
        t0 = time.perf_counter()
        try:
            r = client.post("chat/completions", json=body)
        except httpx.TransportError as e:
            failure, answered = f"{type(e).__name__}: {e}", False
        else:
            seconds = round(time.perf_counter() - t0, 3)
            if r.status_code < 400:
                data = r.json()
                choice = data["choices"][0]
                raw = choice["message"].get("content")
                usage = data.get("usage") or {}
                # a cut answer can still close as valid JSON with fewer skills: not a complete label
                output, repaired = validate(raw) if choice.get("finish_reason") != "length" else (None, False)
                return {
                    "id": item.id,
                    "output": output,
                    "raw": raw,
                    "valid": output is not None,
                    "repaired": repaired,
                    "seconds": seconds,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "input_sha256": sha256_text(item.input),
                    "error": None,
                }
            failure, answered = f"HTTP {r.status_code}: {r.text[:300]}", True
            if r.status_code < 500:
                break
        if attempt < RETRIES:
            log.warning("%s: %s, retry %d of %d", item.id, failure, attempt, RETRIES - 1)
            time.sleep(pause_s * attempt)
    if answered:
        return failed_row(item, seconds, failure)
    raise ServerError(f"{item.id}: {failure} after {RETRIES} attempts")


def check_model(client: httpx.Client, teacher: Teacher) -> None:
    r = client.get("models")
    r.raise_for_status()
    served = [m.get("id") for m in r.json().get("data", [])]
    if teacher.alias not in served:
        raise ServerError(f"the server at LLM_BASE_URL serves {served}, not {teacher.alias}; start the right teacher")


# --- files --------------------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def reusable(rows: Sequence[Mapping[str, Any]], items: Sequence[Item]) -> tuple[list[Mapping[str, Any]], set[str]]:
    """Rows a resumed run keeps: no request error, and the input is the one labeled (records outside
    this selection are kept as they are). Returns the rows and the ids among `items` already done."""
    current = {i.id: sha256_text(i.input) for i in items}
    kept = [
        row
        for row in rows
        if row.get("error") is None and (row["id"] not in current or row.get("input_sha256") == current[row["id"]])
    ]
    return kept, {row["id"] for row in kept if row["id"] in current}


def run_meta(teacher: Teacher, split: str, prompt_sha: str, schema_sha: str) -> dict[str, Any]:
    return {
        "teacher": teacher.name,
        "model": teacher.alias,
        "split": split,
        "prompt_file": PROMPT.relative_to(ROOT).as_posix() if PROMPT.is_relative_to(ROOT) else str(PROMPT),
        "prompt_sha256": prompt_sha,
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha,
        "request": {"temperature": 0, "max_tokens": MAX_TOKENS, **teacher.extra_body},
        "runs": [],
    }


IDENTITY = ("model", "prompt_sha256", "schema_sha256")


# --- labeling -----------------------------------------------------------------------------------


def labeled(
    client: httpx.Client, bodies: Iterable[tuple[Item, dict[str, Any]]], workers: int
) -> Iterator[dict[str, Any]]:
    """Rows as requests finish, with at most `workers` in flight (one per server slot)."""
    todo = iter(bodies)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: set[Future[dict[str, Any]]] = set()
        try:
            for item, body in todo:
                running.add(pool.submit(label_one, client, body, item))
                if len(running) >= workers:
                    done, running = wait(running, return_when=FIRST_COMPLETED)
                    for fut in done:
                        yield fut.result()
            while running:
                done, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield fut.result()
        finally:
            for fut in running:
                fut.cancel()


def cmd_label(args: argparse.Namespace) -> int:
    teacher = TEACHERS[args.teacher]
    seed = load_config().seed
    items = select(args.split, args.limit, args.keep_all, seed)
    prompt = PROMPT.read_text(encoding="utf-8")
    schema = json_schema()
    meta = run_meta(teacher, args.split, sha256_text(prompt), sha256_text(json.dumps(schema, sort_keys=True)))

    out, meta_file = output_path(teacher.name, args.split), meta_path(teacher.name, args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not (args.resume or args.overwrite):
        raise SystemExit(f"{out} exists: --resume continues it, --overwrite starts over")
    if args.resume and out.exists():
        if not meta_file.exists():
            raise SystemExit(f"{meta_file} is missing: cannot tell which prompt labeled {out}")
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        changed = [k for k in IDENTITY if old.get(k) != meta[k]]
        if changed:
            raise SystemExit(f"{out} was labeled with another {', '.join(changed)}; --overwrite starts over")
        meta["runs"] = old.get("runs", [])
        rows, done = reusable(read_rows(out), items)
        write_rows(out, rows)
    else:
        write_rows(out, [])
        done = set()

    todo = [i for i in items if i.id not in done]
    log.info("%s on %s: %d selected, %d done, %d to label", teacher.name, args.split, len(items), len(done), len(todo))
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    n_done = n_valid = failures_in_a_row = 0
    client = httpx.Client(base_url=require("LLM_BASE_URL"), timeout=httpx.Timeout(600.0, connect=10.0))
    try:
        check_model(client, teacher)
        bodies = ((i, request_body(teacher, prompt, i.input, schema)) for i in todo)
        with out.open("a", encoding="utf-8", newline="\n") as f:
            for row in labeled(client, bodies, args.workers):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                n_valid += bool(row["valid"])
                failures_in_a_row = failures_in_a_row + 1 if row["error"] else 0
                if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                    raise ServerError(f"{failures_in_a_row} failed requests in a row, last: {row['error']}")
                if n_done % args.log_every == 0 or n_done == len(todo):
                    elapsed = time.perf_counter() - t0
                    eta_h = (len(todo) - n_done) * elapsed / n_done / 3600
                    log.info(
                        "%d of %d, valid %d of %d, %.2f s per record, eta %.1f h",
                        n_done,
                        len(todo),
                        n_valid,
                        n_done,
                        elapsed / n_done,
                        eta_h,
                    )
    finally:
        client.close()
        elapsed = time.perf_counter() - t0
        meta["runs"].append(
            {
                "started_at": started.isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 1),
                "n_labeled": n_done,
                "n_valid": n_valid,
                "workers": args.workers,
                "limit": args.limit,
                "keep_all": list(args.keep_all),
                "n_selected": len(items),
            }
        )
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("%d labeled, %d valid -> %s", n_done, n_valid, out)
    return 0


# --- probe report -------------------------------------------------------------------------------


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def wall_per_record(runs: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    """Wall seconds per labeled record by worker count, over runs that labeled something."""
    by_workers: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for run in runs:
        if run["n_labeled"]:
            by_workers[run["workers"]][0] += run["elapsed_s"]
            by_workers[run["workers"]][1] += run["n_labeled"]
    return {w: {"seconds": round(s / n, 3), "n_records": int(n)} for w, (s, n) in sorted(by_workers.items())}


def teacher_probe(name: str, ids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = {r["id"]: r for r in read_rows(output_path(name, "train_pool"))}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} probe records not labeled yet, e.g. {missing[0]}")
    probe = [rows[i] for i in ids]
    meta = json.loads(meta_path(name, "train_pool").read_text(encoding="utf-8"))
    dev_meta_file = meta_path(name, "dev")
    dev_meta = json.loads(dev_meta_file.read_text(encoding="utf-8")) if dev_meta_file.exists() else {"runs": []}
    ok = [r for r in probe if r["error"] is None]
    metrics: dict[str, dict[str, Any]] = {
        f"{name}_valid_json": share(sum(r["valid"] for r in probe), len(probe)),
        f"{name}_request_errors": share(len(probe) - len(ok), len(probe)),
        f"{name}_repaired_empty_objects": share(sum(bool(r.get("repaired")) for r in probe), len(probe)),
        f"{name}_latency_s_median": {"value": round(statistics.median(r["seconds"] for r in ok), 2)},
        f"{name}_prompt_tokens_mean": {"value": round(statistics.mean(r["prompt_tokens"] for r in ok), 1)},
        f"{name}_completion_tokens_mean": {"value": round(statistics.mean(r["completion_tokens"] for r in ok), 1)},
        f"{name}_completion_tokens_max": {"value": max(r["completion_tokens"] for r in ok)},
    }
    timing = {"train_pool": wall_per_record(meta["runs"]), "dev": wall_per_record(dev_meta["runs"])}
    for split, by_w in timing.items():
        for w, t in by_w.items():
            metrics[f"{name}_{split}_wall_s_per_record_w{w}"] = {"value": t["seconds"], "records": t["n_records"]}
    identity = {k: meta[k] for k in (*IDENTITY, "request")}
    outputs = {r["id"]: {"output": json.loads(r["output"]) if r["output"] else None} for r in probe}
    return metrics, {"identity": identity, "timing": timing}, outputs


def forecast(pool_size: int, seconds: float) -> dict[str, Any]:
    return {
        "value": round(pool_size * seconds / 3600, 1),
        "unit": "hours",
        "records": pool_size,
        "s_per_record": seconds,
    }


def cmd_probe_report(n: int) -> int:
    cfg = load_config()
    pool = train_pool_items(data_dir() / "corpus", cfg.seed)
    probe = stratified(pool, n)
    ids = [i.id for i in probe]
    metrics: dict[str, dict[str, Any]] = {}
    details: dict[str, Any] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for name in TEACHERS:
        m, d, o = teacher_probe(name, ids)
        metrics |= m
        details[name] = d
        outputs[name] = o
        best = min(d["timing"]["train_pool"].values(), key=lambda t: t["seconds"])
        metrics[f"{name}_forecast_train_pool_hours"] = forecast(len(pool), best["seconds"])

    agreement = compare(outputs["qwen"], outputs["yandex"], cfg.salary_rel_tol)
    both = agreement["records"]["both"] - agreement["invalid"]["a"] - agreement["invalid"]["b"]
    for field in ("role", "skills", "grade", "work_format", "experience_years"):
        metrics[f"agreement_{field}_exact"] = share(
            agreement["exact"][field]["n"], agreement["exact"][field]["denominator"]
        )
    metrics["agreement_all_fields_equal"] = share(
        agreement["all_fields_equal"]["n"], agreement["all_fields_equal"]["denominator"]
    )
    metrics["agreement_skills_f1_yandex_vs_qwen"] = {"value": agreement["f1_b_against_a"]["skills_f1"]}
    metrics["agreement_role_f1_yandex_vs_qwen"] = {"value": agreement["f1_b_against_a"]["role_f1"]}
    strata = Counter(i.stratum for i in probe)
    pool_strata = Counter(i.stratum for i in pool)
    path = write_result(
        "teacher/probe.json",
        metrics,
        source_script="src/radar/extract/teacher.py",
        probe={
            "n": len(ids),
            "ids_sha256": sha256_text("\n".join(sorted(ids))),
            "design": "proportional stratified sample of train_pool by source x role proxy, fixed random order "
            "(seed of configs/eval.toml)",
            "strata": {h: {"probe": strata[h], "pool": pool_strata[h]} for h in sorted(pool_strata)},
        },
        train_pool_records=len(pool),
        teachers=details,
        agreement={
            "a": "qwen",
            "b": "yandex",
            "records_both_valid": both,
            "method": "radar.eval.agreement.compare: exact per field after ordering lists, F1 of b against a; "
            "no gold on these records, so agreement is not accuracy",
            "f1_b_against_a": agreement["f1_b_against_a"],
            "disagreements": agreement["disagreements"],
        },
        forecast_note="wall seconds per record of the fastest train_pool run times train_pool size; "
        "train_pool changes when the corpus is rebuilt",
        dev_results=["results/extract/dev_teacher_qwen.json", "results/extract/dev_teacher_yandex.json"],
    )
    print(path)
    for k, v in metrics.items():
        shown = f"{v['n']} of {v['denominator']}" if "denominator" in v else ""
        print(f"{k:45} {v['value']:>10} {shown}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.extract.teacher", description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--teacher", choices=sorted(TEACHERS))
    mode.add_argument("--probe-report", type=int, metavar="N", help="results/teacher/probe.json from N probe records")
    ap.add_argument("--split", choices=SPLITS, default="train_pool")
    ap.add_argument(
        "--limit",
        type=int,
        help="train_pool: proportional stratified sample of N by source x role proxy; dev: first N in a fixed order",
    )
    ap.add_argument("--keep-all", action="append", default=[], metavar="PROXY", help="take every record of this proxy")
    ap.add_argument("--workers", type=int, default=1, help="requests in flight; at most the server's -np slots")
    run = ap.add_mutually_exclusive_group()
    run.add_argument("--resume", action="store_true", help="continue the output file, skip records already labeled")
    run.add_argument("--overwrite", action="store_true", help="start the output file over")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args(argv)
    load_dotenv()
    if args.probe_report:
        return cmd_probe_report(args.probe_report)
    return cmd_label(args)


if __name__ == "__main__":
    sys.exit(main())
