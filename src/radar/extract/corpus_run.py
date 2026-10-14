"""Extraction of the whole corpus: every current vacancy version through one extractor,
the fields into ClickHouse `extract.vacancy_fields` (infra/clickhouse/002_extract_vacancy_fields.sql).

Versions: for trudvsem (API), trudvsem_snapshot, Habr and SuperJob the latest staged version of
every IT candidate (staging, one row per source_id and day; the latest day wins), its title, text and
payload read from raw.vacancies by the same key. For Telegram the latest stored version of every
post (raw.vacancies FINAL; the latest fetch, then dt and content_hash, as radar.extract.telegram_corpus
orders them) that the corpus rules keep: a vacancy, one vacancy in it, not a digest, enough text of
its own, the channel's language (`drop_reason`). The input is `radar.extract.inputs.input_for_record`,
the one every extractor reads; records with a short text are kept, the fields of their header are
still worth having in the marts.

Skipping: a version is done when the table holds its (source, source_id, content_hash) for this
model_id. model_id names weights and prompt together; the runner refuses a model_id that the table
or the state directory knows with another prompt. Identical inputs (the API and a snapshot carry one
vacancy, templates are posted again unchanged) are sent once: the other versions get the same answer
and `reused` = true. Decoding is greedy, so the answer of an identical input is the same answer.

Extractor: a backend with `model_id`, `prompt_sha256` and `extract(inputs) -> answers`, one batch per
call. `--backend regex` is the regex baseline (dry runs, tests); `--backend module:factory` loads any
other, e.g. the student behind llama-server, built by `factory(**options)` from `--backend-opt k=v`.
`--workers` batches are in flight at once (llama-server: one per slot, batch size 1; a batched
generate: one worker, a larger batch).

State: data/corpus_run/<model_id>/meta.json (identity and every run, written before the first
request) and rows.jsonl (append-only, one line per version, flushed per line). The table gets rows in
chunks; a run that died between the file and the table is caught up by the next run (or `sync`)
before it enumerates anything. A failed request is written to the file with its error, never to the
table, and is tried again on the next run. A run stops safely at any moment: SIGTERM (`kill <pid>`)
lets the requests in flight finish and inserts what is waiting, and a partial last line left by a
harder stop is cut off before the next run reads the file. The same command resumes. `--time-budget`
ends a run the same way once it has taken that long, so a slow extractor works off a backlog over
several runs; `--skip-done-by` leaves out versions another model_id already holds.

    python -m radar.extract.corpus_run plan --rate hf_generate=1.0 --rate llama_server=0.25
    python -m radar.extract.corpus_run run --backend regex --limit 20 --dry-run
    python -m radar.extract.corpus_run run --backend radar.extract.student_server:backend \
        --backend-opt base_url=http://127.0.0.1:8095 --workers 8 --limit 200      (WSL training venv)
    python -m radar.extract.corpus_run run --backend radar.extract.student_server:backend \
        --backend-opt base_url=http://127.0.0.1:8095 --workers 8
    python -m radar.extract.corpus_run run --backend student_cpu:backend --backend-opt base_url=http://student:8080 \
        --skip-done-by qwen3_1p7b_r16_clean-awq --limit 150 --time-budget 3600      (airflow radar_ml_label)
    python -m radar.extract.corpus_run sync --model-id <model_id>
    python -m radar.extract.corpus_run report --model-id <model_id>
    python -m radar.extract.corpus_run dropped        (radar.extract.corpus_drops)
"""

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from radar.config import data_dir, load_dotenv
from radar.extract.baseline import output_for
from radar.extract.corpus import permanent_random
from radar.extract.inputs import input_for_record
from radar.extract.schema import SCHEMA_VERSION, dump, json_schema, try_parse
from radar.extract.telegram_corpus import channel_langs, drop_reason
from radar.extract.telegram_corpus import load_config as telegram_config
from radar.results import write_result
from radar.stream.clickhouse import DDL_DIR, statements

log = logging.getLogger("radar.extract.corpus_run")

TABLE = "extract.vacancy_fields"
DDL = DDL_DIR / "002_extract_vacancy_fields.sql"
SOURCES = ("trudvsem", "trudvsem_snapshot", "habr", "superjob", "telegram")
# Staging table of each portal source and its IT filter (Habr is an IT site, all of it counts).
STAGED = {
    "trudvsem": ("staging.stg_trudvsem_api", "is_it_candidate"),
    "trudvsem_snapshot": ("staging.stg_trudvsem_snapshot", "is_it_candidate"),
    "habr": ("staging.stg_habr", "true"),
    "superjob": ("staging.stg_superjob", "is_it_candidate"),
}
RAW_BATCH = 1000
# Telegram posts are read in slices by a hash of the id, so no single answer holds every text.
TELEGRAM_SLICES = 16
MAX_FAILURES_IN_A_ROW = 10
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

Key = tuple[str, str, str]


# --- versions -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Version:
    source: str
    source_id: str
    content_hash: str
    dt: str
    published_at: str | None
    input: str

    @property
    def key(self) -> Key:
        return (self.source, self.source_id, self.content_hash)

    @property
    def id(self) -> str:
        return f"{self.source}:{self.source_id}"


class Store(Protocol):
    """What the runner needs from ClickHouse; tests give it an in-memory one."""

    def latest_keys(self, source: str) -> list[tuple[str, str, str]]:
        """(source_id, dt, content_hash) of the latest staged version of every IT candidate."""

    def raw_rows(self, source: str, keys: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
        """Raw rows (source_id, dt, content_hash, title, text, payload, published_at) of these keys."""

    def telegram_posts(self, since: date | None, part: int, parts: int) -> list[dict[str, Any]]:
        """The latest version of every Telegram vacancy post without several vacancies, in slice `part`."""

    def table_exists(self) -> bool: ...

    def ensure_table(self) -> None: ...

    def done_keys(self, model_id: str) -> set[Key]: ...

    def prompt_shas(self, model_id: str) -> set[str | None]: ...

    def insert(self, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def summary(self, model_id: str) -> list[dict[str, Any]]:
        """Rows, valid rows, own requests and valid own requests per source, after FINAL."""


def _published(value: Any) -> str | None:
    return str(value) if value else None


def portal_versions(store: Store, source: str, skip: set[Key]) -> Iterator[Version]:
    keys = [k for k in store.latest_keys(source) if (source, k[0], k[2]) not in skip]
    for start in range(0, len(keys), RAW_BATCH):
        batch = keys[start : start + RAW_BATCH]
        rows = {(r["source_id"], r["dt"], r["content_hash"]): r for r in store.raw_rows(source, batch)}
        missing = [k for k in batch if k not in rows]
        if missing:
            raise RuntimeError(f"{source}: {len(missing)} staged versions have no raw row, e.g. {missing[0]}")
        for source_id, dt, content_hash in batch:
            row = rows[(source_id, dt, content_hash)]
            text = input_for_record({"source": source, **row})
            yield Version(source, source_id, content_hash, dt, _published(row.get("published_at")), text)


def telegram_versions(store: Store, since: date | None, skip: set[Key], drops: Counter[str]) -> Iterator[Version]:
    min_chars = telegram_config().min_content_chars
    langs = channel_langs()
    for part in range(TELEGRAM_SLICES):
        for row in store.telegram_posts(since, part, TELEGRAM_SLICES):
            if ("telegram", row["source_id"], row["content_hash"]) in skip:
                continue
            why = drop_reason(row, min_chars, langs.get(str(row["source_id"]).rsplit("/", 1)[0]))
            if why is not None:
                drops[why] += 1
                continue
            text = input_for_record(
                {"source": "telegram", "title": None, "text": row["text"], "payload": row["payload"]}
            )
            yield Version(
                "telegram", row["source_id"], row["content_hash"], row["dt"], _published(row["published_at"]), text
            )


def versions(
    store: Store, sources: Sequence[str], since: date | None, skip: set[Key], drops: Counter[str]
) -> Iterator[Version]:
    for source in sources:
        if source == "telegram":
            yield from telegram_versions(store, since, skip, drops)
        else:
            yield from portal_versions(store, source, skip)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- extractors ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """One record's answer. `raw` is the model text as returned (None when the request failed);
    `error` marks a failed request, which is tried again on the next run."""

    raw: str | None
    finish_reason: str | None = None
    error: str | None = None
    input_truncated: bool | None = None
    seconds: float | None = None


class Backend(Protocol):
    model_id: str
    prompt_sha256: str | None

    def extract(self, inputs: Sequence[str]) -> list[Answer]:
        """Answers in the order of `inputs`. Raise when the extractor itself is gone: the run stops
        with its progress on disk."""


class RegexBackend:
    model_id = "regex_baseline"
    prompt_sha256: str | None = None

    def extract(self, inputs: Sequence[str]) -> list[Answer]:
        answers = []
        for text in inputs:
            t0 = time.perf_counter()
            raw = output_for(text)
            answers.append(Answer(raw, "stop", seconds=round(time.perf_counter() - t0, 4)))
        return answers


def load_backend(spec: str, options: Mapping[str, str]) -> Backend:
    if spec == "regex":
        if options:
            raise SystemExit("the regex backend takes no options")
        return RegexBackend()
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--backend {spec!r}: regex or module:factory")
    factory: Callable[..., Backend] = getattr(importlib.import_module(module), attr)
    backend = factory(**options)
    if not MODEL_ID.match(backend.model_id):
        raise SystemExit(f"model_id {backend.model_id!r}: letters, digits, '.', '_', '-'")
    return backend


def parse_options(pairs: Sequence[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise SystemExit(f"{pair!r}: expected name=value")
        out[name] = value
    return out


# --- rows ---------------------------------------------------------------------------------------


def checked_output(answer: Answer) -> str | None:
    """The answer re-serialized after the schema check; None when it broke the schema or was cut."""
    if answer.raw is None or answer.finish_reason == "length":
        return None
    fields = try_parse(answer.raw)
    return json.dumps(dump(fields), ensure_ascii=False) if fields is not None else None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def row_for(
    version: Version, answer: Answer, backend: Backend, run_id: str, created_at: str, reused: bool
) -> dict[str, Any]:
    output = checked_output(answer)
    return {
        "source": version.source,
        "source_id": version.source_id,
        "content_hash": version.content_hash,
        "dt": version.dt,
        "model_id": backend.model_id,
        "prompt_sha256": backend.prompt_sha256,
        "schema_version": SCHEMA_VERSION,
        "input_sha256": sha256_text(version.input),
        "input_truncated": answer.input_truncated,
        "output": output,
        "raw": answer.raw,
        "valid": output is not None,
        "finish_reason": answer.finish_reason,
        "seconds": None if reused else answer.seconds,
        "error": answer.error,
        "reused": reused,
        "run_id": run_id,
        "created_at": created_at,
    }


def reused_row(version: Version, earlier: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """A version whose input this model answered in an earlier run: that answer, marked as reused."""
    row = dict(earlier)
    row.update(
        source=version.source,
        source_id=version.source_id,
        content_hash=version.content_hash,
        dt=version.dt,
        seconds=None,
        reused=True,
        run_id=run_id,
        created_at=utc_now(),
    )
    return row


def _ch_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def table_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A file row as a row of extract.vacancy_fields; an invalid answer has every field empty."""
    f: dict[str, Any] = json.loads(row["output"]) if row["output"] else {}
    salary = f.get("salary") or {}
    experience = f.get("experience_years") or {}
    return {
        "source": row["source"],
        "source_id": row["source_id"],
        "content_hash": row["content_hash"],
        "model_id": row["model_id"],
        "prompt_sha256": row["prompt_sha256"],
        "schema_version": row["schema_version"],
        "dt": row["dt"],
        "input_sha256": row["input_sha256"],
        "input_truncated": row["input_truncated"],
        "valid": row["valid"],
        "finish_reason": row["finish_reason"],
        "reused": row["reused"],
        "role": f.get("role"),
        "grade": f.get("grade"),
        "skills": f.get("skills") or [],
        "salary_from": salary.get("from"),
        "salary_to": salary.get("to"),
        "salary_currency": salary.get("currency"),
        "salary_gross_net": salary.get("gross_net"),
        "salary_period": salary.get("period"),
        "work_format": f.get("work_format"),
        "cities": f.get("cities") or [],
        "experience_min": experience.get("min"),
        "experience_max": experience.get("max"),
        "employment_type": f.get("employment_type"),
        "lang": f.get("lang"),
        "run_id": row["run_id"],
        "created_at": _ch_time(row["created_at"]),
    }


def row_key(row: Mapping[str, Any]) -> Key:
    return (str(row["source"]), str(row["source_id"]), str(row["content_hash"]))


# --- state --------------------------------------------------------------------------------------


def state_dir(model_id: str, dry_run: bool = False) -> Path:
    root = data_dir() / "corpus_run"
    return root / "dry_run" / model_id if dry_run else root / model_id


def drop_cut_line(path: Path) -> bool:
    """Cut the file back to its last newline. A process killed in the middle of a write leaves a
    partial last line, and the next run would append to it; that version is simply extracted again."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return False
        f.seek(0)
        data = f.read()
        f.truncate(data.rfind(b"\n") + 1)
    log.warning("%s: dropped a partial last line", path)
    return True


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answered_rows(rows: Iterable[Mapping[str, Any]]) -> dict[Key, Mapping[str, Any]]:
    """The latest answered row of every version in the file; failed requests are not answers."""
    return {row_key(r): r for r in rows if r.get("error") is None}


def schema_sha256() -> str:
    return sha256_text(json.dumps(json_schema(), sort_keys=True))


IDENTITY = ("model_id", "prompt_sha256", "schema_sha256")


def open_meta(path: Path, backend: Backend, spec: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model_id": backend.model_id,
        "prompt_sha256": backend.prompt_sha256,
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha256(),
        "backend": spec,
        "table": TABLE,
        "runs": [],
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        changed = [k for k in IDENTITY if old.get(k) != meta[k]]
        if changed:
            raise SystemExit(
                f"{path} belongs to model_id {old.get('model_id')!r} with another {', '.join(changed)}: "
                "a new prompt or schema is a new model_id"
            )
        meta["runs"] = old.get("runs", [])
    return meta


def write_meta(path: Path, meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class Sink:
    """Rows to the file at once, answered rows to the table in chunks."""

    def __init__(self, path: Path, store: Store | None, insert_every: int) -> None:
        self.file = path.open("a", encoding="utf-8", newline="\n")
        self.store = store
        self.insert_every = insert_every
        self.pending: list[dict[str, Any]] = []
        self.inserted = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()
        if self.store is not None and row["error"] is None:
            self.pending.append(table_row(row))
            if len(self.pending) >= self.insert_every:
                self.flush()

    def flush(self) -> None:
        if self.store is not None and self.pending:
            self.store.insert(self.pending)
            self.inserted += len(self.pending)
            self.pending = []

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.file.close()


def sync(store: Store, rows: Iterable[Mapping[str, Any]], model_id: str, chunk: int = 5000) -> int:
    """Insert answered file rows whose version the table does not hold for this model yet."""
    done = store.done_keys(model_id)
    missing = [table_row(r) for k, r in answered_rows(rows).items() if k not in done and r["model_id"] == model_id]
    for start in range(0, len(missing), chunk):
        store.insert(missing[start : start + chunk])
    return len(missing)


# --- the run ------------------------------------------------------------------------------------


@dataclass
class RunStats:
    rows: int = 0
    requests: int = 0
    valid_requests: int = 0
    reused: int = 0
    errors: int = 0
    stopped_by_time_budget: bool = False
    by_source: Counter[str] = field(default_factory=Counter)


def in_flight(fn: Callable[[Any], Any], items: Iterable[Any], workers: int) -> Iterator[tuple[Any, Any]]:
    """(item, fn(item)) as calls finish, with at most `workers` running."""
    todo = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: dict[Future[Any], Any] = {}
        try:
            for item in todo:
                running[pool.submit(fn, item)] = item
                if len(running) >= workers:
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for fut in done:
                        yield running.pop(fut), fut.result()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield running.pop(fut), fut.result()
        finally:
            for fut in running:
                fut.cancel()


def extract_all(
    todo: Sequence[Version],
    backend: Backend,
    sink: Sink,
    run_id: str,
    earlier: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    workers: int,
    stats: RunStats,
    log_every: int = 200,
    deadline: float | None = None,
) -> None:
    """Each distinct input once; versions that share it get the same answer. `stats` is filled as
    rows are written, so a run that stops halfway still reports what it wrote. Past `deadline`
    (a time.perf_counter value) no new batch is sent; the ones in flight finish and are written."""
    groups: dict[str, list[Version]] = defaultdict(list)
    for v in todo:
        groups[sha256_text(v.input)].append(v)

    for sha, members in groups.items():
        if sha in earlier:
            for v in members:
                sink.write(reused_row(v, earlier[sha], run_id))
                stats.rows += 1
                stats.reused += 1
                stats.by_source[v.source] += 1
    pending = [members for sha, members in groups.items() if sha not in earlier]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    log.info(
        "%d versions, %d distinct inputs, %d answered before; %d requests in %d batches",
        len(todo),
        len(groups),
        len(groups) - len(pending),
        len(pending),
        len(batches),
    )

    def call(batch: list[list[Version]]) -> list[Answer]:
        answers = backend.extract([members[0].input for members in batch])
        if len(answers) != len(batch):
            raise RuntimeError(f"the backend answered {len(answers)} of {len(batch)} inputs")
        return answers

    def until_deadline() -> Iterator[list[list[Version]]]:
        for i, batch in enumerate(batches):
            if deadline is not None and time.perf_counter() >= deadline:
                stats.stopped_by_time_budget = True
                log.info("time budget spent: %d of %d batches left for the next run", len(batches) - i, len(batches))
                return
            yield batch

    t0 = time.perf_counter()
    failures_in_a_row = 0
    for batch, answers in in_flight(call, until_deadline(), workers):
        created_at = utc_now()
        for members, answer in zip(batch, answers, strict=True):
            for i, v in enumerate(members):
                row = row_for(v, answer, backend, run_id, created_at, reused=i > 0)
                sink.write(row)
                stats.rows += 1
                stats.by_source[v.source] += 1
                if row["error"] is not None:
                    stats.errors += 1
                elif i > 0:
                    stats.reused += 1
            stats.requests += 1
            stats.valid_requests += checked_output(answer) is not None
            failures_in_a_row = failures_in_a_row + 1 if answer.error else 0
            if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                raise RuntimeError(f"{failures_in_a_row} failed requests in a row, last: {answer.error}")
        if stats.requests % log_every < len(batch) or stats.requests == len(pending):
            elapsed = time.perf_counter() - t0
            left = (len(pending) - stats.requests) * elapsed / stats.requests / 3600
            log.info(
                "%d of %d requests, valid %d, %.3f s per request, eta %.2f h",
                stats.requests,
                len(pending),
                stats.valid_requests,
                elapsed / stats.requests,
                left,
            )


def ordered(items: Iterable[Version], seed: int) -> list[Version]:
    """A fixed random order over all sources: the first N of it is a probe with every source in it."""
    return sorted(items, key=lambda v: (permanent_random(seed, v.id), v.id))


def run_id_now() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{os.getpid()}"


def cmd_run(args: argparse.Namespace, store: Store, backend: Backend, seed: int) -> int:
    # The budget counts from the start of the command: enumeration and catching up the table included.
    deadline = time.perf_counter() + args.time_budget if args.time_budget is not None else None
    directory = args.state_dir or state_dir(backend.model_id, args.dry_run)
    meta_file, rows_file = directory / "meta.json", directory / "rows.jsonl"
    meta = open_meta(meta_file, backend, args.backend)
    drop_cut_line(rows_file)
    file_rows = read_rows(rows_file)

    loaded = 0
    if args.dry_run:
        done = set(answered_rows(file_rows))
        if store.table_exists():
            done |= store.done_keys(backend.model_id)
    else:
        store.ensure_table()
        foreign = store.prompt_shas(backend.model_id) - {backend.prompt_sha256}
        if foreign:
            raise SystemExit(f"{TABLE} holds model_id {backend.model_id} with prompt {sorted(map(str, foreign))}")
        loaded = sync(store, file_rows, backend.model_id)
        done = store.done_keys(backend.model_id)
    # Versions another model has fields for are not pending for this one (a daily run of the CPU
    # student after the corpus run of another quantization); its own rows still count first.
    if args.skip_done_by and store.table_exists():
        for other in args.skip_done_by:
            done |= store.done_keys(other)

    run_id = run_id_now()
    run: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished": False,
        "sources": list(args.sources),
        "telegram_since": args.telegram_since.isoformat() if args.telegram_since else None,
        "limit": args.limit,
        "time_budget_s": args.time_budget,
        "skip_done_by": list(args.skip_done_by),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "dry_run": args.dry_run,
        "synced_from_file": loaded,
    }
    meta["runs"].append(run)
    # On disk before the first request: a run killed from outside leaves its identity behind.
    write_meta(meta_file, meta)

    drops: Counter[str] = Counter()
    t_enum = time.perf_counter()
    found = ordered(versions(store, args.sources, args.telegram_since, done, drops), seed)
    todo = found[: args.limit] if args.limit is not None else found
    run["enumeration_s"] = round(time.perf_counter() - t_enum, 1)
    log.info(
        "%d versions to extract (%d after --limit), Telegram dropped by rule %s", len(found), len(todo), dict(drops)
    )
    earlier = {r["input_sha256"]: r for r in answered_rows(file_rows).values()}

    sink = Sink(rows_file, None if args.dry_run else store, args.insert_every)
    t0 = time.perf_counter()
    stats = RunStats()
    try:
        extract_all(
            todo, backend, sink, run_id, earlier, args.batch_size, args.workers, stats, args.log_every, deadline
        )
    finally:
        try:
            sink.close()
        finally:
            run.update(
                finished=stats.rows == len(todo) and not sink.pending,
                elapsed_s=round(time.perf_counter() - t0, 1),
                versions_found=len(found),
                versions_selected=len(todo),
                telegram_dropped=dict(drops),
                rows=stats.rows,
                requests=stats.requests,
                valid_requests=stats.valid_requests,
                reused=stats.reused,
                errors=stats.errors,
                stopped_by_time_budget=stats.stopped_by_time_budget,
                inserted=sink.inserted,
                rows_by_source=dict(stats.by_source),
            )
            write_meta(meta_file, meta)
    log.info(
        "%d rows (%d requests, %d valid, %d reused, %d failed), %d inserted -> %s",
        stats.rows,
        stats.requests,
        stats.valid_requests,
        stats.reused,
        stats.errors,
        sink.inserted,
        rows_file,
    )
    return 0


# --- plan and report ----------------------------------------------------------------------------


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def parse_rates(pairs: Sequence[str]) -> dict[str, float]:
    rates = {}
    for name, value in parse_options(pairs).items():
        rates[name] = float(value)
        if rates[name] <= 0:
            raise SystemExit(f"--rate {name}: seconds per request must be positive")
    return rates


def plan_metrics(
    found: Iterable[Version], done: set[Key], drops: Counter[str], rates: Mapping[str, float]
) -> dict[str, dict[str, Any]]:
    """Counts per source of what a run would extract now, and hours for each assumed rate. A request
    is a distinct input: the forecast counts requests, not versions."""
    versions_n: Counter[str] = Counter()
    done_n: Counter[str] = Counter()
    todo_n: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    inputs: dict[str, set[str]] = defaultdict(set)
    all_inputs: set[str] = set()
    years: Counter[str] = Counter()
    for v in found:
        versions_n[v.source] += 1
        if v.key in done:
            done_n[v.source] += 1
            continue
        todo_n[v.source] += 1
        chars[v.source] += len(v.input)
        sha = sha256_text(v.input)
        inputs[v.source].add(sha)
        all_inputs.add(sha)
        if v.source == "telegram":
            years[(v.published_at or "unknown")[:4]] += 1
    metrics: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        if source not in versions_n and source not in drops:
            continue
        metrics[f"{source}_versions"] = {"value": versions_n[source]}
        metrics[f"{source}_done"] = {"value": done_n[source]}
        metrics[f"{source}_to_extract"] = {"value": todo_n[source]}
        metrics[f"{source}_distinct_inputs_to_extract"] = {"value": len(inputs[source])}
        metrics[f"{source}_input_chars_mean"] = {
            "value": round(chars[source] / todo_n[source], 1) if todo_n[source] else 0.0
        }
    for why, n in sorted(drops.items()):
        metrics[f"telegram_dropped_{why}"] = {"value": n}
    for year, n in sorted(years.items()):
        metrics[f"telegram_to_extract_published_{year}"] = {"value": n}
    total = sum(todo_n.values())
    metrics["total_to_extract"] = {"value": total}
    metrics["total_requests"] = {"value": len(all_inputs)}
    metrics["requests_saved_by_identical_inputs"] = share(total - len(all_inputs), total)
    for name, seconds in rates.items():
        metrics[f"forecast_{name}_hours"] = {
            "value": round(len(all_inputs) * seconds / 3600, 1),
            "requests": len(all_inputs),
            "assumed_s_per_request": seconds,
        }
        for source in SOURCES:
            if inputs[source]:
                metrics[f"forecast_{name}_{source}_hours"] = {"value": round(len(inputs[source]) * seconds / 3600, 1)}
    return metrics


def cmd_plan(args: argparse.Namespace, store: Store) -> int:
    done = store.done_keys(args.model_id) if args.model_id and store.table_exists() else set()
    drops: Counter[str] = Counter()
    metrics = plan_metrics(versions(store, args.sources, args.telegram_since, set(), drops), done, drops, args.rate)
    path = write_result(
        "extract/corpus_run_plan.json",
        metrics,
        source_script="src/radar/extract/corpus_run.py",
        model_id=args.model_id,
        sources=list(args.sources),
        telegram_since=args.telegram_since.isoformat() if args.telegram_since else None,
        method="versions as `run` enumerates them (latest staged IT version per portal record, latest Telegram "
        "post version kept by the corpus rules); done: rows of model_id already in the table; a request is a "
        "distinct input across all sources (per-source forecasts count distinct inputs within the source, so "
        "they add up to more than the total); forecast hours = requests x the seconds per request given with "
        "--rate, an assumption of the command line, not a measurement of this script",
    )
    print(path)
    for key, value in metrics.items():
        shown = f"{value['n']} of {value['denominator']}" if "denominator" in value else ""
        print(f"{key:55} {value['value']:>12} {shown}")
    return 0


def cmd_sync(args: argparse.Namespace, store: Store) -> int:
    store.ensure_table()
    rows = read_rows(state_dir(args.model_id) / "rows.jsonl")
    print(f"{sync(store, rows, args.model_id)} rows inserted into {TABLE}")
    return 0


def cmd_report(args: argparse.Namespace, store: Store) -> int:
    rows = store.summary(args.model_id)
    if not rows:
        raise SystemExit(f"{TABLE} has no rows of model_id {args.model_id}")
    metrics: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    for r in rows:
        source = r["source"]
        metrics[f"{source}_rows"] = {"value": int(r["rows"])}
        metrics[f"{source}_valid_json_requests"] = share(int(r["valid_requests"]), int(r["requests"]))
        for k in ("rows", "valid_rows", "requests", "valid_requests"):
            totals[k] += int(r[k])
    metrics["valid_json_requests"] = share(totals["valid_requests"], totals["requests"])
    metrics["valid_json_rows"] = share(totals["valid_rows"], totals["rows"])
    path = write_result(
        f"extract/corpus_run_{args.model_id}.json",
        metrics,
        source_script="src/radar/extract/corpus_run.py",
        model_id=args.model_id,
        table=TABLE,
        method="extract.vacancy_fields FINAL for model_id; a request is a row not reused from an identical input, "
        "valid_json_requests is the share of answers that passed the schema among them; valid_json_rows counts "
        "every version, reused answers included",
    )
    print(path)
    for key, value in metrics.items():
        shown = f"{value['n']} of {value['denominator']}" if "denominator" in value else ""
        print(f"{key:40} {value['value']:>10} {shown}")
    return 0


# --- ClickHouse ---------------------------------------------------------------------------------


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _json_lines(body: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in body.splitlines() if line]


def _row(raw: dict[str, Any]) -> dict[str, Any]:
    """The query names its text columns day and published: an alias named like its column would
    shadow the column in WHERE and ORDER BY."""
    raw["dt"] = raw.pop("day")
    if "published" in raw:
        raw["published_at"] = raw.pop("published")
    return raw


_LATEST_SQL = """
SELECT source_id, toString(dt) AS day, content_hash
FROM {table}
WHERE source = {{source:String}} AND {it_filter}
ORDER BY dt DESC
LIMIT 1 BY source_id
FORMAT JSONEachRow
"""

_RAW_SQL = """
SELECT source_id, toString(dt) AS day, content_hash, title, text, payload, toString(published_at) AS published
FROM raw.vacancies
WHERE source = {{source:String}} AND (source_id, dt, content_hash) IN ({keys})
ORDER BY fetched_at DESC
LIMIT 1 BY source_id, dt, content_hash
FORMAT JSONEachRow
"""

_TELEGRAM_SQL = """
SELECT *
FROM (
    SELECT source_id, toString(dt) AS day, content_hash, toString(published_at) AS published,
           text, payload, is_vacancy, multi_vacancy
    FROM raw.vacancies FINAL
    WHERE source = 'telegram' AND cityHash64(source_id) % {{parts:UInt32}} = {{part:UInt32}}{since}
    ORDER BY source_id, fetched_at DESC, dt DESC, content_hash DESC
    LIMIT 1 BY source_id
)
WHERE is_vacancy AND NOT coalesce(multi_vacancy, false)
FORMAT JSONEachRow
"""

_SUMMARY_SQL = f"""
SELECT source, count() AS rows, countIf(valid) AS valid_rows, countIf(NOT reused) AS requests,
       countIf(valid AND NOT reused) AS valid_requests
FROM {TABLE} FINAL
WHERE model_id = {{model:String}}
GROUP BY source
ORDER BY source
FORMAT JSONEachRow
"""


class ClickHouseStore:
    def __init__(self, ch: Any) -> None:
        self.ch = ch

    def latest_keys(self, source: str) -> list[tuple[str, str, str]]:
        table, it_filter = STAGED[source]
        body = self.ch.query(_LATEST_SQL.format(table=table, it_filter=it_filter), source=source)
        return [(r["source_id"], r["day"], r["content_hash"]) for r in _json_lines(body)]

    def raw_rows(self, source: str, keys: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
        literal = ", ".join(f"({_quote(i)}, toDate({_quote(d)}), {_quote(h)})" for i, d, h in keys)
        return [_row(r) for r in _json_lines(self.ch.query(_RAW_SQL.format(keys=literal), source=source))]

    def telegram_posts(self, since: date | None, part: int, parts: int) -> list[dict[str, Any]]:
        params = {"part": str(part), "parts": str(parts)}
        clause = ""
        if since is not None:
            clause = " AND published_at >= {since:Date}"
            params["since"] = since.isoformat()
        return [_row(r) for r in _json_lines(self.ch.query(_TELEGRAM_SQL.format(since=clause), **params))]

    def table_exists(self) -> bool:
        return str(self.ch.query(f"EXISTS TABLE {TABLE}")).strip() == "1"

    def ensure_table(self) -> None:
        for statement in statements(DDL.read_text(encoding="utf-8")):
            self.ch.query(statement)

    def done_keys(self, model_id: str) -> set[Key]:
        body = self.ch.query(
            f"SELECT DISTINCT source, source_id, content_hash FROM {TABLE} WHERE model_id = {{model:String}} "
            "FORMAT JSONEachRow",
            model=model_id,
        )
        return {(r["source"], r["source_id"], r["content_hash"]) for r in _json_lines(body)}

    def prompt_shas(self, model_id: str) -> set[str | None]:
        body = self.ch.query(
            f"SELECT DISTINCT prompt_sha256 FROM {TABLE} WHERE model_id = {{model:String}} FORMAT JSONEachRow",
            model=model_id,
        )
        return {r["prompt_sha256"] for r in _json_lines(body)}

    def insert(self, rows: Sequence[Mapping[str, Any]]) -> None:
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        self.ch.query(f"INSERT INTO {TABLE} FORMAT JSONEachRow\n{lines}\n")

    def summary(self, model_id: str) -> list[dict[str, Any]]:
        return _json_lines(self.ch.query(_SUMMARY_SQL, model=model_id))


# --- command line -------------------------------------------------------------------------------


def _sources(value: str) -> list[str]:
    names = [s.strip() for s in value.split(",") if s.strip()]
    unknown = sorted(set(names) - set(SOURCES))
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"sources from {', '.join(SOURCES)}; got {value!r}")
    return names


def stop_on_signal(signum: int, frame: object) -> None:
    """`kill <pid>` unwinds the run like an error: requests in flight finish and are written, the
    rows waiting for the table are inserted, and meta.json records the run as unfinished."""
    log.warning("signal %d: stopping after the requests in flight", signum)
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.extract.corpus_run", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def with_selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sources", type=_sources, default=list(SOURCES), help="comma-separated, default all")
        p.add_argument("--telegram-since", type=date.fromisoformat, help="Telegram posts published on or after")

    run = sub.add_parser("run", help="extract versions not in the table yet and load them")
    with_selection(run)
    run.add_argument("--backend", required=True, help="regex or module:factory")
    run.add_argument("--backend-opt", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument("--limit", type=int, help="first N versions of a fixed random order (a probe)")
    run.add_argument(
        "--time-budget", type=float, metavar="SECONDS", help="send no new request after this long; the rest waits"
    )
    run.add_argument(
        "--skip-done-by",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="also skip versions this model has rows for (repeatable)",
    )
    run.add_argument("--batch-size", type=int, default=1, help="inputs per backend call")
    run.add_argument("--workers", type=int, default=1, help="backend calls in flight")
    run.add_argument("--insert-every", type=int, default=500, help="rows per insert into the table")
    run.add_argument("--dry-run", action="store_true", help="read ClickHouse only; rows go to the state file")
    run.add_argument("--state-dir", type=Path, help="default data/corpus_run/<model_id>")
    run.add_argument("--log-every", type=int, default=200)

    plan = sub.add_parser("plan", help="results/extract/corpus_run_plan.json: what a run would extract now")
    with_selection(plan)
    plan.add_argument("--model-id", help="count versions this model already has as done")
    plan.add_argument("--rate", action="append", default=[], metavar="NAME=SECONDS", help="assumed s per request")

    for name, text in (("sync", "insert file rows missing from the table"), ("report", "valid JSON share")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--model-id", required=True)
    sub.add_parser("dropped", help="Telegram versions the run never extracts, with the reason (corpus_drops)")

    args = ap.parse_args(argv)
    load_dotenv()
    from radar.eval.cli import load_config
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        store = ClickHouseStore(ch)
        if args.command == "plan":
            args.rate = parse_rates(args.rate)
            return cmd_plan(args, store)
        if args.command == "sync":
            return cmd_sync(args, store)
        if args.command == "report":
            return cmd_report(args, store)
        if args.command == "dropped":
            from radar.extract import corpus_drops

            for line in corpus_drops.write_report(corpus_drops.run(store, ch, TELEGRAM_SLICES)):
                print(line)
            return 0
        backend = load_backend(args.backend, parse_options(args.backend_opt))
        signal.signal(signal.SIGTERM, stop_on_signal)
        return cmd_run(args, store, backend, load_config().seed)


if __name__ == "__main__":
    sys.exit(main())
