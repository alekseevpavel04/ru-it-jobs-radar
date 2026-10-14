"""Weekly quality of the extraction model as agreement with the teacher on a subsample.

There is no gold for new vacancies, so quality is watched against the project's teacher: Qwen3-8B
under the extraction schema, with the prompt version that labeled each family in the teacher labeling (portals:
train_pool, Telegram: tg_train_pool; data/teacher/qwen/<split>.meta.json), whose dev scores against
gold are known. The student did not learn from this teacher alone: its portal labels are the rules
and its Telegram labels the teachers' agreed values with the rules elsewhere (configs/silver.toml),
so the level of agreement differs by family by construction, the portal role most of all.
Agreement is not accuracy either: the teacher has its own errors. What is watched is the change:
whether the student moves away from the teacher on new texts against the same measure on the
reference weeks.

Samples. Units and windows are those of radar.monitor.drift: new vacancies without duplicates, each
source's reference weeks and its latest complete week. Per source group (portals, Telegram) and
window, SAMPLE_SIZE extracted units are drawn: the units with the smallest permanent_random(seed,
source:source_id), so the draw does not depend on row order and the reference sample stays the same
from week to week once the reference is fixed.

Teacher. The student's answer for a unit is the row of extract.vacancy_fields for its version and
the marts' model_id. The teacher sees the same input: rebuilt from the raw layer by
radar.extract.inputs and checked against the row's input_sha256 (a mismatch is counted and left
out). Telegram requests carry the teacher's `exclude` key as in the teacher labeling; a post the
teacher calls a list or not a vacancy is counted and left out, as the gold set leaves such posts
out. Answers are cached in reports/evidently/teacher/qwen.jsonl by input sha256, model, prompt and
schema, so a later run (the Airflow task among them) reuses them; a failed request is not reused.

Without a teacher. The teacher runs in llama-server on the Windows host (scripts/llm_server.sh qwen,
port 8096; 8090 is the Airflow API). When it does not answer, agreement comes from the cache alone,
and a window whose sample the cache does not cover has agreement absent (status
teacher_unreachable), never zero.

Metrics are radar.eval.metrics with the teacher's answer in place of gold and the settings of
configs/eval.toml: per-field exact agreement (<field>_accuracy, n of denominator), F1 for skills and
cities, macro F1, with percentile bootstrap intervals; the change from the reference window to the
current one has an interval from independent resamples of the two samples.

    LLAMA_PORT=8096 LLAMA_CTX_PER_SLOT=8192 bash scripts/llm_server.sh qwen 3     # the teacher, host
    python -m radar.monitor.drift_quality --label-limit 20     # probe: label at most 20 texts
    python -m radar.monitor.drift_quality                       # label what is missing, then agreement
    python -m radar.monitor.drift_quality --no-teacher          # the cache only (the Airflow task)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import ValidationError

from radar.config import ROOT, load_dotenv
from radar.eval.bootstrap import bootstrap, resample_weights
from radar.eval.cli import SKILL_NORMALIZERS, EvalConfig, load_config, metrics_block
from radar.eval.metrics import Stat, score
from radar.extract.corpus import permanent_random
from radar.extract.inputs import input_for_record
from radar.extract.schema import VacancyFields
from radar.extract.teacher import (
    TEACHERS,
    Item,
    check_model,
    labeled,
    request_body,
    request_schema,
    sha256_text,
)
from radar.monitor.drift import BY_NAME, REFERENCE_WEEKS, REPORTS, Windows, source_windows
from radar.monitor.drift_output import OutputReader, extraction_model
from radar.results import write_result

log = logging.getLogger("radar.monitor.drift_quality")

RESULT = "monitor/extraction_quality.json"
CACHE = REPORTS / "teacher" / "qwen.jsonl"
TEACHER_URL = "http://127.0.0.1:8096/v1"
TEACHER = TEACHERS["qwen"]
GROUPS = {"portals": ("trudvsem_snapshot", "trudvsem", "habr", "superjob"), "telegram": ("telegram",)}
WINDOWS = ("reference", "current")
SAMPLE_SIZE = 100
SAMPLE_SEED = 20261014


@dataclass(frozen=True)
class Identity:
    """The teacher as it labeled a family: model, prompt and request schema."""

    split: str
    prompt_sha256: str
    schema_sha256: str
    model: str = TEACHER.alias


IDENTITIES = {
    "portals": Identity(
        "train_pool",
        "5937b5a6fdf2adeb28ed2c8541631ad7cf602fe19efa46fc9e5db06e650f8728",
        "f71af6bb76c28c170911aaa71363fe791142357ef6110d50a53891500a139d12",
    ),
    "telegram": Identity(
        "tg_train_pool",
        "a41ddbf4b47b6172b5be5e12876c718d84614d9db72efe5ab243ec8afd687a62",
        "fd1e39e240e6de5a57ce16aca32a728bac53be93338885994c4911caafa21196",
    ),
}

# Per-field agreement that goes to the flat metrics; the whole block is under `groups`.
REPORTED = (
    "valid_json",
    "macro_f1",
    "role_accuracy",
    "grade_accuracy",
    "work_format_accuracy",
    "employment_type_accuracy",
    "lang_accuracy",
    "salary_accuracy",
    "experience_years_accuracy",
    "skills_f1",
    "cities_f1",
)

Key = tuple[str, str, str]


def schema_sha256(split: str) -> str:
    return sha256_text(json.dumps(request_schema(split), sort_keys=True))


def check_identities() -> None:
    """The request schema is code: if it moved, the cache and the pinned teacher no longer match."""
    for group, identity in IDENTITIES.items():
        if schema_sha256(identity.split) != identity.schema_sha256:
            raise SystemExit(f"{group}: the request schema of {identity.split} changed since the teachers labeled it")


# --- sampling -----------------------------------------------------------------------------------


def draw(keys: Iterable[Key], n: int, seed: int = SAMPLE_SEED) -> list[Key]:
    """The n keys with the smallest permanent_random of source:source_id, in that order."""
    return sorted(keys, key=lambda k: (permanent_random(seed, f"{k[0]}:{k[1]}"), k))[:n]


def candidates(units: Any, weeks: Sequence[Any]) -> list[Key]:
    """Extracted units of these weeks, duplicates left out, as (source, source_id, content_hash)."""
    chosen = units[units["is_extracted"] & ~units["is_duplicate"] & units["week"].isin(list(weeks))]
    return [
        (str(s), str(i), str(h))
        for s, i, h in zip(chosen["source"], chosen["source_id"], chosen["content_hash"], strict=True)
    ]


# --- the student's and the teacher's answers ----------------------------------------------------


def student_fields(row: Mapping[str, Any]) -> VacancyFields | None:
    """VacancyFields of a row of extract.vacancy_fields; None for an invalid answer."""
    if not row["valid"]:
        return None
    salary = None
    if row["salary_from"] is not None or row["salary_to"] is not None:
        salary = {
            "from": row["salary_from"],
            "to": row["salary_to"],
            "currency": row["salary_currency"],
            "gross_net": row["salary_gross_net"],
            "period": row["salary_period"],
        }
    experience = None
    if row["experience_min"] is not None or row["experience_max"] is not None:
        experience = {"min": row["experience_min"], "max": row["experience_max"]}
    return VacancyFields.model_validate(
        {
            "role": row["role"],
            "grade": row["grade"],
            "skills": list(row["skills"]),
            "salary": salary,
            "work_format": row["work_format"],
            "cities": list(row["cities"]),
            "experience_years": experience,
            "employment_type": row["employment_type"],
            "lang": row["lang"],
        }
    )


def cache_key(input_sha256: str, identity: Identity) -> tuple[str, str, str, str]:
    return (input_sha256, identity.model, identity.prompt_sha256, identity.schema_sha256)


def read_cache(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """The latest answered row per input and teacher identity; failed requests are not answers."""
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is None:
                out[(row["input_sha256"], row["model"], row["prompt_sha256"], row["schema_sha256"])] = row
    return out


def teacher_fields(row: Mapping[str, Any]) -> VacancyFields | None:
    if not row["valid"] or row["output"] is None:
        return None
    try:
        return VacancyFields.model_validate_json(row["output"])
    except ValidationError:
        return None


# --- agreement ----------------------------------------------------------------------------------


def window_stats(
    pairs: Sequence[tuple[Mapping[str, Any], VacancyFields | None]], cfg: EvalConfig
) -> tuple[dict[str, Stat] | None, dict[str, Any]]:
    """Field statistics of the student against the teacher on one window's sample, and the counts
    of what was left out. `pairs`: (cache row, student fields or None for an invalid answer)."""
    counts: Counter[str] = Counter()
    teacher, student = [], []
    for row, fields in pairs:
        if row.get("mismatch"):
            counts["input_mismatch"] += 1
            continue
        if row.get("exclude"):
            counts[f"teacher_excluded_{row['exclude']}"] += 1
            continue
        reference = teacher_fields(row)
        if reference is None:
            counts["teacher_invalid"] += 1
            continue
        teacher.append(reference)
        student.append(fields)
    summary = {"sample": len(pairs), **dict(sorted(counts.items())), "scored": len(teacher)}
    if not teacher:
        return None, summary
    stats = score(
        teacher,
        student,
        cfg.salary_rel_tol,
        macro_fields=cfg.macro_fields,
        normalize_skill=SKILL_NORMALIZERS[cfg.skill_normalization],
    )
    return stats, summary


def intervals(stats: Mapping[str, Stat], cfg: EvalConfig) -> dict[str, dict[str, Any]]:
    n = next(iter(stats.values())).columns.shape[0]
    return metrics_block(stats, bootstrap(stats, resample_weights(n, cfg.n_resamples, cfg.seed), cfg.confidence))


def difference(
    current: Mapping[str, Stat], reference: Mapping[str, Stat], cfg: EvalConfig, names: Sequence[str] = REPORTED
) -> dict[str, dict[str, float]]:
    """current minus reference with a percentile interval; the samples are independent, so each is
    resampled on its own."""
    w_cur = resample_weights(current[names[0]].columns.shape[0], cfg.n_resamples, cfg.seed)
    w_ref = resample_weights(reference[names[0]].columns.shape[0], cfg.n_resamples, cfg.seed + 1)
    tail = (1 - cfg.confidence) / 2 * 100
    out = {}
    for name in names:
        if name not in current or name not in reference:
            continue
        point = current[name].value() - reference[name].value()
        samples = current[name].fn(w_cur @ current[name].columns) - reference[name].fn(w_ref @ reference[name].columns)
        samples = samples[~np.isnan(samples)]
        if np.isnan(point) or samples.size == 0:
            continue
        low, high = np.percentile(samples, [tail, 100 - tail])
        out[name] = {"value": round(point, 6), "ci_low": round(float(low), 6), "ci_high": round(float(high), 6)}
    return out


# --- ClickHouse ---------------------------------------------------------------------------------


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _tuples(keys: Iterable[Key]) -> str:
    return ", ".join(f"({_quote(s)}, {_quote(i)}, {_quote(h)})" for s, i, h in keys)


_STUDENT_SQL = """
SELECT
    source, source_id, content_hash, toString(dt) AS day, input_sha256, valid,
    CAST(role AS Nullable(String)) AS role,
    CAST(grade AS Nullable(String)) AS grade,
    skills,
    salary_from, salary_to,
    CAST(salary_currency AS Nullable(String)) AS salary_currency,
    CAST(salary_gross_net AS Nullable(String)) AS salary_gross_net,
    CAST(salary_period AS Nullable(String)) AS salary_period,
    CAST(work_format AS Nullable(String)) AS work_format,
    cities, experience_min, experience_max,
    CAST(employment_type AS Nullable(String)) AS employment_type,
    CAST(lang AS Nullable(String)) AS lang
FROM extract.vacancy_fields FINAL
WHERE model_id = {model:String} AND (source, source_id, content_hash) IN (%(KEYS)s)
SETTINGS output_format_json_quote_64bit_integers = 0
FORMAT JSONEachRow
"""

_RAW_SQL = """
SELECT source, source_id, toString(dt) AS day, content_hash, title, text, payload
FROM raw.vacancies
WHERE (source, source_id, dt, content_hash) IN (%(KEYS)s)
ORDER BY fetched_at DESC
LIMIT 1 BY source, source_id, dt, content_hash
FORMAT JSONEachRow
"""


class QualityReader(OutputReader):
    def _json(self, sql: str, **params: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.ch.query(sql, **params).splitlines() if line]

    def student_rows(self, model: str, keys: Sequence[Key]) -> dict[Key, dict[str, Any]]:
        rows = self._json(_STUDENT_SQL % {"KEYS": _tuples(keys)}, model=model)
        return {(r["source"], r["source_id"], r["content_hash"]): r for r in rows}

    def inputs(self, rows: Sequence[Mapping[str, Any]]) -> dict[Key, str]:
        """The extractor input of each version, rebuilt from the raw layer as the corpus run built it."""
        literal = ", ".join(
            f"({_quote(r['source'])}, {_quote(r['source_id'])}, toDate({_quote(r['day'])}), "
            f"{_quote(r['content_hash'])})"
            for r in rows
        )
        out = {}
        for raw in self._json(_RAW_SQL % {"KEYS": literal}):
            record = {"source": raw["source"], "title": raw["title"], "text": raw["text"], "payload": raw["payload"]}
            if raw["source"] == "telegram":
                record["title"] = None
            out[(raw["source"], raw["source_id"], raw["content_hash"])] = input_for_record(record)
        return out


# --- teacher ------------------------------------------------------------------------------------


def teacher_client(url: str) -> httpx.Client | None:
    """A client of the teacher server, or None when nothing answers at `url`."""
    client = httpx.Client(base_url=url, timeout=httpx.Timeout(600.0, connect=5.0))
    try:
        check_model(client, TEACHER)
    except httpx.TransportError as e:
        client.close()
        log.warning("no teacher at %s (%s): agreement from the cache only", url, type(e).__name__)
        return None
    return client


def teacher_prompt(identity: Identity) -> str:
    # radar.eval.final takes the prompt version out of git and needs psutil, both host-only
    from radar.eval.final import teacher_prompt as prompt_file

    text = prompt_file(identity.prompt_sha256).read_text(encoding="utf-8")
    if sha256_text(text) != identity.prompt_sha256:
        raise SystemExit(f"the prompt of {identity.split} does not hash to {identity.prompt_sha256}")
    return text


def cache_row(identity: Identity, input_sha256: str, unit: str, answer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_sha256": input_sha256,
        "model": identity.model,
        "prompt_sha256": identity.prompt_sha256,
        "schema_sha256": identity.schema_sha256,
        "unit": unit,
        "output": answer.get("output"),
        "valid": bool(answer.get("valid")),
        "exclude": answer.get("exclude"),
        "mismatch": bool(answer.get("mismatch", False)),
        "finish_reason": answer.get("finish_reason"),
        "seconds": answer.get("seconds"),
        "error": answer.get("error"),
        "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def label_missing(
    client: httpx.Client,
    reader: QualityReader,
    todo: Sequence[tuple[Identity, Mapping[str, Any]]],
    cache_path: Path,
    workers: int,
) -> dict[str, Any]:
    """Teacher answers for (identity, student row) pairs, appended to the cache line by line."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = reader.inputs([row for _, row in todo])
    prompts = {identity: teacher_prompt(identity) for identity in {i for i, _ in todo}}
    t0, done, mismatched = time.perf_counter(), 0, 0
    with cache_path.open("a", encoding="utf-8", newline="\n") as f:
        bodies = []
        units = {}
        for identity, row in todo:
            key = (row["source"], row["source_id"], row["content_hash"])
            unit = f"{key[0]}:{key[1]}"
            text = inputs.get(key)
            if text is None or sha256_text(text) != row["input_sha256"]:
                mismatched += 1
                f.write(json.dumps(cache_row(identity, row["input_sha256"], unit, {"mismatch": True})) + "\n")
                continue
            schema = request_schema(identity.split)
            item = Item(f"{identity.split}|{row['input_sha256']}", identity.split, text)
            units[item.id] = (identity, unit)
            bodies.append((item, request_body(TEACHER, prompts[identity], text, schema)))
        for answer in labeled(client, bodies, workers):
            identity, unit = units[answer["id"]]
            f.write(json.dumps(cache_row(identity, answer["input_sha256"], unit, answer), ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 20 == 0 or done == len(bodies):
                elapsed = time.perf_counter() - t0
                log.info(
                    "teacher %d of %d, %.2f s per text, %.1f min left",
                    done,
                    len(bodies),
                    elapsed / done,
                    (len(bodies) - done) * elapsed / done / 60,
                )
    return {"labeled": done, "input_mismatch": mismatched, "seconds": round(time.perf_counter() - t0, 1)}


# --- the run ------------------------------------------------------------------------------------


def samples(
    reader: QualityReader, reference_weeks: int, size: int
) -> tuple[dict[str, dict[str, list[Key]]], dict[str, dict[str, Any]]]:
    """The drawn keys per group and window, and the windows of each source."""
    drawn: dict[str, dict[str, list[Key]]] = {}
    windows: dict[str, dict[str, Any]] = {}
    for group, names in GROUPS.items():
        pool: dict[str, list[Key]] = {w: [] for w in WINDOWS}
        for name in names:
            source = BY_NAME[name]
            win = source_windows(source, reader, reference_weeks)
            if not isinstance(win, Windows):
                windows[name] = {"status": win["status"]}
                continue
            units = reader.outputs(source, win.reference[0], win.last_day)
            units["source"] = name
            windows[name] = {
                "status": "ok",
                "reference_weeks": [w.isoformat() for w in win.reference],
                "current_week": win.current.isoformat(),
            }
            pool["reference"] += candidates(units, win.reference)
            pool["current"] += candidates(units, [win.current])
        drawn[group] = {w: draw(pool[w], size) for w in WINDOWS}
        for w in WINDOWS:
            log.info("%s %s: %d of %d extracted units drawn", group, w, len(drawn[group][w]), len(pool[w]))
    return drawn, windows


def evaluate(
    drawn: Mapping[str, Mapping[str, Sequence[Key]]],
    students: Mapping[Key, Mapping[str, Any]],
    cache: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    cfg: EvalConfig,
    teacher_state: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Flat metrics and the per-group block. A window whose sample the cache does not cover gets a
    status (teacher_unreachable, teacher_skipped, teacher_incomplete) and no agreement at all.
    teacher_state: "answered", "skipped" (--no-teacher) or "unreachable"."""
    metrics: dict[str, dict[str, Any]] = {}
    groups: dict[str, Any] = {}
    for group, by_window in drawn.items():
        identity = IDENTITIES[group]
        groups[group] = {}
        stats_of: dict[str, dict[str, Stat]] = {}
        for w in WINDOWS:
            ks = by_window[w]
            rows = [cache.get(cache_key(students[k]["input_sha256"], identity)) for k in ks]
            by_source = Counter(k[0] for k in ks)
            entry: dict[str, Any] = {"drawn": len(ks), "by_source": dict(sorted(by_source.items()))}
            if not ks:
                groups[group][w] = {"status": "no_window", **entry}
                continue
            covered = sum(r is not None for r in rows)
            if covered < len(ks):
                status = "teacher_incomplete" if teacher_state == "answered" else f"teacher_{teacher_state}"
                groups[group][w] = {
                    "status": status,
                    **entry,
                    "teacher_answers": {"n": covered, "denominator": len(ks)},
                }
                continue
            pairs = [(r, student_fields(students[k])) for r, k in zip(rows, ks, strict=True) if r is not None]
            stats, counts = window_stats(pairs, cfg)
            if stats is None:
                groups[group][w] = {"status": "nothing_scored", **entry, "counts": counts}
                continue
            stats_of[w] = stats
            block = intervals(stats, cfg)
            groups[group][w] = {"status": "ok", **entry, "counts": counts, "agreement": block}
            for name in REPORTED:
                if name in block:
                    metrics[f"{group}.{w}.{name}"] = block[name]
        if len(stats_of) == len(WINDOWS):
            groups[group]["current_minus_reference"] = difference(stats_of["current"], stats_of["reference"], cfg)
    return metrics, groups


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--teacher-url", default=os.environ.get("RADAR_TEACHER_URL", TEACHER_URL))
    parser.add_argument("--no-teacher", action="store_true", help="agreement from the cache only")
    parser.add_argument("--label-limit", type=int, help="label at most this many texts (a probe)")
    parser.add_argument("--workers", type=int, default=3, help="requests in flight, one per server slot")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--reference-weeks", type=int, default=REFERENCE_WEEKS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    check_identities()
    model = extraction_model()
    settings = {
        "sample_size": args.sample_size,
        "sample_seed": SAMPLE_SEED,
        "reference_weeks": args.reference_weeks,
        "teacher": {
            g: {"model": i.model, "split": i.split, "prompt_sha256": i.prompt_sha256} for g, i in IDENTITIES.items()
        },
    }

    cfg = load_config()
    settings["eval"] = {
        "n_resamples": cfg.n_resamples,
        "seed": cfg.seed,
        "confidence": cfg.confidence,
        "salary_rel_tol": cfg.salary_rel_tol,
        "skill_normalization": cfg.skill_normalization,
    }

    reader = QualityReader()
    teacher_run: dict[str, Any] | None = None
    try:
        drawn, windows = samples(reader, args.reference_weeks, args.sample_size)
        keys = [k for group in drawn.values() for w in WINDOWS for k in group[w]]
        students = reader.student_rows(model, keys) if keys else {}
        lost = [k for k in keys if k not in students]
        if lost:
            raise RuntimeError(f"{len(lost)} drawn versions have no row of {model}, e.g. {lost[0]}")

        cache = read_cache(args.cache)
        todo: dict[tuple[str, str, str, str], tuple[Identity, dict[str, Any]]] = {}
        for group, by_window in drawn.items():
            identity = IDENTITIES[group]
            for w in WINDOWS:
                for k in by_window[w]:
                    ck = cache_key(students[k]["input_sha256"], identity)
                    if ck not in cache:
                        todo.setdefault(ck, (identity, students[k]))
        teacher_state = "skipped" if args.no_teacher else "unreachable"
        if todo and not args.no_teacher:
            client = teacher_client(args.teacher_url)
            if client is not None:
                teacher_state = "answered"
                chosen = list(todo.values())[: args.label_limit]
                log.info("teacher: %d texts to label of %d missing", len(chosen), len(todo))
                try:
                    teacher_run = label_missing(client, reader, chosen, args.cache, args.workers)
                finally:
                    client.close()
                cache = read_cache(args.cache)
    finally:
        reader.close()

    metrics, groups = evaluate(drawn, students, cache, cfg, teacher_state)

    path = write_result(
        RESULT,
        metrics,
        source_script="src/radar/monitor/drift_quality.py",
        extraction_model=model,
        settings=settings,
        method="student (extract.vacancy_fields, the marts' model_id) against the teacher's answer on the same "
        "input; <field>_accuracy is exact agreement over the scored sample, skills and cities F1 of the student "
        "against the teacher, macro_f1 over configs/eval.toml fields; teacher-invalid answers, posts the teacher "
        "excludes and inputs that no longer hash to the student's are left out and counted",
        windows=windows,
        groups=groups,
        teacher_run=teacher_run,
    )
    for group, by_window in groups.items():
        for w in WINDOWS:
            res = by_window[w]
            if res["status"] != "ok":
                print(f"{group:9} {w:9} {res['status']}")
                continue
            block, counts = res["agreement"], res["counts"]
            shown = ", ".join(
                f"{n.removesuffix('_accuracy')} {block[n]['n']}/{block[n]['denominator']}"
                for n in REPORTED
                if n.endswith("_accuracy") and n in block
            )
            print(
                f"{group:9} {w:9} scored {counts['scored']} of {counts['sample']}: "
                f"macro F1 {block['macro_f1']['value']:.3f}; {shown}"
            )
    print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
