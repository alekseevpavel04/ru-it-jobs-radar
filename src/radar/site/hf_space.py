"""The Space: a static page that runs the extraction and the salary model in the visitor's browser.

    python -m radar.site.hf_space pick        20 trudvsem texts outside the gold set -> data/hf/space_bench.jsonl,
                                               3 short ones as the page's examples -> data/hf/space_examples.json
    python -m radar.site.hf_space assemble    storage/hf/space: the page, the pinned browser libraries, the
                                               tokenizer, the salary trees, assets/space.json, the card
    python -m radar.site.hf_space check       the page's JavaScript against the Python it repeats, in node
    python -m radar.site.hf_space bench       check, then the assembled Space in Chromium from a local server:
                                               model load, 10 extractions -> results/public/space_static_local.json,
                                               screenshots docs/img/space_<width>.png

Why static. A Space with its own server (Docker, Gradio) on the free CPU tier now needs a paid
account; a static Space is free. The page loads the Q4_K_M GGUF of the student from the model repo
into wllama (llama.cpp compiled to WebAssembly) and keeps it in the browser's storage, so the
visitor's processor does the work and the text never leaves the browser.

What the browser repeats, and how it is checked (`check`, in node with the same files the page loads):
- the prompt: radar.train.dataset cuts the input after max_input_tokens tokens of the training
  tokenizer and wraps it in the system prompt and the Qwen3 chat template with thinking off; the page
  does it with tokenizers.js (js/prompt.js) and must render the dev prompts the quantized student was
  scored on, character for character, and cut texts where the Python tokenizer cuts them;
- the answer check of radar.extract.schema (js/schema.js), on gold answers and broken variants of them;
- the salary model: radar.models.salary_features.feature_row and the HistGradientBoosting trees of
  radar.models.salary, exported as plain trees (`salary_trees`), on rows of the salary dev split and on
  the gold dev fields. sklearn's ONNX converter does not take native categorical splits, which the
  model uses for seven columns, hence the trees and not ONNX.
Decoding is greedy with the extraction JSON schema as the grammar, as the student was scored on dev
(sft_1p7b_clean_q4_k_m_schema); wllama 3 runs llama.cpp's server code, which turns `json_schema`
into a grammar the same way llama-server does.

Texts. Only trudvsem texts are publishable (DECISIONS 2), and only those not reposted from other
sites (source_type "Вакансия интернет ресурса"). The local run uses IT candidates of the latest
snapshot, in a fixed hash order, whose vacancy id is not in any gold split, and that the corpus
extraction (6.7) already answered and read as IT: the page's answer is compared with it.
"""

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar.config import ROOT, data_dir, load_dotenv
from radar.extract.inputs import input_for_record
from radar.extract.schema import dump, json_schema, try_parse
from radar.models.salary import ROUND_RUB, SalaryModel, load_model, predict
from radar.models.salary_data import DEV, STUDENT, records_of, with_features
from radar.models.salary_features import CATEGORICAL, NUMERIC, ROLE_BUCKET, ROLE_GROUP, OutOfScope
from radar.results import write_result
from radar.site.hf_common import HF, LICENSES, STORAGE, load_config, render_card, results_doc, shown, write_json
from radar.stream.clickhouse import ClickHouse

log = logging.getLogger("radar.site.hf_space")

SPACE_SRC = HF / "space"
OUT = STORAGE / "space"
BENCH = data_dir() / "hf" / "space_bench.jsonl"
EXAMPLES = data_dir() / "hf" / "space_examples.json"
WORK = ROOT / "logs" / "hf_space"
RESULT = "public/space_static_local.json"
NODE = Path("D:/VScode_projects/assets/tools/nodejs/node.exe")
CHECK_SCRIPT = ROOT / "scripts" / "hf_space_check.mjs"
BENCH_SCRIPT = ROOT / "scripts" / "hf_space_bench.mjs"
# What the Space repo holds from hf/space; package.json pins the vendored libraries and is not published.
PAGE_FILES = ("index.html", "NOTICE.md")
PAGE_DIRS = ("js",)
# Browser libraries, from hf/space/node_modules at the versions of hf/space/package.json.
VENDOR = {
    "wllama": ("@wllama/wllama", ("esm/index.js", "esm/wasm/wllama.wasm", "LICENCE")),
    "tokenizers": ("@huggingface/tokenizers", ("dist/tokenizers.min.mjs", "LICENSE")),
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
SALARY_MODELS = ("point", "lower", "upper", "bound_from", "bound_to")
REPOSTED = "Вакансия интернет ресурса"
N_BENCH = 20
N_EXAMPLES = 3
EXAMPLE_MAX_CHARS = 1800
MAX_NEW_TOKENS = 512
N_CTX = 4096
# Budgets the input cut is checked at besides max_input_tokens: few dev texts reach the real one.
CUT_BUDGETS = (32, 128, 512)
N_SALARY_ROWS = 50


def gguf_repo_path(name: str, quant: str) -> str:
    """Where a GGUF lies in the model repo (radar.site.hf_model puts it there)."""
    return f"gguf/{name}-{quant}.gguf"


def gguf_entry(name: str, quant: str) -> dict[str, Any]:
    manifest = json.loads((ROOT / "models" / "gguf" / name / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["type"] == quant:
            return {**entry, "manifest_created_at": manifest["created_at"]}
    raise KeyError(f"models/gguf/{name}/manifest.json has no {quant}")


def tokenizer_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / "models" / "merged" / str(cfg["model"]["name"])


def dataset_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((tokenizer_dir(cfg) / "dataset_meta.json").read_text(encoding="utf-8"))
    return meta


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


# --- salary trees -------------------------------------------------------------------------------


def _in_bitset(bitsets: np.ndarray, row: int, value: int) -> bool:
    return bool((int(bitsets[row, value // 32]) >> (value % 32)) & 1)


def hgb_trees(hgb: Any, n_levels: Sequence[int]) -> dict[str, Any]:
    """A fitted HistGradientBoostingRegressor with identity link as plain trees over the columns of
    Encoder.tree_matrix (js/salary.js walks them).

    sklearn routes a categorical value through its own OrdinalEncoder, then left if the category is in
    the node's bitset, right if the model saw it in training, else as missing. That decision depends on
    the encoder level only, so it is resolved here into the list of level codes that go left."""
    pre = hgb._preprocessor
    columns = {name: np.flatnonzero(cols) for name, _, cols in pre.transformers_ if name in pre.output_indices_}
    original = np.empty(pre.n_features_in_, dtype=int)
    for name, out in pre.output_indices_.items():
        original[out] = columns.get(name, np.array([], dtype=int))
    encoder = pre.named_transformers_["encoder"]
    ordinal = {
        int(col): {int(v): i for i, v in enumerate(cats) if not np.isnan(v)}
        for col, cats in zip(columns["encoder"], encoder.categories_, strict=True)
    }
    known, f_idx_map = hgb._bin_mapper.make_known_categories_bitsets()

    def goes_left(node: Any, raw_left: np.ndarray, code: int) -> bool:
        o = ordinal[int(original[node["feature_idx"]])].get(code)
        if o is not None and _in_bitset(raw_left, int(node["bitset_idx"]), o):
            return True
        if o is not None and _in_bitset(known, int(f_idx_map[node["feature_idx"]]), o):
            return False
        return bool(node["missing_go_to_left"])

    trees = []
    for (predictor,) in hgb._predictors:
        nodes = []
        for node in predictor.nodes:
            if node["is_leaf"]:
                nodes.append([float(node["value"])])
                continue
            col = int(original[node["feature_idx"]])
            if node["is_categorical"]:
                split: Any = [c for c in range(n_levels[col]) if goes_left(node, predictor.raw_left_cat_bitsets, c)]
            else:
                # A split that sends every number left has an infinite threshold, which JSON cannot hold.
                threshold = float(node["num_threshold"])
                split = threshold if math.isfinite(threshold) else ("inf" if threshold > 0 else "-inf")
            nodes.append([col, split, int(node["missing_go_to_left"]), int(node["left"]), int(node["right"])])
        trees.append(nodes)
    return {"baseline": float(np.ravel(hgb._baseline_prediction)[0]), "trees": trees}


def salary_trees(model: SalaryModel) -> dict[str, Any]:
    """What js/salary.js needs to repeat radar.models.salary.predict: encoder, trees, margin, cells."""
    n_levels = [len(model.encoder.levels[c]) for c in CATEGORICAL]
    for name in SALARY_MODELS:
        if getattr(model, name)._loss.__class__.__name__ not in ("AbsoluteError", "PinballLoss"):
            raise TypeError(f"{name}: the export assumes an identity link")
    return {
        "version": model.version,
        "trained_at": model.trained_at,
        "level": model.level,
        "margin": model.margin,
        "round_rub": ROUND_RUB,
        "maps": {"role_bucket": ROLE_BUCKET, "role_group": ROLE_GROUP},
        "encoder": {
            "categorical": [[c, list(model.encoder.levels[c])] for c in CATEGORICAL],
            "numeric": list(NUMERIC),
            "skill_vocab": list(model.encoder.skill_vocab),
        },
        "models": {name: hgb_trees(getattr(model, name), n_levels) for name in SALARY_MODELS},
        "n_similar": {"|".join(k): v[0] for k, v in model.cells.tables[0].items()},
    }


# --- assemble -----------------------------------------------------------------------------------


def package_version(package: str) -> str:
    doc = json.loads((SPACE_SRC / "node_modules" / package / "package.json").read_text(encoding="utf-8"))
    pinned = json.loads((SPACE_SRC / "package.json").read_text(encoding="utf-8"))["dependencies"][package]
    if doc["version"] != pinned:
        raise SystemExit(f"{package} {doc['version']} in hf/space/node_modules, {pinned} pinned: run npm ci there")
    return str(doc["version"])


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
    meta = dataset_meta(cfg)
    entry = gguf_entry(name, quant)
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"]
    repos = cfg["repos"]
    gguf_file = gguf_repo_path(name, quant)
    revision = cfg["space"]["gguf_revision"]
    return {
        "model": {
            "name": name,
            "base": cfg["model"]["base"],
            "gguf_type": quant,
            "gguf_sha256": entry["sha256"],
            "gguf_size_bytes": entry["size_bytes"],
            "gguf_size_mib": entry["size_mib"],
            "built_at": entry["manifest_created_at"][:10],
            "prompt_sha256": meta["prompt_sha256"],
        },
        "gguf": {
            "repo": repos["model"],
            "file": gguf_file,
            "revision": revision,
            "url": f"https://huggingface.co/{repos['model']}/resolve/{revision}/{gguf_file}",
        },
        "prompt": {"system_prompt": meta["system_prompt"], "max_input_tokens": meta["max_input_tokens"]},
        "decoding": {"max_new_tokens": MAX_NEW_TOKENS, "n_ctx": N_CTX, "temperature": 0},
        "json_schema": json_schema(),
        "libraries": {pkg: package_version(pkg) for pkg, _ in VENDOR.values()},
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
            "space_direct": cfg["space"]["direct_url"],
        },
        "examples": examples,
        "speed": speed(),
    }


def speed() -> dict[str, Any] | None:
    """The local browser run's numbers, for the page's wait note."""
    path = ROOT / "results" / RESULT
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    m = doc["metrics"]
    return {
        "extract_p50_s": m["extract_p50_s"]["value"],
        "extract_single_thread_p50_s": m["extract_single_thread_p50_s"]["value"],
        "first_extract_s": m["first_extract_s"]["value"],
        "threads": m["threads"]["value"],
        "load_cached_s": m["load_cached_s"]["value"],
        "cpu": doc["setup"]["host_cpu"],
        "measured_at": doc["measured_at"][:10],
        "source": f"results/{RESULT}",
    }


def cmd_assemble() -> int:
    cfg = load_config()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for file in PAGE_FILES:
        shutil.copy2(SPACE_SRC / file, OUT / file)
    for folder in PAGE_DIRS:
        shutil.copytree(SPACE_SRC / folder, OUT / folder)
    shutil.copytree(LICENSES, OUT / "licenses")
    for name, (package, files) in VENDOR.items():
        package_version(package)
        for rel in files:
            dst = OUT / "vendor" / name / Path(rel).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SPACE_SRC / "node_modules" / package / rel, dst)
    for file in TOKENIZER_FILES:
        dst = OUT / "assets" / "tokenizer" / file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tokenizer_dir(cfg) / file, dst)
    salary = load_model(ROOT / cfg["salary"]["artifact"])
    (OUT / "assets" / "salary.json").write_text(
        json.dumps(salary_trees(salary), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    write_json(OUT / "assets" / "space.json", space_info(cfg))
    card = render_card((SPACE_SRC / "README.md.in").read_text(encoding="utf-8"))
    (OUT / "README.md").write_text(card.published, encoding="utf-8", newline="\n")
    log.info("space -> %s (%d numbers in the card, pending %s)", shown(OUT), card.numbers, card.pending or "none")
    return 0


# --- check --------------------------------------------------------------------------------------


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def python_cut(tokenizer: Any, text: str, budget: int) -> str:
    """radar.train.dataset.cap_input with the offsets of the training tokenizer."""
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    return text if len(offsets) <= budget else text[: offsets[budget - 1][1]]


def broken_answers(fields: dict[str, Any]) -> list[str]:
    """Variants of a valid answer that radar.extract.schema rejects, or accepts after a fix-up (stripped
    items, the field name `from_`), so the page's check is compared on both sides of each rule."""
    salary = {"from": 100000, "to": 150000, "currency": "RUB", "gross_net": None, "period": "month"}
    variants: list[dict[str, Any]] = [
        {**fields, "salary": {**salary, "from": 200000}},
        {**fields, "salary": {**salary, "from": None, "to": None}},
        {**fields, "salary": {**salary, "from": 0}},
        {**fields, "salary": {**salary, "from": 1500.5}},
        {**fields, "salary": {**salary, "from": 1500.0}},
        {**fields, "salary": {k if k != "from" else "from_": v for k, v in salary.items()}},
        {**fields, "salary": {**salary, "currency": "rub"}},
        {**fields, "experience_years": {"min": 5, "max": 2}},
        {**fields, "experience_years": {"min": None, "max": None}},
        {**fields, "experience_years": {"min": 51, "max": None}},
        {**fields, "experience_years": {"min": 1.5, "max": 3}},
        {**fields, "grade": "master"},
        {**fields, "lang": None},
        {**fields, "skills": ["  Python  ", "SQL"]},
        {**fields, "skills": [" "]},
        {**fields, "skills": ["x" * 81]},
        {**fields, "cities": [f"c{i}" for i in range(31)]},
        {**fields, "extra": 1},
        {k: v for k, v in fields.items() if k != "lang"},
    ]
    return [json.dumps(v, ensure_ascii=False) for v in variants] + ["", "{", "[]", "null"]


def salary_dev_rows(n: int) -> list[dict[str, Any]]:
    """The first n rows of the salary dev split by entity id, with the student features the model uses."""
    frame = pd.read_parquet(data_dir() / "salary" / "frame.parquet")
    dev = with_features(frame[frame["split"] == DEV].sort_values("entity_id").head(n), STUDENT)
    rows = []
    for r in records_of(dev):
        exp = r["experience_min"]
        rows.append(
            {
                "role_bucket": r["role_bucket"],
                "role_group": r["role_group"],
                "grade": r["grade"],
                "region_group": r["region_group"],
                "work_format": r["work_format"],
                "employment_type": r["employment_type"],
                "source": r["source"],
                "experience_min": None if exp is None or math.isnan(exp) else float(exp),
                "skills": sorted(str(s) for s in r["skills"]),
            }
        )
    return rows


def node(script: Path, *args: str) -> None:
    """Run a node script with its output in logs/hf_space/<script>.log (the browser run takes half an hour)."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{script.stem}.log"
    with path.open("w", encoding="utf-8") as out:
        done = subprocess.run([str(NODE), str(script), *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        tail = path.read_text(encoding="utf-8").splitlines()[-30:]
        raise RuntimeError(f"{script.name}: exit {done.returncode}, {shown(path)}:\n" + "\n".join(tail))


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else None, "n": n, "denominator": d}


def run_check(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The page's prompt rendering, input cut, answer check and salary model against the Python
    they repeat; the JavaScript runs in node on the files the assembled Space serves."""
    from tokenizers import Tokenizer

    gold = data_dir() / "gold"
    prompts = tokenizer_dir(cfg) / "prompts"
    rendered = {s: {r["id"]: r for r in jsonl(prompts / f"{s}.jsonl")} for s in ("dev", "tg_dev")}
    inputs = {s: jsonl(gold / f"{s}_inputs.jsonl") for s in ("dev", "tg_dev")}
    tokenizer = Tokenizer.from_file(str(OUT / "assets" / "tokenizer" / "tokenizer.json"))
    texts = [r for s in ("dev", "tg_dev") for r in inputs[s]]
    cut_cases = [{"id": r["id"], "input": r["input"], "budget": b} for r in texts for b in CUT_BUDGETS]

    gold_fields = [r["fields"] for r in jsonl(gold / "dev.jsonl")]
    answers = []
    for f in gold_fields:
        parsed = try_parse(json.dumps(f, ensure_ascii=False))
        if parsed is None:
            raise ValueError(f"a gold answer does not parse: {f}")
        answers.append(json.dumps(dump(parsed), ensure_ascii=False))
    answers += broken_answers(json.loads(answers[0]))

    model = load_model(ROOT / cfg["salary"]["artifact"])
    rows = salary_dev_rows(N_SALARY_ROWS)
    source = cfg["salary"]["source"]
    cases = {
        "space": str(OUT),
        "prompts": [{"id": r["id"], "input": r["input"]} for r in texts],
        "cuts": cut_cases,
        "answers": answers,
        "salary_rows": rows,
        "salary_fields": [{"fields": f, "source": source} for f in gold_fields],
    }
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        cases_path, out_path = Path(tmp) / "cases.json", Path(tmp) / "out.json"
        cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
        node(CHECK_SCRIPT, str(cases_path), str(out_path))
        js = json.loads(out_path.read_text(encoding="utf-8"))

    metrics: dict[str, dict[str, Any]] = {}
    for split in ("dev", "tg_dev"):
        ids = [r["id"] for r in inputs[split]]
        same = sum(js["prompts"][i]["prompt"] == rendered[split][i]["prompt"] for i in ids)
        flag = sum(js["prompts"][i]["truncated"] == rendered[split][i]["truncated_input"] for i in ids)
        metrics[f"{split}_prompt_render_match"] = share(same, len(ids))
        metrics[f"{split}_prompt_truncated_flag_match"] = share(flag, len(ids))
        metrics[f"{split}_prompts_truncated"] = {"value": sum(rendered[split][i]["truncated_input"] for i in ids)}
    by_text = {r["id"]: r["input"] for r in texts}
    cut_same = sum(
        js["cuts"][k] == python_cut(tokenizer, by_text[c["id"]], c["budget"]) for k, c in enumerate(cut_cases)
    )
    n_tokens = {i: len(tokenizer.encode(t, add_special_tokens=False).ids) for i, t in by_text.items()}
    cut_cut = sum(n_tokens[c["id"]] > c["budget"] for c in cut_cases)
    metrics["input_cut_match"] = share(cut_same, len(cut_cases)) | {"budgets": list(CUT_BUDGETS), "cut": cut_cut}

    expected = [(dump(p) if (p := try_parse(a)) is not None else None) for a in answers]
    same_answers = sum(e == g for e, g in zip(expected, js["answers"], strict=True))
    metrics["answer_check_match"] = share(same_answers, len(answers))
    metrics["answer_check_valid"] = {"value": sum(e is not None for e in expected)}

    X = model.predict_rows(rows)
    keys = ("point", "low", "high", "from", "to")
    raw_diff = max(abs(float(X[k][i]) - js["salary_rows"][i][k]) for i in range(len(rows)) for k in keys)
    metrics["salary_rows_max_abs_diff_rub"] = {"value": raw_diff, "n_rows": len(rows), "outputs": list(keys)}
    want: list[dict[str, Any] | None] = []
    for f in gold_fields:
        try:
            want.append(predict(f, source, model=model))
        except OutOfScope:
            want.append(None)
    got = js["salary_fields"]
    metrics["salary_fields_match"] = share(sum(w == g for w, g in zip(want, got, strict=True)), len(want))
    metrics["salary_fields_out_of_scope"] = {"value": sum(w is None for w in want)}
    return metrics


def cmd_check() -> int:
    for name, m in run_check(load_config()).items():
        log.info("%s %s", name, {k: v for k, v in m.items() if k in ("value", "n", "denominator")})
    return 0


# --- bench --------------------------------------------------------------------------------------


def windows_cpu_load(samples: int = 5) -> float:
    """Mean processor load of the host over `samples` seconds, percent (Win32_Processor.LoadPercentage;
    the performance counters are not readable on this machine)."""
    script = (
        f"$v = 1..{samples} | ForEach-Object {{ (Get-CimInstance Win32_Processor).LoadPercentage; Start-Sleep 1 }}; "
        "[string]($v | Measure-Object -Average).Average"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=True)
    return round(float(out.stdout.strip().replace(",", ".")), 1)


def quantile(values: Sequence[float], q: float) -> float:
    from radar.extract.student_server import quantile as nearest_rank

    return nearest_rank(values, q)


def timed(values: Sequence[dict[str, Any]], q: float, pick: Callable[[dict[str, Any]], float]) -> float:
    return round(quantile([pick(v) for v in values], q), 2)


def cmd_bench(n_texts: int, n_single: int, reuse: bool) -> int:
    cfg = load_config()
    if not (OUT / "assets" / "space.json").exists():
        raise SystemExit("assemble the Space first: python -m radar.site.hf_space assemble")
    checks = run_check(cfg)
    name, quant = cfg["model"]["name"], cfg["model"]["space_gguf"]
    gguf = ROOT / "models" / "gguf" / name / f"{name}-{quant}.gguf"
    rows = jsonl(BENCH)[:n_texts]
    WORK.mkdir(parents=True, exist_ok=True)
    texts_path, raw_path = WORK / "bench_texts.json", WORK / "bench_raw.json"
    texts_path.write_text(json.dumps([{"id": r["id"], "input": r["input"]} for r in rows], ensure_ascii=False),
                          encoding="utf-8")  # fmt: skip
    if not reuse:
        load_before = windows_cpu_load()
        node(BENCH_SCRIPT, "--space", str(OUT), "--gguf", str(gguf), "--texts", str(texts_path), "--out",
             str(raw_path), "--single", str(n_single), "--shots", str(ROOT / "docs" / "img"))  # fmt: skip
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["host_cpu_load_pct"] = {"before": load_before, "after": windows_cpu_load()}
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    load_before, load_after = raw["host_cpu_load_pct"]["before"], raw["host_cpu_load_pct"]["after"]

    by_id = {r["id"]: r for r in rows}
    multi, single = raw["multi"], raw["single"]
    n = len(multi)
    valid = sum(a["valid"] for a in multi)
    same = sum(a["valid"] and a["fields"] == by_id[a["id"]]["corpus_fields"] for a in multi)
    with_salary = sum(a["salary"] is not None for a in multi)
    single_same = sum(a["fields"] == m["fields"] for a, m in zip(single, multi, strict=False))

    def wall(a: dict[str, Any]) -> float:
        return float(a["wall_s"])

    metrics: dict[str, dict[str, Any]] = {
        "load_cold_s": {"value": round(raw["load"]["cold_s"], 1)},
        "load_cached_s": {"value": round(raw["load"]["cached_s"], 1)},
        "first_extract_s": {"value": round(raw["first"]["wall_s"], 1)},
        "first_extract_single_thread_s": {"value": round(raw["first_single"]["wall_s"], 1)},
        "extract_p50_s": {"value": timed(multi, 0.5, wall)},
        "extract_p95_s": {"value": timed(multi, 0.95, wall)},
        "extract_max_s": {"value": round(max(wall(a) for a in multi), 2)},
        "prompt_p50_s": {"value": timed(multi, 0.5, lambda a: a["prompt_ms"] / 1000)},
        "generate_p50_s": {"value": timed(multi, 0.5, lambda a: a["predicted_ms"] / 1000)},
        "prompt_tokens_per_s_p50": {"value": timed(multi, 0.5, lambda a: a["prompt_n"] / a["prompt_ms"] * 1000)},
        "generate_tokens_per_s_p50": {
            "value": timed(multi, 0.5, lambda a: a["predicted_n"] / a["predicted_ms"] * 1000)
        },
        "threads": {"value": raw["threads"]},
        "extract_single_thread_p50_s": {"value": timed(single, 0.5, wall), "texts": len(single)},
        "single_thread_same_fields": share(single_same, len(single)),
        "valid_json": share(valid, n),
        "salary_estimates": share(with_salary, n),
        "same_fields_as_corpus_awq": share(same, n),
        "grammar_enforced": {"value": raw["grammar_enforced"]},
        "host_cpu_load_pct": {"value": max(load_before, load_after), "before": load_before, "after": load_after},
        **checks,
    }
    write_result(
        RESULT,
        metrics,
        source_script="src/radar/site/hf_space.py",
        setup={
            "browser": raw["browser"],
            "host_cpu": raw["host_cpu"],
            "logical_cpus": raw["logical_cpus"],
            "wllama": raw["wllama"],
            "libllama": raw["libllama"],
            "gguf": f"models/gguf/{name}/{name}-{quant}.gguf, served by a local HTTP server in place of the model repo",
            "load_cold": "the page's startModel (what the load button calls) in a fresh browser profile, to the model "
            "ready: the GGUF read from the local server into the browser storage, then into WebAssembly",
            "load_cached": "a press of the load button in a new page of the same profile: the GGUF read from the "
            "browser storage",
            "extract": "wall time of the page's run function (what the run button calls) from the call to the fields "
            "and the salary on the page, one text at a time; prompt and generate are llama.cpp's own timings",
            "first_extract": "the page's first example right after the load, not in the quantiles: it also reads the "
            "system prompt, which later texts find in the KV cache (cache_prompt)",
            "single_thread": "the page served without the cross-origin isolation headers, as when the browser refuses "
            "them: wllama falls back to one thread",
            "texts": "trudvsem snapshot IT candidates outside the gold set (python -m radar.site.hf_space pick); "
            "same_fields_as_corpus_awq compares with the 6.7 corpus answer of the AWQ checkpoint of the same model",
            "host_cpu_load_pct": "Windows processor load over 5 s before and after the browser run; other jobs ran on "
            "the machine (the project's services in WSL, the corpus extraction with vLLM, the editor, a browser)",
            "checks": "python -m radar.site.hf_space check: the page's JavaScript in node against its Python",
        },
        per_text=[
            {k: a[k] for k in ("id", "wall_s", "prompt_ms", "predicted_ms", "prompt_n", "predicted_n", "valid")}
            | {"same_fields_as_corpus": a["valid"] and a["fields"] == by_id[a["id"]]["corpus_fields"]}
            for a in multi
        ],
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    log.info("load %.1f s cold, %.1f s cached; extract p50 %.1f s (%d threads), one thread p50 %.1f s -> results/%s",
             raw["load"]["cold_s"], raw["load"]["cached_s"], metrics["extract_p50_s"]["value"], raw["threads"],
             metrics["extract_single_thread_p50_s"]["value"], RESULT)  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.site.hf_space", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick", help="texts for the local run and the page's examples (ClickHouse)")
    pick.add_argument("--model-id", help="corpus extraction to compare with; default [dataset] model_id of hf.toml")
    sub.add_parser("assemble", help="lay out storage/hf/space")
    sub.add_parser("check", help="the page's JavaScript against the Python it repeats (node)")
    bench = sub.add_parser("bench", help="check, then the Space in Chromium from a local server")
    bench.add_argument("--texts", type=int, default=10, help="texts timed with all threads")
    bench.add_argument("--single", type=int, default=5, help="of them, timed again with one thread")
    bench.add_argument(
        "--reuse", action="store_true", help="the result from logs/hf_space/bench_raw.json, no browser run"
    )
    args = ap.parse_args(argv)
    if args.command == "pick":
        return cmd_pick(args.model_id or load_config()["dataset"]["model_id"])
    if args.command == "assemble":
        return cmd_assemble()
    if args.command == "check":
        return cmd_check()
    return cmd_bench(args.texts, args.single, args.reuse)


if __name__ == "__main__":
    sys.exit(main())
