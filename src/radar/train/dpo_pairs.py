"""Preference pairs for one DPO iteration on the student.

Every pair is built on a row of data/sft (radar.train.dataset): a silver label of a train_pool
group, with the system prompt and the input cut as the student saw them. The chosen answer is
always the silver label in the training format. The rejected answer is one of two kinds:

- student: the SFT student's own greedy answer on that input, where it differs from the label on
  role, work_format, grade or employment_type, or is not valid JSON. The student answers a sample
  of SFT train rows (half per source family) and every SFT val row: val groups were held out of SFT,
  so there the student makes the mistakes it makes on unseen vacancies.
- corrupted: a copy of the label with one defect, on SFT train rows that gave no student pair:
  work_format swapped for another value (null included), role swapped for a neighbouring role,
  one required key dropped, or the JSON cut short. Enum swaps keep the rest of the answer byte for
  byte, so the pair differs in that one value.

A silver label can be wrong where the student is right; such a pair pushes towards the label, as
SFT did. No dev or test record is used: build refuses a pair whose record id or input text (after
whitespace and case folding) is also among data/gold/*_inputs.jsonl.

Pairs of a share of groups (seeded hash, independent of the SFT val split) go to the DPO eval
file; the rest to the train file.

    python -m radar.train.dpo_pairs inputs --config configs/dpo/qwen3_1p7b_clean.toml
    bash scripts/train_run.sh predict --adapter models/sft/qwen3_1p7b_r16_clean/adapter \
        --inputs data/dpo/student_inputs.jsonl --system sft_1p7b_r16_clean --out data/dpo/student_answers.jsonl
    python -m radar.train.dpo_pairs build --config configs/dpo/qwen3_1p7b_clean.toml

Output of build: data/dpo/{train,val}.jsonl ({"id", "record_id", "kind", "family", "group", "part",
"fields", "user", "chosen", "rejected"}), data/dpo/meta.json (hashes of every input file and of the
outputs) and results/dpo/pairs.json (counts by kind, family and part; how often the student
disagrees with the label, per field).
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar.config import ROOT, data_dir
from radar.extract.schema import try_parse
from radar.results import git_commit, write_result
from radar.train.config import CORRUPTIONS, DpoConfig, Pairs, load_dpo
from radar.train.dataset import FAMILIES, read_jsonl, sha256_file, write_jsonl

log = logging.getLogger("radar.train.dpo_pairs")

COMPARED = ("role", "work_format", "grade", "employment_type")
WORK_FORMATS: tuple[str | None, ...] = ("office", "hybrid", "remote", None)
GOLD_SPLITS = ("dev", "test", "tg_dev", "tg_test")

# Roles a vacancy is plausibly confused with: a swap to any role would teach the student to avoid
# answers it never gives (designer for an ML vacancy).
ROLE_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "ml_engineer": ("data_scientist", "mlops", "backend"),
    "data_scientist": ("ml_engineer", "data_analyst"),
    "mlops": ("ml_engineer", "devops", "data_engineer"),
    "data_engineer": ("data_analyst", "backend", "mlops"),
    "data_analyst": ("data_scientist", "business_analyst", "data_engineer"),
    "business_analyst": ("system_analyst", "data_analyst", "manager"),
    "system_analyst": ("business_analyst", "data_analyst"),
    "backend": ("fullstack", "software_engineer", "data_engineer"),
    "frontend": ("fullstack", "designer", "mobile"),
    "fullstack": ("backend", "frontend"),
    "mobile": ("frontend", "software_engineer"),
    "software_engineer": ("backend", "embedded", "developer_1c"),
    "developer_1c": ("software_engineer", "backend", "support"),
    "embedded": ("software_engineer", "backend"),
    "devops": ("sysadmin", "mlops", "backend"),
    "qa": ("software_engineer", "support"),
    "sysadmin": ("devops", "support", "security"),
    "support": ("sysadmin", "other_it"),
    "security": ("sysadmin", "devops"),
    "designer": ("frontend", "other_it"),
    "manager": ("business_analyst", "other_it"),
    "other_it": ("non_it", "support", "manager"),
    "non_it": ("other_it", "manager"),
}


def compact(obj: Mapping[str, Any]) -> str:
    """An answer in the training format of radar.train.dataset.answer_text."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def shown(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def normalized_sha(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()


def in_holdout(group: str, share: float, seed: int) -> bool:
    digest = hashlib.sha256(f"dpo:{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


# --- student inputs -----------------------------------------------------------------------------


def student_inputs(
    train: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]], p: Pairs
) -> list[dict[str, Any]]:
    """The SFT rows the student answers: a seeded sample of train, half per family, and all of val."""
    rng = random.Random(p.seed)
    picked: list[Mapping[str, Any]] = []
    per_family = p.train_sample // len(FAMILIES)
    for family in FAMILIES:
        rows = sorted((r for r in train if r["family"] == family), key=lambda r: r["id"])
        picked += rng.sample(rows, min(per_family, len(rows)))
    out = [input_row(r, "train") for r in sorted(picked, key=lambda r: r["id"])]
    if p.with_val:
        out += [input_row(r, "val") for r in sorted(val, key=lambda r: r["id"])]
    return out


def input_row(example: Mapping[str, Any], part: str) -> dict[str, Any]:
    return {
        "id": example["id"],
        "input": example["messages"][1]["content"],
        "family": example["family"],
        "group": example["group"],
        "part": part,
    }


# --- pairs --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    kind: str
    example: Mapping[str, Any]
    part: str
    fields: tuple[str, ...]
    rejected: str

    def row(self) -> dict[str, Any]:
        e = self.example
        return {
            "id": f"{self.kind}:{e['id']}",
            "record_id": e["id"],
            "kind": self.kind,
            "family": e["family"],
            "group": e["group"],
            "part": self.part,
            "fields": list(self.fields),
            "user": e["messages"][1]["content"],
            "chosen": e["messages"][2]["content"],
            "rejected": self.rejected,
        }


def disagreement(chosen: str, answer: str | None) -> tuple[str, ...]:
    """The compared fields where `answer` differs from the label, ("invalid_json",) for an answer
    that breaks the schema, () when they agree on every compared field."""
    gold, pred = try_parse(chosen), try_parse(answer)
    if gold is None:
        raise ValueError(f"the silver answer does not parse: {chosen[:80]}")
    if pred is None:
        return ("invalid_json",)
    return tuple(f for f in COMPARED if getattr(gold, f) != getattr(pred, f))


def student_pairs(
    examples: Mapping[str, Mapping[str, Any]], inputs: Sequence[Mapping[str, Any]], answers: Mapping[str, Any]
) -> tuple[list[Pair], Counter[str]]:
    """Pairs from the student's answers, and per part and field how often it disagreed."""
    pairs: list[Pair] = []
    stats: Counter[str] = Counter()
    for row in inputs:
        if row["id"] not in answers:
            raise SystemExit(f"no student answer for {row['id']}: run the predict step over the whole inputs file")
        example = examples[row["id"]]
        answer = answers[row["id"]]
        fields = disagreement(example["messages"][2]["content"], answer)
        stats[f"{row['part']}.{example['family']}.answered"] += 1
        for f in fields:
            stats[f"{row['part']}.{example['family']}.{f}"] += 1
        if fields:
            stats[f"{row['part']}.{example['family']}.pairs"] += 1
            pairs.append(Pair("student", example, row["part"], fields, (answer or "").strip()))
    return pairs, stats


def corrupt(kind: str, answer: str, rng: random.Random) -> str:
    obj = json.loads(answer)
    if kind == "work_format":
        obj["work_format"] = rng.choice([v for v in WORK_FORMATS if v != obj["work_format"]])
        return compact(obj)
    if kind == "role":
        role = obj["role"]
        obj["role"] = rng.choice(ROLE_NEIGHBOURS[role]) if role is not None else "other_it"
        return compact(obj)
    if kind == "drop_field":
        del obj[rng.choice(sorted(obj))]
        return compact(obj)
    if kind == "broken_json":
        return answer[: rng.randrange(len(answer) // 3, len(answer) - 1)]
    raise ValueError(f"unknown corruption {kind!r}")


def corrupted_pairs(pool: Sequence[Mapping[str, Any]], counts: Mapping[str, int], seed: int) -> list[Pair]:
    """`counts[kind]` corrupted copies per kind, half from each family, no example used twice."""
    rng = random.Random(seed)
    by_family = {f: sorted((e for e in pool if e["family"] == f), key=lambda e: e["id"]) for f in FAMILIES}
    for rows in by_family.values():
        rng.shuffle(rows)
    pairs = []
    for kind in CORRUPTIONS:
        for i, family in enumerate(FAMILIES):
            n = counts[kind] // len(FAMILIES) + (1 if i < counts[kind] % len(FAMILIES) else 0)
            rows = by_family[family]
            if len(rows) < n:
                raise SystemExit(f"{kind}: {n} {family} examples wanted, {len(rows)} left")
            taken, by_family[family] = rows[:n], rows[n:]
            for e in taken:
                chosen = e["messages"][2]["content"]
                rejected = corrupt(kind, chosen, rng)
                fields = (kind,) if kind in COMPARED else ()
                pairs.append(Pair(kind, e, "train", fields, rejected))
    return pairs


def check_no_gold(pairs: Sequence[Pair], gold_inputs: Sequence[Mapping[str, Any]]) -> None:
    ids = {r["id"] for r in gold_inputs}
    texts = {normalized_sha(r["input"]) for r in gold_inputs}
    for p in pairs:
        if p.example["id"] in ids or normalized_sha(p.example["messages"][1]["content"]) in texts:
            raise SystemExit(f"{p.example['id']} is a gold record or has a gold input: the pairs leak dev or test")


def build_pairs(
    cfg: Pairs,
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Any],
    gold_inputs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    examples = {e["id"]: e for e in [*train, *val]}
    from_student, stats = student_pairs(examples, inputs, answers)
    used = {p.example["id"] for p in from_student}
    pool = [e for e in train if e["id"] not in used]
    pairs = from_student + corrupted_pairs(pool, cfg.corrupted(), cfg.seed)
    check_no_gold(pairs, gold_inputs)
    for p in pairs:
        if p.rejected == p.example["messages"][2]["content"]:
            raise AssertionError(f"{p.kind}:{p.example['id']}: rejected equals chosen")
    rows = [p.row() for p in pairs]
    held = [r for r in rows if in_holdout(r["group"], cfg.holdout_share, cfg.seed)]
    kept = [r for r in rows if not in_holdout(r["group"], cfg.holdout_share, cfg.seed)]
    return kept, held, stats


# --- report -------------------------------------------------------------------------------------


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def report(
    train: Sequence[Mapping[str, Any]], held: Sequence[Mapping[str, Any]], stats: Counter[str]
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {
        "pairs_train": {"value": len(train)},
        "pairs_eval": {"value": len(held)},
    }
    for split, rows in (("train", train), ("eval", held)):
        c = Counter((r["kind"], r["family"]) for r in rows)
        for (kind, family), n in sorted(c.items()):
            metrics[f"{split}.{kind}.{family}"] = {"value": n}
    for part in ("train", "val"):
        for family in FAMILIES:
            answered = stats[f"{part}.{family}.answered"]
            if not answered:
                continue
            prefix = f"student_disagrees.{part}.{family}"
            metrics[f"{prefix}.any"] = share(stats[f"{part}.{family}.pairs"], answered)
            for f in (*COMPARED, "invalid_json"):
                metrics[f"{prefix}.{f}"] = share(stats[f"{part}.{family}.{f}"], answered)
    return metrics


# --- commands -----------------------------------------------------------------------------------


def gold_inputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in GOLD_SPLITS:
        path = data_dir() / "gold" / f"{split}_inputs.jsonl"
        if path.exists():
            rows += read_jsonl(path)
    return rows


def cmd_inputs(cfg: DpoConfig) -> int:
    train = read_jsonl(cfg.sft_dir / "train.jsonl")
    val = read_jsonl(cfg.sft_dir / "val.jsonl")
    rows = student_inputs(train, val, cfg.pairs)
    out = cfg.pairs_dir / "student_inputs.jsonl"
    write_jsonl(out, rows)
    parts = Counter(f"{r['part']}.{r['family']}" for r in rows)
    log.info("%d inputs for the student -> %s: %s", len(rows), out, dict(sorted(parts.items())))
    return 0


def cmd_build(cfg: DpoConfig, config_file: Path) -> int:
    train = read_jsonl(cfg.sft_dir / "train.jsonl")
    val = read_jsonl(cfg.sft_dir / "val.jsonl")
    inputs_file = cfg.pairs_dir / "student_inputs.jsonl"
    answers_file = cfg.pairs_dir / "student_answers.jsonl"
    inputs = read_jsonl(inputs_file)
    if inputs != student_inputs(train, val, cfg.pairs):
        raise SystemExit(f"{inputs_file} is not what `inputs` gives for this config and data/sft: run it again")
    answers = {r["id"]: r["output"] for r in read_jsonl(answers_file)}
    kept, held, stats = build_pairs(cfg.pairs, train, val, inputs, answers, gold_inputs())
    write_jsonl(cfg.pairs_dir / "train.jsonl", kept)
    write_jsonl(cfg.pairs_dir / "val.jsonl", held)
    answers_meta = answers_file.with_suffix(".meta.json")
    sft_meta = json.loads((cfg.sft_dir / "meta.json").read_text(encoding="utf-8"))
    meta = {
        "config_file": shown(config_file),
        "config_sha256": sha256_file(config_file),
        "sft_adapter": shown(cfg.sft_adapter),
        "sft_train_sha256": sha256_file(cfg.sft_dir / "train.jsonl"),
        "sft_val_sha256": sha256_file(cfg.sft_dir / "val.jsonl"),
        "prompt_sha256": sft_meta["prompt_sha256"],
        "student_inputs_sha256": sha256_file(inputs_file),
        "student_answers_sha256": sha256_file(answers_file),
        "student_answers_adapter_sha256": (
            json.loads(answers_meta.read_text(encoding="utf-8")).get("adapter_sha256")
            if answers_meta.exists()
            else None
        ),
        "train_sha256": sha256_file(cfg.pairs_dir / "train.jsonl"),
        "val_sha256": sha256_file(cfg.pairs_dir / "val.jsonl"),
        "commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (cfg.pairs_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    path = write_result(
        "dpo/pairs.json",
        report(kept, held, stats),
        source_script="src/radar/train/dpo_pairs.py",
        **{k: v for k, v in meta.items() if k not in ("commit", "created_at")},
        method="chosen: the silver label of a data/sft row; rejected: the SFT student's greedy answer where it "
        f"differs on {', '.join(COMPARED)} or is invalid (student_disagrees.* over the answered rows of each part "
        "and family), or a corrupted copy of the label (work_format swap, role swap to a neighbouring role, one key "
        "dropped, JSON cut short) on train rows without a student pair; pairs_eval: groups held out by a seeded hash",
    )
    log.info("%d train pairs, %d eval pairs -> %s", len(kept), len(held), path)
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.dpo_pairs", description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("inputs", "build"))
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_dpo(args.config)
    if args.cmd == "inputs":
        return cmd_inputs(cfg)
    return cmd_build(cfg, args.config)


if __name__ == "__main__":
    sys.exit(main())
