"""SFT dataset for the Qwen3 students: silver labels as chat examples.

Input is data/silver/train.jsonl ({"id", "input", "output"}, `python -m radar.extract.silver build`)
with both source families. Each record takes its source, group and role proxy from its corpus
(data/corpus for trudvsem and Habr, data/corpus/telegram for posts), and must still be a member of
a train_pool group and of no other: a corpus rebuild after the silver build can move a record, and
such a record is dropped and counted rather than trained on.

System prompt. The student gets configs/student_prompt.md (the task, the keys and their allowed
values, about 300 tokens), not the teacher's prompt of about 6 500 tokens. The teacher's rules
(how a role is chosen, when salary is null) are what the labels already carry; repeating them in
every example would make the median example about nine times longer and the epoch as much slower,
and would not fit max_seq_len. The allowed values stay in the prompt: they are the schema contract
 and keep the answer on the enums before the weights have learned them.

The answer is the label as compact JSON in schema key order, followed by the end-of-turn token.
The input is cut to max_input_tokens at a token boundary; radar.train.predict cuts the same way
with the value stored next to the adapter. An example whose prompt plus answer exceeds max_seq_len
is dropped, never cut. Train and validation are split by group, so reposts and cross-posts of one
vacancy stay on one side.

Output:
    data/sft/train.jsonl, data/sft/val.jsonl   {"id", "family", "source", "group", "role_proxy", "role",
        "input_tokens", "truncated", "prompt_tokens", "completion_tokens", "messages"}
    data/sft/meta.json                          prompt, tokenizer, max_input_tokens, max_seq_len, hashes
    results/sft/dataset.json                    counts per split, family, source, role proxy; token lengths

Runs in the WSL training venv, where the tokenizer is:
    bash scripts/train_run.sh dataset
"""

import argparse
import hashlib
import json
import logging
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from radar.config import RESULTS, ROOT, data_dir
from radar.extract.schema import dump, parse
from radar.results import git_commit, write_result

log = logging.getLogger("radar.train.dataset")

CONFIG = ROOT / "configs" / "sft" / "dataset.toml"
FAMILIES = ("trudvsem_habr", "telegram")


@dataclass(frozen=True)
class DatasetConfig:
    tokenizer: str
    prompt: Path
    max_input_tokens: int
    max_seq_len: int
    val_share: float
    seed: int


def load_config(path: Path = CONFIG) -> DatasetConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = DatasetConfig(
        tokenizer=str(raw["tokenizer"]),
        prompt=ROOT / str(raw["prompt"]),
        max_input_tokens=int(raw["max_input_tokens"]),
        max_seq_len=int(raw["max_seq_len"]),
        val_share=float(raw["val_share"]),
        seed=int(raw["seed"]),
    )
    if not 0 < cfg.val_share < 0.5:
        raise ValueError(f"val_share {cfg.val_share}: a share of groups between 0 and 0.5")
    if not 0 < cfg.max_input_tokens < cfg.max_seq_len:
        raise ValueError("max_input_tokens must be positive and below max_seq_len")
    return cfg


def system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- text ---------------------------------------------------------------------------------------


class Tokens(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token of `text`, no special tokens."""

    def lengths(self, system: str, user: str, answer: str) -> tuple[int, int]:
        """Tokens of the rendered prompt and of the answer with its end-of-turn token."""


def cap_input(text: str, spans: Sequence[tuple[int, int]], max_tokens: int) -> tuple[str, bool]:
    """`text` cut after its first `max_tokens` tokens, and whether it was cut."""
    if len(spans) <= max_tokens:
        return text, False
    return text[: spans[max_tokens - 1][1]], True


def answer_text(output: str) -> str:
    """The label as the student writes it: validated, compact, in schema key order."""
    return json.dumps(dump(parse(output)), ensure_ascii=False, separators=(",", ":"))


def messages(system: str, user: str, answer: str | None = None) -> list[dict[str, str]]:
    turns = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if answer is not None:
        turns.append({"role": "assistant", "content": answer})
    return turns


def render_prompt(tokenizer: Any, system: str, user: str) -> str:
    """The chat prompt up to the answer, Qwen3 thinking off (an empty think block, as at inference)."""
    text: str = tokenizer.apply_chat_template(
        messages(system, user), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return text


def token_spans(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(int(a), int(b)) for a, b in enc["offset_mapping"]]


class HFTokens:
    """A Hugging Face tokenizer from the local cache; needs transformers (training venv)."""

    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def spans(self, text: str) -> list[tuple[int, int]]:
        return token_spans(self.tokenizer, text)

    def lengths(self, system: str, user: str, answer: str) -> tuple[int, int]:
        prompt = render_prompt(self.tokenizer, system, user)
        completion = answer + self.tokenizer.eos_token
        n_prompt = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        return n_prompt, len(self.tokenizer(completion, add_special_tokens=False).input_ids)


# --- records ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Origin:
    family: str
    source: str
    group: str  # prefixed with the family: the two corpora number their groups independently
    role_proxy: str


def _pool(split_path: Path) -> tuple[set[str], set[str]]:
    rows = pq.read_table(split_path, columns=["split", "member_ids"]).to_pylist()
    pool = {rid for r in rows if r["split"] == "train_pool" for rid in r["member_ids"]}
    held = {rid for r in rows if r["split"] != "train_pool" for rid in r["member_ids"]}
    return pool, held


def origins(corpus: Path, wanted: Iterable[str]) -> dict[str, Origin]:
    """Origin of every wanted id that is a member of a train_pool group of its corpus and of no other
    group. Group and source come from the group members, so a labeled record keeps its group after
    a corpus rebuild picked a newer version of its vacancy; the role proxy is the corpus record's,
    "not_in_corpus" for such an older version."""
    wanted = set(wanted)
    out: dict[str, Origin] = {}
    if any(not i.startswith("telegram:") for i in wanted):
        pool, held = _pool(corpus / "split.parquet")
        proxy = {
            r["id"]: r["role_proxy"]
            for r in pq.read_table(corpus / "corpus.parquet", columns=["id", "role_proxy"]).to_pylist()
            if r["id"] in wanted
        }
        for r in pq.read_table(corpus / "members.parquet", columns=["id", "source", "group_id"]).to_pylist():
            if r["id"] in wanted and r["id"] in pool and r["id"] not in held:
                group = f"trudvsem_habr:{r['group_id']}"
                out[r["id"]] = Origin("trudvsem_habr", r["source"], group, proxy.get(r["id"], "not_in_corpus"))
    if any(i.startswith("telegram:") for i in wanted):
        tg = corpus / "telegram"
        pool, held = _pool(tg / "split.parquet")
        for r in pq.read_table(tg / "records.parquet", columns=["id", "group_id", "role_proxy"]).to_pylist():
            if r["id"] in wanted and r["id"] in pool and r["id"] not in held:
                out[r["id"]] = Origin("telegram", "telegram", f"telegram:{r['group_id']}", r["role_proxy"])
    return out


def in_val(group: str, share: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


@dataclass(frozen=True)
class Built:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    too_long: list[str]
    outside: list[str]  # no longer in a train_pool group: the corpus was rebuilt after the silver build


def build(
    rows: Sequence[Mapping[str, Any]],
    where: Mapping[str, Origin],
    cfg: DatasetConfig,
    system: str,
    tokens: Tokens,
) -> Built:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    too_long: list[str] = []
    outside = [r["id"] for r in rows if r["id"] not in where]
    for row in rows:
        origin = where.get(row["id"])
        if origin is None:
            continue
        spans = tokens.spans(row["input"])
        user, truncated = cap_input(row["input"], spans, cfg.max_input_tokens)
        answer = answer_text(row["output"])
        n_prompt, n_completion = tokens.lengths(system, user, answer)
        if n_prompt + n_completion > cfg.max_seq_len:
            too_long.append(row["id"])
            continue
        example = {
            "id": row["id"],
            "family": origin.family,
            "source": origin.source,
            "group": origin.group,
            "role_proxy": origin.role_proxy,
            "role": json.loads(answer)["role"],
            "input_tokens": len(spans),
            "truncated": truncated,
            "prompt_tokens": n_prompt,
            "completion_tokens": n_completion,
            "messages": messages(system, user, answer),
        }
        (val if in_val(origin.group, cfg.val_share, cfg.seed) else train).append(example)
    return Built(train, val, too_long, outside)


# --- report -------------------------------------------------------------------------------------


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def quantiles(values: Sequence[int]) -> dict[str, int]:
    """Nearest-rank p50, p90, p99 and the maximum."""
    if not values:
        return {}
    xs = sorted(values)
    out = {f"p{p}": xs[min(len(xs) - 1, max(0, -(-p * len(xs) // 100) - 1))] for p in (50, 90, 99)}
    return out | {"max": xs[-1]}


def counts(examples: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    c = Counter(str(e[key]) for e in examples)
    return {k: c[k] for k in sorted(c)}


def report(built: Built, n_silver: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Metrics and the breakdowns that go next to them in results/sft/dataset.json."""
    both = built.train + built.val
    groups = {s: {e["group"] for e in ex} for s, ex in (("train", built.train), ("val", built.val))}
    overlap = groups["train"] & groups["val"]
    if overlap:
        raise AssertionError(f"groups on both sides: {sorted(overlap)[:3]}")
    metrics: dict[str, dict[str, Any]] = {
        "silver_records": {"value": n_silver},
        "train_examples": share(len(built.train), n_silver),
        "val_examples": share(len(built.val), n_silver),
        "dropped_too_long": share(len(built.too_long), n_silver),
        "dropped_not_in_train_pool": share(len(built.outside), n_silver),
        "truncated_inputs": share(sum(e["truncated"] for e in both), len(both)),
        "train_groups": {"value": len(groups["train"])},
        "val_groups": {"value": len(groups["val"])},
        "train_sequence_tokens": {"value": sum(e["prompt_tokens"] + e["completion_tokens"] for e in built.train)},
    }
    for family in FAMILIES:
        for split, ex in (("train", built.train), ("val", built.val)):
            n = sum(e["family"] == family for e in ex)
            if n:
                metrics[f"{family}.{split}_examples"] = {"value": n}
                metrics[f"{family}.{split}_truncated"] = share(
                    sum(e["truncated"] for e in ex if e["family"] == family), n
                )
    splits = {"train": built.train, "val": built.val}
    extra: dict[str, Any] = {
        "by_source": {s: counts(ex, "source") for s, ex in splits.items()},
        "by_role_proxy": {
            s: counts([{"k": f"{e['family']}|{e['role_proxy']}"} for e in ex], "k") for s, ex in splits.items()
        },
        "by_label_role": {s: counts(ex, "role") for s, ex in splits.items()},
        "tokens": {
            family: {
                "input_before_cap": quantiles([e["input_tokens"] for e in both if e["family"] == family]),
                "prompt": quantiles([e["prompt_tokens"] for e in both if e["family"] == family]),
                "completion": quantiles([e["completion_tokens"] for e in both if e["family"] == family]),
                "sequence": quantiles(
                    [e["prompt_tokens"] + e["completion_tokens"] for e in both if e["family"] == family]
                ),
            }
            for family in FAMILIES
            if any(e["family"] == family for e in both)
        },
    }
    return metrics, extra


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def silver_recorded(silver: Path) -> str:
    """sha256 of the silver file, checked against the one its build recorded."""
    actual = sha256_file(silver)
    record = RESULTS / "silver" / "build.json"
    if record.exists():
        recorded = json.loads(record.read_text(encoding="utf-8")).get("output_sha256")
        if recorded != actual:
            raise SystemExit(f"{silver} is not the file {record} describes: run the silver build again")
    return actual


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.dataset", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    silver = data_dir() / "silver" / "train.jsonl"
    silver_sha = silver_recorded(silver)
    rows = read_jsonl(silver)
    system = system_prompt(cfg.prompt)
    where = origins(data_dir() / "corpus", (r["id"] for r in rows))
    built = build(rows, where, cfg, system, HFTokens(cfg.tokenizer))
    log.info(
        "%d train, %d val; dropped %d too long, %d no longer in a train_pool group",
        len(built.train),
        len(built.val),
        len(built.too_long),
        len(built.outside),
    )

    out = data_dir() / "sft"
    write_jsonl(out / "train.jsonl", built.train)
    write_jsonl(out / "val.jsonl", built.val)
    meta = {
        "prompt_file": cfg.prompt.relative_to(ROOT).as_posix(),
        "prompt_sha256": sha256_file(cfg.prompt),
        "system_prompt": system,
        "tokenizer": cfg.tokenizer,
        "max_input_tokens": cfg.max_input_tokens,
        "max_seq_len": cfg.max_seq_len,
        "silver_sha256": silver_sha,
        "train_sha256": sha256_file(out / "train.jsonl"),
        "val_sha256": sha256_file(out / "val.jsonl"),
        "commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics, extra = report(built, len(rows))
    path = write_result(
        "sft/dataset.json",
        metrics,
        source_script="src/radar/train/dataset.py",
        config_file=args.config.resolve().relative_to(ROOT).as_posix(),
        config_sha256=sha256_file(args.config),
        **{k: meta[k] for k in ("prompt_file", "prompt_sha256", "tokenizer", "max_input_tokens", "max_seq_len")},
        silver_file="data/silver/train.jsonl",
        silver_sha256=silver_sha,
        train_sha256=meta["train_sha256"],
        val_sha256=meta["val_sha256"],
        dropped_too_long_ids=built.too_long,
        dropped_not_in_train_pool_ids=built.outside,
        **extra,
        method="silver labels per source family (results/silver/build.json); input cut to max_input_tokens "
        "tokens at a token boundary; an example over max_seq_len tokens with its answer dropped; train and "
        "val split by group with a seeded hash, val_share of groups; token lengths with the tokenizer above, "
        "nearest-rank quantiles; truncated_inputs over written examples",
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
