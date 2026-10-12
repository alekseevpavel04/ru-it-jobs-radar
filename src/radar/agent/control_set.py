"""The control set of the analyst agent: questions, reference SQL and answers on the frozen snapshot.

`eval/agent/questions.toml` is written by hand: a question in Russian and the reference SQL over the
marts. `build` assigns the split and runs every reference query on the snapshot as the read-only
account, through the same check as the agent, and writes `eval/agent/control_set.jsonl`. Nobody
types an answer.

Split: within each kind of question, the items ordered by sha256 of seed and id; the first
`dev_share` go to dev (prompt tuning), the rest to test (scored once, `--final-test`). `quick` marks
two dev items of every kind for the regression run after each change.
"""

import hashlib
import json
import math
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import sqlglot
from pydantic import BaseModel

from radar.agent.clickhouse import ReadOnlyClickHouse
from radar.agent.profiles import frozen
from radar.agent.snapshot import pin_id, pinned
from radar.agent.sqlcheck import check
from radar.config import ROOT

QUESTIONS = ROOT / "eval" / "agent" / "questions.toml"
CONTROL_SET = ROOT / "eval" / "agent" / "control_set.jsonl"
QUICK_PER_KIND = 2

Split = Literal["dev", "test"]


class Question(BaseModel):
    id: str
    kind: str
    question: str
    sql: str


class Unsafe(BaseModel):
    id: str
    question: str


class Reference(BaseModel):
    columns: list[str]
    types: list[str]
    rows: list[list[Any]]


class Item(BaseModel):
    id: str
    kind: str
    split: Split
    quick: bool
    question: str
    sql: str
    answer: Reference
    snapshot: str


class Spec(BaseModel):
    seed: int
    dev_share: float
    questions: list[Question]
    unsafe: list[Unsafe]


def load_spec(path: Path = QUESTIONS) -> Spec:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Spec(
        seed=raw["seed"],
        dev_share=raw["dev_share"],
        questions=[Question(**q) for q in raw["question"]],
        unsafe=[Unsafe(**u) for u in raw["unsafe"]],
    )


def assign_splits(questions: list[Question], seed: int, dev_share: float) -> dict[str, tuple[Split, bool]]:
    by_kind: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_kind[q.kind].append(q)
    out: dict[str, tuple[Split, bool]] = {}
    for items in by_kind.values():
        ordered = sorted(items, key=lambda q: hashlib.sha256(f"{seed}:{q.id}".encode()).hexdigest())
        n_dev = max(1, math.floor(len(ordered) * dev_share + 0.5))
        for rank, q in enumerate(ordered):
            split: Split = "dev" if rank < n_dev else "test"
            out[q.id] = (split, split == "dev" and rank < QUICK_PER_KIND)
    return out


def _stable_order(sql: str, rows: list[list[Any]]) -> list[list[Any]]:
    """Rows of a query without ORDER BY come in any order; sort them so a rebuild rewrites no line."""
    tree = sqlglot.parse_one(sql, dialect="clickhouse")
    if tree.args.get("order") is not None:
        return rows
    return sorted(rows, key=lambda row: json.dumps(row, ensure_ascii=False, default=str))


def build(ro: ReadOnlyClickHouse, spec: Spec) -> list[Item]:
    ids = [q.id for q in spec.questions]
    if len(set(ids)) != len(ids):
        raise ValueError("question ids repeat")
    snapshot = pin_id(pinned())
    profile = frozen()
    splits = assign_splits(spec.questions, spec.seed, spec.dev_share)
    items = []
    for q in spec.questions:
        result = ro.run(check(q.sql, database=profile.database, tables=profile.tables).sql)
        if not result.rows:
            raise ValueError(f"{q.id}: the reference query returns no rows")
        split, quick = splits[q.id]
        answer = Reference(columns=result.columns, types=result.types, rows=_stable_order(q.sql, result.rows))
        items.append(
            Item(
                id=q.id,
                kind=q.kind,
                split=split,
                quick=quick,
                question=q.question,
                sql=q.sql.strip(),
                answer=answer,
                snapshot=snapshot,
            )
        )
    return items


def write(items: list[Item], path: Path = CONTROL_SET) -> None:
    lines = [json.dumps(item.model_dump(), ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load(path: Path = CONTROL_SET) -> list[Item]:
    return [Item.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
