"""Corpus refresh: the corpus rebuilt from current staging while test and dev stay as drawn (S1).

`build` and `sample` drew test and dev on 2026-10-08 (EXPERIMENTS S1), and the gold labels hang on
those record ids. Snapshot days, API days and Habr cards keep arriving, and stored partitions were
rewritten by the PII cleaner, so the corpus is rebuilt, the gold rows never are:

- `data/corpus/s1/` holds the corpus, members, split and gold inputs as drawn. Its split file must
  hash to the S1 value in results/gold/sample_stats.json, otherwise the command stops.
- Test and dev rows of the split are copied unchanged: group id, gold id, members, inclusion
  probabilities. A gold record keeps the version it was labeled on (its S1 day); the input is rebuilt
  from the current raw Parquet partition of that day, which the PII rewrite cleaned. When the
  partition holds several rows of the record, the one closest to the labeled input is taken.
- Groups are recomputed over the current entities and text clusters. A component that reaches a test
  or dev record puts its other records into a withheld group `<gold group>_linked`; a component that
  reaches a group withheld in S1 stays withheld; the rest is the training pool.
- The near-duplicate check of `sample` runs again against the rebuilt gold inputs; a training-pool
  group with a neighbour is withheld, and the check is repeated on the result.

The output is a function of S1, staging and raw only, so the command is rerun after every new day.
"""

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.extract.corpus import (
    MEMBERS_SCHEMA,
    SOURCES,
    SPLIT_SCHEMA,
    SPLITS,
    TRAIN_POOL,
    WITHHELD,
    GroupRow,
    LeakReport,
    Record,
    SampleConfig,
    UnionFind,
    build_records,
    id_list_digest,
    make_record,
    near_duplicates,
    read_corpus,
    record_id,
    records_in_two_splits,
    representatives,
    share,
    split_digest,
    stratum,
    text_chars,
    withhold,
)
from radar.extract.inputs import input_for_record
from radar.extract.role_proxy import BUCKETS, ML_DS_DATA
from radar.pii.text import PLACEHOLDER

log = logging.getLogger("radar.extract.refresh")

LINKED = "_linked"
RAW_COLUMNS = ["source", "source_id", "fetched_at", "title", "text", "payload", "content_hash"]
_PLACEHOLDER = re.compile("|".join(re.escape(p) for p in sorted(set(PLACEHOLDER.values()))))

# (source, day, source ids) -> source id -> raw rows of that partition
VersionsOf = Callable[[str, str, Sequence[str]], dict[str, list[dict[str, Any]]]]


@dataclass(frozen=True)
class S1:
    records: list[Record]
    members: list[dict[str, Any]]
    rows: list[GroupRow]
    # split -> labeled inputs in file order
    inputs: dict[str, list[dict[str, str]]]


def read_split(path: Path) -> list[GroupRow]:
    return [GroupRow(**{**r, "member_ids": tuple(r["member_ids"])}) for r in pq.read_table(path).to_pylist()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_s1(directory: Path, split_file_sha256: str) -> S1:
    split_path = directory / "split.parquet"
    actual = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if actual != split_file_sha256:
        raise SystemExit(f"{split_path} hashes to {actual}, S1 recorded {split_file_sha256}")
    return S1(
        records=read_corpus(directory / "corpus.parquet"),
        members=pq.read_table(directory / "members.parquet").to_pylist(),
        rows=read_split(split_path),
        inputs={s: read_jsonl(directory / f"{s}_inputs.jsonl") for s in SPLITS},
    )


# --- gold inputs ----------------------------------------------------------------------------


def change_kind(labeled: str, current: str) -> str:
    """`same`, `pii` when the current text is the labeled one with spans replaced by PII
    placeholders, or `other`."""
    if labeled == current:
        return "same"
    parts = _PLACEHOLDER.split(current)
    if len(parts) > 1 and re.fullmatch(".+?".join(re.escape(p) for p in parts), labeled, re.DOTALL):
        return "pii"
    return "other"


def placeholders_added(labeled: str, current: str) -> Counter[str]:
    added = Counter(_PLACEHOLDER.findall(current))
    added.subtract(Counter(_PLACEHOLDER.findall(labeled)))
    return +added


def closest_version(labeled: str, rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str]:
    """The stored row the label was made on: an identical input, else one that differs only by
    placeholders, else the most similar text; the later fetch breaks ties."""
    scored = []
    for row in rows:
        current = input_for_record(row)
        kind = change_kind(labeled, current)
        similarity = SequenceMatcher(None, labeled, current).ratio() if len(rows) > 1 else 0.0
        scored.append(((kind == "same", kind == "pii", similarity, str(row["fetched_at"])), row, current))
    _, row, current = max(scored, key=lambda s: s[0])
    return row, current


def raw_versions(root: Path, source: str, day: str, source_ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    files = sorted((root / source / f"dt={day}").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw partition {source} dt={day} under {root}")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wanted = sorted(set(source_ids))
    for path in files:
        for row in pq.read_table(path, columns=RAW_COLUMNS, filters=[("source_id", "in", wanted)]).to_pylist():
            out[str(row["source_id"])].append(row)
    return out


def gold_records(
    s1: S1, staged_by_id: Mapping[str, Mapping[str, Any]], versions_of: VersionsOf
) -> tuple[dict[str, Record], dict[str, str]]:
    """Gold id -> its record at the labeled version with a cleaned input, and gold id -> change kind."""
    labeled = {x["id"]: x["input"] for s in SPLITS for x in s1.inputs[s]}
    old = {r.id: r for r in s1.records if r.id in labeled}
    by_partition: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in old.values():
        by_partition[(r.source, r.dt)].append(r.source_id)

    out: dict[str, Record] = {}
    kinds: dict[str, str] = {}
    for (source, day), ids in sorted(by_partition.items()):
        found = versions_of(source, day, ids)
        for source_id in ids:
            rid = record_id(source, source_id)
            if not found.get(source_id):
                raise RuntimeError(f"gold record {rid} is missing from raw partition dt={day}")
            row, current = closest_version(labeled[rid], found[source_id])
            prev = old[rid]
            staged = staged_by_id.get(rid)
            rec = make_record(
                {"source": source, "source_id": source_id, "dt": day, "content_hash": row["content_hash"]},
                row,
                str(staged["vacancy_entity_id"]) if staged else prev.vacancy_entity_id,
                str(staged["text_cluster_id"]) if staged else prev.text_cluster_id,
                prev.group_id,
            )
            if rec.input != current:
                raise AssertionError(f"{rid}: input differs between version choice and record")
            out[rid] = rec
            kinds[rid] = change_kind(labeled[rid], current)
    return out, kinds


# --- groups ---------------------------------------------------------------------------------


def regroup(
    staged: Sequence[Mapping[str, Any]], frozen: Sequence[GroupRow], sticky: Sequence[GroupRow]
) -> tuple[dict[str, tuple[str, str]], list[list[str]]]:
    """Record id -> (group id, split) for every staged record, and the sets of test and dev groups
    that current entities and text clusters join into one component (a leak between them that a
    frozen split cannot undo, so it is reported).

    Components join records through entities, text clusters and their S1 group if it is frozen
    (test, dev) or sticky (withheld in S1). Members of a frozen group stay in it; the other records
    of its component form `<group>_linked`, withheld. A component reaching a sticky group is
    withheld; the rest is the training pool, named like `group_ids` after its smallest entity.
    """
    frozen_of = {m: r for r in frozen for m in r.member_ids}
    sticky_of = {m: r.group_id for r in sticky for m in r.member_ids}
    uf = UnionFind()
    ids = []
    for s in staged:
        rid = record_id(s["source"], s["source_id"])
        ids.append(rid)
        uf.union(f"r:{rid}", f"e:{s['vacancy_entity_id']}")
        uf.union(f"r:{rid}", f"t:{s['text_cluster_id']}")
        if rid in frozen_of:
            uf.union(f"r:{rid}", f"g:{frozen_of[rid].group_id}")
        if rid in sticky_of:
            uf.union(f"r:{rid}", f"g:{sticky_of[rid]}")
    entity_of = {record_id(s["source"], s["source_id"]): str(s["vacancy_entity_id"]) for s in staged}

    components: dict[str, list[str]] = defaultdict(list)
    for rid in ids:
        components[uf.find(f"r:{rid}")].append(rid)

    out: dict[str, tuple[str, str]] = {}
    joined = []
    for members in components.values():
        gold = sorted({frozen_of[rid].group_id for rid in members if rid in frozen_of})
        if len(gold) > 1:
            joined.append(gold)
        rest = [rid for rid in members if rid not in frozen_of]
        for rid in members:
            if rid in frozen_of:
                out[rid] = (frozen_of[rid].group_id, frozen_of[rid].split)
        if not rest:
            continue
        if gold:
            group, split = gold[0] + LINKED, WITHHELD
        else:
            group = "grp_" + min(entity_of[rid] for rid in rest).removeprefix("ve_")
            split = WITHHELD if any(rid in sticky_of for rid in rest) else TRAIN_POOL
        for rid in rest:
            out[rid] = (group, split)

    frozen_groups = {r.group_id for r in frozen}
    clash = {g for g, s in out.values() if g in frozen_groups and s not in SPLITS}
    if clash:
        raise RuntimeError(f"new groups reuse frozen group ids: {sorted(clash)[:5]}")
    return out, sorted(joined)


# --- refresh --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refreshed:
    records: list[Record]
    members: list[dict[str, Any]]
    rows: list[GroupRow]
    kinds: dict[str, str]
    leak: LeakReport
    leak_after: LeakReport
    withheld_reason: dict[str, str]
    gold_groups_joined: list[list[str]]
    # gold id -> change kind of its latest staged version against the labeled input, when that
    # version is from a later day than the labeled one
    newer_kinds: dict[str, str]


def refresh(
    staged: Sequence[Mapping[str, Any]],
    raw: Mapping[tuple[str, str], Mapping[str, Any]],
    s1: S1,
    cfg: SampleConfig,
    versions_of: VersionsOf,
) -> Refreshed:
    frozen = [r for r in s1.rows if r.split in SPLITS]
    sticky = [r for r in s1.rows if r.split == WITHHELD]
    staged_by_id = {record_id(s["source"], s["source_id"]): s for s in staged}
    assigned, joined = regroup(staged, frozen, sticky)

    gold, kinds = gold_records(s1, staged_by_id, versions_of)
    labeled = {x["id"]: x["input"] for s in SPLITS for x in s1.inputs[s]}
    newer_kinds = {
        rid: change_kind(labeled[rid], input_for_record(raw[(str(st["source"]), str(st["source_id"]))]))
        for rid, rec in gold.items()
        if (st := staged_by_id.get(rid)) is not None and str(st["dt"]) > rec.dt
    }
    built = build_records(staged, raw, {rid: g for rid, (g, _) in assigned.items()})
    gold_keys = {(r.vacancy_entity_id, r.group_id) for r in gold.values()}
    records = sorted(
        [r for r in built if r.id not in gold and (r.vacancy_entity_id, r.group_id) not in gold_keys]
        + list(gold.values()),
        key=lambda r: r.id,
    )

    member_ids: dict[str, list[str]] = defaultdict(list)
    split_of: dict[str, str] = {}
    for rid, (g, s) in assigned.items():
        member_ids[g].append(rid)
        split_of[g] = s
    reps = representatives(records)
    n_corpus = Counter(r.group_id for r in records)
    frozen_ids = {r.group_id for r in frozen}
    rows = list(frozen) + [
        GroupRow(
            group_id=g,
            split=split_of[g],
            stratum=stratum(reps[g].source, reps[g].role_proxy) if g in reps else None,
            gold_id=None,
            pi_test=None,
            pi_dev=None,
            n_corpus=n_corpus.get(g, 0),
            member_ids=tuple(sorted(ids)),
        )
        for g, ids in member_ids.items()
        if g not in frozen_ids
    ]
    reason = {r.group_id: ("linked" if r.group_id.endswith(LINKED) else "s1") for r in rows if r.split == WITHHELD}

    leak = near_duplicates(rows, records, cfg)
    rows = withhold(rows, records, leak)
    reason.update({r.group_id: "near_duplicate" for r in rows if r.split == WITHHELD and r.group_id not in reason})
    leak_after = near_duplicates(rows, records, cfg)
    rows = sorted(rows, key=lambda r: r.group_id)

    staged_members = [
        {
            "id": rid,
            "source": str(staged_by_id[rid]["source"]),
            "vacancy_entity_id": str(staged_by_id[rid]["vacancy_entity_id"]),
            "group_id": g,
            "text_chars": text_chars(
                raw[(str(staged_by_id[rid]["source"]), str(staged_by_id[rid]["source_id"]))].get("text")
            ),
        }
        for rid, (g, _) in assigned.items()
    ]
    frozen_members = {m for r in frozen for m in r.member_ids}
    kept = [m for m in s1.members if m["id"] in frozen_members and m["id"] not in assigned]
    members = sorted(staged_members + kept, key=lambda m: str(m["id"]))
    return Refreshed(records, members, rows, kinds, leak, leak_after, reason, joined, newer_kinds)


def frozen_rows_unchanged(before: Sequence[GroupRow], after: Sequence[GroupRow]) -> bool:
    old = sorted((asdict(r) for r in before if r.split in SPLITS), key=lambda d: d["group_id"])
    new = sorted((asdict(r) for r in after if r.split in SPLITS), key=lambda d: d["group_id"])
    return old == new


def refresh_stats(
    out: Refreshed, s1: S1, staged: Sequence[Mapping[str, Any]], s1_hashes: Mapping[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    records, rows = out.records, out.rows
    staged_n = Counter(str(s["source"]) for s in staged)
    corpus_n = Counter(r.source for r in records)
    m["staged.records"] = {"value": len(staged)}
    m["corpus.records"] = share(len(records), len(staged))
    m["corpus.groups"] = {"value": len({r.group_id for r in records})}
    for s in SOURCES:
        m[f"staged.records.{s}"] = {"value": staged_n[s]}
        m[f"corpus.records.{s}"] = share(corpus_n[s], len(records))

    split_of = {r.group_id: r.split for r in rows}
    for split in (*SPLITS, TRAIN_POOL, WITHHELD):
        m[f"split.{split}.groups"] = {"value": sum(r.split == split for r in rows)}
        m[f"split.{split}.corpus_records"] = {"value": sum(split_of[r.group_id] == split for r in records)}
    for why in ("s1", "linked", "near_duplicate"):
        m[f"split.withheld.groups.{why}"] = {"value": sum(v == why for v in out.withheld_reason.values())}

    pool = [r for r in records if split_of[r.group_id] == TRAIN_POOL]
    m["decisions10.train_pool.ml_ds_data"] = share(sum(r.role_proxy == ML_DS_DATA for r in pool), len(pool))
    for s in SOURCES:
        per = [r for r in pool if r.source == s]
        m[f"decisions10.train_pool.ml_ds_data.{s}"] = share(sum(r.role_proxy == ML_DS_DATA for r in per), len(per))
    pool_reps = [r for g, r in representatives(pool).items()]
    m["decisions10.train_pool_groups.ml_ds_data"] = share(
        sum(r.role_proxy == ML_DS_DATA for r in pool_reps), len(pool_reps)
    )
    for b in BUCKETS:
        m[f"role_proxy.train_pool.{b}"] = share(sum(r.role_proxy == b for r in pool), len(pool))

    staged_ids = {record_id(s["source"], s["source_id"]): s for s in staged}
    by_id = {r.id: r for r in records}
    other: dict[str, list[str]] = {}
    newer_other: dict[str, list[str]] = {}
    for split in SPLITS:
        ids = [x["id"] for x in s1.inputs[split]]
        kinds = Counter(out.kinds[i] for i in ids)
        changed = kinds["pii"] + kinds["other"]
        m[f"gold.{split}.inputs_changed"] = share(changed, len(ids))
        m[f"gold.{split}.inputs_changed_pii_only"] = share(kinds["pii"], changed)
        m[f"gold.{split}.inputs_changed_other"] = share(kinds["other"], changed)
        other[split] = sorted(i for i in ids if out.kinds[i] == "other")
        labeled = {x["id"]: x["input"] for x in s1.inputs[split]}
        added: Counter[str] = Counter()
        for i in ids:
            added.update(placeholders_added(labeled[i], by_id[i].input))
        for p in sorted(set(PLACEHOLDER.values())):
            m[f"gold.{split}.placeholders_added.{p.strip('[]')}"] = {"value": added[p]}
        m[f"gold.{split}.not_in_staging"] = share(sum(i not in staged_ids for i in ids), len(ids))
        newer = Counter(out.newer_kinds[i] for i in ids if i in out.newer_kinds)
        m[f"gold.{split}.newer_version_in_staging"] = share(sum(newer.values()), len(ids))
        m[f"gold.{split}.newer_version_other_text"] = share(newer["other"], sum(newer.values()))
        newer_other[split] = sorted(i for i in ids if out.newer_kinds.get(i) == "other")
        m[f"leak.{split}_with_train_neighbour_before"] = share(len(out.leak.neighbours[split]), len(ids))
        m[f"leak.{split}_with_train_neighbour"] = share(len(out.leak_after.neighbours[split]), len(ids))
    m["leak.test_dev_pairs"] = {"value": len(out.leak.test_dev_pairs)}
    m["leak.records_in_two_splits"] = {"value": records_in_two_splits(rows)}
    m["leak.gold_groups_joined"] = {"value": len(out.gold_groups_joined)}

    gold_ids = {s: [r.gold_id for r in rows if r.split == s and r.gold_id] for s in SPLITS}
    hashes = {
        "test_ids_sha256": id_list_digest(gold_ids["test"]),
        "dev_ids_sha256": id_list_digest(gold_ids["dev"]),
        "train_pool_corpus_ids_sha256": id_list_digest(r.id for r in pool),
        "split_content_sha256": split_digest(rows),
    }
    checks = {
        "test_ids_equal_s1": hashes["test_ids_sha256"] == s1_hashes["test_ids_sha256"],
        "dev_ids_equal_s1": hashes["dev_ids_sha256"] == s1_hashes["dev_ids_sha256"],
        "test_dev_rows_equal_s1": frozen_rows_unchanged(s1.rows, rows),
    }
    for k, v in checks.items():
        m[f"check.{k}"] = {"value": str(v).lower()}
    extra = {
        "hashes": hashes,
        "s1_hashes": {k: s1_hashes[k] for k in ("test_ids_sha256", "dev_ids_sha256", "split_file_sha256")},
        "gold_inputs_changed_other": other,
        "gold_newer_version_other_text": newer_other,
        "leak_neighbours_before": out.leak.neighbours,
        "leak_test_dev_pairs": out.leak.test_dev_pairs,
        "gold_groups_joined": out.gold_groups_joined,
    }
    return m, extra


def write_split(rows: Iterable[GroupRow], path: Path) -> None:
    table = pa.Table.from_pylist([{**asdict(r), "member_ids": list(r.member_ids)} for r in rows], schema=SPLIT_SCHEMA)
    pq.write_table(table, path)


def write_members(members: Sequence[Mapping[str, Any]], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(list(members), schema=MEMBERS_SCHEMA), path)


def write_gold_inputs(path: Path, order: Sequence[Mapping[str, str]], records: Mapping[str, Record]) -> None:
    """Same ids in the same order as the labeled file, current inputs."""
    lines = [
        json.dumps(
            {"id": x["id"], "source": records[x["id"]].source, "input": records[x["id"]].input}, ensure_ascii=False
        )
        for x in order
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
