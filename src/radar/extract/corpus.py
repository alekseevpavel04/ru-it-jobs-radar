"""The extraction corpus and the gold samples drawn from it.

Corpus: what the teachers label (6.1) and the students learn from. One record per vacancy entity
(`staging.vacancy_entities`), the entity's latest version, over the IT candidates of the trudvsem
API and snapshots and over Habr. A record whose free text is shorter than MIN_TEXT_CHARS is not a
corpus record: the extractor would see a title and a header only (Habr cards not read yet), so
there is nothing to learn from or label beyond the source's own fields. Telegram posts form their own
corpus (radar.extract.telegram_corpus).

Groups: connected components over both `vacancy_entity_id` and `text_cluster_id`, computed over
every staged record, not only corpus records. The text cluster ignores place, so one template
posted in 70 regions is one group: the same text in train and test would leak whatever the city.

Samples: test and dev are drawn per stratum (source x role proxy) as simple random samples of
groups without replacement, one record per group (the group's latest corpus record). Every group
gets a permanent random number from the seed of configs/eval.toml; inside a stratum the lowest
numbers go to test and the next to dev, so the marginal inclusion probability of a group is
n_h / N_h for each split and is stored, and the natural distribution is recoverable by weighting.
Groups whose records appear in repository fixtures were read while writing parsers and the input
format, so they are kept out of test and dev.

Leak check: every test and dev record is compared with every training-pool record at a looser
Jaccard than the text clusters; a training-pool group with such a near duplicate is withheld from
training (split `withheld`), and the check is run again on the result.

Splits are frozen once drawn: `sample` refuses to overwrite them. `refresh` (radar.extract.refresh)
rebuilds the corpus from grown staging around the frozen test and dev rows; a training set takes
the `train_pool` groups of the split it writes.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.config import ROOT, data_dir, load_dotenv
from radar.extract.inputs import input_for_record
from radar.extract.role_proxy import BUCKETS, ML_DS_DATA, role_proxy

log = logging.getLogger("radar.extract.corpus")

SOURCES = ("trudvsem", "trudvsem_snapshot", "habr")
# The API is fresher than a snapshot of the same day and carries the same vacancy id.
SOURCE_RANK = {s: i for i, s in enumerate(SOURCES)}

# trudvsem's free text is "Обязанности: ...\n\nТребования: ..."; below 40 characters it holds the
# two labels and a word or two ("Обязанности: -"), above it a real if short description starts.
MIN_TEXT_CHARS = 40

EVAL_CONFIG = ROOT / "configs" / "eval.toml"
SAMPLE_CONFIG = ROOT / "configs" / "gold_sample.toml"
FIXTURES = ROOT / "tests" / "fixtures"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HABR_ID = re.compile(r"(?<!\d)1000\d{6}(?!\d)")


@dataclass(frozen=True)
class Record:
    id: str
    source: str
    source_id: str
    dt: str
    content_hash: str
    vacancy_entity_id: str
    text_cluster_id: str
    group_id: str
    title: str
    specializations: tuple[str, ...]
    role_proxy: str
    text_chars: int
    input: str


def record_id(source: str, source_id: str) -> str:
    return f"{source}:{source_id}"


def corpus_dir() -> Path:
    return data_dir() / "corpus"


def gold_dir() -> Path:
    return data_dir() / "gold"


# --- groups ---------------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_ids(memberships: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Entity id -> group id: components of the graph where an entity and a text cluster are linked
    whenever one record carries both. The group is named after its smallest entity id."""
    uf = UnionFind()
    entities: set[str] = set()
    for entity, cluster in memberships:
        entities.add(entity)
        uf.union(f"e:{entity}", f"t:{cluster}")
    smallest: dict[str, str] = {}
    for entity in entities:
        root = uf.find(f"e:{entity}")
        if root not in smallest or entity < smallest[root]:
            smallest[root] = entity
    return {e: "grp_" + smallest[uf.find(f"e:{e}")].removeprefix("ve_") for e in entities}


def version_key(dt: str, source: str, rid: str) -> tuple[str, int, str]:
    """Order of versions: the latest day, the API before a snapshot of that day, then the id."""
    return (dt, -SOURCE_RANK[source], rid)


def latest(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(rows, key=lambda r: version_key(str(r["dt"]), str(r["source"]), record_id(r["source"], r["source_id"])))


# --- build ----------------------------------------------------------------------------------

_LATEST_SQL = """
SELECT e.source AS source, e.source_id AS source_id, e.vacancy_entity_id AS vacancy_entity_id,
       e.text_cluster_id AS text_cluster_id, toString(s.dt) AS dt, s.content_hash AS content_hash
FROM staging.vacancy_entities AS e
INNER JOIN (
    SELECT source, source_id, dt, content_hash
    FROM staging.stg_vacancies
    WHERE is_it_candidate AND source IN ('trudvsem', 'trudvsem_snapshot', 'habr')
    ORDER BY dt DESC
    LIMIT 1 BY source, source_id
) AS s ON e.source = s.source AND e.source_id = s.source_id
FORMAT JSONEachRow
"""

_RAW_SQL = """
SELECT source, source_id, toString(dt) AS day, content_hash, title, text, payload
FROM raw.vacancies
WHERE (source, source_id, dt, content_hash) IN ({keys})
ORDER BY fetched_at DESC
LIMIT 1 BY source, source_id, dt, content_hash
FORMAT JSONEachRow
"""

RAW_BATCH = 1000


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def fetch_raw(ch: Any, keys: Sequence[tuple[str, str, str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for start in range(0, len(keys), RAW_BATCH):
        batch = keys[start : start + RAW_BATCH]
        literal = ", ".join(f"({_quote(s)}, {_quote(i)}, toDate({_quote(d)}), {_quote(h)})" for s, i, d, h in batch)
        body = ch.query(_RAW_SQL.format(keys=literal))
        for line in body.splitlines():
            if line:
                row = json.loads(line)
                out[(row["source"], row["source_id"])] = row
    missing = len(keys) - len(out)
    if missing:
        raise RuntimeError(f"{missing} staged records have no raw row with the same content_hash")
    return out


def text_chars(text: str | None) -> int:
    return len((text or "").strip())


def specializations(source: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    if source != "habr":
        return ()
    return tuple(str(d["title"]) for d in payload.get("divisions") or () if d.get("title"))


def make_record(staged: Mapping[str, Any], raw_row: Mapping[str, Any], entity: str, cluster: str, group: str) -> Record:
    source, source_id = str(staged["source"]), str(staged["source_id"])
    payload = json.loads(raw_row["payload"]) if isinstance(raw_row["payload"], str) else raw_row["payload"]
    specs = specializations(source, payload)
    title = str(raw_row.get("title") or "")
    return Record(
        id=record_id(source, source_id),
        source=source,
        source_id=source_id,
        dt=str(staged["dt"]),
        content_hash=str(staged["content_hash"]),
        vacancy_entity_id=entity,
        text_cluster_id=cluster,
        group_id=group,
        title=title,
        specializations=specs,
        role_proxy=role_proxy(title, specs),
        text_chars=text_chars(raw_row.get("text")),
        input=input_for_record({**raw_row, "payload": payload}),
    )


def build_records(
    staged: Sequence[Mapping[str, Any]],
    raw: Mapping[tuple[str, str], Mapping[str, Any]],
    group_of: Mapping[str, str],
) -> list[Record]:
    """One record per entity and group (record id -> group in `group_of`): its latest version among
    records with enough text. In `build` an entity lies inside one group; `refresh` can split an
    entity between a frozen gold group and the records that joined it later."""
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in staged:
        raw_row = raw[(str(r["source"]), str(r["source_id"]))]
        if text_chars(raw_row.get("text")) >= MIN_TEXT_CHARS:
            rid = record_id(r["source"], r["source_id"])
            by_key[(str(r["vacancy_entity_id"]), group_of[rid])].append(r)
    records = []
    for (entity, group), rows in by_key.items():
        r = latest(rows)
        raw_row = raw[(str(r["source"]), str(r["source_id"]))]
        records.append(make_record(r, raw_row, entity, str(r["text_cluster_id"]), group))
    return sorted(records, key=lambda rec: rec.id)


def fields(*columns: "pa.Field[Any]") -> pa.Schema:
    return pa.schema(columns)


CORPUS_SCHEMA = fields(
    pa.field("id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("dt", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("vacancy_entity_id", pa.string()),
    pa.field("text_cluster_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("specializations", pa.list_(pa.string())),
    pa.field("role_proxy", pa.string()),
    pa.field("text_chars", pa.int32()),
    pa.field("input", pa.string()),
)

# Every staged record, corpus or not: groups are closed over all of them.
MEMBERS_SCHEMA = fields(
    pa.field("id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("vacancy_entity_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("text_chars", pa.int32()),
)


def write_corpus(records: Sequence[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(r), "specializations": list(r.specializations)} for r in records]
    pq.write_table(pa.Table.from_pylist(rows, schema=CORPUS_SCHEMA), path)


def read_corpus(path: Path) -> list[Record]:
    return [
        Record(**{**row, "specializations": tuple(row["specializations"])}) for row in pq.read_table(path).to_pylist()
    ]


def load_staged() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """The latest staged version of every IT candidate with its entity, and its raw row."""
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        staged = [json.loads(line) for line in ch.query(_LATEST_SQL).splitlines() if line]
        keys = sorted({(r["source"], r["source_id"], r["dt"], r["content_hash"]) for r in staged})
        raw = fetch_raw(ch, keys)
    return staged, raw


def cmd_build(_: argparse.Namespace) -> int:
    staged, raw = load_staged()
    groups = group_ids((str(r["vacancy_entity_id"]), str(r["text_cluster_id"])) for r in staged)
    records = build_records(
        staged, raw, {record_id(r["source"], r["source_id"]): groups[str(r["vacancy_entity_id"])] for r in staged}
    )
    members = [
        {
            "id": record_id(r["source"], r["source_id"]),
            "source": r["source"],
            "vacancy_entity_id": r["vacancy_entity_id"],
            "group_id": groups[str(r["vacancy_entity_id"])],
            "text_chars": text_chars(raw[(r["source"], r["source_id"])].get("text")),
        }
        for r in sorted(staged, key=lambda r: (r["source"], r["source_id"]))
    ]
    out = corpus_dir()
    write_corpus(records, out / "corpus.parquet")
    pq.write_table(pa.Table.from_pylist(members, schema=MEMBERS_SCHEMA), out / "members.parquet")
    log.info(
        "%d staged records, %d corpus records in %d groups: %s",
        len(staged),
        len(records),
        len({r.group_id for r in records}),
        dict(Counter(r.source for r in records)),
    )
    return 0


# --- samples --------------------------------------------------------------------------------

SPLITS = ("test", "dev")
TRAIN_POOL = "train_pool"
# A training-pool group with a near duplicate of a test or dev record: out of training, not gold.
WITHHELD = "withheld"


@dataclass(frozen=True)
class SampleConfig:
    targets: dict[str, dict[str, int]]
    power: float
    max_share: float
    leak_jaccard: float
    shingle_k: int
    num_perm: int
    seed: int


def load_sample_config(path: Path = SAMPLE_CONFIG, eval_path: Path = EVAL_CONFIG) -> SampleConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    targets = {split: {b: int(raw[split][b]) for b in BUCKETS} for split in SPLITS}
    return SampleConfig(
        targets=targets,
        power=float(raw["allocation"]["power"]),
        max_share=float(raw["allocation"]["max_share"]),
        leak_jaccard=float(raw["leak"]["jaccard"]),
        shingle_k=int(raw["leak"]["shingle_k"]),
        num_perm=int(raw["leak"]["num_perm"]),
        seed=int(tomllib.loads(eval_path.read_text(encoding="utf-8"))["bootstrap"]["seed"]),
    )


def fixture_ids(root: Path = FIXTURES) -> set[str]:
    """Record ids that appear in repository fixtures, read while writing parsers and inputs."""
    out: set[str] = set()
    for path in sorted((root / "trudvsem").rglob("*")):
        if path.is_file():
            for uuid in _UUID.findall(path.read_text(encoding="utf-8", errors="replace").lower()):
                out.update({record_id("trudvsem", uuid), record_id("trudvsem_snapshot", uuid)})
    for path in sorted((root / "habr").rglob("*")):
        if path.is_file():
            out.update(
                record_id("habr", i) for i in _HABR_ID.findall(path.read_text(encoding="utf-8", errors="replace"))
            )
    return out


def permanent_random(seed: int, key: str) -> float:
    """A fixed uniform number per key: adding groups later does not reorder the existing ones."""
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64


def largest_remainder(shares: Mapping[str, float], total: int) -> dict[str, int]:
    """Integers summing to `total` whose distance to `shares` is smallest (Hamilton's method)."""
    base = {k: math.floor(v) for k, v in shares.items()}
    left = total - sum(base.values())
    for k in sorted(shares, key=lambda k: (-(shares[k] - base[k]), k))[: max(left, 0)]:
        base[k] += 1
    return base


def allocate(target: int, sizes: Mapping[str, int], power: float, max_share: float) -> dict[str, int]:
    """Spread `target` over strata in proportion to size ** power, none above max_share of its size.

    What a capped stratum cannot take goes to the others; if all are capped the result is short of
    the target, and the caller reports the shortfall.
    """
    caps = {k: math.floor(max_share * n) for k, n in sizes.items()}
    alloc = dict.fromkeys(sizes, 0)
    while True:
        active = [k for k in sizes if alloc[k] < caps[k]]
        remaining = target - sum(alloc.values())
        if remaining <= 0 or not active:
            return alloc
        weights = {k: sizes[k] ** power for k in active}
        total = sum(weights.values())
        give = largest_remainder({k: remaining * w / total for k, w in weights.items()}, remaining)
        for k in active:
            alloc[k] = min(caps[k], alloc[k] + give[k])


def stratum(source: str, bucket: str) -> str:
    return f"{source}|{bucket}"


@dataclass(frozen=True)
class Plan:
    """Groups to draw per stratum and split, and the frame they are drawn from."""

    frame: dict[str, int]
    draw: dict[str, dict[str, int]]
    shortfall: dict[str, dict[str, int]]


def plan(frame: Mapping[str, int], cfg: SampleConfig) -> Plan:
    draw: dict[str, dict[str, int]] = {s: dict.fromkeys(frame, 0) for s in SPLITS}
    shortfall: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for bucket in BUCKETS:
        sizes = {h: n for h, n in frame.items() if h.endswith(f"|{bucket}")}
        want = {s: cfg.targets[s][bucket] for s in SPLITS}
        both = allocate(sum(want.values()), sizes, cfg.power, cfg.max_share)
        got = sum(both.values())
        test_total = round(got * want["test"] / sum(want.values())) if got else 0
        test = largest_remainder({h: n * test_total / got for h, n in both.items()} if got else {}, test_total)
        for h, n in both.items():
            draw["test"][h] = test.get(h, 0)
            draw["dev"][h] = n - test.get(h, 0)
        for s in SPLITS:
            short = want[s] - sum(draw[s][h] for h in sizes)
            if short > 0:
                shortfall[s][bucket] = short
    return Plan(frame=dict(frame), draw=draw, shortfall=shortfall)


@dataclass(frozen=True)
class GroupRow:
    group_id: str
    split: str
    stratum: str | None
    gold_id: str | None
    pi_test: float | None
    pi_dev: float | None
    n_corpus: int
    member_ids: tuple[str, ...]


def representatives(records: Iterable[Record]) -> dict[str, Record]:
    """The group's latest corpus record: the one a test or dev group contributes to gold."""
    best: dict[str, Record] = {}
    for r in records:
        cur = best.get(r.group_id)
        if cur is None or version_key(r.dt, r.source, r.id) > version_key(cur.dt, cur.source, cur.id):
            best[r.group_id] = r
    return best


def draw_splits(
    records: Sequence[Record], members: Sequence[Mapping[str, Any]], excluded_ids: set[str], cfg: SampleConfig
) -> tuple[list[GroupRow], Plan]:
    reps = representatives(records)
    member_ids: dict[str, list[str]] = defaultdict(list)
    for m in members:
        member_ids[str(m["group_id"])].append(str(m["id"]))
    n_corpus = Counter(r.group_id for r in records)
    blocked = {g for g, ids in member_ids.items() if excluded_ids.intersection(ids)}

    strata: dict[str, list[str]] = defaultdict(list)
    for g, rep in reps.items():
        if g not in blocked:
            strata[stratum(rep.source, rep.role_proxy)].append(g)
    the_plan = plan({h: len(gs) for h, gs in strata.items()}, cfg)

    split_of: dict[str, str] = {}
    for h, gs in strata.items():
        ordered = sorted(gs, key=lambda g: (permanent_random(cfg.seed, g), g))
        n_test, n_dev = the_plan.draw["test"][h], the_plan.draw["dev"][h]
        for g in ordered[:n_test]:
            split_of[g] = "test"
        for g in ordered[n_test : n_test + n_dev]:
            split_of[g] = "dev"

    rows = []
    for g in sorted(member_ids):
        rep_of = reps.get(g)
        cell = stratum(rep_of.source, rep_of.role_proxy) if rep_of and g not in blocked else None
        split = split_of.get(g, TRAIN_POOL)
        rows.append(
            GroupRow(
                group_id=g,
                split=split,
                stratum=cell,
                gold_id=rep_of.id if rep_of and split in SPLITS else None,
                pi_test=the_plan.draw["test"][cell] / the_plan.frame[cell] if cell else None,
                pi_dev=the_plan.draw["dev"][cell] / the_plan.frame[cell] if cell else None,
                n_corpus=n_corpus.get(g, 0),
                member_ids=tuple(sorted(member_ids[g])),
            )
        )
    return rows, the_plan


@dataclass(frozen=True)
class LeakReport:
    jaccard: float
    neighbours: dict[str, dict[str, list[str]]]
    test_dev_pairs: list[tuple[str, str, float]]


def near_duplicates(rows: Sequence[GroupRow], records: Sequence[Record], cfg: SampleConfig) -> LeakReport:
    """Gold records of each split against every training-pool corpus record, by exact Jaccard of
    word shingles over the extractor input; MinHash LSH only proposes the pairs."""
    from datasketch import MinHashLSH  # type: ignore[import-untyped]

    from radar.dedup.minhash import signatures
    from radar.dedup.normalize import jaccard, shingles

    split_of = {r.group_id: r.split for r in rows}
    gold = {s: sorted(r.gold_id for r in rows if r.split == s and r.gold_id) for s in SPLITS}
    by_id = {r.id: r for r in records}
    pool = [r for r in records if split_of[r.group_id] == TRAIN_POOL]
    query_ids = gold["test"] + gold["dev"]

    docs = [shingles(r.input, cfg.shingle_k) for r in pool] + [
        shingles(by_id[i].input, cfg.shingle_k) for i in query_ids
    ]
    sigs = signatures(docs, cfg.num_perm, cfg.seed)
    lsh = MinHashLSH(threshold=cfg.leak_jaccard, num_perm=cfg.num_perm)
    with lsh.insertion_session() as session:
        for i, sig in enumerate(sigs[: len(pool)]):
            if sig is not None:
                session.insert(i, sig, check_duplication=False)

    neighbours: dict[str, dict[str, list[str]]] = {s: {} for s in SPLITS}
    for k, gid in enumerate(query_ids):
        q = len(pool) + k
        if sigs[q] is None:
            continue
        hits = sorted(pool[j].id for j in lsh.query(sigs[q]) if jaccard(docs[q], docs[j]) >= cfg.leak_jaccard)
        if hits:
            neighbours["test" if k < len(gold["test"]) else "dev"][gid] = hits

    test_docs = {i: docs[len(pool) + k] for k, i in enumerate(query_ids)}
    cross = [
        (t, d, round(jaccard(test_docs[t], test_docs[d]), 4))
        for t in gold["test"]
        for d in gold["dev"]
        if jaccard(test_docs[t], test_docs[d]) >= cfg.leak_jaccard
    ]
    return LeakReport(jaccard=cfg.leak_jaccard, neighbours=neighbours, test_dev_pairs=cross)


def withhold(rows: Sequence[GroupRow], records: Sequence[Record], leak: LeakReport) -> list[GroupRow]:
    group_of = {r.id: r.group_id for r in records}
    near = {group_of[i] for per_split in leak.neighbours.values() for ids in per_split.values() for i in ids}
    return [
        GroupRow(**{**asdict(r), "split": WITHHELD}) if r.group_id in near and r.split == TRAIN_POOL else r
        for r in rows
    ]


SPLIT_SCHEMA = fields(
    pa.field("group_id", pa.string()),
    pa.field("split", pa.string()),
    pa.field("stratum", pa.string()),
    pa.field("gold_id", pa.string()),
    pa.field("pi_test", pa.float64()),
    pa.field("pi_dev", pa.float64()),
    pa.field("n_corpus", pa.int32()),
    pa.field("member_ids", pa.list_(pa.string())),
)


def id_list_digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def split_digest(rows: Iterable[GroupRow]) -> str:
    """Content digest of the split, independent of the parquet encoder: record id, group, split."""
    lines = sorted(f"{m}\t{r.group_id}\t{r.split}" for r in rows for m in r.member_ids)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def sample_stats(
    records: Sequence[Record],
    members: Sequence[Mapping[str, Any]],
    rows: Sequence[GroupRow],
    the_plan: Plan,
    leak: LeakReport,
    leak_after: LeakReport,
    n_blocked: int,
) -> dict[str, dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    staged = Counter(str(x["source"]) for x in members)
    short = Counter(str(x["source"]) for x in members if int(x["text_chars"]) < MIN_TEXT_CHARS)
    corpus = Counter(r.source for r in records)
    m["staged.records"] = {"value": len(members)}
    m["corpus.records"] = share(len(records), len(members))
    m["corpus.groups"] = {"value": len({r.group_id for r in records})}
    for s in SOURCES:
        m[f"staged.records.{s}"] = {"value": staged[s]}
        m[f"staged.short_text.{s}"] = share(short[s], staged[s])
        m[f"corpus.records.{s}"] = share(corpus[s], len(records))

    by_bucket = Counter(r.role_proxy for r in records)
    for b in BUCKETS:
        m[f"role_proxy.corpus.{b}"] = share(by_bucket[b], len(records))
    for s in SOURCES:
        per = Counter(r.role_proxy for r in records if r.source == s)
        for b in BUCKETS:
            m[f"role_proxy.corpus.{s}.{b}"] = share(per[b], corpus[s])
    reps = representatives(records)
    m["role_proxy.corpus_groups.ml_ds_data"] = share(sum(r.role_proxy == ML_DS_DATA for r in reps.values()), len(reps))

    gold_by_id = {r.id: r for r in records}
    for split in SPLITS:
        gold = [gold_by_id[r.gold_id] for r in rows if r.split == split and r.gold_id]
        m[f"{split}.records"] = {"value": len(gold)}
        per_bucket = Counter(r.role_proxy for r in gold)
        for b in BUCKETS:
            m[f"{split}.role_proxy.{b}"] = share(per_bucket[b], len(gold))
        per_source = Counter(r.source for r in gold)
        for s in SOURCES:
            m[f"{split}.source.{s}"] = share(per_source[s], len(gold))
            ml = sum(r.role_proxy == ML_DS_DATA for r in gold if r.source == s)
            m[f"{split}.ml_ds_data.{s}"] = share(ml, per_source[s])
        m[f"leak.{split}_with_train_neighbour_drawn"] = share(len(leak.neighbours[split]), len(gold))
        m[f"leak.{split}_with_train_neighbour"] = share(len(leak_after.neighbours[split]), len(gold))
    m["leak.test_dev_pairs"] = {"value": len(leak.test_dev_pairs)}
    m["leak.withheld_groups"] = {"value": sum(r.split == WITHHELD for r in rows)}
    m["leak.records_in_two_splits"] = {"value": records_in_two_splits(rows)}
    m["frame.groups"] = {"value": sum(the_plan.frame.values())}
    m["frame.blocked_by_fixtures"] = {"value": n_blocked}
    for s in SOURCES:
        ml = sum(r.role_proxy == ML_DS_DATA for r in records if r.source == s)
        m[f"decisions10.corpus.ml_ds_data.{s}"] = share(ml, corpus[s])
    pool = {r.group_id for r in rows if r.split == TRAIN_POOL}
    pool_ml = sum(1 for r in records if r.group_id in pool and r.role_proxy == ML_DS_DATA)
    m["decisions10.train_pool.ml_ds_data"] = share(pool_ml, sum(1 for r in records if r.group_id in pool))
    return m


def records_in_two_splits(rows: Sequence[GroupRow]) -> int:
    """Record ids listed under more than one split; a group is one row, so this checks members."""
    seen: dict[str, str] = {}
    clashes = 0
    for r in rows:
        for i in r.member_ids:
            if i in seen and seen[i] != r.split:
                clashes += 1
            seen[i] = r.split
    return clashes


def write_inputs(path: Path, gold: Sequence[Record], seed: int) -> None:
    """id, source and input only: no proxy or stratum, so the annotators see what the extractor sees."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(gold, key=lambda r: (permanent_random(seed, r.id), r.id))
    lines = [json.dumps({"id": r.id, "source": r.source, "input": r.input}, ensure_ascii=False) for r in ordered]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_sample(args: argparse.Namespace) -> int:
    from radar.results import write_result

    cfg = load_sample_config()
    out_gold, out_corpus = gold_dir(), corpus_dir()
    targets = [out_gold / f"{s}_inputs.jsonl" for s in SPLITS] + [out_corpus / "split.parquet"]
    labeled = [out_gold / f"{s}.jsonl" for s in SPLITS]
    if any(p.exists() for p in labeled):
        raise SystemExit("gold labels exist: test and dev are fixed, redrawing would detach them")
    if any(p.exists() for p in targets) and not args.resample:
        raise SystemExit("test and dev are drawn already; --resample redraws them (record it in EXPERIMENTS)")

    records = read_corpus(out_corpus / "corpus.parquet")
    members = pq.read_table(out_corpus / "members.parquet").to_pylist()
    excluded = fixture_ids()
    rows, the_plan = draw_splits(records, members, excluded, cfg)
    n_blocked = len({str(m["group_id"]) for m in members if str(m["id"]) in excluded})
    leak = near_duplicates(rows, records, cfg)
    rows = withhold(rows, records, leak)
    leak_after = near_duplicates(rows, records, cfg)

    by_id = {r.id: r for r in records}
    for split in SPLITS:
        gold = [by_id[r.gold_id] for r in rows if r.split == split and r.gold_id]
        write_inputs(out_gold / f"{split}_inputs.jsonl", gold, cfg.seed)
    split_path = out_corpus / "split.parquet"
    pq.write_table(
        pa.Table.from_pylist([{**asdict(r), "member_ids": list(r.member_ids)} for r in rows], schema=SPLIT_SCHEMA),
        split_path,
    )

    gold_ids = {s: [r.gold_id for r in rows if r.split == s and r.gold_id] for s in SPLITS}
    pool_groups = {r.group_id for r in rows if r.split == TRAIN_POOL}
    hashes = {
        "test_ids_sha256": id_list_digest(gold_ids["test"]),
        "dev_ids_sha256": id_list_digest(gold_ids["dev"]),
        "train_pool_corpus_ids_sha256": id_list_digest(r.id for r in records if r.group_id in pool_groups),
        "split_content_sha256": split_digest(rows),
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    strata = [
        {
            "stratum": h,
            "frame_groups": n,
            "test": the_plan.draw["test"][h],
            "dev": the_plan.draw["dev"][h],
            "pi_test": round(the_plan.draw["test"][h] / n, 6),
            "pi_dev": round(the_plan.draw["dev"][h] / n, 6),
        }
        for h, n in sorted(the_plan.frame.items())
    ]
    path = write_result(
        "gold/sample_stats.json",
        sample_stats(records, members, rows, the_plan, leak, leak_after, n_blocked),
        source_script="src/radar/extract/corpus.py",
        min_text_chars=MIN_TEXT_CHARS,
        config={
            "targets": cfg.targets,
            "power": cfg.power,
            "max_share": cfg.max_share,
            "seed": cfg.seed,
            "leak_jaccard": cfg.leak_jaccard,
        },
        shortfall=the_plan.shortfall,
        strata=strata,
        hashes=hashes,
        leak_neighbours=leak.neighbours,
        leak_test_dev_pairs=leak.test_dev_pairs,
    )
    log.info("test %d, dev %d; stats in %s", len(gold_ids["test"]), len(gold_ids["dev"]), path)
    for k, v in hashes.items():
        log.info("%s %s", k, v)
    return 0


S1_DIR = "s1"


def cmd_refresh(_: argparse.Namespace) -> int:
    """See radar.extract.refresh."""
    from radar.collect.writer import raw_root
    from radar.extract import refresh as rf
    from radar.results import write_result

    cfg = load_sample_config()
    out_corpus, out_gold = corpus_dir(), gold_dir()
    s1_hashes = json.loads((ROOT / "results" / "gold" / "sample_stats.json").read_text(encoding="utf-8"))["hashes"]
    s1 = rf.load_s1(out_corpus / S1_DIR, s1_hashes["split_file_sha256"])
    staged, raw = load_staged()
    root = raw_root()
    result = rf.refresh(staged, raw, s1, cfg, lambda source, day, ids: rf.raw_versions(root, source, day, ids))
    metrics, extra = rf.refresh_stats(result, s1, staged, s1_hashes)
    failed = [k for k, v in metrics.items() if k.startswith("check.") and v["value"] != "true"]
    if failed or result.leak_after.neighbours != {s: {} for s in SPLITS}:
        raise SystemExit(f"refresh checks failed, nothing written: {failed or 'training pool still has neighbours'}")

    write_corpus(result.records, out_corpus / "corpus.parquet")
    rf.write_members(result.members, out_corpus / "members.parquet")
    split_path = out_corpus / "split.parquet"
    rf.write_split(result.rows, split_path)
    by_id = {r.id: r for r in result.records}
    for split in SPLITS:
        rf.write_gold_inputs(out_gold / f"{split}_inputs.jsonl", s1.inputs[split], by_id)
    extra["hashes"]["split_file_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    path = write_result(
        "corpus/refresh.json",
        metrics,
        source_script="src/radar/extract/refresh.py",
        min_text_chars=MIN_TEXT_CHARS,
        leak_jaccard=cfg.leak_jaccard,
        staging=staging_coverage(),
        **extra,
    )
    log.info(
        "%d corpus records; train_pool %s, withheld %s; ML/DS/data in train_pool %s; %s",
        len(result.records),
        metrics["split.train_pool.corpus_records"]["value"],
        metrics["split.withheld.corpus_records"]["value"],
        metrics["decisions10.train_pool.ml_ds_data"],
        path,
    )
    return 0


_COVERAGE_SQL = """
SELECT source, toString(min(dt)) AS first, toString(max(dt)) AS last, uniqExact(dt) AS days,
       uniqExactIf((source, source_id), is_it_candidate) AS it_candidates,
       uniqExactIf((source, source_id), is_it_candidate
                   AND (source, source_id) NOT IN (SELECT source, source_id FROM staging.vacancy_entities))
           AS it_candidates_without_entity
FROM staging.stg_vacancies
WHERE source IN ('trudvsem', 'trudvsem_snapshot', 'habr')
GROUP BY source
ORDER BY source
FORMAT JSONEachRow
"""


def staging_coverage() -> dict[str, dict[str, Any]]:
    """Days in staging per source, and IT candidates the dedup has not given an entity yet (they
    are not in the corpus until `radar.dedup` runs again)."""
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        rows = [json.loads(line) for line in ch.query(_COVERAGE_SQL).splitlines() if line]
    return {r.pop("source"): r for r in rows}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m radar.extract.corpus", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="corpus from ClickHouse staging and raw").set_defaults(func=cmd_build)
    sample = sub.add_parser("sample", help="draw test and dev, write the split and results/gold/sample_stats.json")
    sample.add_argument("--resample", action="store_true", help="redraw unlabeled samples")
    sample.set_defaults(func=cmd_sample)
    sub.add_parser(
        "refresh", help="rebuild corpus and gold inputs from current staging, test and dev as in S1"
    ).set_defaults(func=cmd_refresh)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
