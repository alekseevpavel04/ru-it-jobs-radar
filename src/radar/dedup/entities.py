"""Entities: groups of records that are one vacancy, and text clusters for the gold split.

`vacancy_entity_id` is a connected component of the decision rule over candidate pairs, on top of
the id links between the trudvsem API and snapshots. Links are transitive, so templates chain: a
card that states only the region joins offices in two towns, a short legal name joins two
subsidiaries. So a component carries what its records state, and a link is refused when the merged
component would hold two INNs, two Habr employers, two settlements or two streets (remote cards
state no place). Links are taken strongest Jaccard first, so a refused link loses to the closer one.
A component still above `max_component` is recut by raising the Jaccard threshold inside it in
steps of 0.05; how many were recut is reported.

`text_cluster_id` ignores place and employer: records whose texts are near-identical (Jaccard at
or above the text_cluster threshold) share it, including one template posted in 70 regions. That
is the unit for splitting the gold set (5.2): the same text in train and test leaks whatever the
place. No constraints apply there, so chains are expected and the largest cluster is reported.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from radar.dedup.evaluate import links
from radar.dedup.normalize import normalize_title
from radar.dedup.place import Place, place_of, same_place
from radar.dedup.settings import Rule
from radar.dedup.units import Unit
from radar.stream.clickhouse import ClickHouse

TABLE = "staging.vacancy_entities"
STEP = 0.05


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
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

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    jaccard: float
    cosine: float


@dataclass(frozen=True)
class Traits:
    """What a record states that two records of one vacancy cannot disagree on."""

    inn: str | None
    habr_employer: str | None
    settlement: str | None
    street: frozenset[str] | None


def traits_of(u: Unit) -> Traits:
    place = place_of(u.location, u.address, u.remote, u.source)
    # A remote card states no place, except for a sole proprietor: the employer is hidden («ИП»),
    # so the place is all that tells two proprietors with one template apart (guide, «ИП»).
    stated = not place.remote or u.employer_key is None
    return Traits(
        inn=u.employer_inn,
        habr_employer=u.employer_name_key if u.source == "habr" else None,
        settlement=place.settlement if stated else None,
        street=place.street if stated else None,
    )


class ConstrainedUnionFind(UnionFind):
    """Union that refuses a merge whose component would state two different values of a trait."""

    FIELDS = ("inn", "habr_employer", "settlement", "street")

    def __init__(self, items: Iterable[str], traits: dict[str, Traits]) -> None:
        super().__init__(items)
        self.stated: dict[str, dict[str, set[object]]] = {
            x: {f: {v} if (v := getattr(traits[x], f)) is not None else set() for f in self.FIELDS} for x in self.parent
        }
        self.refused = 0

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        merged = {f: self.stated[ra][f] | self.stated[rb][f] for f in self.FIELDS}
        if any(len(v) > 1 for v in merged.values()):
            self.refused += 1
            return
        root, child = min(ra, rb), max(ra, rb)
        self.parent[child] = root
        self.stated[root] = merged


def components(
    nodes: Sequence[str],
    edges: Iterable[Edge],
    min_jaccard: float | None = None,
    traits: dict[str, Traits] | None = None,
) -> tuple[list[list[str]], int]:
    """Components, and how many links the traits refused (0 without traits)."""
    uf = UnionFind(nodes) if traits is None else ConstrainedUnionFind(nodes, traits)
    for e in sorted(edges, key=lambda e: (-e.jaccard, e.a, e.b)):
        if min_jaccard is None or e.jaccard >= min_jaccard:
            uf.union(e.a, e.b)
    refused = uf.refused if isinstance(uf, ConstrainedUnionFind) else 0
    return sorted((sorted(g) for g in uf.groups().values()), key=lambda g: g[0]), refused


def split_large(
    nodes: Sequence[str], edges: Sequence[Edge], start: float, max_size: int, traits: dict[str, Traits] | None = None
) -> tuple[list[list[str]], int, int]:
    """Components of size at most max_size: oversized ones are recut with a stricter Jaccard.

    Returns the components, how many links the traits refused and how many components were recut.
    """
    out: list[list[str]] = []
    recut = 0
    first, refused = components(nodes, edges, traits=traits)
    for comp in first:
        if len(comp) <= max_size:
            out.append(comp)
            continue
        recut += 1
        inside = set(comp)
        local = [e for e in edges if e.a in inside and e.b in inside]
        threshold = start
        parts = [comp]
        while any(len(p) > max_size for p in parts) and threshold < 1.0:
            threshold = round(threshold + STEP, 4)
            parts, _ = components(comp, local, min_jaccard=threshold, traits=traits)
        # Identical texts in more than max_size records of one place: nothing left to cut by.
        out.extend(parts)
    return sorted(out, key=lambda g: g[0]), refused, recut


def rule_edges(reps: dict[str, Unit], candidates: Iterable[Edge], rule: Rule) -> list[Edge]:
    """The candidate pairs (as edges with both scores) that the rule links."""
    places: dict[str, Place] = {}

    def place(u: Unit) -> Place:
        if u.uid not in places:
            places[u.uid] = place_of(u.location, u.address, u.remote, u.source)
        return places[u.uid]

    out = []
    for c in candidates:
        a, b = reps[c.a], reps[c.b]
        if links(
            rule,
            c.jaccard,
            c.cosine,
            same_place(place(a), place(b)),
            normalize_title(a.title) == normalize_title(b.title),
        ):
            out.append(c)
    return out


def stable_id(prefix: str, members: Sequence[str]) -> str:
    """Named after its smallest member uid, so an id survives reruns while that member stays in."""
    return prefix + hashlib.sha1(min(members).encode("utf-8")).hexdigest()[:12]


def mapping(
    all_units: Sequence[Unit], entity_groups: Sequence[Sequence[str]], text_groups: Sequence[Sequence[str]]
) -> list[dict[str, str]]:
    """One row per unit; groups are given over representatives and spread to the id-linked members."""
    members: dict[str, list[Unit]] = defaultdict(list)
    for u in all_units:
        members[u.group].append(u)
    rep_group = {u.uid: u.group for u in all_units}

    def spread(prefix: str, groups: Sequence[Sequence[str]]) -> dict[str, str]:
        out = {}
        for g in groups:
            uids = [m.uid for rep in g for m in members[rep_group[rep]]]
            ident = stable_id(prefix, uids)
            for uid in uids:
                out[uid] = ident
        return out

    entity, text = spread("ve_", entity_groups), spread("tc_", text_groups)
    return [
        {
            "source": u.source,
            "source_id": u.source_id,
            "vacancy_entity_id": entity[u.uid],
            "text_cluster_id": text[u.uid],
        }
        for u in sorted(all_units, key=lambda u: u.uid)
    ]


SCHEMA = pa.schema([pa.field(n, pa.string()) for n in ("source", "source_id", "vacancy_entity_id", "text_cluster_id")])


def write_mapping(rows: Sequence[dict[str, str]], path: str) -> None:
    pq.write_table(pa.Table.from_pylist(list(rows), schema=SCHEMA), path)


def load_clickhouse(ch: ClickHouse, rows: Sequence[dict[str, str]]) -> int:
    """Replace the table whole: fill a fresh copy, then swap it in, so readers never see it half full."""
    tmp = TABLE + "_new"
    ch.query(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (source LowCardinality(String), source_id String, "
        "vacancy_entity_id String, text_cluster_id String) ENGINE = MergeTree ORDER BY (source, source_id)"
    )
    ch.query(f"DROP TABLE IF EXISTS {tmp}")
    ch.query(f"CREATE TABLE {tmp} AS {TABLE}")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    ch.query(f"INSERT INTO {tmp} FORMAT JSONEachRow\n{body}")
    ch.query(f"EXCHANGE TABLES {tmp} AND {TABLE}")
    ch.query(f"DROP TABLE {tmp}")
    return ch.scalar(f"SELECT count() FROM {TABLE}")
