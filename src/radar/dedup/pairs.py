"""Candidate pairs of the two methods, their bands, and the stratified sample for labeling.

Blocking: a pair of groups is a candidate when MinHash LSH puts the two texts in a common band
(word shingles, units with text only) or when one is among the other's top-k e5 neighbours above a
cosine floor. Nothing else restricts a candidate: sources and employers may differ.

The labeling sample covers what the candidates can miss as well as what they find. Strata are
disjoint and taken in this order, a pair belongs to the first that holds:

  high_minhash      candidate, jaccard >= jaccard_high
  high_embed        candidate, cosine >= cosine_high
  border_minhash    candidate, jaccard >= jaccard_border
  border_embed      candidate, cosine >= cosine_border
  cross_habr        Habr against trudvsem, same normalized employer name or title
  cross_trudvsem    trudvsem API against a snapshot-only id, same employer
  same_employer_low same source, same employer

A pair is drawn through a unit, not uniformly over pairs: a unit uniformly among those with a
partner in the stratum, then a partner uniformly. One employer posting the same text in 70 regions
makes thousands of pairs, and uniform pairs would spend a whole stratum on it. Each drawn pair
carries its per-draw probability (1/A)(1/d_a + 1/d_b), A units with a partner, d their partner
counts, so stage B can weight estimates back to pairs (Hansen-Hurwitz). A stratum no larger than
its allocation is taken whole.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from radar.dedup.settings import Params

STRATA = (
    "high_minhash",
    "high_embed",
    "border_minhash",
    "border_embed",
    "cross_habr",
    "cross_trudvsem",
    "same_employer_low",
)
BAND_STRATA = STRATA[:4]


@dataclass(frozen=True)
class Candidate:
    i: int
    j: int
    jaccard: float
    cosine: float
    by_minhash: bool
    by_embedding: bool


def band(c: Candidate, p: Params) -> str | None:
    if c.jaccard >= p.jaccard_high:
        return "high_minhash"
    if c.cosine >= p.cosine_high:
        return "high_embed"
    if c.jaccard >= p.jaccard_border:
        return "border_minhash"
    if c.cosine >= p.cosine_border:
        return "border_embed"
    return None


def merge_candidates(
    lsh: set[tuple[int, int]],
    knn: dict[tuple[int, int], float],
    jaccard_of: Callable[[int, int], float],
    cosine_of: Callable[[int, int], float],
) -> list[Candidate]:
    out = []
    for i, j in sorted(lsh | knn.keys()):
        cos = knn.get((i, j))
        out.append(
            Candidate(
                i=i,
                j=j,
                jaccard=jaccard_of(i, j),
                cosine=cos if cos is not None else cosine_of(i, j),
                by_minhash=(i, j) in lsh,
                by_embedding=(i, j) in knn,
            )
        )
    return out


class PairSpace(Protocol):
    def degree(self, i: int) -> int: ...

    def partner(self, i: int, rng: random.Random) -> int: ...

    def partners(self, i: int) -> list[int]: ...

    def anchors(self) -> list[int]: ...


class ListedPairs:
    """A stratum whose pairs are all listed."""

    def __init__(self, pairs: Iterable[tuple[int, int]]) -> None:
        adj: dict[int, list[int]] = defaultdict(list)
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)
        self.adj = {i: sorted(js) for i, js in adj.items()}

    def degree(self, i: int) -> int:
        return len(self.adj.get(i, ()))

    def partner(self, i: int, rng: random.Random) -> int:
        return rng.choice(self.adj[i])

    def partners(self, i: int) -> list[int]:
        return self.adj.get(i, [])

    def anchors(self) -> list[int]:
        return sorted(self.adj)


class BlockPairs:
    """Pairs of units with one block key, minus excluded pairs, without listing them all.

    With `side` given, a pair takes one unit from each side (API against snapshot); without it, any
    two units of the block. `excluded` is an adjacency of pairs that belong to an earlier stratum.
    """

    def __init__(
        self,
        key_of: Sequence[Hashable | None],
        excluded: dict[int, set[int]],
        side: Sequence[int] | None = None,
    ) -> None:
        self.key_of = key_of
        self.side = side
        self.excluded = excluded
        self.mates: dict[tuple[Hashable, int], list[int]] = defaultdict(list)
        for i, key in enumerate(key_of):
            if key is not None:
                self.mates[(key, side[i] if side is not None else 0)].append(i)
        self._degree = {i: self._count(i) for i, key in enumerate(key_of) if key is not None}

    def _others(self, i: int) -> list[int]:
        key = self.key_of[i]
        if self.side is None:
            return self.mates[(key, 0)]
        return self.mates.get((key, 1 - self.side[i]), [])

    def _eligible(self, i: int, j: int) -> bool:
        if j == i or self.key_of[j] != self.key_of[i]:
            return False
        return self.side is None or self.side[i] != self.side[j]

    def _count(self, i: int) -> int:
        others = len(self._others(i)) - (1 if self.side is None else 0)
        return others - sum(1 for j in self.excluded.get(i, ()) if self._eligible(i, j))

    def degree(self, i: int) -> int:
        return self._degree.get(i, 0)

    def partner(self, i: int, rng: random.Random) -> int:
        others = self._others(i)
        while True:
            j = rng.choice(others)
            if j != i and j not in self.excluded.get(i, ()):
                return j

    def partners(self, i: int) -> list[int]:
        return [j for j in self._others(i) if self._eligible(i, j) and j not in self.excluded.get(i, ())]

    def anchors(self) -> list[int]:
        return sorted(i for i, d in self._degree.items() if d > 0)


@dataclass
class Drawn:
    pair: tuple[int, int]
    p_draw: float


@dataclass
class Stratum:
    name: str
    population: int
    anchors: int
    census: bool
    draws: int
    pairs: list[Drawn]


def population(space: PairSpace) -> int:
    return sum(space.degree(i) for i in space.anchors()) // 2


def sample_stratum(name: str, space: PairSpace, k: int, rng: random.Random) -> Stratum:
    anchors = space.anchors()
    total = population(space)
    if total <= k:
        listed = sorted({(min(i, j), max(i, j)) for i in anchors for j in space.partners(i)})
        return Stratum(name, total, len(anchors), True, 0, [Drawn(pair, 1.0) for pair in listed])
    chosen: dict[tuple[int, int], Drawn] = {}
    draws = 0
    while len(chosen) < k:
        draws += 1
        if draws > 1000 * k:
            raise RuntimeError(f"{name}: sampling stalled at {len(chosen)} of {k} after {draws} draws")
        a = rng.choice(anchors)
        b = space.partner(a, rng)
        pair = (min(a, b), max(a, b))
        if pair not in chosen:
            p = (1 / space.degree(a) + 1 / space.degree(b)) / len(anchors)
            chosen[pair] = Drawn(pair, p)
    return Stratum(name, total, len(anchors), False, draws, list(chosen.values()))


def allocate(populations: dict[str, int], p: Params) -> dict[str, int]:
    """Allocation per stratum, capped by its population, the shortfall spilled in `p.spill` order."""
    out = {s: min(p.allocation[s], populations[s]) for s in STRATA}
    short = sum(p.allocation[s] for s in STRATA) - sum(out.values())
    for s in p.spill:
        room = populations[s] - out[s]
        take = min(room, short)
        out[s] += take
        short -= take
    return out


def split_tuning(pairs_by_stratum: dict[str, list[Any]], share: float, rng: random.Random) -> dict[Any, str]:
    """tuning or heldout per pair, half of each stratum each (the odd one to heldout)."""
    out: dict[Any, str] = {}
    for name in STRATA:
        items = list(pairs_by_stratum.get(name, []))
        rng.shuffle(items)
        cut = int(len(items) * share)
        for n, item in enumerate(items):
            out[item] = "tuning" if n < cut else "heldout"
    return out


def group_by(keys: Sequence[Hashable | None], members: Iterable[int]) -> dict[Hashable, list[int]]:
    out: dict[Hashable, list[int]] = defaultdict(list)
    for i in members:
        if keys[i] is not None:
            out[keys[i]].append(i)
    return dict(out)


def cross_pairs(left: dict[Hashable, list[int]], right: dict[Hashable, list[int]]) -> set[tuple[int, int]]:
    out = set()
    for key, members in left.items():
        for a in members:
            for b in right.get(key, []):
                out.add((min(a, b), max(a, b)))
    return out


@dataclass(frozen=True)
class Keys:
    """Per unit index: source, employer key (INN or name), employer name key, title key."""

    source: Sequence[str]
    employer: Sequence[str | None]
    employer_name: Sequence[str | None]
    title: Sequence[str | None]


def spaces(keys: Keys, candidates: Sequence[Candidate], p: Params) -> dict[str, PairSpace]:
    by_band: dict[str, list[tuple[int, int]]] = {s: [] for s in BAND_STRATA}
    for c in candidates:
        b = band(c, p)
        if b is not None:
            by_band[b].append((c.i, c.j))
    banded: dict[int, set[int]] = defaultdict(set)
    for pairs in by_band.values():
        for i, j in pairs:
            banded[i].add(j)
            banded[j].add(i)

    n = len(keys.source)
    habr = [i for i in range(n) if keys.source[i] == "habr"]
    trud = [i for i in range(n) if keys.source[i] in ("trudvsem", "trudvsem_snapshot")]
    cross_habr = cross_pairs(group_by(keys.employer_name, habr), group_by(keys.employer_name, trud)) | cross_pairs(
        group_by(keys.title, habr), group_by(keys.title, trud)
    )
    shared_portal = ("trudvsem", "trudvsem_snapshot")
    return {
        **{s: ListedPairs(by_band[s]) for s in BAND_STRATA},
        "cross_habr": ListedPairs((i, j) for i, j in cross_habr if j not in banded.get(i, ())),
        "cross_trudvsem": BlockPairs(
            [keys.employer[i] if keys.source[i] in shared_portal else None for i in range(n)],
            banded,
            side=[shared_portal.index(s) if s in shared_portal else -1 for s in keys.source],
        ),
        "same_employer_low": BlockPairs(
            [(keys.source[i], keys.employer[i]) if keys.employer[i] else None for i in range(n)], banded
        ),
    }


def build_strata(
    keys: Keys, candidates: Sequence[Candidate], p: Params, rng: random.Random
) -> tuple[list[Stratum], dict[str, int]]:
    """Strata with their populations and the drawn pairs; also the allocation actually used."""
    space = spaces(keys, candidates, p)
    take = allocate({s: population(space[s]) for s in STRATA}, p)
    return [sample_stratum(s, space[s], take[s], rng) for s in STRATA], take
