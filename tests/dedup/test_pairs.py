import random
from collections import Counter
from dataclasses import replace

import pytest

from radar.dedup.pairs import (
    STRATA,
    BlockPairs,
    Candidate,
    Keys,
    ListedPairs,
    allocate,
    band,
    build_strata,
    merge_candidates,
    population,
    sample_stratum,
    split_tuning,
)
from radar.dedup.settings import load

P = load().params


def cand(jaccard: float, cosine: float) -> Candidate:
    return Candidate(0, 1, jaccard, cosine, by_minhash=True, by_embedding=True)


@pytest.mark.parametrize(
    ("jaccard", "cosine", "expected"),
    [
        (0.9, 0.99, "high_minhash"),
        (0.5, 0.985, "high_embed"),
        (0.5, 0.96, "border_minhash"),
        (0.1, 0.96, "border_embed"),
        (0.1, 0.94, None),
    ],
)
def test_band_takes_the_first_rule_that_holds(jaccard: float, cosine: float, expected: str | None) -> None:
    assert band(cand(jaccard, cosine), P) == expected


def test_merge_scores_pairs_found_by_one_method_only() -> None:
    merged = merge_candidates({(0, 1)}, {(1, 2): 0.93}, lambda i, j: 0.7, lambda i, j: 0.5)
    assert merged == [Candidate(0, 1, 0.7, 0.5, True, False), Candidate(1, 2, 0.7, 0.93, False, True)]


def test_allocation_spills_a_short_stratum() -> None:
    populations = dict.fromkeys(STRATA, 1000) | {"high_embed": 10}
    out = allocate(populations, P)
    assert out["high_embed"] == 10
    assert out["border_embed"] == P.allocation["border_embed"] + P.allocation["high_embed"] - 10
    assert sum(out.values()) == sum(P.allocation.values())


def test_block_pairs_within_counts_partners_minus_excluded() -> None:
    # Block a: units 0, 1, 2, 3 with the pair 0-1 excluded; block b: 4, 5; unit 6 has no key.
    space = BlockPairs(["a", "a", "a", "a", "b", "b", None], excluded={0: {1}, 1: {0}})
    assert [space.degree(i) for i in range(7)] == [2, 2, 3, 3, 1, 1, 0]
    assert space.anchors() == [0, 1, 2, 3, 4, 5]
    assert population(space) == 6
    assert sorted(space.partners(0)) == [2, 3]
    rng = random.Random(0)
    assert {space.partner(0, rng) for _ in range(200)} == {2, 3}


def test_block_pairs_across_sides() -> None:
    # Units 0, 1 on side 0 and 2 on side 1 share key x; 3 is alone on its key.
    space = BlockPairs(["x", "x", "x", "y"], excluded={}, side=[0, 0, 1, 1])
    assert [space.degree(i) for i in range(4)] == [1, 1, 2, 0]
    assert population(space) == 2
    assert sorted(space.partners(2)) == [0, 1]


def test_draw_probability_matches_the_formula() -> None:
    # A star 0-1, 0-2, 0-3 and a lone pair 4-5: pairs through the hub are drawn less often each.
    space = ListedPairs([(0, 1), (0, 2), (0, 3), (4, 5)])
    rng = random.Random(11)
    n = 20000
    counts: Counter[tuple[int, int]] = Counter()
    probs: dict[tuple[int, int], float] = {}
    for _ in range(n):
        drawn = sample_stratum("s", space, 1, rng).pairs[0]
        counts[drawn.pair] += 1
        probs[drawn.pair] = drawn.p_draw
    assert probs[(4, 5)] == pytest.approx(2 / 6)
    assert probs[(0, 1)] == pytest.approx((1 / 3 + 1) / 6)
    assert sum(probs.values()) == pytest.approx(1.0)
    for pair, p in probs.items():
        assert counts[pair] / n == pytest.approx(p, abs=0.015)


def test_small_stratum_is_taken_whole() -> None:
    s = sample_stratum("s", ListedPairs([(0, 1), (2, 3)]), 5, random.Random(0))
    assert s.census and s.population == 2
    assert [(d.pair, d.p_draw) for d in s.pairs] == [((0, 1), 1.0), ((2, 3), 1.0)]


def synthetic() -> tuple[Keys, list[Candidate]]:
    # 0..9 API, 10..19 snapshot-only, 20..24 Habr. Employers e0..e2 cycle; Habr shares names e0, e1.
    n = 25
    source = ["trudvsem"] * 10 + ["trudvsem_snapshot"] * 10 + ["habr"] * 5
    employer = [f"inn:e{i % 3}" for i in range(20)] + [None] * 5
    employer_name = [f"e{i % 3}" for i in range(20)] + ["e0", "e1", "e9", None, None]
    title = [f"t{i % 4}" for i in range(n)]
    candidates = [
        Candidate(0, 3, 0.95, 0.99, True, True),
        Candidate(1, 4, 0.3, 0.985, False, True),
        Candidate(2, 5, 0.5, 0.96, True, True),
        Candidate(6, 9, 0.1, 0.96, False, True),
        Candidate(0, 13, 0.1, 0.94, False, True),
    ]
    return Keys(source, employer, employer_name, title), candidates


def test_build_strata_is_disjoint_and_deterministic() -> None:
    keys, candidates = synthetic()
    p = replace(P, allocation=dict.fromkeys(STRATA, 3), spill=())
    strata, take = build_strata(keys, candidates, p, random.Random(7))
    again, _ = build_strata(keys, candidates, p, random.Random(7))
    assert [[d.pair for d in s.pairs] for s in strata] == [[d.pair for d in s.pairs] for s in again]

    by_name = {s.name: s for s in strata}
    assert [by_name[s].population for s in STRATA[:4]] == [1, 1, 1, 1]
    drawn = [d.pair for s in strata for d in s.pairs]
    assert len(drawn) == len(set(drawn)) == sum(take.values())
    banded = {(c.i, c.j) for c in candidates[:4]}
    assert not banded & {d.pair for s in strata[4:] for d in s.pairs}
    for d in by_name["cross_habr"].pairs:
        i, j = d.pair
        assert "habr" in {keys.source[i], keys.source[j]} and keys.source[i] != keys.source[j]
    for d in by_name["same_employer_low"].pairs:
        i, j = d.pair
        assert keys.source[i] == keys.source[j] and keys.employer[i] == keys.employer[j]
    for d in by_name["cross_trudvsem"].pairs:
        i, j = d.pair
        assert {keys.source[i], keys.source[j]} == {"trudvsem", "trudvsem_snapshot"}


def test_build_strata_populations_of_the_low_strata() -> None:
    keys, candidates = synthetic()
    strata, _ = build_strata(keys, candidates, P, random.Random(0))
    by_name = {s.name: s.population for s in strata}
    # API 0..9 and snapshot 10..19 by employer e0: 4x3, e1: 3x4, e2: 3x3 = 33; none banded.
    assert by_name["cross_trudvsem"] == 33
    # Within the API: employers of sizes 4, 3, 3 give 6+3+3 = 12 pairs, likewise the snapshot;
    # four banded pairs are within the API and same employer (0-3, 1-4, 2-5, 6-9).
    assert by_name["same_employer_low"] == 24 - 4


def test_split_halves_each_stratum() -> None:
    items = {"high_minhash": list(range(10)), "cross_habr": list(range(10, 15))}
    out = split_tuning(items, 0.5, random.Random(0))
    assert Counter(out[i] for i in range(10)) == {"tuning": 5, "heldout": 5}
    assert Counter(out[i] for i in range(10, 15)) == {"tuning": 2, "heldout": 3}
