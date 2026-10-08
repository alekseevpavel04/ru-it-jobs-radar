import numpy as np
import pytest

from radar.dedup.evaluate import (
    Labeled,
    Weighted,
    cohen_kappa,
    decide,
    grid,
    interval,
    labeled_pairs,
    missed_upper,
    select,
)
from radar.dedup.settings import CONFIG, Rule, load


def pair(
    pid: str,
    stratum: str,
    p: float,
    same: bool,
    jaccard: float = 0.9,
    place_ok: bool = True,
    candidate: bool = True,
    split: str = "heldout",
) -> Labeled:
    return Labeled(pid, stratum, split, p, jaccard, 0.99, candidate, place_ok, True, same)


def test_config_loads_and_rule_is_in_the_grid() -> None:
    s = load(CONFIG)
    assert s.rule in grid(s.grid_jaccard, s.grid_cosine)
    assert sum(s.params.allocation.values()) == 400


def test_decide_respects_candidates_and_conditions() -> None:
    rule = Rule(jaccard=0.7, cosine=None, same_place=True, same_title=False)
    assert decide(rule, pair("a", "s", 0.1, True))
    assert not decide(rule, pair("a", "s", 0.1, True, jaccard=0.6))
    assert not decide(rule, pair("a", "s", 0.1, True, place_ok=False))
    assert not decide(rule, pair("a", "s", 0.1, True, candidate=False))
    assert decide(Rule(None, 0.98, False, False), pair("a", "s", 0.1, True, jaccard=0.1))


def test_hansen_hurwitz_and_hajek_totals() -> None:
    # One stratum, two drawn pairs with per-draw probabilities 0.5 and 0.25; the second is positive
    # and linked, the first linked but negative.
    xs = [pair("a", "s", 0.5, False), pair("b", "s", 0.25, True)]
    hh = Weighted(xs).estimate(lambda x: True)
    assert hh.linked_pairs == pytest.approx((2 + 4) / 2)
    assert hh.positive_pairs == pytest.approx(4 / 2)
    assert hh.precision == pytest.approx(4 / 6)
    hajek = Weighted(xs, {"s": 12}).estimate(lambda x: True)
    assert hajek.linked_pairs == pytest.approx(12)
    assert hajek.positive_pairs == pytest.approx(12 * 4 / 6)
    assert hajek.recall == pytest.approx(1.0)


def test_strata_add_up() -> None:
    xs = [pair("a", "s1", 1.0, True), pair("b", "s2", 1.0, False)]
    est = Weighted(xs, {"s1": 10, "s2": 30}).estimate(lambda x: True)
    assert est.precision == pytest.approx(10 / 40)


def test_bootstrap_is_seeded_and_brackets_the_point() -> None:
    xs = [pair(str(i), "s", 0.1, i % 3 == 0, jaccard=0.9 if i % 2 else 0.1) for i in range(30)]
    w = Weighted(xs, {"s": 100})
    p1, r1 = w.bootstrap(lambda x: decide(Rule(0.5, None, True, False), x), 500, 7)
    p2, _ = w.bootstrap(lambda x: decide(Rule(0.5, None, True, False), x), 500, 7)
    assert np.array_equal(p1, p2, equal_nan=True)
    point = w.estimate(lambda x: decide(Rule(0.5, None, True, False), x))
    low, high, n = interval(r1, 0.95)
    assert low <= point.recall <= high and n == 500


def test_select_prefers_f1_then_the_stricter_rule() -> None:
    tuning = [
        pair("a", "s", 0.1, True, jaccard=0.95),
        pair("b", "s", 0.1, True, jaccard=0.75),
        pair("c", "s", 0.1, False, jaccard=0.65),
    ]
    rules = [Rule(0.6, None, True, False), Rule(0.7, None, True, False), Rule(0.72, None, True, False)]
    best, est = select(rules, tuning, None)
    assert best == Rule(0.72, None, True, False)
    assert est.f1 == pytest.approx(1.0)


def test_missed_upper_only_for_strata_without_positives() -> None:
    xs = [pair("a", "s1", 0.1, True), *(pair(str(i), "s2", 0.1, False) for i in range(30))]
    assert missed_upper(xs, {"s1": 100, "s2": 900}) == {"s2": pytest.approx(90.0)}


def test_kappa() -> None:
    assert cohen_kappa(["same", "different"], ["same", "different"]) == 1.0
    assert cohen_kappa(["same", "same", "different", "different"], ["same", "different", "same", "different"]) == 0.0


def test_labeled_pairs_drop_unsure_and_read_the_views() -> None:
    card = {"source": "trudvsem", "title": "Программист", "city": "Город Москва", "address": None, "remote": None}
    views = {"d1": {"a": card, "b": card}, "d2": {"a": card, "b": card}}
    meta = {
        pid: {"stratum": "s", "split": "tuning", "p_draw": 0.1, "jaccard": 0.8, "cosine": 0.9}
        | {"by_minhash": True, "by_embedding": False}
        for pid in views
    }
    out = labeled_pairs(views, meta, {"d1": "same", "d2": "unsure"})
    assert [(x.pair_id, x.same, x.place_ok, x.same_title, x.candidate) for x in out] == [("d1", True, True, True, True)]
