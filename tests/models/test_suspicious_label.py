import numpy as np
import pytest

from radar.models.suspicious_label import (
    Design,
    fit_label_model,
    lf_statistics,
    majority_vote,
    score_systems,
    totals,
    weak_labels,
    weighted_scores,
)
from radar.models.suspicious_lfs import ABSTAIN as A
from radar.models.suspicious_lfs import OK as O
from radar.models.suspicious_lfs import SUSPICIOUS as S

VOTES = np.array(
    [
        [S, A, A],  # one suspicious vote
        [S, A, O],  # tie
        [S, S, O],  # suspicious wins
        [A, A, O],  # ok
        [A, A, A],  # no votes
    ],
    dtype=np.int8,
)


def test_majority_vote_and_weak_labels() -> None:
    assert majority_vote(VOTES).tolist() == [S, A, S, O, A]
    y, w = weak_labels(VOTES)
    assert y.tolist() == [True, False, True, False, False]
    assert w.tolist() == [1.0, 0.0, 1.0, 1.0, 1.0]


def test_lf_statistics_and_totals() -> None:
    stats = lf_statistics(VOTES, ["a", "b", "c"])
    assert stats["a"] == {"coverage": 3, "overlap": 2, "conflict": 2, "agrees_with_majority": 2, "denominator": 5}
    assert stats["c"]["conflict"] == 2 and stats["c"]["agrees_with_majority"] == 1
    t = totals(VOTES)
    assert (t["covered"], t["conflicted"], t["majority_suspicious"], t["majority_tie_or_none"]) == (4, 2, 2, 2)


def test_label_model_recovers_independent_functions() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    truth = rng.random(n) < 0.2
    accuracy = np.array([0.9, 0.8, 0.7, 0.85])
    coverage = np.array([0.6, 0.5, 0.7, 0.4])
    votes = np.full((n, 4), A, dtype=np.int8)
    for j in range(4):
        fires = rng.random(n) < coverage[j]
        right = rng.random(n) < accuracy[j]
        label = np.where(right, truth, ~truth)
        votes[fires, j] = np.where(label[fires], S, O)
    mv = majority_vote(votes)
    model = fit_label_model(votes, np.where(mv == S, 0.9, np.where(mv == O, 0.1, 0.5)))
    assert model.prior == pytest.approx(0.2, abs=0.02)
    p_right = model.tables[:, 1, 2] / (model.tables[:, 1, 1] + model.tables[:, 1, 2])
    assert np.allclose(p_right, accuracy, atol=0.05)
    assert model.converged
    # Weighting functions by accuracy beats counting them.
    assert ((model.predict_proba(votes) >= 0.5) == truth).mean() > ((mv == S) == truth).mean()


def test_design_weights_and_weighted_scores() -> None:
    # Stratum h: 2 of 100 cards sampled; stratum r: 4 of 1000.
    design = Design(
        truth=np.array([True, False, True, False, False, False]),
        strata=np.array(["h", "h", "r", "r", "r", "r"]),
        population={"h": 100, "r": 1000},
    )
    w = design.weights()
    assert w.tolist() == [50, 50, 250, 250, 250, 250]
    pred = np.array([True, True, False, False, False, False])
    s = weighted_scores(design.truth, pred, w)
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(50 / 300)


def test_score_systems_reports_raw_counts_and_interval() -> None:
    rng = np.random.default_rng(1)
    truth = rng.random(200) < 0.3
    design = Design(truth=truth, strata=np.where(np.arange(200) < 100, "a", "b"), population={"a": 500, "b": 5000})
    pred = truth.copy()
    pred[:10] = ~pred[:10]
    out = score_systems(design, {"sys": pred}, n_resamples=200, seed=3, confidence=0.95)["sys"]
    assert out["sample_tp"] == int((truth & pred).sum())
    assert out["sample_flagged"] == int(pred.sum())
    for k in ("precision", "recall", "f1"):
        assert out[k]["ci_low"] <= out[k]["value"] <= out[k]["ci_high"]
