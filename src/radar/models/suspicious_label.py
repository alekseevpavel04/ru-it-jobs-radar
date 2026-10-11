"""Weak labels from the votes, their statistics, and the scoring of a stratified labeled sample.

Two ways to combine votes. The majority vote counts SUSPICIOUS against OK; the weak label used for
training takes it, treats a card without votes as OK and leaves ties out. OK by default rests on the
base rate: red-flag phrases are rare in the sources, and the classifier has to see ordinary
commercial cards as negatives, not only the state institutions and experienced roles the OK
functions cover. The `rest` stratum of the sample measures what this default misses.

The label model is Dawid-Skene fitted by EM: every labeling function has, for each true class, a
distribution over its outputs (abstain, OK, SUSPICIOUS), the functions are assumed independent given
the class, and the prior and the tables are found from how the functions agree. It is the generative
model Snorkel's LabelModel approximates. It needs overlapping votes to tell a function's accuracy
from its coverage; here the suspicious functions almost never overlap, and EM settles on the largest
structure in the votes instead (which OK function fired). It is kept as a candidate scored on dev,
not as the training label.

Scoring: the sample is stratified, so every labeled card carries the weight N_h / n_h of its
stratum. Precision and recall are ratios of weighted sums (Hajek); the interval resamples cards
within each stratum.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar.models.suspicious_lfs import ABSTAIN, OK, SUSPICIOUS, Votes

Array = NDArray[np.float64]

# Outputs of a labeling function in the order of the label model's tables.
OUTPUTS = (ABSTAIN, OK, SUSPICIOUS)


def majority_vote(votes: Votes) -> NDArray[np.int8]:
    pos = (votes == SUSPICIOUS).sum(axis=1)
    neg = (votes == OK).sum(axis=1)
    out = np.full(len(votes), ABSTAIN, dtype=np.int8)
    out[pos > neg] = SUSPICIOUS
    out[neg > pos] = OK
    return out


def weak_labels(votes: Votes) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Training target and weight: majority vote, no votes count as OK, ties get weight 0."""
    pos = (votes == SUSPICIOUS).sum(axis=1)
    neg = (votes == OK).sum(axis=1)
    tie = (pos == neg) & (pos > 0)
    return pos > neg, np.where(tie, 0.0, 1.0)


@dataclass
class LabelModel:
    """Dawid-Skene with two classes; `tables[j, y, k]` is P(function j outputs OUTPUTS[k] | class y)."""

    prior: float
    tables: Array
    iterations: int
    converged: bool

    def log_likelihoods(self, votes: Votes) -> Array:
        index = np.searchsorted(np.array(OUTPUTS), votes)
        logs = np.log(self.tables)
        ll = np.zeros((len(votes), 2))
        for j in range(votes.shape[1]):
            ll += logs[j][:, index[:, j]].T
        return ll

    def predict_proba(self, votes: Votes) -> Array:
        """P(suspicious) for each card."""
        ll = self.log_likelihoods(votes)
        ll[:, 0] += np.log1p(-self.prior)
        ll[:, 1] += np.log(self.prior)
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        out: Array = p[:, 1] / p.sum(axis=1)
        return out


def fit_label_model(
    votes: Votes, init: Array, smoothing: float = 1.0, max_iter: int = 1000, tol: float = 1e-6
) -> LabelModel:
    """EM from soft labels `init` (P(suspicious) per card); `smoothing` is a Dirichlet pseudo-count.

    Starting from the majority vote keeps class 1 the suspicious one: EM alone cannot tell the two
    classes apart by name.
    """
    index = np.searchsorted(np.array(OUTPUTS), votes)
    onehot = np.stack([(index == k) for k in range(len(OUTPUTS))], axis=2).astype(float)
    q = np.clip(init, 1e-6, 1 - 1e-6)
    model = LabelModel(prior=float(q.mean()), tables=np.empty(0), iterations=0, converged=False)
    for it in range(1, max_iter + 1):
        resp = np.stack([1 - q, q], axis=1)
        counts = np.einsum("iy,ijk->jyk", resp, onehot) + smoothing
        model = LabelModel(
            prior=float(np.clip(q.mean(), 1e-6, 1 - 1e-6)),
            tables=counts / counts.sum(axis=2, keepdims=True),
            iterations=it,
            converged=False,
        )
        q_new = model.predict_proba(votes)
        if np.max(np.abs(q_new - q)) < tol:
            model.converged = True
            q = q_new
            break
        q = q_new
    return model


def lf_statistics(votes: Votes, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per function: coverage, overlap (another function also voted) and conflict (another voted the
    other way), each as n of all cards; plus the share of its votes that agree with the majority."""
    n = len(votes)
    voted = votes != ABSTAIN
    pos_any = (votes == SUSPICIOUS).sum(axis=1)
    neg_any = (votes == OK).sum(axis=1)
    mv = majority_vote(votes)
    out: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        mine = voted[:, j]
        others = voted.sum(axis=1) - mine
        opposite = np.where(votes[:, j] == SUSPICIOUS, neg_any, pos_any)
        out[name] = {
            "coverage": int(mine.sum()),
            "overlap": int((mine & (others > 0)).sum()),
            "conflict": int((mine & (opposite > 0)).sum()),
            "agrees_with_majority": int((mine & (mv == votes[:, j])).sum()),
            "denominator": n,
        }
    return out


def totals(votes: Votes) -> dict[str, int]:
    voted = votes != ABSTAIN
    pos = (votes == SUSPICIOUS).any(axis=1)
    neg = (votes == OK).any(axis=1)
    mv = majority_vote(votes)
    return {
        "cards": len(votes),
        "covered": int(voted.any(axis=1).sum()),
        "any_suspicious_vote": int(pos.sum()),
        "any_ok_vote": int(neg.sum()),
        "conflicted": int((pos & neg).sum()),
        "majority_suspicious": int((mv == SUSPICIOUS).sum()),
        "majority_ok": int((mv == OK).sum()),
        "majority_tie_or_none": int((mv == ABSTAIN).sum()),
    }


@dataclass(frozen=True)
class Design:
    """A labeled stratified sample: truth and stratum per card, population size of each stratum."""

    truth: NDArray[np.bool_]
    strata: NDArray[np.str_]
    population: Mapping[str, int]

    def weights(self, mask: NDArray[np.bool_] | None = None) -> Array:
        keep = np.ones(len(self.truth), dtype=bool) if mask is None else mask
        w = np.zeros(len(self.truth))
        for h, size in self.population.items():
            in_h = keep & (self.strata == h)
            if in_h.any():
                w[in_h] = size / in_h.sum()
        return w


def weighted_scores(truth: NDArray[np.bool_], pred: NDArray[np.bool_], w: Array) -> dict[str, float]:
    tp = float((w * (truth & pred)).sum())
    fp = float((w * (~truth & pred)).sum())
    fn = float((w * (truth & ~pred)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1}


def score_systems(
    design: Design, predictions: Mapping[str, NDArray[np.bool_]], n_resamples: int, seed: int, confidence: float
) -> dict[str, dict[str, Any]]:
    """Weighted precision, recall and F1 of each system, interval by resampling within strata.

    The raw counts on the sample sit next to the weighted estimates: tp is n, the sample cards the
    system flags are the precision denominator, the suspicious sample cards the recall one.
    """
    rng = np.random.default_rng(seed)
    strata = {h: np.flatnonzero(design.strata == h) for h in np.unique(design.strata)}
    draws = [
        np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata.values()])
        for _ in range(n_resamples)
    ]
    w = design.weights()
    alpha = (1 - confidence) / 2
    out: dict[str, dict[str, Any]] = {}
    for name, pred in predictions.items():
        point = weighted_scores(design.truth, pred, w)
        samples: dict[str, list[float]] = {k: [] for k in point}
        for ix in draws:
            sub = Design(design.truth[ix], design.strata[ix], design.population)
            s = weighted_scores(sub.truth, pred[ix], sub.weights())
            for k, v in s.items():
                if not np.isnan(v):
                    samples[k].append(v)
        block: dict[str, Any] = {}
        for k, v in point.items():
            entry: dict[str, Any] = {"value": None if np.isnan(v) else round(v, 4)}
            if samples[k]:
                lo, hi = np.quantile(samples[k], [alpha, 1 - alpha])
                entry |= {"ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}
            block[k] = entry
        block["sample_tp"] = int((design.truth & pred).sum())
        block["sample_flagged"] = int(pred.sum())
        block["sample_suspicious"] = int(design.truth.sum())
        out[name] = block
    return out
