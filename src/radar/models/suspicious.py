"""Detector of reposts and suspicious vacancies: weak labels, a classifier, a labeled sample.

Definitions and labeling functions: `radar.models.suspicious_lfs`. Combining votes and scoring the
sample: `radar.models.suspicious_label`.

The classifier is a logistic regression on TF-IDF of the title and text plus the card's numeric
context (salary z, text spread over employers, repost chain, experience). It learns the weak label
(majority vote, no votes as OK, ties left out). Folds of its cross-validation are grouped by text, so
a template never sits on both sides; agreement with the weak labels is reported as what it is, a
check that the model learned them, not its quality. Quality comes only from the labeled sample.

Sample: cards are split into dev and test halves by a seeded hash of their text group, then each half
is sampled by strata (the rarest suspicious vote that fired; classifier flags without a suspicious
vote; the rest) and frozen with the population size of every stratum, so estimates are weighted
back to the population. Labels come from two independent annotation passes
(docs/suspicious_guide.md); the labeled dev half is scored on every run, the test half only with
--final-test.

    python -m radar.models.suspicious run                  votes, weak labels, classifier, results
    python -m radar.models.suspicious sample               freeze the labeling sample (once)
    python -m radar.models.suspicious run --final-test     also score the labeled test half (once)
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from radar.config import data_dir, load_dotenv
from radar.eval.cli import EvalConfig, load_config, log_test_run
from radar.models.suspicious_label import (
    Design,
    fit_label_model,
    lf_statistics,
    majority_vote,
    score_systems,
    totals,
    weak_labels,
)
from radar.models.suspicious_lfs import (
    ABSTAIN,
    CATEGORIES,
    LFS,
    OK,
    REPOST_CHAIN,
    SUSPICIOUS,
    add_context,
    apply_lfs,
    combined,
)
from radar.results import write_result

log = logging.getLogger("radar.models.suspicious")

RESULT = "models/suspicious.json"
THRESHOLD = 0.5
CV_FOLDS = 5
TOP_TERMS = 15
TEXT_CHARS = 5000
SAMPLE_PER_SPLIT = 150
# Per split: cards for the suspicious-vote strata (shared out evenly, capped by stratum size) and for
# the classifier-only stratum; the rest of the 150 goes to cards without a suspicious vote.
SAMPLE_LF_BUDGET = 70
SAMPLE_MODEL_ONLY = 20
SPLITS = ("dev", "test")
LABELS = ("suspicious", "ok", "unsure")

CARD_COLUMNS = (
    "family",
    "source_id",
    "source",
    "dt",
    "first_seen",
    "last_seen",
    "title",
    "text",
    "url",
    "employer_key",
    "employer_name",
    "is_sole_proprietor",
    "region",
    "cities",
    "region_group",
    "salary_text",
    "salary_from",
    "salary_to",
    "salary_currency_code",
    "role_bucket",
    "grade",
    "grade_is_stated",
    "work_format",
    "experience_min_years",
    "no_experience",
    "schedule",
    "employment",
    "skills",
    "repost_of",
)

NUMERIC = (
    "salary_z",
    "salary_missing",
    "no_experience",
    "sole_proprietor",
    "group_employers",
    "group_region_share",
    "employer_cards",
    "repost_chain",
    "text_length",
    "skills",
    "habr",
)

RECORDS_SQL = """
WITH texts AS (
    SELECT source, source_id, dt, text, region, CAST([] AS Array(String)) AS cities, schedule, employment,
        salary_text, url
    FROM staging.stg_trudvsem_api
    WHERE (source_id, dt) IN (SELECT source_id, dt FROM staging.int_vacancy_records WHERE source = 'trudvsem')
    UNION ALL
    SELECT source, source_id, dt, text, region, CAST([] AS Array(String)), schedule, employment, salary_text, url
    FROM staging.stg_trudvsem_snapshot
    WHERE (source_id, dt) IN (
        SELECT source_id, dt FROM staging.int_vacancy_records WHERE source = 'trudvsem_snapshot'
    )
    UNION ALL
    SELECT source, source_id, dt, text, CAST(NULL AS Nullable(String)), cities, NULL, employment, salary_text, url
    FROM staging.stg_habr
    WHERE (source_id, dt) IN (SELECT source_id, dt FROM staging.int_vacancy_records WHERE source = 'habr')
)
SELECT
    r.family AS family, r.source_id AS source_id, r.source AS source, r.dt AS dt,
    r.first_seen AS first_seen, r.last_seen AS last_seen, r.title AS title, t.text AS text, t.url AS url,
    r.employer_key AS employer_key, r.employer_name AS employer_name, r.is_sole_proprietor AS is_sole_proprietor,
    t.region AS region, t.cities AS cities, r.region_group AS region_group, t.salary_text AS salary_text,
    r.salary_from AS salary_from, r.salary_to AS salary_to, r.salary_currency_code AS salary_currency_code,
    r.role_bucket AS role_bucket, r.grade AS grade, r.grade_is_stated AS grade_is_stated,
    r.work_format AS work_format, r.experience_min_years AS experience_min_years,
    r.no_experience AS no_experience, t.schedule AS schedule, t.employment AS employment, r.skills AS skills,
    l.repost_of_source_id AS repost_of
FROM staging.int_vacancy_records AS r
LEFT JOIN (SELECT * FROM texts ORDER BY source, source_id LIMIT 1 BY source, source_id) AS t
    ON r.source = t.source AND r.source_id = t.source_id
LEFT JOIN staging.hist_vacancy_lifetime AS l ON r.source = l.source AND r.source_id = l.source_id
FORMAT JSONEachRow
SETTINGS join_use_nulls = 1
"""


def work_dir() -> Path:
    return data_dir() / "suspicious"


def load_records() -> pd.DataFrame:
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse(timeout_s=900) as ch:
        body = ch.query(RECORDS_SQL)
    records = pd.DataFrame([json.loads(line) for line in body.splitlines() if line], columns=list(CARD_COLUMNS))
    for column in ("dt", "first_seen", "last_seen"):
        records[column] = pd.to_datetime(records[column])
    return records


def cards_from_records(records: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, source_id): fields of the latest row, the API before the snapshot on a day.

    The card spans all its rows: first and last sighting over them, whether the snapshot has it, and
    the repost link of the snapshot row (only the snapshot has a history that closes).
    """
    rows = records.assign(api_last=records["source"].ne("trudvsem"))
    rows = rows.sort_values(["family", "source_id", "dt", "api_last"], ascending=[True, True, False, True])
    cards = rows.drop_duplicates(["family", "source_id"]).drop(columns="api_last").set_index(["family", "source_id"])
    spans = rows.assign(in_snapshot=rows["source"].eq("trudvsem_snapshot")).groupby(["family", "source_id"])
    cards["first_seen"] = spans["first_seen"].min()
    cards["last_seen"] = spans["last_seen"].max()
    cards["repost_of"] = spans["repost_of"].first()
    cards["in_snapshot"] = spans["in_snapshot"].any()
    out = cards.reset_index()
    out["key"] = out["family"] + ":" + out["source_id"]
    out["skills"] = out["skills"].map(lambda s: list(s) if s is not None else [])
    return out


@dataclass
class CardClassifier:
    """TF-IDF of title and text with the numeric context, logistic regression."""

    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    model: LogisticRegression

    @staticmethod
    def numeric(cards: pd.DataFrame) -> NDArray[np.float64]:
        columns = {
            "salary_z": cards["salary_z"].fillna(0.0).clip(-6, 6),
            "salary_missing": cards["salary_z"].isna().astype(float),
            "no_experience": pd.to_numeric(cards["no_experience"]).fillna(0).astype(float),
            "sole_proprietor": cards["is_sole_proprietor"].astype(float),
            "group_employers": np.log1p(cards["group_employers"]),
            "group_region_share": cards["group_region_share"],
            "employer_cards": np.log1p(cards["employer_cards"]),
            "repost_chain": np.log(cards["repost_chain"]),
            "text_length": np.log1p(cards["text"].fillna("").str.len()),
            "skills": np.log1p(cards["skills"].map(len)),
            "habr": cards["family"].eq("habr").astype(float),
        }
        return pd.DataFrame({k: columns[k] for k in NUMERIC}).to_numpy(dtype=np.float64)

    @classmethod
    def fit(cls, cards: pd.DataFrame, y: NDArray[np.bool_], weight: NDArray[np.float64], seed: int) -> "CardClassifier":
        keep = weight > 0
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True)
        scaler = StandardScaler()
        x = sparse.hstack(
            [vectorizer.fit_transform(cards["body"][keep]), scaler.fit_transform(cls.numeric(cards[keep]))],
            format="csr",
        )
        model = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed)
        model.fit(x, y[keep].astype(int))
        return cls(vectorizer, scaler, model)

    def predict_proba(self, cards: pd.DataFrame) -> NDArray[np.float64]:
        x = sparse.hstack(
            [self.vectorizer.transform(cards["body"]), self.scaler.transform(self.numeric(cards))], format="csr"
        )
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)

    def top_features(self, k: int = TOP_TERMS) -> dict[str, list[str]]:
        names = np.concatenate([self.vectorizer.get_feature_names_out(), [f"#{n}" for n in NUMERIC]])
        order = np.argsort(self.model.coef_[0])
        return {"suspicious": [str(names[i]) for i in order[::-1][:k]], "ok": [str(names[i]) for i in order[:k]]}


def cross_validate(
    cards: pd.DataFrame, y: NDArray[np.bool_], weight: NDArray[np.float64], groups: NDArray[np.str_], seed: int
) -> dict[str, Any]:
    """Held-out P(suspicious) per card from folds grouped by text, against the weak labels."""
    held_out = np.zeros(len(cards))
    for train, test in GroupKFold(n_splits=CV_FOLDS).split(cards, y, groups):
        clf = CardClassifier.fit(cards.iloc[train], y[train], weight[train], seed)
        held_out[test] = clf.predict_proba(cards.iloc[test])
    scored = weight > 0
    flagged = held_out >= THRESHOLD
    return {
        "held_out": held_out,
        "auc": float(roc_auc_score(y[scored], held_out[scored])),
        "agree": int((flagged == y)[scored].sum()),
        "scored": int(scored.sum()),
        "positive": int((y & scored).sum()),
        "positive_recovered": int((flagged & y & scored).sum()),
    }


def card_groups(cards: pd.DataFrame) -> NDArray[np.str_]:
    return cards["text_group"].fillna(cards["key"]).to_numpy(dtype=str)


def split_of(group: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return SPLITS[digest[0] % 2]


def strata_of(votes: NDArray[np.int8], classifier: NDArray[np.float64], names: Sequence[str]) -> NDArray[np.str_]:
    """`lf:<name>` of the rarest suspicious vote of a card, `model_only` for a classifier flag without
    one, `rest` otherwise. Rarest first, so that a function firing on a handful of cards is sampled."""
    suspicious = votes == SUSPICIOUS
    coverage = suspicious.sum(axis=0)
    rank = np.where(suspicious, coverage[None, :], np.iinfo(np.int64).max)
    primary = np.argmin(rank, axis=1)
    out = np.where(classifier >= THRESHOLD, "model_only", "rest").astype(object)
    has_vote = suspicious.any(axis=1)
    out[has_vote] = [f"lf:{names[j]}" for j in primary[has_vote]]
    return out.astype(str)


def allocate(sizes: dict[str, int], total: int, lf_budget: int, model_only: int) -> dict[str, int]:
    """Even shares of lf_budget over the lf: strata, capped by size, leftovers passed on; then the
    classifier-only stratum; the rest of `total` to cards without a suspicious vote."""
    alloc = {h: 0 for h in sizes}
    lf = sorted((h for h in sizes if h.startswith("lf:")), key=lambda h: (sizes[h], h))
    budget = lf_budget
    for i, h in enumerate(lf):
        alloc[h] = min(sizes[h], budget // (len(lf) - i))
        budget -= alloc[h]
    if "model_only" in sizes:
        alloc["model_only"] = min(sizes["model_only"], model_only)
    if "rest" in sizes:
        alloc["rest"] = min(sizes["rest"], total - sum(alloc.values()))
    return alloc


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def annotation_input(card: pd.Series, item_id: str) -> dict[str, Any]:
    """What the annotator sees: the card as a job seeker sees it, no votes, scores or history."""
    text = card["text"] or ""
    cities = card["cities"]
    return {
        "item_id": item_id,
        "source": "habr" if card["family"] == "habr" else "trudvsem",
        "url": _plain(card["url"]),
        "title": _plain(card["title"]),
        "employer": _plain(card["employer_name"]),
        "region": _plain(card["region"]),
        "cities": [str(c) for c in cities] if cities is not None else [],
        "salary_text": _plain(card["salary_text"]),
        "salary_from": _plain(card["salary_from"]),
        "salary_to": _plain(card["salary_to"]),
        "currency": _plain(card["salary_currency_code"]),
        "experience_min_years": _plain(card["experience_min_years"]),
        "grade_stated": _plain(card["grade"]) if card["grade_is_stated"] else None,
        "schedule": _plain(card["schedule"]),
        "employment": _plain(card["employment"]),
        "text": text[:TEXT_CHARS],
        "text_truncated": len(text) > TEXT_CHARS,
    }


def _sha256(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def draw_sample(cards: pd.DataFrame, strata: NDArray[np.str_], seed: int) -> dict[str, Any]:
    """Stratified sample of each half: the design (items, population per stratum) and the inputs."""
    rng = np.random.default_rng(seed)
    split = np.array([split_of(g, seed) for g in card_groups(cards)])
    design: dict[str, Any] = {"seed": seed, "per_split": SAMPLE_PER_SPLIT, "splits": {}}
    inputs: dict[str, list[dict[str, Any]]] = {}
    for name in SPLITS:
        in_split = split == name
        labels, counts = np.unique(strata[in_split], return_counts=True)
        sizes = {str(h): int(c) for h, c in zip(labels, counts, strict=True)}
        alloc = allocate(sizes, SAMPLE_PER_SPLIT, SAMPLE_LF_BUDGET, SAMPLE_MODEL_ONLY)
        chosen: list[tuple[int, str]] = []
        for h in sorted(alloc):
            pool = np.flatnonzero(in_split & (strata == h))
            chosen += [(int(i), h) for i in rng.choice(pool, size=alloc[h], replace=False)]
        items, rows = [], []
        for n, k in enumerate(rng.permutation(len(chosen)), start=1):
            i, h = chosen[k]
            item_id = f"{name[0]}{n:03d}"
            card = cards.iloc[i]
            items.append({"item_id": item_id, "key": str(card["key"]), "stratum": h})
            rows.append(annotation_input(card, item_id))
        inputs[name] = rows
        design["splits"][name] = {
            "population_cards": int(in_split.sum()),
            "population": sizes,
            "allocation": alloc,
            "cards_sha256": _sha256(sorted(it["key"] for it in items)),
            "items": items,
        }
    return {"design": design, "inputs": inputs}


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Final labels of one split: one JSON object per line with item_id, label and categories."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["label"] not in LABELS:
            raise ValueError(f"{path.name}: label {row['label']!r} of {row['item_id']} is not one of {LABELS}")
        unknown = set(row.get("categories") or []) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"{path.name}: unknown categories {sorted(unknown)} of {row['item_id']}")
        out[row["item_id"]] = row
    return out


def score_split(
    split: dict[str, Any], labels: dict[str, dict[str, Any]], predictions: pd.DataFrame, cfg: EvalConfig
) -> dict[str, Any]:
    """Every suspicious function, the vote combinations and the classifier on one labeled half."""
    items = pd.DataFrame(split["items"])
    items["label"] = items["item_id"].map(lambda i: labels.get(i, {}).get("label"))
    unlabeled = int(items["label"].isna().sum())
    unsure = int(items["label"].eq("unsure").sum())
    items = items[items["label"].isin(["suspicious", "ok"])].merge(predictions, on="key", how="left")
    missing = int(items["classifier"].isna().sum())
    items = items[items["classifier"].notna()].reset_index(drop=True)
    design = Design(
        truth=items["label"].eq("suspicious").to_numpy(),
        strata=items["stratum"].to_numpy(dtype=str),
        population=split["population"],
    )
    systems: dict[str, NDArray[np.bool_]] = {
        f"lf:{lf.name}": items[f"lf:{lf.name}"].eq(SUSPICIOUS).to_numpy() for lf in LFS if lf.vote == SUSPICIOUS
    }
    systems["any_combined_suspicious_vote"] = np.asarray(
        np.column_stack([systems[f"lf:{lf.name}"] for lf in LFS if lf.vote == SUSPICIOUS and lf.combine]).any(axis=1),
        dtype=bool,
    )
    systems["majority_vote"] = items["majority"].eq(SUSPICIOUS).to_numpy()
    systems["label_model"] = (items["label_model"] >= THRESHOLD).to_numpy()
    systems["classifier"] = (items["classifier"] >= THRESHOLD).to_numpy()
    categories: dict[str, int] = {}
    for item_id in items["item_id"]:
        for c in labels[item_id].get("categories") or []:
            categories[c] = categories.get(c, 0) + 1
    return {
        "labeled": len(items),
        "unsure": unsure,
        "unlabeled": unlabeled,
        "missing_predictions": missing,
        "suspicious": int(design.truth.sum()),
        "categories": categories,
        "systems": score_systems(design, systems, cfg.n_resamples, cfg.seed, cfg.confidence),
    }


def _day(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": int(n), "denominator": int(d)}


def run(final_test: bool) -> dict[str, Any]:
    cfg = load_config()
    records = load_records()
    cards = add_context(cards_from_records(records))
    log.info("%d records, %d cards", len(records), len(cards))
    names = [lf.name for lf in LFS]
    votes = apply_lfs(cards)
    used = votes[:, combined()]
    mv = majority_vote(used)
    y, weight = weak_labels(used)
    lm = fit_label_model(used, np.where(mv == SUSPICIOUS, 0.9, np.where(mv == OK, 0.1, 0.3)))
    posterior = lm.predict_proba(used)
    cv = cross_validate(cards, y, weight, card_groups(cards), cfg.seed)
    clf = CardClassifier.fit(cards, y, weight, cfg.seed)
    prob = clf.predict_proba(cards)

    predictions = pd.DataFrame({"key": cards["key"], "majority": mv, "label_model": posterior, "classifier": prob})
    for j, name in enumerate(names):
        predictions[f"lf:{name}"] = votes[:, j]
    predictions["classifier_held_out"] = cv["held_out"]
    predictions["is_repost"] = cards["repost_of"].notna().to_numpy()
    predictions["repost_chain"] = cards["repost_chain"].to_numpy()
    out_dir = work_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    # `sample` draws from these, so the frozen sample matches the run it was drawn from.
    cards.drop(columns=["body"]).to_parquet(out_dir / "cards.parquet", index=False)

    n = len(cards)
    metrics: dict[str, dict[str, Any]] = {"cards": {"value": n}, "records": {"value": len(records)}}
    for name, st in lf_statistics(votes, names).items():
        metrics[f"lf.{name}.coverage"] = _share(st["coverage"], n)
        metrics[f"lf.{name}.overlap"] = _share(st["overlap"], st["coverage"])
        metrics[f"lf.{name}.conflict"] = _share(st["conflict"], st["coverage"])
    for k, v in totals(used).items():
        if k != "cards":
            metrics[f"weak.{k}"] = _share(v, n)
    metrics["weak.training_suspicious"] = _share(int((y & (weight > 0)).sum()), int((weight > 0).sum()))
    lm_pos = posterior >= THRESHOLD
    metrics["label_model.suspicious"] = _share(int(lm_pos.sum()), n)
    metrics["label_model.prior"] = {"value": round(lm.prior, 4)}
    decided = mv != ABSTAIN
    metrics["label_model.agrees_with_majority"] = _share(
        int((lm_pos[decided] == (mv[decided] == SUSPICIOUS)).sum()), int(decided.sum())
    )
    metrics["classifier.flagged"] = _share(int((prob >= THRESHOLD).sum()), n)
    no_vote = ~(used == SUSPICIOUS).any(axis=1)
    metrics["classifier.flagged_without_suspicious_vote"] = _share(
        int(((prob >= THRESHOLD) & no_vote).sum()), int(no_vote.sum())
    )
    metrics["classifier.cv_auc_vs_weak_labels"] = {"value": round(cv["auc"], 4)}
    metrics["classifier.cv_agrees_with_weak_labels"] = _share(cv["agree"], cv["scored"])
    metrics["classifier.cv_recovers_weak_positives"] = _share(cv["positive_recovered"], cv["positive"])
    in_snapshot = cards["in_snapshot"].to_numpy()
    metrics["reposts.cards_reposting_a_closed_card"] = _share(
        int(cards["repost_of"].notna().sum()), int(in_snapshot.sum())
    )
    metrics["reposts.cards_in_long_chain"] = _share(
        int((cards["repost_chain"].to_numpy() >= REPOST_CHAIN).sum()), int(in_snapshot.sum())
    )

    by_family = cards.groupby("family").agg(
        cards=("key", "size"), first_seen=("first_seen", "min"), last_seen=("last_seen", "max")
    )
    doc: dict[str, Any] = {
        "metrics": metrics,
        "data": {
            str(f): {"cards": int(r["cards"]), "first_seen": _day(r["first_seen"]), "last_seen": _day(r["last_seen"])}
            for f, r in by_family.iterrows()
        },
        "labeling_functions": [
            {
                "name": lf.name,
                "votes": "suspicious" if lf.vote == SUSPICIOUS else "ok",
                "category": lf.category,
                "in_weak_label": lf.combine,
                "rule": lf.rule,
            }
            for lf in LFS
        ],
        "weak_label": "majority vote of the functions in_weak_label; a card without votes is OK, a tie is "
        "left out of training",
        "label_model": {
            "kind": "Dawid-Skene, two classes, EM from the majority vote, Dirichlet pseudo-count 1; "
            "a candidate scored on dev, not the training label",
            "iterations": lm.iterations,
            "converged": lm.converged,
            "p_vote_given_class": {
                name: {
                    "suspicious_vote": [round(float(lm.tables[j, y_, 2]), 4) for y_ in (0, 1)],
                    "ok_vote": [round(float(lm.tables[j, y_, 1]), 4) for y_ in (0, 1)],
                }
                for j, name in enumerate(n for n, keep in zip(names, combined(), strict=True) if keep)
            },
        },
        "classifier": {
            "kind": "logistic regression, liblinear, C=1, class_weight balanced; TF-IDF word 1-2 grams "
            f"(min_df 3, up to 100k features, sublinear tf) of title and text plus {len(NUMERIC)} scaled "
            "numeric features",
            "threshold": THRESHOLD,
            "cv": f"{CV_FOLDS} folds grouped by text group",
            "top_features": clf.top_features(),
        },
    }

    design_path = out_dir / "sample_design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text(encoding="utf-8"))
        doc["sample"] = {
            "seed": design["seed"],
            "created_at": design["created_at"],
            "data_last_seen": design["data_last_seen"],
            "inputs_sha256": design["inputs_sha256"],
            "splits": {
                s: {k: v for k, v in d.items() if k != "items"} | {"items": len(d["items"])}
                for s, d in design["splits"].items()
            },
        }
        for s in SPLITS:
            path = out_dir / f"labels_{s}.jsonl"
            if (s == "test" and not final_test) or not path.exists():
                continue
            if s == "test":
                log_test_run("suspicious", None)
            doc[f"scores_{s}"] = score_split(design["splits"][s], load_labels(path), predictions, cfg)
    if final_test and "scores_test" not in doc:
        raise SystemExit("--final-test: no frozen sample or no data/suspicious/labels_test.jsonl")
    return doc


def sample(resample: bool) -> Path:
    cfg = load_config()
    out_dir = work_dir()
    design_path = out_dir / "sample_design.json"
    labeled = sorted(p.name for p in out_dir.glob("labels_*.jsonl"))
    if design_path.exists() and labeled:
        raise SystemExit(f"refused: {design_path} is frozen and labeled ({', '.join(labeled)})")
    if design_path.exists() and not resample:
        raise SystemExit(f"refused: {design_path} is frozen; --resample redraws an unlabeled sample")
    cards = pd.read_parquet(out_dir / "cards.parquet")
    predictions = pd.read_parquet(out_dir / "predictions.parquet")
    if not cards["key"].equals(predictions["key"]):
        raise SystemExit("cards.parquet and predictions.parquet come from different runs; rerun `run`")
    names = [lf.name for lf in LFS]
    votes = predictions[[f"lf:{n}" for n in names]].to_numpy(dtype=np.int8)
    drawn = draw_sample(cards, strata_of(votes, predictions["classifier"].to_numpy(), names), cfg.seed)
    design = drawn["design"]
    design["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    design["data_last_seen"] = _day(cards["last_seen"].max())
    design["inputs_sha256"] = {}
    for split, rows in drawn["inputs"].items():
        path = out_dir / f"sample_{split}_inputs.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        design["inputs_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return design_path


LIMITS_RU = [
    "Метки слабые: их дают правила, а не люди. Согласие классификатора с ними показывает, что он их выучил, "
    "а не что он прав. Точность и полнота считаются только по выборке с предразметкой LLM и ручной проверкой.",
    "Подозрительность судится по карточке. Цель публикации (например, формальная вакансия ради разрешения "
    "на найм) по карточке не устанавливается, поэтому вид formal описывает признаки, а не намерение.",
    "Перевыкладка определяется по истории снимков trudvsem (окно с 30.06); у Хабра и среза API истории "
    "закрытий пока нет. Перевыкладка сама по себе не подозрительна: карточка портала истекает примерно "
    "через 30 дней.",
    "Название работодателя-ИП вырезано при чистке персональных данных, поэтому «нет работодателя» здесь "
    "артефакт конвейера и функцией разметки не служит.",
    "Выборка заморожена на дату создания; веса страт относятся к совокупности карточек на эту дату.",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.models.suspicious", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="votes, weak labels, classifier, results/models/suspicious.json")
    r.add_argument("--final-test", action="store_true", help="the one scoring of the labeled test half")
    s = sub.add_parser("sample", help="freeze the stratified labeling sample from the last run")
    s.add_argument("--resample", action="store_true", help="redraw a sample that has no labels yet")
    args = ap.parse_args(argv)
    load_dotenv()
    if args.cmd == "sample":
        log.info("wrote %s", sample(args.resample))
        return 0
    doc = run(args.final_test)
    path = write_result(
        RESULT, doc.pop("metrics"), source_script="src/radar/models/suspicious.py", limits_ru=LIMITS_RU, **doc
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
