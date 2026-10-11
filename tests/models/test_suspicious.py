import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar.eval.cli import EvalConfig
from radar.models.suspicious import (
    CARD_COLUMNS,
    LFS,
    CardClassifier,
    allocate,
    annotation_input,
    card_groups,
    cards_from_records,
    draw_sample,
    load_labels,
    score_split,
    split_of,
    strata_of,
)
from radar.models.suspicious_lfs import ABSTAIN, SUSPICIOUS, add_context

CFG = EvalConfig(n_resamples=200, seed=7, confidence=0.95, salary_rel_tol=0.01, macro_fields=())


def record(source: str, source_id: str, dt: str, **fields: object) -> dict[str, object]:
    row: dict[str, object] = {c: None for c in CARD_COLUMNS}
    row |= {
        "family": "habr" if source == "habr" else "trudvsem",
        "source": source,
        "source_id": source_id,
        "dt": pd.Timestamp(dt),
        "first_seen": pd.Timestamp(dt),
        "last_seen": pd.Timestamp(dt),
        "title": "Программист",
        "text": "текст",
        "employer_name": "ООО Ромашка",
        "is_sole_proprietor": 0,
        "region_group": "other",
        "salary_currency_code": "RUB",
        "role_bucket": "software_engineer",
        "grade_is_stated": 0,
        "skills": ["Python"],
    }
    return row | fields


def test_cards_from_records_merge_api_and_snapshot_rows_of_one_id() -> None:
    records = pd.DataFrame(
        [
            record("trudvsem_snapshot", "v1", "2026-07-01", first_seen=pd.Timestamp("2026-06-30"), repost_of="v0"),
            record("trudvsem_snapshot", "v1", "2026-10-09", title="snapshot title"),
            record("trudvsem", "v1", "2026-10-09", title="api title", last_seen=pd.Timestamp("2026-10-10")),
            record("habr", "1", "2026-10-09"),
        ]
    )
    cards = cards_from_records(records).set_index("key")
    assert list(cards.index) == ["habr:1", "trudvsem:v1"]
    v1 = cards.loc["trudvsem:v1"]
    assert v1["title"] == "api title"
    assert v1["first_seen"] == pd.Timestamp("2026-06-30") and v1["last_seen"] == pd.Timestamp("2026-10-10")
    assert v1["repost_of"] == "v0" and bool(v1["in_snapshot"])
    assert not bool(cards.loc["habr:1", "in_snapshot"])


def test_allocate_shares_the_budget_and_passes_leftovers_on() -> None:
    sizes = {"lf:rare": 1, "lf:mid": 5, "lf:big": 300, "lf:huge": 900, "model_only": 12, "rest": 10_000}
    alloc = allocate(sizes, total=150, lf_budget=70, model_only=20)
    assert alloc["lf:rare"] == 1 and alloc["lf:mid"] == 5
    assert alloc["lf:big"] + alloc["lf:huge"] == 64
    assert alloc["model_only"] == 12
    assert sum(alloc.values()) == 150


def test_strata_take_the_rarest_suspicious_vote() -> None:
    votes = np.array(
        [
            [SUSPICIOUS, SUSPICIOUS],
            [SUSPICIOUS, ABSTAIN],
            [SUSPICIOUS, ABSTAIN],
            [ABSTAIN, ABSTAIN],
            [ABSTAIN, ABSTAIN],
        ],
        dtype=np.int8,
    )
    strata = strata_of(votes, np.array([0.1, 0.1, 0.9, 0.9, 0.1]), ["common", "rare"])
    assert strata.tolist() == ["lf:rare", "lf:common", "lf:common", "model_only", "rest"]


def test_split_is_a_function_of_the_text_group() -> None:
    assert split_of("g1", 1) == split_of("g1", 1)
    assert {split_of(f"g{i}", 1) for i in range(50)} == {"dev", "test"}


def synthetic_cards(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        template = i % 3 == 0
        text = ("Шаблон обязанностей программиста по справочнику профессий " * 5) if template else f"Текст {i} " * 30
        rows.append(
            record(
                "trudvsem_snapshot",
                f"id{i}",
                "2026-10-01",
                text=text + f" {i % 7}",
                employer_key=f"inn{i % 40}",
                region=f"Регион {i % 5}",
                salary_from=float(rng.integers(30, 200) * 1000),
                url=f"https://example.org/{i}",
                cities=np.array([], dtype=object),
            )
        )
    cards = add_context(cards_from_records(pd.DataFrame(rows)))
    return cards


def test_draw_sample_keeps_text_groups_in_one_split_and_hides_the_design() -> None:
    cards = synthetic_cards()
    strata = np.where(np.arange(len(cards)) % 10 == 0, "lf:x", "rest")
    drawn = draw_sample(cards, strata, seed=5)
    design, inputs = drawn["design"], drawn["inputs"]
    group = dict(zip(cards["key"], card_groups(cards), strict=True))
    seen: dict[str, set[str]] = {}
    for split, d in design["splits"].items():
        assert sum(d["population"].values()) == d["population_cards"]
        assert len(d["items"]) == sum(d["allocation"].values()) == len(inputs[split])
        for item in d["items"]:
            seen.setdefault(group[item["key"]], set()).add(split)
    assert all(len(s) == 1 for s in seen.values())
    row = inputs["dev"][0]
    assert set(row) >= {"item_id", "title", "employer", "text"}
    assert not {"key", "stratum", "classifier", "majority"} & set(row)
    json.dumps(row, ensure_ascii=False)


def test_annotation_input_cuts_long_text() -> None:
    card = synthetic_cards(3).iloc[0].copy()
    card["text"] = "а" * 6000
    row = annotation_input(card, "d001")
    assert len(row["text"]) == 5000 and row["text_truncated"]


def test_load_labels_rejects_unknown_values(tmp_path: Path) -> None:
    path = tmp_path / "labels_dev.jsonl"
    path.write_text('{"item_id": "d001", "label": "spam"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="spam"):
        load_labels(path)
    path.write_text('{"item_id": "d001", "label": "suspicious", "categories": ["scam"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="scam"):
        load_labels(path)


def test_score_split_on_a_labeled_half() -> None:
    items = [
        {"item_id": f"d{i:03d}", "key": f"k{i}", "stratum": "lf:mass_text" if i < 10 else "rest"} for i in range(40)
    ]
    split = {"population": {"lf:mass_text": 20, "rest": 3000}, "items": items}
    labels = {
        f"d{i:03d}": {"label": "suspicious" if i < 7 else "ok", "categories": ["formal"] if i < 7 else []}
        for i in range(39)
    }
    labels["d000"] = {"label": "unsure"}
    predictions = pd.DataFrame({"key": [f"k{i}" for i in range(40)]})
    predictions["majority"] = np.where(np.arange(40) < 10, SUSPICIOUS, ABSTAIN)
    predictions["label_model"] = 0.1
    predictions["classifier"] = np.where(np.arange(40) < 8, 0.9, 0.1)
    for lf in LFS:
        predictions[f"lf:{lf.name}"] = ABSTAIN
    predictions["lf:mass_text"] = predictions["majority"]
    out = score_split(split, labels, predictions, CFG)
    assert (out["labeled"], out["unsure"], out["unlabeled"], out["suspicious"]) == (38, 1, 1, 6)
    assert out["categories"] == {"formal": 6}
    mass = out["systems"]["lf:mass_text"]
    assert (mass["sample_tp"], mass["sample_flagged"]) == (6, 9)
    assert mass["precision"]["value"] == pytest.approx(6 / 9, abs=1e-4)
    assert out["systems"]["label_model"]["precision"]["value"] is None


def test_classifier_fits_the_weak_labels_and_names_its_features() -> None:
    cards = synthetic_cards()
    y = cards["text"].str.startswith("Шаблон").to_numpy()
    clf = CardClassifier.fit(cards, y, np.ones(len(cards)), seed=0)
    p = clf.predict_proba(cards)
    assert ((p >= 0.5) == y).mean() > 0.95
    top = clf.top_features(5)
    assert len(top["suspicious"]) == 5 and len(top["ok"]) == 5
