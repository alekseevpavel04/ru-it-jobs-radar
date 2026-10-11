import pytest

from radar.models.suspicious_agreement import merge


def row(item_id: str, label: str, categories: list[str]) -> dict[str, object]:
    return {"item_id": item_id, "label": label, "categories": categories, "reason": "-"}


def test_agreed_cards_keep_their_label_and_differing_ones_take_the_adjudicated_label() -> None:
    a = [row("d1", "ok", []), row("d2", "suspicious", ["formal", "bait"]), row("d3", "unsure", [])]
    b = [row("d1", "ok", []), row("d2", "suspicious", ["bait", "formal"]), row("d3", "ok", [])]
    labels, counts = merge(a, b, {"d3": row("d3", "ok", [])})
    assert [(r["item_id"], r["provenance"]) for r in labels] == [
        ("d1", "agreed"),
        ("d2", "agreed"),
        ("d3", "adjudicated"),
    ]
    assert counts["label_and_categories_equal"] == {"n": 2, "denominator": 3}
    assert counts["suspicious_final"] == {"n": 1, "denominator": 3}


def test_a_disagreement_without_adjudication_fails() -> None:
    with pytest.raises(ValueError, match="no adjudicated label"):
        merge([row("d1", "ok", [])], [row("d1", "suspicious", ["grey"])], {})


def test_passes_over_different_cards_fail() -> None:
    with pytest.raises(ValueError, match="different cards"):
        merge([row("d1", "ok", [])], [row("d2", "ok", [])], {})
