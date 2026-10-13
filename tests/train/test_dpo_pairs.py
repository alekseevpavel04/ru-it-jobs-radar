import json
import random
from collections import Counter
from typing import Any

import pytest

from radar.extract.schema import ROLES, try_parse
from radar.train.config import Pairs
from radar.train.dpo_pairs import (
    ROLE_NEIGHBOURS,
    build_pairs,
    compact,
    corrupt,
    disagreement,
    report,
    student_inputs,
)


def label(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "role": "backend",
        "grade": "middle",
        "skills": ["Python", "PostgreSQL"],
        "salary": None,
        "work_format": "remote",
        "cities": ["Москва"],
        "experience_years": None,
        "employment_type": "labor_contract",
        "lang": "ru",
    }
    return base | kw


def example(i: int, family: str, answer: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"{family}:{i}",
        "family": family,
        "group": f"{family}:g{i}",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": f"Вакансия номер {i} для {family}"},
            {"role": "assistant", "content": compact(answer or label())},
        ],
    }


def pool(n: int) -> list[dict[str, Any]]:
    return [example(i, f) for f in ("trudvsem_habr", "telegram") for i in range(n)]


NO_CORRUPTION = Pairs(train_sample=4, work_format=0, role=0, drop_field=0, broken_json=0, holdout_share=0.0)


def test_every_role_has_valid_neighbours() -> None:
    assert set(ROLE_NEIGHBOURS) == set(ROLES)
    for role, near in ROLE_NEIGHBOURS.items():
        assert near and role not in near and set(near) <= set(ROLES)


def test_enum_swaps_change_one_value_and_stay_valid() -> None:
    rng = random.Random(1)
    chosen = compact(label())
    for kind, key in (("work_format", "work_format"), ("role", "role")):
        rejected = corrupt(kind, chosen, rng)
        a, b = json.loads(chosen), json.loads(rejected)
        assert try_parse(rejected) is not None
        assert a[key] != b[key]
        assert {k for k in a if a[k] != b[k]} == {key}
    assert json.loads(corrupt("role", chosen, rng))["role"] in ROLE_NEIGHBOURS["backend"]
    assert json.loads(corrupt("role", compact(label(role=None)), rng))["role"] == "other_it"


def test_work_format_swap_can_guess_or_drop() -> None:
    rng = random.Random(3)
    from_null = {
        json.loads(corrupt("work_format", compact(label(work_format=None)), rng))["work_format"] for _ in range(30)
    }
    assert from_null == {"office", "hybrid", "remote"}
    to_any = {json.loads(corrupt("work_format", compact(label()), rng))["work_format"] for _ in range(30)}
    assert to_any == {"office", "hybrid", None}


def test_format_corruptions_break_the_schema() -> None:
    rng = random.Random(2)
    chosen = compact(label())
    dropped = corrupt("drop_field", chosen, rng)
    assert try_parse(dropped) is None and len(json.loads(dropped)) == 8
    cut = corrupt("broken_json", chosen, rng)
    assert try_parse(cut) is None and chosen.startswith(cut) and len(cut) < len(chosen)


def test_disagreement_looks_at_the_compared_fields_only() -> None:
    chosen = compact(label())
    assert disagreement(chosen, compact(label(skills=["Go"], cities=[]))) == ()
    assert disagreement(chosen, compact(label(work_format="office", grade=None))) == ("work_format", "grade")
    assert disagreement(chosen, '{"role": "backend"') == ("invalid_json",)
    assert disagreement(chosen, None) == ("invalid_json",)


def test_student_inputs_sample_half_per_family_and_all_of_val() -> None:
    train, val = pool(10), [example(100, "telegram"), example(101, "trudvsem_habr")]
    rows = student_inputs(train, val, Pairs(train_sample=6))
    assert Counter((r["part"], r["family"]) for r in rows) == {
        ("train", "trudvsem_habr"): 3,
        ("train", "telegram"): 3,
        ("val", "telegram"): 1,
        ("val", "trudvsem_habr"): 1,
    }
    assert rows == student_inputs(list(reversed(train)), val, Pairs(train_sample=6))
    assert rows[0]["input"] == next(e for e in train if e["id"] == rows[0]["id"])["messages"][1]["content"]
    assert len(student_inputs(train, val, Pairs(train_sample=6, with_val=False))) == 6


def answers_for(inputs: list[dict[str, Any]], wrong: dict[str, str | None]) -> dict[str, str | None]:
    return {r["id"]: wrong.get(r["id"], compact(label())) for r in inputs}


def test_student_pairs_only_where_the_student_is_wrong() -> None:
    train = pool(4)
    inputs = student_inputs(train, [], NO_CORRUPTION)
    first, second, third = (r["id"] for r in inputs[:3])
    wrong = {first: compact(label(work_format="office")), second: "not json", third: compact(label(skills=[]))}
    kept, held, stats = build_pairs(NO_CORRUPTION, train, [], inputs, answers_for(inputs, wrong), [])
    assert held == []
    assert {r["record_id"]: r["fields"] for r in kept} == {first: ["work_format"], second: ["invalid_json"]}
    assert all(r["kind"] == "student" and r["chosen"] == compact(label()) for r in kept)
    assert sum(v for k, v in stats.items() if k.endswith(".answered")) == 4


def test_corrupted_pairs_skip_records_with_a_student_pair() -> None:
    train = pool(20)
    cfg = Pairs(train_sample=4, work_format=4, role=3, drop_field=2, broken_json=2, holdout_share=0.0)
    inputs = student_inputs(train, [], cfg)
    wrong = {inputs[0]["id"]: compact(label(role="frontend"))}
    kept, _, _ = build_pairs(cfg, train, [], inputs, answers_for(inputs, wrong), [])
    kinds = Counter(r["kind"] for r in kept)
    assert kinds == {"student": 1, "work_format": 4, "role": 3, "drop_field": 2, "broken_json": 2}
    corrupted = [r for r in kept if r["kind"] != "student"]
    assert inputs[0]["id"] not in {r["record_id"] for r in corrupted}
    assert len({r["record_id"] for r in corrupted}) == len(corrupted)
    assert Counter(r["family"] for r in corrupted if r["kind"] == "work_format") == {"trudvsem_habr": 2, "telegram": 2}
    assert all(r["chosen"] != r["rejected"] for r in kept)


def test_pairs_never_use_a_gold_record() -> None:
    train = pool(4)
    inputs = student_inputs(train, [], NO_CORRUPTION)
    wrong = {r["id"]: compact(label(role="qa")) for r in inputs}
    by_id = [{"id": inputs[0]["id"], "input": "other text"}]
    with pytest.raises(SystemExit, match="leak"):
        build_pairs(NO_CORRUPTION, train, [], inputs, answers_for(inputs, wrong), by_id)
    same_text = [{"id": "dev:1", "input": "  ВАКАНСИЯ номер 0\nдля   telegram "}]
    with pytest.raises(SystemExit, match="leak"):
        build_pairs(NO_CORRUPTION, train, [], inputs, answers_for(inputs, wrong), same_text)


def test_every_input_needs_a_student_answer() -> None:
    train = pool(4)
    inputs = student_inputs(train, [], NO_CORRUPTION)
    with pytest.raises(SystemExit, match="no student answer"):
        build_pairs(NO_CORRUPTION, train, [], inputs, {}, [])


def test_holdout_is_by_group_and_stable() -> None:
    train = pool(200)
    cfg = Pairs(train_sample=0, work_format=100, role=100, drop_field=0, broken_json=0, holdout_share=0.2)
    kept, held, _ = build_pairs(cfg, train, [], [], {}, [])
    again, held_again, _ = build_pairs(cfg, train, [], [], {}, [])
    assert kept == again and held == held_again
    assert {r["group"] for r in kept}.isdisjoint({r["group"] for r in held})
    assert 0.1 < len(held) / (len(kept) + len(held)) < 0.3


def test_report_has_denominators() -> None:
    stats = Counter({"val.telegram.answered": 10, "val.telegram.pairs": 3, "val.telegram.work_format": 2})
    rows = [{"kind": "student", "family": "telegram"}]
    metrics = report(rows, [], stats)
    assert metrics["student_disagrees.val.telegram.any"] == {"value": 0.3, "n": 3, "denominator": 10}
    assert metrics["student_disagrees.val.telegram.work_format"]["n"] == 2
    assert metrics["train.student.telegram"] == {"value": 1}
    assert "student_disagrees.train.telegram.any" not in metrics
