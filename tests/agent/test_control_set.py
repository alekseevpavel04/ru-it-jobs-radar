"""The control set in git: shape, safety of every reference query, split and sync with the questions."""

from collections import Counter

from radar.agent.control_set import Question, _stable_order, assign_splits, load, load_spec
from radar.agent.snapshot import pin_id, pinned
from radar.agent.sqlcheck import check


def test_at_least_100_questions_with_unique_ids() -> None:
    spec = load_spec()
    ids = [q.id for q in spec.questions]
    assert len(ids) >= 100 and len(set(ids)) == len(ids)
    assert len({u.id for u in spec.unsafe}) == len(spec.unsafe) >= 10


def test_every_reference_query_passes_the_agent_check() -> None:
    for q in load_spec().questions:
        assert check(q.sql).tables, q.id


def test_employer_names_are_asked_and_answered_for_trudvsem_only() -> None:
    # Habr and SuperJob allow aggregates only: their employers are asked about by rank and counts.
    for q in load_spec().questions:
        if "mart_employers" in q.sql and ("employer_name" in q.sql or "employer_key" in q.sql):
            assert "source = 'trudvsem'" in q.sql, q.id
    for item in load():
        for column in ("employer_name", "employer_key"):
            if column in item.answer.columns:
                assert "source = 'trudvsem'" in item.sql, item.id


def test_rows_without_order_by_are_sorted_ordered_ones_kept() -> None:
    rows = [["senior", 11], ["intern", 4]]
    assert _stable_order("SELECT grade, n FROM t", rows) == [["intern", 4], ["senior", 11]]
    assert _stable_order("SELECT grade, n FROM t ORDER BY n DESC", rows) == rows


def test_every_kind_has_dev_and_test() -> None:
    items = load()
    kinds = {i.kind for i in items}
    per = Counter((i.kind, i.split) for i in items)
    for kind in kinds:
        assert per[(kind, "dev")] >= 1 and per[(kind, "test")] >= 1, kind


def test_quick_subset_is_dev_only() -> None:
    quick = [i for i in load() if i.quick]
    assert quick and all(i.split == "dev" for i in quick)


def test_control_set_is_in_sync_with_the_questions_and_the_pin() -> None:
    spec = load_spec()
    items = {i.id: i for i in load()}
    assert set(items) == {q.id for q in spec.questions}
    splits = assign_splits(spec.questions, spec.seed, spec.dev_share)
    for q in spec.questions:
        item = items[q.id]
        assert (item.question, item.sql, item.kind) == (q.question, q.sql.strip(), q.kind), q.id
        assert (item.split, item.quick) == splits[q.id], q.id
        assert item.answer.rows, q.id
    assert {i.snapshot for i in items.values()} == {pin_id(pinned())}


def test_split_is_deterministic_and_independent_of_order() -> None:
    qs = [Question(id=f"q{i:02}", kind="k", question="?", sql="SELECT 1") for i in range(10)]
    first = assign_splits(qs, seed=1, dev_share=0.4)
    assert first == assign_splits(list(reversed(qs)), seed=1, dev_share=0.4)
    assert sum(split == "dev" for split, _ in first.values()) == 4
    assert sum(quick for _, quick in first.values()) == 2
