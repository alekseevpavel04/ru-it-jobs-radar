import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from radar.extract.corpus import (
    MIN_TEXT_CHARS,
    TRAIN_POOL,
    WITHHELD,
    GroupRow,
    Record,
    SampleConfig,
    allocate,
    build_records,
    draw_splits,
    fixture_ids,
    group_ids,
    id_list_digest,
    largest_remainder,
    latest,
    near_duplicates,
    permanent_random,
    plan,
    records_in_two_splits,
    split_digest,
    withhold,
)
from radar.extract.role_proxy import BUCKETS, DEV, ML_DS_DATA

LONG = "Обязанности: разработка и сопровождение внутренних сервисов компании на Python и PostgreSQL. " * 2


def config(**over: Any) -> SampleConfig:
    base: dict[str, Any] = {
        "targets": {"test": dict.fromkeys(BUCKETS, 0), "dev": dict.fromkeys(BUCKETS, 0)},
        "power": 0.5,
        "max_share": 0.6,
        "leak_jaccard": 0.5,
        "shingle_k": 3,
        "num_perm": 128,
        "seed": 7,
    }
    return SampleConfig(**{**base, **over})


def record(rid: str, group: str, bucket: str = DEV, text: str = LONG, dt: str = "2026-10-08") -> Record:
    source, source_id = rid.split(":")
    return Record(
        id=rid,
        source=source,
        source_id=source_id,
        dt=dt,
        content_hash="h",
        vacancy_entity_id=f"ve_{source_id}",
        text_cluster_id=f"tc_{source_id}",
        group_id=group,
        title="t",
        specializations=(),
        role_proxy=bucket,
        text_chars=len(text),
        input=f"Должность: t\n\n{text}",
    )


def members_of(records: list[Record]) -> list[dict[str, Any]]:
    return [{"id": r.id, "source": r.source, "group_id": r.group_id, "text_chars": r.text_chars} for r in records]


def test_groups_join_entities_through_a_shared_text_cluster() -> None:
    groups = group_ids([("ve_b", "tc_1"), ("ve_c", "tc_1"), ("ve_c", "tc_2"), ("ve_d", "tc_2"), ("ve_a", "tc_9")])
    assert groups == {"ve_b": "grp_b", "ve_c": "grp_b", "ve_d": "grp_b", "ve_a": "grp_a"}


def test_latest_prefers_the_later_day_then_the_api() -> None:
    rows = [
        {"source": "trudvsem_snapshot", "source_id": "1", "dt": "2026-07-21"},
        {"source": "trudvsem_snapshot", "source_id": "1", "dt": "2026-10-08"},
        {"source": "trudvsem", "source_id": "1", "dt": "2026-10-08"},
    ]
    assert latest(rows)["source"] == "trudvsem"
    assert latest(rows[:2])["dt"] == "2026-10-08"


def test_build_records_skips_short_text_and_takes_the_latest_version_with_text() -> None:
    staged = [
        {"source": "trudvsem", "source_id": "1", "dt": "2026-10-08", "content_hash": "a",
         "vacancy_entity_id": "ve_1", "text_cluster_id": "tc_1"},
        {"source": "trudvsem_snapshot", "source_id": "2", "dt": "2026-07-01", "content_hash": "b",
         "vacancy_entity_id": "ve_1", "text_cluster_id": "tc_1"},
        {"source": "habr", "source_id": "3", "dt": "2026-10-08", "content_hash": "c",
         "vacancy_entity_id": "ve_3", "text_cluster_id": "tc_3"},
    ]  # fmt: skip
    raw = {
        ("trudvsem", "1"): {"source": "trudvsem", "title": "Программист", "text": "Обязанности: -", "payload": "{}"},
        ("trudvsem_snapshot", "2"): {
            "source": "trudvsem_snapshot",
            "title": "Программист",
            "text": LONG,
            "payload": json.dumps({"salaryMin": "50000", "salaryMax": "0"}),
        },
        ("habr", "3"): {"source": "habr", "title": "Data Engineer", "text": None, "payload": "{}"},
    }
    groups = group_ids((r["vacancy_entity_id"], r["text_cluster_id"]) for r in staged)
    records = build_records(staged, raw, groups)
    assert [r.id for r in records] == ["trudvsem_snapshot:2"]
    rec = records[0]
    assert rec.group_id == "grp_1"
    assert rec.role_proxy == DEV
    assert rec.input.startswith("Должность: Программист\nЗарплата: от 50 000 руб.\n\nОбязанности:")
    assert rec.text_chars >= MIN_TEXT_CHARS


def test_largest_remainder_hits_the_total() -> None:
    assert largest_remainder({"a": 1.6, "b": 1.3, "c": 0.1}, 3) == {"a": 2, "b": 1, "c": 0}
    assert sum(largest_remainder({"a": 0.5, "b": 0.5}, 1).values()) == 1


def test_allocate_power_and_caps() -> None:
    assert allocate(10, {"a": 100, "b": 100}, 0.5, 0.6) == {"a": 5, "b": 5}
    sqrt = allocate(30, {"big": 900, "small": 100}, 0.5, 0.6)
    assert sqrt == {"big": 23, "small": 7}  # 22.5 and 7.5, the tie goes by name
    proportional = allocate(30, {"big": 900, "small": 100}, 1.0, 0.6)
    assert proportional == {"big": 27, "small": 3}
    capped = allocate(20, {"big": 900, "tiny": 10}, 0.0, 0.6)
    assert capped == {"big": 14, "tiny": 6}
    assert allocate(20, {"x": 10, "y": 5}, 0.5, 0.6) == {"x": 6, "y": 3}


def test_plan_splits_each_bucket_into_test_and_dev_and_reports_shortfall() -> None:
    targets = {"test": dict.fromkeys(BUCKETS, 0), "dev": dict.fromkeys(BUCKETS, 0)}
    targets["test"][ML_DS_DATA], targets["dev"][ML_DS_DATA] = 8, 4
    targets["test"][DEV], targets["dev"][DEV] = 30, 10
    frame = {"habr|ml_ds_data": 10, "trudvsem|ml_ds_data": 5, "habr|dev": 100, "trudvsem|dev": 400}
    p = plan(frame, config(targets=targets))
    ml = ("habr|ml_ds_data", "trudvsem|ml_ds_data")
    assert sum(p.draw["test"][h] + p.draw["dev"][h] for h in ml) == 9  # caps 6 and 3
    assert p.shortfall["test"].get(ML_DS_DATA, 0) + p.shortfall["dev"].get(ML_DS_DATA, 0) == 3
    assert p.draw["test"]["habr|dev"] + p.draw["test"]["trudvsem|dev"] == 30
    assert p.draw["dev"]["habr|dev"] + p.draw["dev"]["trudvsem|dev"] == 10


def test_permanent_random_is_fixed_and_uniform_range() -> None:
    assert permanent_random(7, "grp_a") == permanent_random(7, "grp_a")
    assert permanent_random(7, "grp_a") != permanent_random(8, "grp_a")
    assert all(0 <= permanent_random(7, f"g{i}") < 1 for i in range(100))


def frame_records() -> list[Record]:
    out = []
    for i in range(40):
        out.append(record(f"trudvsem:{i:03d}", f"grp_t{i:03d}", ML_DS_DATA if i < 10 else DEV, text=f"{LONG} {i}"))
    # a second, older record in one group: the group still gives one gold record, its latest
    out.append(record("trudvsem_snapshot:000", "grp_t000", ML_DS_DATA, dt="2026-07-01"))
    return out


def test_draw_splits_is_disjoint_deterministic_and_stores_inclusion_probability() -> None:
    targets = {"test": dict.fromkeys(BUCKETS, 0), "dev": dict.fromkeys(BUCKETS, 0)}
    targets["test"][ML_DS_DATA], targets["dev"][ML_DS_DATA] = 3, 2
    targets["test"][DEV], targets["dev"][DEV] = 6, 3
    cfg = config(targets=targets)
    records = frame_records()
    blocked = {"trudvsem:001"}
    rows, p = draw_splits(records, members_of(records), blocked, cfg)
    again, _ = draw_splits(records, members_of(records), blocked, cfg)
    assert rows == again

    by_split = Counter(r.split for r in rows)
    assert by_split == {"test": 9, "dev": 5, TRAIN_POOL: 26}
    assert records_in_two_splits(rows) == 0
    blocked_row = next(r for r in rows if r.group_id == "grp_t001")
    assert blocked_row.split == TRAIN_POOL and blocked_row.stratum is None and blocked_row.pi_test is None
    ml_row = next(r for r in rows if r.stratum == "trudvsem|ml_ds_data")
    assert ml_row.pi_test == pytest.approx(3 / 9) and ml_row.pi_dev == pytest.approx(2 / 9)
    assert p.frame["trudvsem|ml_ds_data"] == 9
    two = next(r for r in rows if r.group_id == "grp_t000")
    assert two.member_ids == ("trudvsem:000", "trudvsem_snapshot:000")
    assert two.n_corpus == 2
    if two.split in ("test", "dev"):
        assert two.gold_id == "trudvsem:000"


def test_near_duplicate_of_a_gold_record_is_withheld_from_training() -> None:
    gold = record("trudvsem:g", "grp_g", text=LONG + " Офис в Казани.")
    twin = record("trudvsem:twin", "grp_twin", text=LONG + " Офис в Самаре.")
    other = record(
        "trudvsem:other", "grp_other", text="Требования: знание бухгалтерского учета и 1С, опыт от трёх лет."
    )
    records = [gold, twin, other]
    rows = [
        GroupRow("grp_g", "test", "trudvsem|dev", "trudvsem:g", 1.0, 0.0, 1, ("trudvsem:g",)),
        GroupRow("grp_twin", TRAIN_POOL, "trudvsem|dev", None, 0.0, 0.0, 1, ("trudvsem:twin",)),
        GroupRow("grp_other", TRAIN_POOL, "trudvsem|dev", None, 0.0, 0.0, 1, ("trudvsem:other",)),
    ]
    cfg = config()
    leak = near_duplicates(rows, records, cfg)
    assert leak.neighbours["test"] == {"trudvsem:g": ["trudvsem:twin"]}
    after = withhold(rows, records, leak)
    assert [r.split for r in after] == ["test", WITHHELD, TRAIN_POOL]
    assert near_duplicates(after, records, cfg).neighbours == {"test": {}, "dev": {}}


def test_fixture_ids(tmp_path: Path) -> None:
    (tmp_path / "trudvsem").mkdir()
    (tmp_path / "habr").mkdir()
    (tmp_path / "trudvsem" / "page.json").write_text('{"id": "C50D9CD8-bbe3-11f1-9a11-2f8460579df4"}')
    (tmp_path / "habr" / "list.json").write_text('{"id": 1000150058, "company": 21000150058}')
    assert fixture_ids(tmp_path) == {
        "trudvsem:c50d9cd8-bbe3-11f1-9a11-2f8460579df4",
        "trudvsem_snapshot:c50d9cd8-bbe3-11f1-9a11-2f8460579df4",
        "habr:1000150058",
    }


def test_digests_ignore_order() -> None:
    assert id_list_digest(["b", "a"]) == id_list_digest(["a", "b"])
    rows = [
        GroupRow("g1", "test", None, None, None, None, 1, ("x", "y")),
        GroupRow("g2", TRAIN_POOL, None, None, None, None, 1, ("z",)),
    ]
    assert split_digest(rows) == split_digest(list(reversed(rows)))
    moved = [rows[0], GroupRow("g2", "dev", None, None, None, None, 1, ("z",))]
    assert split_digest(rows) != split_digest(moved)
