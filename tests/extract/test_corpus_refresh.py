import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar.extract.corpus import TRAIN_POOL, WITHHELD, GroupRow, Record, SampleConfig
from radar.extract.inputs import input_for_record
from radar.extract.refresh import (
    S1,
    change_kind,
    closest_version,
    frozen_rows_unchanged,
    placeholders_added,
    raw_versions,
    refresh,
    refresh_stats,
    regroup,
    write_gold_inputs,
)
from radar.extract.role_proxy import BUCKETS, DEV

TEXT = (
    "Обязанности: разработка серверной части платежного сервиса на Go, проектирование API, ревью кода. "
    "Требования: опыт от трех лет, PostgreSQL, Kafka, Kubernetes. Контакт: Анна, пишите в чат портала."
)
OTHER = (
    "Обязанности: настройка и сопровождение сетевого оборудования филиалов, мониторинг каналов связи. "
    "Требования: CCNA, опыт администрирования Linux, готовность к командировкам по области."
)
FRESH = (
    "Обязанности: поддержка пользователей первой линии, учет техники, заявки в системе обращений. "
    "Требования: внимательность, базовые знания Windows и офисных программ, грамотная речь."
)


def config() -> SampleConfig:
    return SampleConfig(
        targets={"test": dict.fromkeys(BUCKETS, 0), "dev": dict.fromkeys(BUCKETS, 0)},
        power=0.5,
        max_share=0.6,
        leak_jaccard=0.5,
        shingle_k=3,
        num_perm=128,
        seed=7,
    )


def raw_row(source_id: str, text: str, fetched_at: str = "2026-10-08T10:00:00+00:00", h: str = "h") -> dict[str, Any]:
    return {
        "source": "trudvsem",
        "source_id": source_id,
        "fetched_at": fetched_at,
        "title": "Разработчик",
        "text": text,
        "payload": "{}",
        "content_hash": h,
    }


def staged_row(source_id: str, entity: str, cluster: str, dt: str = "2026-10-08") -> dict[str, Any]:
    return {
        "source": "trudvsem",
        "source_id": source_id,
        "dt": dt,
        "content_hash": f"h{source_id}",
        "vacancy_entity_id": entity,
        "text_cluster_id": cluster,
    }


def group_row(group: str, split: str, members: tuple[str, ...], gold: str | None = None) -> GroupRow:
    return GroupRow(
        group_id=group,
        split=split,
        stratum="trudvsem|dev",
        gold_id=gold,
        pi_test=0.1 if gold else None,
        pi_dev=0.05 if gold else None,
        n_corpus=1,
        member_ids=members,
    )


def s1_record(source_id: str, group: str, text: str) -> Record:
    return Record(
        id=f"trudvsem:{source_id}",
        source="trudvsem",
        source_id=source_id,
        dt="2026-10-08",
        content_hash="old",
        vacancy_entity_id=f"ve_{source_id}",
        text_cluster_id=f"tc_{source_id}",
        group_id=group,
        title="Разработчик",
        specializations=(),
        role_proxy=DEV,
        text_chars=len(text),
        input=input_for_record(raw_row(source_id, text)),
    )


def test_change_kind_tells_placeholders_from_other_edits() -> None:
    labeled = "Контакт: Анна Смирнова, тел. +7 900 000-00-00. Офис у метро."
    assert change_kind(labeled, labeled) == "same"
    assert change_kind(labeled, "Контакт: [name], тел. +7 900 000-00-00. Офис у метро.") == "pii"
    assert change_kind(labeled, "Контакт: [name], тел. [phone]. Офис у метро.") == "pii"
    assert change_kind(labeled, "Контакт: [name], тел. [phone]. Офис в центре.") == "other"
    assert change_kind("Пишите [email], Анна", "Пишите [email], [name]") == "pii"
    assert change_kind(labeled, labeled + " Новая строка.") == "other"
    assert placeholders_added("Пишите [email], Анна", "Пишите [email], [name]") == {"[name]": 1}


def test_closest_version_takes_the_cleaned_labeled_row_over_a_later_rewrite() -> None:
    labeled = input_for_record(raw_row("1", TEXT))
    cleaned = raw_row("1", TEXT.replace("Анна", "[name]"), "2026-10-08T09:00:00+00:00")
    later = raw_row("1", OTHER, "2026-10-08T18:00:00+00:00")
    row, current = closest_version(labeled, [later, cleaned])
    assert row is cleaned
    assert change_kind(labeled, current) == "pii"


def test_regroup_keeps_frozen_groups_links_new_records_and_names_the_rest() -> None:
    frozen = [
        group_row("grp_t", "test", ("trudvsem:1",), "trudvsem:1"),
        group_row("grp_d", "dev", ("trudvsem:2",), "trudvsem:2"),
    ]
    sticky = [group_row("grp_w", WITHHELD, ("trudvsem:3",))]
    staged = [
        staged_row("1", "ve_1", "tc_1"),
        staged_row("2", "ve_2", "tc_2"),
        staged_row("3", "ve_3", "tc_3"),
        staged_row("4", "ve_1", "tc_4"),  # a new record of the test vacancy
        staged_row("5", "ve_5", "tc_3"),  # the same text as an S1 withheld record
        staged_row("6", "ve_7", "tc_6"),
        staged_row("7", "ve_6", "tc_6"),
    ]
    assigned, joined = regroup(staged, frozen, sticky)
    assert assigned["trudvsem:1"] == ("grp_t", "test")
    assert assigned["trudvsem:2"] == ("grp_d", "dev")
    assert assigned["trudvsem:4"] == ("grp_t_linked", WITHHELD)
    assert assigned["trudvsem:3"] == assigned["trudvsem:5"] == ("grp_3", WITHHELD)
    assert assigned["trudvsem:6"] == assigned["trudvsem:7"] == ("grp_6", TRAIN_POOL)
    assert joined == []

    assigned, joined = regroup(
        [*staged, staged_row("8", "ve_8", "tc_2"), staged_row("9", "ve_8", "tc_1")], frozen, sticky
    )
    assert joined == [["grp_d", "grp_t"]]
    assert assigned["trudvsem:8"] == ("grp_d_linked", WITHHELD)


def test_raw_versions_reads_every_part_of_the_day(tmp_path: Path) -> None:
    day = tmp_path / "trudvsem" / "dt=2026-10-08"
    day.mkdir(parents=True)
    for n, rows in enumerate(
        [[raw_row("1", TEXT), raw_row("2", OTHER)], [raw_row("1", FRESH, "2026-10-08T12:00:00+00:00")]]
    ):
        pq.write_table(pa.Table.from_pylist(rows), day / f"part-{n}.parquet")
    found = raw_versions(tmp_path, "trudvsem", "2026-10-08", ["1"])
    assert sorted(r["text"] for r in found["1"]) == sorted([TEXT, FRESH])
    assert "2" not in found
    with pytest.raises(FileNotFoundError):
        raw_versions(tmp_path, "trudvsem", "2026-10-09", ["1"])


def test_refresh_keeps_gold_rows_rebuilds_inputs_and_withholds_neighbours(tmp_path: Path) -> None:
    test_gold, dev_gold, pool = (
        s1_record("1", "grp_t", TEXT),
        s1_record("2", "grp_d", OTHER),
        s1_record("3", "grp_p", FRESH),
    )
    rows = [
        group_row("grp_t", "test", ("trudvsem:1",), "trudvsem:1"),
        group_row("grp_d", "dev", ("trudvsem:2",), "trudvsem:2"),
        group_row("grp_p", TRAIN_POOL, ("trudvsem:3",)),
    ]
    s1 = S1(
        records=[test_gold, dev_gold, pool],
        members=[
            {
                "id": r.id,
                "source": r.source,
                "vacancy_entity_id": r.vacancy_entity_id,
                "group_id": r.group_id,
                "text_chars": 1,
            }
            for r in (test_gold, dev_gold, pool)
        ],
        rows=rows,
        inputs={
            "test": [{"id": "trudvsem:1", "source": "trudvsem", "input": test_gold.input}],
            "dev": [{"id": "trudvsem:2", "source": "trudvsem", "input": dev_gold.input}],
        },
    )
    cleaned = TEXT.replace("Анна", "[name]")
    raw_parquet = {"1": [raw_row("1", cleaned, h="new1")], "2": [raw_row("2", OTHER, h="h2")]}
    staged = [
        staged_row("1", "ve_1", "tc_1", dt="2026-10-09"),  # a newer version of the test record
        staged_row("2", "ve_2", "tc_2"),
        staged_row("3", "ve_3", "tc_3"),
        staged_row("4", "ve_1", "tc_4"),  # the test vacancy under a new id
        staged_row("5", "ve_5", "tc_5"),  # a new near copy of the test text, not linked by dedup
        staged_row("6", "ve_6", "tc_6"),
    ]
    raw = {
        ("trudvsem", "1"): raw_row("1", FRESH + " Обновлено."),
        ("trudvsem", "2"): raw_row("2", OTHER),
        ("trudvsem", "3"): raw_row("3", FRESH),
        ("trudvsem", "4"): raw_row("4", TEXT),
        ("trudvsem", "5"): raw_row("5", TEXT.replace("Kubernetes", "Docker")),
        ("trudvsem", "6"): raw_row("6", OTHER.replace("CCNA", "MikroTik").replace("Linux", "FreeBSD") + " " + FRESH),
    }
    out = refresh(staged, raw, s1, config(), lambda source, day, ids: {i: raw_parquet[i] for i in ids})

    by_id = {r.id: r for r in out.records}
    assert by_id["trudvsem:1"].input == input_for_record(raw_row("1", cleaned))
    assert by_id["trudvsem:1"].dt == "2026-10-08"
    assert by_id["trudvsem:1"].content_hash == "new1"
    assert out.kinds == {"trudvsem:1": "pii", "trudvsem:2": "same"}
    split = {r.group_id: r.split for r in out.rows}
    group = {m["id"]: m["group_id"] for m in out.members}
    assert split[group["trudvsem:4"]] == WITHHELD and group["trudvsem:4"] == "grp_t_linked"
    assert split[group["trudvsem:5"]] == WITHHELD and out.withheld_reason[group["trudvsem:5"]] == "near_duplicate"
    assert split[group["trudvsem:3"]] == TRAIN_POOL
    assert out.leak_after.neighbours == {"test": {}, "dev": {}}
    assert frozen_rows_unchanged(rows, out.rows)

    s1_hashes = {"test_ids_sha256": "", "dev_ids_sha256": "", "split_file_sha256": ""}
    metrics, extra = refresh_stats(out, s1, staged, s1_hashes)
    assert metrics["check.test_dev_rows_equal_s1"]["value"] == "true"
    assert metrics["check.test_ids_equal_s1"]["value"] == "false"
    assert metrics["gold.test.inputs_changed_pii_only"] == {"value": 1.0, "n": 1, "denominator": 1}
    assert metrics["gold.test.newer_version_in_staging"]["n"] == 1
    assert extra["gold_newer_version_other_text"] == {"test": ["trudvsem:1"], "dev": []}
    assert metrics["gold.test.placeholders_added.name"]["value"] == 1
    assert extra["gold_inputs_changed_other"] == {"test": [], "dev": []}

    path = tmp_path / "test_inputs.jsonl"
    write_gold_inputs(path, s1.inputs["test"], by_id)
    assert [json.loads(line)["input"] for line in path.read_text(encoding="utf-8").splitlines()] == [
        by_id["trudvsem:1"].input
    ]
