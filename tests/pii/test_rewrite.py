import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from radar.collect import habr, superjob, trudvsem, trudvsem_snapshot
from radar.collect.envelope import RawRecord, content_hash
from radar.collect.summary import read_meta
from radar.collect.telegram import MSK, VOLATILE, keep_handles
from radar.collect.writer import PartitionWriter, partition_dir
from radar.pii import clean_habr_card_company, clean_text, records, rewrite, text

KEEP = keep_handles()
NOW = datetime.now(UTC)
TODAY = NOW.astimezone(MSK).date()
PAST = TODAY - timedelta(days=7)
# Stored by the cleaner before name lines were caught: the nick is gone, the name line is not.
WITH_NAME = "Иванова Мария\nДата-инженер, удалённо\nОтклик: [nick]"
CLEAN = "Python-разработчик, офис в Казани"


def record(post_id: int, text: str, pii: dict[str, int] | None = None) -> RawRecord:
    payload: dict[str, Any] = {"channel": "job_python", "views": post_id, "rule_kind": "vacancy"}
    return RawRecord(
        source="telegram",
        source_id=f"job_python/{post_id}",
        fetched_at=NOW,
        published_at=NOW - timedelta(days=30),
        modified_at=None,
        url=f"https://t.me/job_python/{post_id}",
        title=None,
        text=text,
        payload=payload,
        content_hash=content_hash(None, text, payload, ignore=VOLATILE),
        pii_removed=pii or {},
        is_vacancy=True,
        multi_vacancy=False,
    )


def write(root: Path, dt: Any, records: list[RawRecord], age: timedelta | None = None) -> list[Path]:
    with PartitionWriter("telegram", dt, root=root) as writer:
        writer.add_many(records)
    if age is not None:
        stamp = time.time() - age.total_seconds()
        for part in writer.parts:
            os.utime(part, (stamp, stamp))
    return writer.parts


def rows(part: Path) -> list[dict[str, Any]]:
    return pq.read_table(part).to_pylist()


def test_rewrites_old_parts_keeps_rows_and_skips_fresh_ones(tmp_path: Path) -> None:
    hour = timedelta(hours=1)
    (old,) = write(tmp_path, PAST, [record(1, WITH_NAME, {"nick": 1}), record(2, CLEAN)], age=hour)
    (fresh,) = write(tmp_path, PAST, [record(3, WITH_NAME)])
    marker = (partition_dir("telegram", PAST, tmp_path) / "_SUCCESS").read_bytes()
    (today,) = write(tmp_path, TODAY, [record(4, WITH_NAME)], age=hour)
    before = rows(old)
    untouched = {p: p.read_bytes() for p in (fresh, today)}

    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)

    after = rows(old)
    assert [r["source_id"] for r in after] == [r["source_id"] for r in before]
    changed, same = after
    assert changed["text"] == "[name]\nДата-инженер, удалённо\nОтклик: [nick]"
    assert dict(changed["pii_removed"]) == {"nick": 1, "person_name": 1}
    payload = json.loads(changed["payload"])
    assert changed["content_hash"] == content_hash(None, changed["text"], payload, ignore=VOLATILE)
    assert changed["content_hash"] != before[0]["content_hash"]
    for key in ("fetched_at", "published_at", "url", "payload", "is_vacancy"):
        assert changed[key] == before[0][key]
    assert same == before[1]
    assert {p: p.read_bytes() for p in (fresh, today)} == untouched
    assert (partition_dir("telegram", PAST, tmp_path) / "_SUCCESS").read_bytes() == marker
    assert not list(tmp_path.rglob("*.tmp"))

    assert summary["rows_checked"] == 2
    assert summary["rows_changed"] == 1
    assert summary["pii_added"] == {"person_name": 1}
    assert summary["hash_not_reproduced"] == 0
    assert summary["parts_rewritten"] == 1
    assert {(s["dt"], s.get("part"), s["reason"]) for s in summary["skipped"]} == {
        (PAST.isoformat(), fresh.name, "modified in the last 30 minutes"),
        (TODAY.isoformat(), None, "partition of today (Moscow)"),
    }
    (entry,) = read_meta("telegram", rewrite.META, tmp_path)
    assert entry["part"] == old.name and entry["rows_changed"] == 1 and entry["pii_added"] == {"person_name": 1}

    again = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)
    assert again["rows_changed"] == 0
    assert rows(old) == after


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    (part,) = write(tmp_path, PAST, [record(1, WITH_NAME)], age=timedelta(hours=1))
    content = part.read_bytes()
    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW, dry_run=True)
    assert summary["rows_changed"] == 1
    assert part.read_bytes() == content
    assert read_meta("telegram", rewrite.META, tmp_path) == []


def test_versions_that_differed_only_in_a_name_keep_both_rows(tmp_path: Path) -> None:
    other = WITH_NAME.replace("Иванова Мария", "Петрова Анна")
    (part,) = write(tmp_path, PAST, [record(1, WITH_NAME), record(1, other)], age=timedelta(hours=1))
    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)
    after = rows(part)
    assert len(after) == 2
    assert after[0]["content_hash"] == after[1]["content_hash"]
    assert summary["duplicate_keys_in_rewritten_partitions"] == 1


def test_scan_counts_what_is_left(tmp_path: Path) -> None:
    write(tmp_path, PAST, [record(1, WITH_NAME), record(2, CLEAN), record(3, "Пишите @someone_private")])
    found = rewrite.scan("telegram", KEEP, tmp_path)
    assert found["rows"] == 3
    assert found["rows_with_findings"] == 2
    assert found["findings"] == {"nick": 1, "person_name": 1}
    assert found["person_name_by_rule"] == {"name_line": 1}


def test_verify_refuses_a_lost_row(tmp_path: Path) -> None:
    (part,) = write(tmp_path, PAST, [record(1, WITH_NAME), record(2, CLEAN)])
    table = pq.read_table(part)
    cleaned = rewrite.SPECS["telegram"].cleaned
    with pytest.raises(rewrite.RewriteError, match="rows"):
        rewrite._verify(table, table.slice(0, 1), cleaned)
    with pytest.raises(rewrite.RewriteError, match="column url"):
        rewrite._verify(table, table.set_column(6, "url", table.column("title")), cleaned)


def test_partition_without_success_is_skipped(tmp_path: Path) -> None:
    writer = PartitionWriter("telegram", PAST, root=tmp_path)
    writer.add(record(1, WITH_NAME))
    writer.close(complete=False)
    (part,) = writer.parts
    stamp = time.time() - 3600
    os.utime(part, (stamp, stamp))
    content = part.read_bytes()
    summary = rewrite.rewrite("telegram", KEEP, root=tmp_path, now=NOW)
    assert part.read_bytes() == content
    assert summary["skipped"] == [{"dt": PAST.isoformat(), "parts": 1, "reason": "partition without _SUCCESS"}]


# The other collectors, end to end: a record stored by the cleaner before the name rules, then
# rewritten, must equal the record the collector builds today from the same input.


@contextmanager
def rules_before_names() -> Iterator[None]:
    """The text cleaner without the first-name rules of `first_names.csv`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(text, "_person_names", lambda _text, _context: iter(()))
        text._clean_short.cache_clear()
        yield
    text._clean_short.cache_clear()


# Caught only by the new rules: a first name with a word that is not surname-shaped, alone on its
# line (so a section heading in front of it would hide it), and one in a signature line.
NAME_LINE = "Мария Шмидт"
SIGNED = "HR: Анна Шмидт"

HABR_ITEM: dict[str, Any] = {
    "id": 1000000201,
    "href": "/vacancies/1000000201",
    "title": "Python-разработчик",
    "isMarked": False,
    "remoteWork": True,
    "publishedDate": {"date": "2026-10-01T10:00:00+03:00", "title": "1 октября"},
    # A sole trader listed under the own name, without the "ИП" the employer rules look for.
    "company": {"id": 1000000077, "alias_name": "shop-77", "title": NAME_LINE, "accredited": False},
    "employment": "full_time",
    "salary": {"from": 200000, "to": None, "currency": "rur", "formatted": "от 200 000 ₽"},
    "skills": [{"title": "Python", "href": "/vacancies/skills/python"}],
    "locations": [{"title": "Москва", "href": "/vacancies?city_id=678"}],
    "favorite": False,
    "archived": False,
    "hidden": False,
}
HABR_DESCRIPTION = f"Пишем сервисы на Python.\n{NAME_LINE}\nКонтакт в профиле, {SIGNED}"


def habr_record() -> RawRecord:
    # As habr.fetch_card and habr.card_fields build it.
    description, removed = clean_text(HABR_DESCRIPTION)
    company, company_removed = clean_habr_card_company({"id": 1000000077, "name": NAME_LINE, "description": SIGNED})
    card = {
        "text": description,
        "card": {"employmentType": "FULL_TIME"},
        "company_card": company,
        "card_fetched_at": NOW.isoformat(),
        "card_list_hash": habr.list_entry_hash(HABR_ITEM),
        "published_at": NOW - timedelta(days=7),
        "pii_removed": dict(removed + company_removed),
    }
    return habr.build_record(HABR_ITEM, NOW, ["all_by_date"], card)


API_VACANCY: dict[str, Any] = {
    "id": "a1b2c3d4-0000-4000-8000-000000000001",
    "source": "Электронная служба",
    "region": {"region_code": "7700000000000", "name": "Город Москва"},
    "company": {"name": "ООО Ромашка", "inn": "7701234567", "ogrn": "1027700000001"},
    "creation-date": "2026-09-01",
    "date_modify": "2026-10-01",
    "job-name": "Программист Python",
    "duty": f"{NAME_LINE}\nРазработка сервисов",
    "requirements": "Опыт от 3 лет",
    "conditions": f"Офис в центре\n{SIGNED}",
    "requirement": {"education": "Высшее", "experience": 3},
    "contact_list": [{"contact_type": "Телефон", "contact_value": "+7 900 000-00-00"}],
}


def api_record() -> RawRecord:
    return trudvsem.to_record(API_VACANCY, ["industry_it", "text_python"], NOW)


SNAPSHOT_VACANCY: dict[str, Any] = {
    "id": "b1b2c3d4-0000-4000-8000-000000000002",
    "vacancyName": "Аналитик данных",
    "creationDate": "2026-09-01",
    "dateModify": "2026-10-01",
    "deleted": "false",
    "regionName": "Москва",
    "fullCompanyName": "ООО Ромашка",
    "responsibilities": f"{NAME_LINE}\nОтчёты",
    # Equal once the names are gone: the collector then keeps only the first of them.
    "requirements": "SQL\nМария Шмидт",
    "qualifications": "SQL\nАнна Шмидт",
    "benefitDetails": f"ДМС\n{SIGNED}",
    "contactPerson": "Иванова Мария Петровна",
}


def snapshot_record() -> RawRecord:
    version = trudvsem_snapshot.Version("20261001T000000", "")
    return trudvsem_snapshot.to_record(SNAPSHOT_VACANCY, version, NOW)


SUPERJOB_VACANCY: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "superjob" / "vacancies_page.json").read_text(encoding="utf-8")
)["objects"][0]


def superjob_record() -> RawRecord:
    vacancy = {**SUPERJOB_VACANCY, "vacancyRichText": SUPERJOB_VACANCY["vacancyRichText"] + f"<p>{NAME_LINE}</p>"}
    return superjob.to_record(vacancy, ["catalogue_33"], NOW)


@pytest.mark.parametrize(
    ("source", "build"),
    [
        ("habr", habr_record),
        ("trudvsem", api_record),
        ("trudvsem_snapshot", snapshot_record),
        ("superjob", superjob_record),
    ],
)
def test_rewrite_gives_the_record_the_collector_builds_now(
    tmp_path: Path, source: str, build: Callable[[], RawRecord]
) -> None:
    with rules_before_names():
        stored = build()
    collected = build()
    assert stored.content_hash != collected.content_hash
    with PartitionWriter(source, PAST, root=tmp_path) as writer:
        writer.add(stored)
    (part,) = writer.parts
    stamp = time.time() - 3600
    os.utime(part, (stamp, stamp))
    keep = rewrite.keep_for(source)

    summary = rewrite.rewrite(source, keep, root=tmp_path, now=NOW)

    (row,) = rows(part)
    assert summary["rows_changed"] == 1
    assert summary["hash_not_reproduced"] == 0
    assert summary["changed_rows_scan"]["rows_with_person_name_after"] == 0
    assert row["content_hash"] == collected.content_hash
    assert (row["title"], row["text"]) == (collected.title, collected.text)
    assert row["payload"] == json.dumps(collected.payload, ensure_ascii=False, separators=(",", ":"))
    assert dict(row["pii_removed"]) == collected.pii_removed
    assert rewrite.scan(source, keep, tmp_path)["rows_with_person_name"] == 0


# A person employer's address as stored by a cleaner that did not cut it, and one the current
# cleaner cuts after the town: a Zelenograd building with no street word in front.
PERSON_ADDRESS = "г Москва, г Зеленоград, к. 1001"


def api_person_record() -> RawRecord:
    vacancy = {
        **API_VACANCY,
        "company": {"name": "ИП Иванов Иван Иванович", "inn": "770000000001"},
        "addresses": {"address": [{"location": PERSON_ADDRESS}]},
    }
    return trudvsem.to_record(vacancy, ["industry_it"], NOW)


def snapshot_person_record() -> RawRecord:
    vacancy = {
        **SNAPSHOT_VACANCY,
        "fullCompanyName": "ИП Иванов Иван Иванович",
        "company": {"inn": "770000000001"},
        "vacancyAddress": PERSON_ADDRESS,
    }
    return trudvsem_snapshot.to_record(vacancy, trudvsem_snapshot.Version("20261001T000000", ""), NOW)


def superjob_town_record() -> RawRecord:
    return superjob.to_record({**SUPERJOB_VACANCY, "address": "Москва, Примерный р-н, кв-л 12"}, ["catalogue_33"], NOW)


@pytest.mark.parametrize(
    ("source", "build"),
    [
        ("trudvsem", api_person_record),
        ("trudvsem_snapshot", snapshot_person_record),
        ("superjob", superjob_town_record),
    ],
)
def test_rewrite_cuts_an_address_the_collector_cuts_now(
    tmp_path: Path, source: str, build: Callable[[], RawRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as mp:
        mp.setattr(records, "coarse_address", lambda address: address)
        stored = build()
    collected = build()
    assert stored.content_hash != collected.content_hash
    with PartitionWriter(source, PAST, root=tmp_path) as writer:
        writer.add(stored)
    (part,) = writer.parts
    stamp = time.time() - 3600
    os.utime(part, (stamp, stamp))

    summary = rewrite.rewrite(source, rewrite.keep_for(source), root=tmp_path, now=NOW)

    (row,) = rows(part)
    assert summary["pii_added"] == {"address": 1}
    assert summary["hash_not_reproduced"] == 0
    assert row["content_hash"] == collected.content_hash
    assert row["payload"] == json.dumps(collected.payload, ensure_ascii=False, separators=(",", ":"))
    assert dict(row["pii_removed"]) == collected.pii_removed


@pytest.mark.parametrize(
    ("source", "build"),
    [
        ("trudvsem", api_person_record),
        ("trudvsem_snapshot", snapshot_person_record),
        ("superjob", superjob_town_record),
    ],
)
def test_an_address_cut_by_the_current_cleaner_is_left_alone(
    tmp_path: Path, source: str, build: Callable[[], RawRecord]
) -> None:
    with PartitionWriter(source, PAST, root=tmp_path) as writer:
        writer.add(build())
    stamp = time.time() - 3600
    os.utime(writer.parts[0], (stamp, stamp))
    summary = rewrite.rewrite(source, rewrite.keep_for(source), root=tmp_path, now=NOW)
    assert (summary["rows_changed"], summary["parts_rewritten"]) == (0, 0)


def test_a_company_address_is_not_cut_again(tmp_path: Path) -> None:
    # A company named after a person: the name holds «[name]», the INN says it is a company.
    company = {**API_VACANCY["company"], "name": "ООО [name] и партнёры"}
    vacancy = {**API_VACANCY, "company": company, "addresses": {"address": [{"location": PERSON_ADDRESS}]}}
    stored = trudvsem.to_record(vacancy, ["industry_it"], NOW)
    with PartitionWriter("trudvsem", PAST, root=tmp_path) as writer:
        writer.add(stored)
    stamp = time.time() - 3600
    os.utime(writer.parts[0], (stamp, stamp))
    summary = rewrite.rewrite("trudvsem", rewrite.keep_for("trudvsem"), root=tmp_path, now=NOW)
    assert summary["rows_changed"] == 0


def test_habr_card_is_not_read_again_after_a_rewrite(tmp_path: Path) -> None:
    with rules_before_names():
        stored = habr_record()
    with PartitionWriter("habr", PAST, root=tmp_path) as writer:
        writer.add(stored)
    stamp = time.time() - 3600
    os.utime(writer.parts[0], (stamp, stamp))
    assert habr.latest_previous(TODAY, tmp_path)[stored.source_id]["card_list_hash"] != habr.list_entry_hash(HABR_ITEM)

    summary = rewrite.rewrite("habr", rewrite.keep_for("habr"), root=tmp_path, now=NOW)

    assert summary["notes"] == {"list_hash_changed": 1}
    known = habr.latest_previous(TODAY, tmp_path)[stored.source_id]
    # What the collector compares to decide whether the card is stale.
    assert known["card_list_hash"] == habr.list_entry_hash(HABR_ITEM)
    assert known["text"] == habr_record().text


def test_habr_card_read_for_an_older_list_entry_stays_stale(tmp_path: Path) -> None:
    with rules_before_names():
        stored = habr_record()
    payload = {**stored.payload, "card_list_hash": "0" * 64}
    older = stored.model_copy(update={"payload": payload})
    with PartitionWriter("habr", PAST, root=tmp_path) as writer:
        writer.add(older)
    stamp = time.time() - 3600
    os.utime(writer.parts[0], (stamp, stamp))
    rewrite.rewrite("habr", rewrite.keep_for("habr"), root=tmp_path, now=NOW)
    (row,) = rows(writer.parts[0])
    assert json.loads(row["payload"])["card_list_hash"] == "0" * 64


def test_sections_split_on_headings_in_order() -> None:
    sections = rewrite.SectionedText(("Обязанности", "Требования", "Условия"), dedupe=False)
    head = sections.head
    text_ = "Обязанности: код\n\nУсловия: офис\n\nТребования: опыт\n\nПримечание: нет"
    parts, ambiguous = sections.split(text_, head)
    assert parts == [("Обязанности", "код"), ("Условия", "офис\n\nТребования: опыт\n\nПримечание: нет")]
    assert ambiguous
    with pytest.raises(rewrite.RewriteError):
        sections.split("код без заголовка", head)
