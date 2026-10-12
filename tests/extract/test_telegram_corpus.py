import json
import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar.extract.corpus import TRAIN_POOL, WITHHELD, GroupRow, SampleConfig
from radar.extract.role_proxy import BUCKETS, DEV, ML_DS_DATA, NON_IT_NOISE, QA_DEVOPS_SYS
from radar.extract.telegram_corpus import (
    RAW_COLUMNS,
    Config,
    Frozen,
    S1Corpus,
    TgRecord,
    assign,
    baseline_check,
    build_records,
    channel_langs,
    draw,
    drop_reason,
    file_sha256,
    fixture_ids,
    group_records,
    group_rows,
    hashtags,
    is_digest,
    link_ids,
    load_frozen,
    make_record,
    own_text,
    partitions,
    post_lang,
    read_posts,
    refresh_records,
    representatives,
    s1_links,
    telegram_role_proxy,
    with_groups,
    write_inputs,
    write_split,
)

GO = (
    "Backend-разработчик Go в Альфа Тех\n\n"
    "Задачи: разработка платежного шлюза, проектирование API, ревью кода коллег.\n"
    "Требования: Go от 3 лет, PostgreSQL, Kafka, Kubernetes.\n"
    "Условия: удалённо, от 250 000 руб. на руки.\n#вакансия #golang #middle"
)
ML = (
    "Вакансия: ML-инженер (рекомендательная система)\n\n"
    "Компания: Бета Маркет, Москва, гибрид.\n"
    "Обучение и вывод в прод моделей ранжирования, A/B-тесты, PyTorch, Spark, Airflow.\n"
    "Вилка 300-400к gross."
)
QA = (
    "Ищем QA Engineer в команду мобильного банка\n\n"
    "Ручное и автоматизированное тестирование, Python, pytest, Allure, CI в GitLab.\n"
    "Офис в Казани, оформление по ТК, опыт от 2 лет."
)
EN = (
    "Senior Data Engineer at Gamma Labs\n\n"
    "Gamma Labs builds fraud detection for marketplaces. Spark, Airflow, dbt, ClickHouse.\n"
    "Remote within Europe, $6-8k net, B2B contract. #remote #dataengineer"
)


def cfg(**targets: int) -> Config:
    test = dict.fromkeys(BUCKETS, 0) | targets
    return Config(
        min_content_chars=100,
        group_jaccard=0.8,
        sample=SampleConfig(
            targets={"test": test, "dev": dict.fromkeys(BUCKETS, 0)},
            power=0.5,
            max_share=1.0,
            leak_jaccard=0.5,
            shingle_k=3,
            num_perm=128,
            seed=7,
        ),
        baseline_pool_sample=10,
    )


def row(
    source_id: str,
    text: str,
    *,
    vacancy: bool = True,
    multi: bool = False,
    fetched: str = "2026-10-08T10:00:00+00:00",
    published: str = "2026-09-01T10:00:00+00:00",
    job_links: list[dict[str, Any]] | None = None,
    h: str = "h1",
    dt: str = "2026-10-08",
) -> dict[str, Any]:
    payload = {
        "channel": source_id.split("/")[0],
        "job_links": job_links or [],
        "rule_kind": "vacancy" if vacancy else "ad",
    }
    return {
        "source_id": source_id,
        "fetched_at": datetime.fromisoformat(fetched),
        "published_at": datetime.fromisoformat(published),
        "text": text,
        "payload": json.dumps(payload),
        "is_vacancy": vacancy,
        "multi_vacancy": multi,
        "content_hash": h,
        "dt": dt,
    }


def rec(source_id: str, text: str, network: str = "", **kw: Any) -> TgRecord:
    channel = source_id.split("/")[0]
    return make_record(row(source_id, text, **kw), {channel: network or channel})


def test_own_text_cuts_links_hashtags_handles_and_placeholders() -> None:
    text = "Python Developer https://u.habr.com/AbC12 #remote #python пишите @hr_bot или [nick]"
    assert own_text(text) == "Python Developer пишите или"
    assert hashtags("#data_science #backend@habr_career #ml, #ml") == ("data science", "backend", "ml")


def test_link_ids_from_payload_and_text() -> None:
    payload = {"job_links": [{"host": "career.habr.com", "habr_vacancy_id": 1000123456}, {"host": "geekjob.ru"}]}
    text = "Отклик: https://u.habr.com/XyZ1 и https://career.habr.com/vacancies/1000999999?f=telegram_c"
    assert link_ids(text, payload) == ("habr:1000123456", "habr:1000999999", "u.habr.com/XyZ1")


def test_drop_reasons_in_order() -> None:
    assert drop_reason(row("c/1", GO, vacancy=False), 100) == "not_vacancy"
    assert drop_reason(row("c/1", GO, multi=True), 100) == "multi_vacancy"
    digest = "Вакансии для лидов на Хабр Карьере:\n\n" + GO
    assert drop_reason(row("c/1", digest), 100) == "digest"
    assert drop_reason(row("c/1", "Middle/Senior DevOps https://u.habr.com/v9z4l #devops"), 100) == "link_stub"
    assert drop_reason(row("c/1", GO), 100) is None


def test_digest_by_links_and_link_lines() -> None:
    assert is_digest(GO, ("habr:1", "habr:2"))
    lines = GO + "\n→ http://amp.gs/a\n→ http://amp.gs/b\nhttps://bit.ly/c"
    assert is_digest(lines, ())
    assert not is_digest(GO, ("habr:1",))
    assert not is_digest("Вакансия дня: Backend Ruby on Rails developer в meyvn", ())
    assert is_digest("Работа для отлетевших\n\nСобрали топ из 5 вакансий на удалёнку.\n\nРазработка / Удалённо", ())
    assert is_digest("#стажировка\n\nКРОК приглашает на стажировки по направлениям:\n\n✅ Тестировщик", ())
    footer = GO + "\n" + "x" * 500 + "\nПодборки в бигтех, для джунов, канал со стажировками"
    assert not is_digest(footer, ())


def test_role_proxy_from_title_hashtags_and_a_role_line() -> None:
    assert telegram_role_proxy(GO, hashtags(GO)) == DEV
    assert telegram_role_proxy("Всем привет!\nМы растём\n#ml #вакансия", ("ml", "вакансия")) == ML_DS_DATA
    assert telegram_role_proxy("Всем привет!\nМы растём.\nПозиция: QA Engineer", ()) == QA_DEVOPS_SYS
    assert telegram_role_proxy("Всем привет!\nМы растём и ждём вас.", ()) == NON_IT_NOISE


def test_record_input_is_the_post_text_without_header_or_channel() -> None:
    r = rec("datasciencejobs/10", "  " + ML + "\n")
    assert r.input == ML
    assert "datasciencejobs" not in r.input
    assert r.id == "telegram:datasciencejobs/10"
    assert r.lang == "ru"
    assert rec("jobs/1", EN).lang == "en"


def write_partition(root: Path, day: str, rows: list[dict[str, Any]], complete: bool) -> None:
    d = root / "telegram" / f"dt={day}"
    d.mkdir(parents=True)
    table = pa.Table.from_pylist([{k: r[k] for k in RAW_COLUMNS} for r in rows])
    pq.write_table(table, d / "part-0.parquet")
    if complete:
        (d / "_SUCCESS").write_text("{}", encoding="utf-8")


def test_only_complete_partitions_are_read_and_the_latest_version_wins(tmp_path: Path) -> None:
    write_partition(tmp_path, "2026-10-08", [row("a/1", GO, h="old")], complete=True)
    write_partition(tmp_path, "2026-10-09", [row("a/1", QA, fetched="2026-10-09T10:00:00+00:00", h="new")], True)
    write_partition(tmp_path, "2026-10-10", [row("a/2", ML, fetched="2026-10-10T10:00:00+00:00")], False)
    complete, partial = partitions(tmp_path)
    assert (complete, partial) == (["2026-10-08", "2026-10-09"], ["2026-10-10"])
    posts, n_rows = read_posts(tmp_path, complete)
    assert n_rows == 2
    assert [(p["source_id"], p["content_hash"], p["dt"]) for p in posts] == [("a/1", "new", "2026-10-09")]


def test_build_records_counts_drops_per_reason_and_channel() -> None:
    rows = [
        row("a/1", GO),
        row("a/2", GO, vacancy=False),
        row("a/3", GO, multi=True),
        row("b/1", "Senior Java https://u.habr.com/x1"),
    ]
    posts = build_records(rows, 4, {}, cfg())
    assert [r.id for r in posts.records] == ["telegram:a/1"]
    assert posts.dropped == {"not_vacancy": 1, "multi_vacancy": 1, "digest": 0, "link_stub": 1, "channel_lang": 0}
    assert posts.not_vacancy_kinds == {"ad": 1}
    assert posts.per_channel["b"]["link_stub"] == 1


def test_cross_posts_and_shared_links_are_one_group() -> None:
    habr = [{"host": "career.habr.com", "habr_vacancy_id": 1000000001}]
    records = [
        rec("geekjobs/1", GO, network="geekjob"),
        rec("jobforjunior/7", GO + "\nБольше вакансий для junior: @jobforjunior", network="geekjob"),
        rec("habr_career/5", ML, job_links=habr),
        rec("progjob/9", QA, job_links=habr),
        rec("remotejun/3", EN),
    ]
    grouping = group_records(records, cfg())
    g = grouping.group_of
    assert g["telegram:geekjobs/1"] == g["telegram:jobforjunior/7"]
    assert g["telegram:habr_career/5"] == g["telegram:progjob/9"]
    assert len(set(g.values())) == 3
    assert grouping.edges["same_link"] == 1


def test_a_new_record_joining_a_frozen_gold_group_is_linked() -> None:
    frozen = [GroupRow("tg_x", "test", "c|dev", "telegram:c/1", 0.1, 0.0, 1, ("telegram:c/1",))]
    records = [rec("c/1", GO), rec("d/2", GO), rec("e/3", QA)]
    grouping = group_records(records, cfg(), frozen)
    assert grouping.group_of["telegram:c/1"] == "tg_x"
    assert grouping.group_of["telegram:d/2"] == "tg_x_linked"
    assert grouping.linked == {"tg_x": ["telegram:d/2"]}


VOCAB = [a + b for a in ("ка", "ло", "ми", "ну", "пе", "ру", "са", "ти") for b in ("бор", "дел", "жук", "зон", "лис")]


def distinct_posts(n: int) -> list[TgRecord]:
    """Posts that share no word 3-shingle beyond chance: numbers normalize to 0, so words differ."""
    out = []
    for i in range(n):
        body = " ".join(random.Random(i).sample(VOCAB, 20))
        out.append(rec(f"ch{i % 3}/{i}", f"Backend-разработчик Python\n\n{body}"))
    return out


def grouped(records: list[TgRecord], c: Config) -> list[TgRecord]:
    return with_groups(records, group_records(records, c).group_of)


def test_draw_is_deterministic_disjoint_and_skips_blocked_groups() -> None:
    c = cfg(dev=4)
    records = grouped(distinct_posts(12), c)
    reps = representatives(records)
    blocked = {records[0].group_id}
    a = draw(reps, blocked, c.sample)
    b = draw(reps, blocked, c.sample)
    assert a.split_of == b.split_of
    assert sum(s == "test" for s in a.split_of.values()) == 4
    assert records[0].group_id not in a.split_of
    rows = group_rows(records, a.split_of, a.plan)
    assert {r.gold_id for r in rows if r.split == "test"} <= {r.id for r in records}
    assert all(r.pi_test is not None and 0 < r.pi_test <= 1 for r in rows if r.split == "test")


def test_draw_skips_a_near_duplicate_of_a_drawn_record() -> None:
    c = cfg(dev=2)
    base = distinct_posts(1)[0].input
    twin_a = rec("ch0/100", base + "\nЕщё строка про условия работы в офисе.")
    twin_b = rec("ch1/101", base + "\nДругая строка: гибрид, ДМС, обучение за счёт компании.")
    records = grouped([twin_a, twin_b], c)
    assert len({r.group_id for r in records}) == 2
    out = draw(representatives(records), set(), c.sample)
    assert len(out.split_of) == 1
    assert out.skipped_near_gold == 1
    assert out.shortfall["test"] == {DEV: 1}


def test_assign_withholds_pool_neighbours_of_gold_and_s1_gold_links() -> None:
    c = cfg(dev=1)
    gold = rec("a/1", GO)
    near = rec("b/2", GO.replace("Kubernetes", "Docker").replace("ревью кода коллег", "ревью кода"))
    linked = rec("c/3", QA, job_links=[{"host": "career.habr.com", "habr_vacancy_id": 1000000042}])
    free = rec("d/4", EN)
    records = grouped([gold, near, linked, free], c)
    group_of = {r.id: r.group_id for r in records}
    assert group_of[gold.id] != group_of[near.id]
    rows = group_rows(records, {group_of[gold.id]: "test"}, None)
    s1 = S1Corpus(split_of={"habr:1000000042": "dev"}, inputs={})
    links = s1_links(records, s1, c.sample)
    assert links == {linked.id: {"link": ["habr:1000000042"]}}
    out = assign(rows, records, links, s1, c.sample, {})
    split = {r.group_id: r.split for r in out.rows}
    assert split[group_of[near.id]] == WITHHELD
    assert out.reason[group_of[near.id]] == "near_duplicate"
    assert split[group_of[linked.id]] == WITHHELD
    assert out.reason[group_of[linked.id]] == "s1_gold"
    assert split[group_of[free.id]] == TRAIN_POOL
    assert out.leak_after == {"test": {}, "dev": {}}


def test_s1_links_by_text() -> None:
    c = cfg()
    tg = grouped([rec("a/1", GO)], c)
    s1 = S1Corpus(split_of={"habr:1": "train_pool"}, inputs={"habr:1": "Должность: Go\n\n" + GO})
    assert s1_links(tg, s1, c.sample) == {"telegram:a/1": {"text": ["habr:1"]}}


def test_refresh_keeps_the_drawn_input_unless_only_pii_changed() -> None:
    drawn = [rec("a/1", GO + "\nПишите Анне"), rec("b/2", QA), rec("c/3", ML)]
    drawn = [replace(r, group_id=f"g{i}") for i, r in enumerate(drawn)]
    frozen = Frozen(records={r.id: r for r in drawn}, rows=[])
    current = [rec("a/1", GO + "\nПишите [name]"), rec("b/2", QA + "\nОбновлено: удалёнка"), rec("z/9", EN)]
    records, kinds = refresh_records(current, frozen)
    by_id = {r.id: r for r in records}
    assert kinds == {"telegram:a/1": "pii", "telegram:b/2": "other", "telegram:c/3": "missing"}
    assert by_id["telegram:a/1"].input.endswith("[name]")
    assert by_id["telegram:a/1"].group_id == "g0"
    assert by_id["telegram:b/2"].input == QA
    assert "telegram:z/9" in by_id


def test_gold_inputs_have_the_line_format_of_the_s1_gold_inputs(tmp_path: Path) -> None:
    path = tmp_path / "tg_dev_inputs.jsonl"
    write_inputs(path, [rec("a/1", GO), rec("b/2", QA)], seed=7)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert all(set(x) == {"id", "source", "input"} and x["source"] == "telegram" for x in lines)
    assert {x["id"] for x in lines} == {"telegram:a/1", "telegram:b/2"}


def test_load_frozen_refuses_a_changed_split(tmp_path: Path) -> None:
    rows = [GroupRow("g", "test", "c|dev", "telegram:a/1", 0.5, 0.0, 1, ("telegram:a/1",))]
    write_split(rows, tmp_path / "split.parquet")
    for name in ("tg_test_inputs.jsonl", "tg_dev_inputs.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    recorded = {
        "split_file_sha256": "0" * 64,
        "test_inputs_sha256": file_sha256(tmp_path / "tg_test_inputs.jsonl"),
        "dev_inputs_sha256": file_sha256(tmp_path / "tg_dev_inputs.jsonl"),
    }
    with pytest.raises(SystemExit, match="hashes to"):
        load_frozen(tmp_path, recorded)


def test_fixture_ids(tmp_path: Path) -> None:
    (tmp_path / "page.html").write_text('<div data-post="datasciencejobs/3364"></div>', encoding="utf-8")
    assert fixture_ids(tmp_path) == {"telegram:datasciencejobs/3364"}


def test_regex_baseline_accepts_telegram_inputs() -> None:
    texts = [
        GO,
        ML,
        QA,
        EN,
        "🔥 Ищем дизайнера!\n\nФигма, удалёнка, 150к. #design",
        "Позиция: 1С программист\nЗП 200 000",
    ]
    inputs = [(f"telegram:x/{i}", rec(f"x/{i}", t).input) for i, t in enumerate(texts)]
    assert baseline_check(inputs) == (len(texts), [])


def test_post_lang_reads_what_the_post_itself_says() -> None:
    # Links and hashtags are Latin in any post: with them this title is 11 Cyrillic of 49 letters.
    assert post_lang("Java разработчик\nhttps://telegra.ph/Java-razrabotchik-01-25 #java #remote #itjob") == "ru"
    assert post_lang(GO) == "ru"
    assert post_lang(EN) == "en"
    assert post_lang("Senior PHP Developer\nhttps://telegra.ph/Senior-PHP-Developer-04-08 #remote #php") == "en"
    assert post_lang("https://telegra.ph/Data-Scientist-06-28 [link]") == "other"
    assert post_lang(None) == "other"


def test_a_channel_kept_to_russian_drops_its_other_posts_only() -> None:
    assert drop_reason(row("remoteit/1", EN), 100, "ru") == "channel_lang"
    assert drop_reason(row("remoteit/2", GO), 100, "ru") is None
    assert drop_reason(row("jobs/1", EN), 100) is None
    # the earlier reasons come first, so their counts do not move
    assert drop_reason(row("remoteit/3", EN, vacancy=False), 100, "ru") == "not_vacancy"
    rows = [row("remoteit/1", EN), row("remoteit/2", GO), row("jobs/1", EN)]
    posts = build_records(rows, 3, {}, cfg(), {"remoteit": "ru"})
    assert [r.id for r in posts.records] == ["telegram:remoteit/2", "telegram:jobs/1"]
    assert posts.dropped["channel_lang"] == 1
    assert posts.per_channel["remoteit"]["channel_lang"] == 1


def test_channel_langs_from_the_config(tmp_path: Path) -> None:
    path = tmp_path / "channels.toml"
    path.write_text('[[channel]]\nname = "a"\nlang = "ru"\n\n[[channel]]\nname = "b"\n', encoding="utf-8")
    assert channel_langs(path) == {"a": "ru"}
    assert channel_langs()["remoteit"] == "ru"


def test_published_at_is_iso_utc() -> None:
    r = make_record(row("a/1", GO, published="2026-09-01T10:00:00+00:00"), {})
    assert datetime.fromisoformat(r.published_at) == datetime(2026, 9, 1, 10, tzinfo=UTC)
