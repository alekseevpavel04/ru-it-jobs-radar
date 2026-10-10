import json
import re
from pathlib import Path

import numpy as np
import pytest

from radar.extract import baseline, skill_review
from radar.extract import skill_dictionary as sd
from radar.extract import skills as sk
from radar.extract.inputs import build_input


def found(text: str, matcher: sd.Matcher | None = None, english: bool = False) -> list[str]:
    return sd.find_skills(text, baseline.normalize(text), english, matcher)


def toy(tmp_path: Path, body: str) -> sk.Dictionary:
    path = tmp_path / "skills.toml"
    path.write_text(body, encoding="utf-8")
    return sk.load(path)


def test_literal_makes_inner_separators_optional() -> None:
    regex, anchor = sd.literal("Node.js")
    assert anchor == "node"
    for text in ("node.js", "nodejs", "node js", "node-js"):
        assert re.fullmatch(regex, text)
    assert re.fullmatch(sd.literal(".NET Core")[0], ".net core")


@pytest.mark.parametrize(
    ("pattern", "anchor"),
    [
        ("питон[а-я]*", "питон"),
        ("websockets?", "websocket"),
        ("fine-?tun[a-z]*", "fine"),
        ("a/?b[- ]тест[а-я]*", "a"),
        ("1[сc]-?битрикс", "1"),
        ("битрикс\\s*24", "битрикс"),
    ],
)
def test_regex_anchor(pattern: str, anchor: str) -> None:
    assert sd.regex_anchor(pattern) == anchor


def test_v1_finds_canonical_names_in_order() -> None:
    assert found("Опыт: Python 3.10, Postgres, k8s, sklearn") == ["Python", "PostgreSQL", "Kubernetes", "scikit-learn"]


def test_v1_keeps_the_parts_the_gold_keeps_apart() -> None:
    assert found("Linux (Ubuntu, Astra), Hadoop: HDFS, Hive") == [
        "Linux",
        "Ubuntu",
        "Astra Linux",
        "Hadoop",
        "HDFS",
        "Hive",
    ]


def test_v1_vendor_prefix_is_not_a_second_skill() -> None:
    assert found("Apache Kafka, Apache Spark и Apache") == ["Kafka", "Spark", "Apache"]


def test_v1_configuration_adds_1c() -> None:
    assert found("Знание 1С:ЗУП и 1С: Бухгалтерия предприятия 3.0") == ["1С", "1С:ЗУП", "1С:Бухгалтерия"]


def test_v1_names_not_searched_keep_their_aliases() -> None:
    assert found("MS Word, язык R; работа с word-файлами и r-значением") == ["Word", "R"]
    assert found("смета, рубеж, парус") == []


def test_gold_only_entries_are_not_searched(tmp_path: Path) -> None:
    d = toy(
        tmp_path,
        '[skills.data]\n"Kafka" = []\n"Kreuzberg" = []\n[provenance.gold_only]\n"Kreuzberg" = "test"\n',
    )
    assert found("Kafka и Kreuzberg", sd.build(d)) == ["Kafka"]
    assert found("Kafka и Kreuzberg", sd.build(d, search_gold_only=True)) == ["Kafka", "Kreuzberg"]


def test_v0_matcher_is_the_e1_dictionary() -> None:
    assert found("MS SQL и 1С-Битрикс", sd.load_v0()) == ["MS SQL Server", "Bitrix"]
    assert found("Ubuntu", sd.load_v0()) == ["Linux"]


def test_skill_source_is_what_the_baseline_searches() -> None:
    text = build_input(
        "Data Scientist", "Требования: Python, Postgres, Apache Airflow", [("Навыки", "Pandas, MatPlotLib, Golang")]
    )
    source, english = skill_review.skill_source(text)
    assert found(source, english=english) == baseline.extract(text).skills


def test_text_candidates_take_habr_header_and_latin_words_of_russian_text() -> None:
    text = build_input(
        "Аналитик",
        "Нужен опыт работы с Camunda и Apache Camel, знание процессов компании",
        [("Навыки", "SQL, Анализ данных")],
    )
    got = skill_review.text_candidates(text)
    assert {"sql", "анализ данных", "camunda", "apache camel", "apache", "camel"} <= got


def test_char_vectors_are_unit_length_and_closer_for_shared_spelling() -> None:
    v = skill_review.char_vectors(["PostgreSQL", "Postgres", "Kubernetes"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_alias_retrieval_counts_right_neighbours(tmp_path: Path) -> None:
    d = toy(tmp_path, '[skills.database]\n"PostgreSQL" = ["Postgres"]\n"ClickHouse" = ["Click House DB"]\n')
    strings = ["PostgreSQL", "Postgres", "ClickHouse", "Click House DB"]
    r = skill_review.alias_retrieval(d, strings, skill_review.char_vectors(strings), min_precision=0.9)
    assert (r["queries"], r["top1_right"], r["above_threshold"]) == (2, 2, 2)


def test_components() -> None:
    assert skill_review._components(4, [(0, 1), (2, 3), (1, 0)]) == [1, 1, 3, 3]


def test_provenance_marks_label_only_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = toy(
        tmp_path,
        '[skills.data]\n"Kafka" = []\n"Teradata" = []\n"Kreuzberg" = []\n"Docling" = []\n"Shared" = []\n',
    )
    data = tmp_path / "data"
    gold = data / "gold"
    gold.mkdir(parents=True)
    for split, names in (("dev", ["Kafka", "Docling", "Shared"]), ("test", ["Teradata", "Kreuzberg", "Shared"])):
        row = {"id": "x", "fields": {"skills": names}}
        (gold / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    skills_data = tmp_path / "skills"
    skills_data.mkdir()
    (skills_data / "pool_counts.json").write_text(
        json.dumps({"records": 100, "counts": {"Teradata": 40, "Kreuzberg": 2}}), encoding="utf-8"
    )
    monkeypatch.setenv("RADAR_DATA_DIR", str(data))
    monkeypatch.setattr(skill_review, "SKILLS_DATA", skills_data)
    # Kafka is in v0; Teradata has train_pool evidence; the rest only the labels name.
    assert skill_review.provenance(d, floor=10) == {"Docling": "dev", "Kreuzberg": "test", "Shared": "dev+test"}
