"""The marts keep their contract without a database: where they land, the tests that
guard their grain and their counts, and the seeds they read.

Building them on real data is `bash scripts/dbt.sh build`; the manifest comes from `dbt parse` with
placeholder credentials (fixture in test_dbt_project.py).
"""

import csv
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARTS = {
    "mart_weekly_dynamics": ["source", "week", "role_bucket"],
    "mart_salaries": ["source", "role_bucket", "grade", "region_group"],
    "mart_skills": ["source", "role_bucket", "grade", "skill"],
    "mart_work_format": ["source", "role_bucket", "work_format"],
    "mart_employers": ["source", "employer_key"],
    "mart_junior": ["source", "role_bucket"],
}
SEEDS = ROOT / "dbt" / "seeds"


def _models(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["name"]: n for n in manifest["nodes"].values() if n["resource_type"] == "model"}


def _tests(manifest: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        n
        for n in manifest["nodes"].values()
        if n["resource_type"] == "test" and n.get("test_metadata", {}).get("name") == name
    ]


def test_marts_land_in_marts_and_their_inputs_in_staging(manifest: dict[str, Any]) -> None:
    models = {name: m for name, m in _models(manifest).items() if m["fqn"][1] == "marts"}
    assert set(MARTS) | {"int_vacancy_records", "int_vacancy_entities"} == set(models)
    for name in MARTS:
        assert models[name]["schema"] == "marts", name
    # the read-only account sees only marts, so nothing intermediate may land there
    assert models["int_vacancy_records"]["schema"] == "staging"
    assert models["int_vacancy_entities"]["schema"] == "staging"
    assert _models(manifest)["stg_telegram"]["schema"] == "staging"


def test_every_mart_is_non_empty_and_has_its_grain_tested(manifest: dict[str, Any]) -> None:
    non_empty = {n["attached_node"].rsplit(".", 1)[-1] for n in _tests(manifest, "has_rows")}
    assert set(MARTS) <= non_empty
    grains = {
        (n["attached_node"].rsplit(".", 1)[-1], tuple(n["test_metadata"]["kwargs"]["columns"]))
        for n in _tests(manifest, "unique_combination")
    }
    for name, grain in MARTS.items():
        assert (name, tuple(grain)) in grains, name


def test_counts_never_exceed_their_denominators(manifest: dict[str, Any]) -> None:
    checked = {
        (n["attached_node"].rsplit(".", 1)[-1], n["test_metadata"]["kwargs"]["column_name"])
        for n in _tests(manifest, "not_above")
    }
    for pair in [
        ("mart_salaries", "with_salary_rub"),
        ("mart_skills", "vacancies_mentioning"),
        ("mart_work_format", "vacancies"),
        ("mart_junior", "open_to_junior"),
        ("mart_employers", "open_it_vacancies"),
        ("mart_weekly_dynamics", "telegram_posts"),
    ]:
        assert pair in checked, pair
    singular = {
        n["name"] for n in manifest["nodes"].values() if n["resource_type"] == "test" and "test_metadata" not in n
    }
    assert "assert_marts_consistent" in singular


def _buckets_in_macros() -> list[str]:
    text = (ROOT / "dbt" / "macros" / "marts.sql").read_text(encoding="utf-8")
    block = text[text.index("{% macro role_buckets()") : text.index("{% macro role_bucket_values()")]
    names: list[str] = []
    for name in re.findall(r"^\s*\('([a-z0-9_]+)', \[", block, re.M):
        if name not in names:
            names.append(name)
    return names


def test_bucket_values_tested_on_marts_are_those_the_rules_produce(manifest: dict[str, Any]) -> None:
    expected = {b for b in _buckets_in_macros() if b != "not_it"} | {"multi_post"}
    tested = [
        n
        for n in _tests(manifest, "accepted_values")
        if n["test_metadata"]["kwargs"]["column_name"] == "role_bucket"
        and n["attached_node"].rsplit(".", 1)[-1] in MARTS
    ]
    assert {n["attached_node"].rsplit(".", 1)[-1] for n in tested} == set(MARTS) - {"mart_employers"}
    for node in tested:
        assert set(node["test_metadata"]["kwargs"]["values"]) == expected


def test_telegram_channels_seed_matches_the_config() -> None:
    config = tomllib.loads((ROOT / "configs" / "telegram_channels.toml").read_text(encoding="utf-8"))["channel"]
    with (SEEDS / "telegram_channels.csv").open(encoding="utf-8", newline="") as f:
        seed = list(csv.DictReader(f))
    assert [(r["channel"], r["network"], r["topic"], r["enabled"]) for r in seed] == [
        (c["name"], c["network"], c["topic"], str(c["enabled"]).lower()) for c in config
    ]


@pytest.fixture(scope="module")
def dictionary() -> list[dict[str, str]]:
    with (SEEDS / "skill_dictionary.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_skill_dictionary_is_v0_sized_and_well_formed(dictionary: list[dict[str, str]]) -> None:
    assert 150 <= len(dictionary) <= 300
    skills = [r["skill"] for r in dictionary]
    assert len(skills) == len(set(skills))
    for row in dictionary:
        synonyms = row["synonyms"]
        assert synonyms and row["category"], row
        # matched against lowercased text with ё folded to е
        assert synonyms == synonyms.lower() and "ё" not in synonyms, row
        re.compile(f"(?:{synonyms})")


def _matches(dictionary: list[dict[str, str]], text: str) -> set[str]:
    # The same boundaries as macro skill_dictionary_constant: + and # are word characters.
    found = set()
    for row in dictionary:
        pattern = rf"(^|[^a-zа-я0-9_+#])({row['synonyms']})([^a-zа-я0-9_+#]|$)"
        if re.search(pattern, text):
            found.add(row["skill"])
    return found


def test_dictionary_tells_close_spellings_apart(dictionary: list[dict[str, str]]) -> None:
    text = "опыт на c++ и c#, знание java и spring, postgresql и ms sql, питон"
    found = _matches(dictionary, text)
    assert {"C++", "C#", "Java", "PostgreSQL", "MS SQL Server", "SQL", "Python"} <= found
    assert "JavaScript" not in found
    assert "Python" not in _matches(dictionary, "pythonista")
