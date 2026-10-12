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
    "mart_salaries": ["source", "level", "role_group", "role_bucket", "grade", "region_group"],
    "mart_skills": ["source", "role_bucket", "grade", "skill"],
    "mart_work_format": ["source", "role_bucket", "work_format"],
    "mart_employers": ["source", "employer_key"],
    "mart_junior": ["source", "role_bucket"],
    "mart_skills_weekly": ["source", "week", "skill"],
    "mart_lifetime": ["source", "level", "role_group"],
}
# Marts without a role bucket column: per employer, per week and skill, per role group.
WITHOUT_BUCKET = {"mart_employers", "mart_skills_weekly", "mart_lifetime"}
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


def test_history_selector_rebuilds_the_intermediate_models_and_marts_with_it(manifest: dict[str, Any]) -> None:
    # int_vacancy_records reads hist_*; a history-only rebuild would leave it and the marts stale
    selector = manifest["selectors"]["history_and_marts"]["definition"]
    assert selector == {"method": "path", "value": "models/history", "children": True}
    parents = _models(manifest)["int_vacancy_records"]["depends_on"]["nodes"]
    assert {"model.radar.hist_vacancy_lifetime", "model.radar.hist_snapshot_days"} <= set(parents)


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
        ("mart_weekly_dynamics", "telegram_channels"),
        ("mart_skills_weekly", "vacancies_mentioning"),
        ("mart_lifetime", "closed_in_quantiles"),
    ]:
        assert pair in checked, pair
    singular = {
        n["name"] for n in manifest["nodes"].values() if n["resource_type"] == "test" and "test_metadata" not in n
    }
    assert {"assert_marts_consistent", "assert_skills_found_matches_re2"} <= singular


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
    assert {n["attached_node"].rsplit(".", 1)[-1] for n in tested} == set(MARTS) - WITHOUT_BUCKET
    for node in tested:
        # the salary totals mark a level that does not split by bucket with 'all'; no Telegram there
        if node["attached_node"].endswith(".mart_salaries"):
            assert set(node["test_metadata"]["kwargs"]["values"]) == expected - {"multi_post"} | {"all"}
        else:
            assert set(node["test_metadata"]["kwargs"]["values"]) == expected


def test_role_groups_cover_every_bucket_as_the_survival_analysis_groups_them() -> None:
    from radar.models.survival import ROLE_GROUPS

    with (SEEDS / "role_groups.csv").open(encoding="utf-8", newline="") as f:
        seed = {r["role_bucket"]: r["role_group"] for r in csv.DictReader(f)}
    assert seed == ROLE_GROUPS
    assert set(seed) == {b for b in _buckets_in_macros() if b != "not_it"}


def test_salary_levels_are_one_grouping_set_each() -> None:
    sql = (ROOT / "dbt" / "models" / "marts" / "mart_salaries.sql").read_text(encoding="utf-8")
    block = sql[sql.index("{%- set levels = [") : sql.index("] %}")]
    levels = re.findall(r"\('(\w+)', \[([^\]]*)\]\)", block)
    names = [name for name, _ in levels]
    assert names[0] == "cell" and names[-1] == "source" and len(set(names)) == len(names)
    keys = [frozenset(re.findall(r"'(\w+)'", k)) for _, k in levels]
    # the level of a row is read from which dimensions are NULL, so no two sets may share a key set
    assert len(set(keys)) == len(keys)
    # role_bucket implies its role_group, so a set with the bucket carries the group too
    assert all("role_group" in k for k in keys if "role_bucket" in k)


def test_telegram_channels_seed_matches_the_config() -> None:
    config = tomllib.loads((ROOT / "configs" / "telegram_channels.toml").read_text(encoding="utf-8"))["channel"]
    with (SEEDS / "telegram_channels.csv").open(encoding="utf-8", newline="") as f:
        seed = list(csv.DictReader(f))
    assert [(r["channel"], r["network"], r["topic"], r["enabled"], r["lang"]) for r in seed] == [
        (c["name"], c["network"], c["topic"], str(c["enabled"]).lower(), c.get("lang", "")) for c in config
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
        for piece in synonyms.split("|"):
            # radar.extract.skill_dictionary.load_v0 splits on every |, so a group must not hold one
            re.compile(piece)
            # Hyperscan shortlists by the bare synonyms; it mishandles negated classes (macros/marts.sql)
            assert "[^" not in piece, row


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


@pytest.mark.parametrize(
    ("text", "skill", "expected"),
    [
        ("знание трудового законодательства", "Node.js", False),
        ("управление мастер-нодами кластера", "Node.js", False),
        ("multi-node обучение моделей", "Node.js", False),
        ("опыт на node.js и nodejs", "Node.js", True),
        ("please send your cv to the recruiter", "Computer vision", False),
        ("cv-инженер, opencv", "Computer vision", True),
        ("интеграционное тестирование", "API integration", False),
        ("интеграция ии в повседневную работу", "API integration", False),
        ("интеграция с внешними системами через http-сервисы", "API integration", True),
        ("работа с api, json", "API integration", True),
        ("доработка функционала по тз", "Requirements", False),
        ("в соответствии с требованиями технического задания", "Requirements", False),
        ("написание технических заданий разработчикам", "Requirements", True),
        ("опыт подготовки тз", "Requirements", True),
    ],
)
def test_narrowed_synonyms(dictionary: list[dict[str, str]], text: str, skill: str, expected: bool) -> None:
    assert (skill in _matches(dictionary, text)) is expected
