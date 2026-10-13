"""The models over extract.vacancy_fields: the extract views and the marts on extracted
fields (models/extracted) are off while var extraction_model is empty; with a model named they parse
into staging and marts with their grain, bounds and enumerations tested, and the skill seed matches
the dictionary v1.

`dbt parse` with placeholder credentials, as in conftest.py; no ClickHouse is contacted.
"""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, get_args

import pytest

from radar.api import vacancies
from radar.extract import schema

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = {"stg_vacancy_fields", "int_vacancy_fields"}
INTERMEDIATE = {"int_extracted_fields", "int_extracted_entities", "int_extracted_weekly_units"}
MARTS = {
    "mart_extracted_skills_weekly": ["source", "week", "skill"],
    "mart_extracted_fields_weekly": ["source", "week", "field", "value"],
    "mart_extracted_salaries": ["source", "level", "role", "grade"],
    "mart_extraction_coverage": ["source", "level", "week"],
    "mart_open_vacancies": ["vacancy_entity_id"],
}
SINGULAR = {
    "assert_open_vacancies_match_entities",
    "assert_extracted_marts_consistent",
    "assert_extracted_field_values",
    "assert_skill_seed_covers_extracted_spellings",
}


def _parse(out: Path, extraction_model: str) -> dict[str, Any]:
    env = os.environ | {
        "CLICKHOUSE_URL": "http://localhost:1",
        "CLICKHOUSE_USER": "parse_only",
        "CLICKHOUSE_PASSWORD": "parse_only",
        "DBT_PROJECT_DIR": str(ROOT / "dbt"),
        "DBT_PROFILES_DIR": str(ROOT / "dbt"),
        "PYTHONIOENCODING": "utf-8",
    }
    args = [
        "parse",
        "--no-partial-parse",
        "--vars",
        json.dumps({"extraction_model": extraction_model}),
        "--target-path",
        str(out / "target"),
        "--log-path",
        str(out / "logs"),
    ]
    run = subprocess.run(
        [sys.executable, "-P", "-c", "from dbt.cli.main import cli; cli()", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    doc: dict[str, Any] = json.loads((out / "target" / "manifest.json").read_text(encoding="utf-8"))
    return doc


@pytest.fixture(scope="module")
def manifest_on(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _parse(tmp_path_factory.mktemp("dbt_extract_on"), "student-q4")


@pytest.fixture(scope="module")
def manifest_off(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _parse(tmp_path_factory.mktemp("dbt_extract_off"), "")


def _models(manifest: dict[str, Any], folder: str) -> dict[str, dict[str, Any]]:
    return {n["name"]: n for n in manifest["nodes"].values() if n["resource_type"] == "model" and n["fqn"][1] == folder}


def _tests(manifest: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        n
        for n in manifest["nodes"].values()
        if n["resource_type"] == "test" and n.get("test_metadata", {}).get("name") == name
    ]


def _accepted(manifest: dict[str, Any], model: str, column: str) -> set[str]:
    found = [
        n["test_metadata"]["kwargs"]["values"]
        for n in _tests(manifest, "accepted_values")
        if n["attached_node"].endswith(f".{model}") and n["test_metadata"]["kwargs"]["column_name"] == column
    ]
    assert len(found) == 1, (model, column)
    return {str(v) for v in found[0]}


def test_extract_models_are_off_without_an_extraction_model(manifest_off: dict[str, Any]) -> None:
    assert not _models(manifest_off, "extract")
    assert not _models(manifest_off, "extracted")
    disabled = {n[0]["name"] for n in manifest_off["disabled"].values()}
    assert disabled >= EXTRACT | INTERMEDIATE | set(MARTS) | SINGULAR


def test_extract_models_are_staging_views_with_their_grain(manifest_on: dict[str, Any]) -> None:
    models = _models(manifest_on, "extract")
    assert set(models) == EXTRACT
    for name, model in models.items():
        assert model["schema"] == "staging", name
        assert model["config"]["materialized"] == "view", name
        assert model["description"].strip() and model["columns"], name
    grain = {
        (n["attached_node"].rsplit(".", 1)[-1], tuple(n["test_metadata"]["kwargs"]["columns"]))
        for n in _tests(manifest_on, "unique_combination")
    }
    assert ("stg_vacancy_fields", ("source", "source_id", "content_hash", "model_id")) in grain
    assert ("int_vacancy_fields", ("source", "source_id")) in grain
    sources = {s["name"] for s in manifest_on["sources"].values() if s["source_name"] == "extract"}
    assert sources == {"vacancy_fields", "vacancy_dropped"}


def test_extracted_marts_land_in_marts_with_their_grain_tested(manifest_on: dict[str, Any]) -> None:
    models = _models(manifest_on, "extracted")
    assert set(models) == INTERMEDIATE | set(MARTS)
    for name, model in models.items():
        # the read-only account sees only marts, so nothing intermediate may land there
        assert model["schema"] == ("marts" if name in MARTS else "staging"), name
        assert model["config"]["materialized"] == "table", name
        assert model["description"].strip() and model["columns"], name
    grains = {
        (n["attached_node"].rsplit(".", 1)[-1], tuple(n["test_metadata"]["kwargs"]["columns"]))
        for n in _tests(manifest_on, "unique_combination")
    }
    non_empty = {n["attached_node"].rsplit(".", 1)[-1] for n in _tests(manifest_on, "has_rows")}
    for name, grain in MARTS.items():
        assert (name, tuple(grain)) in grains, name
        assert name in non_empty, name
    singular = {
        n["name"] for n in manifest_on["nodes"].values() if n["resource_type"] == "test" and "test_metadata" not in n
    }
    assert singular >= SINGULAR | {"assert_skill_key_matches_seed"}


def test_counts_never_exceed_their_denominators_and_portals_stay_covered(manifest_on: dict[str, Any]) -> None:
    checked = {
        (n["attached_node"].rsplit(".", 1)[-1], n["test_metadata"]["kwargs"]["column_name"])
        for n in _tests(manifest_on, "not_above")
    }
    for pair in [
        ("mart_extracted_skills_weekly", "vacancies_mentioning"),
        ("mart_extracted_skills_weekly", "vacancies_with_fields"),
        ("mart_extracted_fields_weekly", "vacancies"),
        ("mart_extracted_salaries", "with_salary_rub_month"),
        ("mart_extraction_coverage", "units_with_fields"),
        ("mart_extraction_coverage", "units_valid"),
        ("mart_extraction_coverage", "skills_in_dictionary"),
    ]:
        assert pair in checked, pair
    bounds = {
        (n["test_metadata"]["kwargs"]["min_share"], n["config"]["severity"].lower(), n["config"]["where"])
        for n in _tests(manifest_on, "share_at_least")
    }
    portals = "level = 'source' AND source != 'telegram'"
    assert bounds == {(0.95, "warn", portals), (0.5, "error", portals)}


def test_open_vacancies_publish_names_and_links_only_where_the_rules_allow(manifest_on: dict[str, Any]) -> None:
    # docs/sources.md: titles and employer names of the open data only, no SuperJob links
    guarded = {
        (n["test_metadata"]["kwargs"]["column_name"], n["config"]["where"])
        for n in _tests(manifest_on, "is_null")
        if n["attached_node"].endswith(".mart_open_vacancies")
    }
    assert guarded == {
        ("title", "source != 'trudvsem'"),
        ("employer_name", "source != 'trudvsem'"),
        ("url", "source = 'superjob'"),
    }
    assert _accepted(manifest_on, "mart_open_vacancies", "source") == {"trudvsem", "habr", "superjob"}
    columns = set(_models(manifest_on, "extracted")["mart_open_vacancies"]["columns"])
    selected = re.search(r"SELECT (.*?)\s+FROM", vacancies.SEARCH_SQL, re.S)
    assert selected is not None
    read = {c.strip().split(" AS ")[-1] for c in selected.group(1).split(",")}
    assert read | {"skills"} == columns


def test_enumerations_are_those_of_the_extraction_schema(manifest_on: dict[str, Any]) -> None:
    enums = {
        "role": schema.ROLES,
        "grade": schema.GRADES,
        "work_format": get_args(schema.WorkFormat),
        "employment_type": get_args(schema.EmploymentType),
        "salary_currency": get_args(schema.Currency),
        "salary_gross_net": get_args(schema.GrossNet),
        "salary_period": get_args(schema.Period),
    }
    for column, values in enums.items():
        assert _accepted(manifest_on, "int_extracted_fields", column) == set(values), column
    assert _accepted(manifest_on, "mart_extracted_salaries", "role") == set(schema.ROLES) - {"non_it"} | {
        "unknown",
        "all",
    }
    assert _accepted(manifest_on, "mart_extracted_salaries", "grade") == set(schema.GRADES) | {"unknown", "all"}

    # the macro that the field-value test and the bands read
    text = (ROOT / "dbt" / "macros" / "extracted.sql").read_text(encoding="utf-8")
    block = text[text.index("{% macro extraction_enums()") : text.index("{% macro experience_bands()")]
    macro = {k: ast.literal_eval(v) for k, v in re.findall(r"'(\w+)': (\[[^\]]*\])", block)}
    assert macro == {
        "role": list(schema.ROLES),
        "grade": list(schema.GRADES),
        "work_format": list(get_args(schema.WorkFormat)),
        "employment_type": list(get_args(schema.EmploymentType)),
        "currency": list(get_args(schema.Currency)),
        "gross_net": list(get_args(schema.GrossNet)),
        "period": list(get_args(schema.Period)),
        "lang": list(get_args(schema.Lang)),
    }
    bands = _accepted(manifest_on, "int_extracted_fields", "experience_band")
    assert bands == {"none", "under_1", "1_3", "3_6", "6_plus", "unknown"}


def _skill_seed() -> Any:
    spec = importlib.util.spec_from_file_location("skill_seed", ROOT / "dbt" / "skill_seed.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_seed_dictionary_rows_match_the_dictionary_v1() -> None:
    from radar.extract import skills

    module = _skill_seed()
    assert module.main(["--check"]) == 0, "python dbt/skill_seed.py"
    seed = module.read_seed()
    d = skills.load()
    dictionary = [r for r in seed if r["origin"] == "dictionary"]
    assert {r["skill"] for r in dictionary} == set(d.skills)
    assert {r["spelling_key"]: r["skill"] for r in dictionary} == d.by_key
    assert all(r["key_hash"] == module.key_hash(r["spelling_key"]) for r in dictionary)


def test_skill_seed_corpus_rows_hold_hashes_and_dictionary_names_only() -> None:
    from radar.extract import skills

    seed = _skill_seed().read_seed()
    corpus = [r for r in seed if r["origin"] == "corpus"]
    names = set(skills.load().skills)
    # the spellings come from Habr and Telegram texts, which are not published
    assert all(r["spelling"] == "" and r["spelling_key"] == "" for r in corpus)
    assert all(re.fullmatch(r"[0-9a-f]{16}", r["key_hash"]) for r in seed)
    assert all((r["skill"] in names) == (r["status"] == "known") for r in corpus)
    assert all(r["status"] in ("known", "non_skill", "unknown") for r in corpus)
    pairs = [(r["key_hash"], r["skill"]) for r in seed]
    assert len(pairs) == len(set(pairs))


def test_corpus_spellings_resolve_by_normalize_list() -> None:
    module = _skill_seed()
    rows, conflicts = module.seed_rows(["Python 3.10", "Microsoft SQL Server 2019", "C#/.NET", "Soft Skills", "Xyzzy"])
    corpus = {(r["skill"], r["status"]) for r in rows if r["origin"] == "corpus"}
    assert {("C#", "known"), (".NET", "known"), ("", "non_skill"), ("", "unknown")} <= corpus
    by_hash = {r["key_hash"]: r for r in rows if r["origin"] == "corpus"}
    python = module.key_hash(module.skills.key("Python 3.10"))
    assert by_hash[python]["skill"] == "Python" and conflicts == 0
