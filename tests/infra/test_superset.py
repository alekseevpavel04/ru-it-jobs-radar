"""Static checks of Superset: the compose service and the assets in infra/superset/assets."""

import csv
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "infra" / "compose"
COMPOSE: dict[str, Any] = yaml.safe_load((COMPOSE_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICE: dict[str, Any] = COMPOSE["services"]["superset"]
LIB = (COMPOSE_DIR / "lib.sh").read_text(encoding="utf-8")
ASSETS = ROOT / "infra" / "superset" / "assets"
SECRETS = {"SUPERSET_SECRET_KEY", "SUPERSET_ADMIN_PASSWORD"}


def load(kind: str) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(ASSETS).as_posix(): yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((ASSETS / kind).rglob("*.yaml"))
    }


DATABASES = load("databases")
DATASETS = load("datasets")
CHARTS = load("charts")
DASHBOARDS = load("dashboards")
DATASET_BY_UUID = {d["uuid"]: d for d in DATASETS.values()}


def dataset_of(chart: dict[str, Any]) -> dict[str, Any]:
    return DATASET_BY_UUID[chart["dataset_uuid"]]


def reads_salaries(chart: dict[str, Any]) -> bool:
    dataset = dataset_of(chart)
    return dataset["table_name"] == "mart_salaries" or "mart_salaries" in (dataset["sql"] or "")


def simple_filters(chart: dict[str, Any]) -> dict[str, tuple[str, Any]]:
    return {
        f["subject"]: (f["operator"], f["comparator"])
        for f in chart["params"]["adhoc_filters"]
        if f["expressionType"] == "SIMPLE"
    }


def test_superset_is_the_bi_profile() -> None:
    bi = {name for name, svc in COMPOSE["services"].items() if "bi" in svc.get("profiles", [])}
    assert bi == {"superset"}
    assert SERVICE["profiles"] == ["bi"]


def test_superset_listens_on_loopback_with_capped_memory() -> None:
    assert SERVICE["ports"] == ["127.0.0.1:8088:8088"]
    assert SERVICE["mem_limit"] == "2g"
    assert SERVICE["image"] == "radar/superset:6.1.0"
    dockerfile = (COMPOSE_DIR / SERVICE["build"] / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM apache/superset:6\.1\.0$", dockerfile, flags=re.MULTILINE)
    assert re.search(r"clickhouse-connect==\d+\.\d+\.\d+", dockerfile)


def test_superset_secrets_are_profile_secrets_and_no_clickhouse_account() -> None:
    env = SERVICE["environment"]
    for name in SECRETS:
        assert env[name] == f"${{{name}:-}}"
    bi = re.search(r'^  \[bi\]="(.+)"$', LIB, flags=re.MULTILINE)
    assert bi is not None and set(bi.group(1).split()) == SECRETS
    # The read-only account reaches Superset only through scripts/superset_import.sh.
    assert not [name for name in env if name.startswith("CLICKHOUSE")]
    assert "depends_on" not in SERVICE


def test_metadata_volume_and_readonly_config() -> None:
    assert SERVICE["volumes"] == ["superset-home:/app/superset_home", "../superset:/app/radar:ro"]
    assert "superset-home" in COMPOSE["volumes"]


def test_database_points_at_marts_without_credentials() -> None:
    assert len(DATABASES) == 1
    (database,) = DATABASES.values()
    assert database["sqlalchemy_uri"] == "clickhousedb://clickhouse:8123/marts"
    assert database["allow_dml"] is False
    assert all(d["database_uuid"] == database["uuid"] for d in DATASETS.values())


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_datasets_read_marts_only(name: str) -> None:
    dataset = DATASETS[name]
    assert dataset["schema"] == "marts"
    if dataset["sql"]:
        sql = re.sub(r"--[^\n]*", "", dataset["sql"])
        tables = re.findall(r"\b(?:FROM|JOIN)\s+([\w.]+)", sql, flags=re.IGNORECASE)
        assert tables and all(t.startswith("marts.mart_") for t in tables), tables


def test_charts_and_dashboards_reference_existing_objects() -> None:
    chart_uuids = {c["uuid"] for c in CHARTS.values()}
    for chart in CHARTS.values():
        assert chart["dataset_uuid"] in DATASET_BY_UUID, chart["slice_name"]
    used = set()
    for dashboard in DASHBOARDS.values():
        for node in dashboard["position"].values():
            if isinstance(node, dict) and node.get("type") == "CHART":
                assert node["meta"]["uuid"] in chart_uuids
                used.add(node["meta"]["uuid"])
    assert used == chart_uuids


def test_expected_dashboards() -> None:
    slugs = {d["slug"] for d in DASHBOARDS.values()}
    assert slugs == {"radar-market", "radar-skills", "radar-salaries", "radar-junior", "radar-lifetime"}


@pytest.mark.parametrize("name", sorted(DASHBOARDS))
def test_every_dashboard_states_the_slice_and_update_date(name: str) -> None:
    position = DASHBOARDS[name]["position"]
    text = " ".join(n["meta"]["code"] for n in position.values() if isinstance(n, dict) and n.get("type") == "MARKDOWN")
    assert "без hh.ru" in text
    period = {c["uuid"] for c in CHARTS.values() if dataset_of(c)["table_name"] == "collection_period"}
    charts = {n["meta"]["uuid"] for n in position.values() if isinstance(n, dict) and n.get("type") == "CHART"}
    assert period & charts


SALARY_LEVELS = {"cell", "bucket_grade", "bucket", "group_grade", "group", "grade", "region", "source"}
SALARY_CHARTS = sorted(name for name, chart in CHARTS.items() if reads_salaries(chart))


def test_there_are_salary_charts() -> None:
    assert len(SALARY_CHARTS) >= 4


@pytest.mark.parametrize("name", SALARY_CHARTS)
def test_every_salary_chart_filters_one_level_and_n(name: str) -> None:
    chart = CHARTS[name]
    filters = simple_filters(chart)
    # mart_salaries holds several aggregation levels; rows of different levels must never meet.
    assert filters.get("level", (None, None))[0] == "==", f"{name} does not filter level by equality"
    assert filters["level"][1] in SALARY_LEVELS
    assert filters.get("with_from") == (">=", 9), f"{name} shows cells with n below 9"
    params = chart["params"]
    split_by_source = "source" in filters or "source_ru" in params.get("groupby", [])
    assert split_by_source, f"{name} mixes sources"


def test_role_group_mapping_matches_the_seed() -> None:
    with (ROOT / "dbt" / "seeds" / "role_groups.csv").open(encoding="utf-8") as f:
        seed = {row["role_bucket"]: row["role_group"] for row in csv.DictReader(f)}
    for dataset in DATASETS.values():
        for column in dataset["columns"]:
            expression = column["expression"] or ""
            inner = re.match(r"transform\(transform\(role_bucket, \[(.*?)\], \[(.*?)\], role_bucket\)", expression)
            if inner:
                buckets, groups = (re.findall(r"'([^']*)'", part) for part in inner.groups())
                assert dict(zip(buckets, groups, strict=True)) == seed, dataset["table_name"]


def test_texts_have_no_long_dash() -> None:
    for path in ASSETS.rglob("*.yaml"):
        assert "—" not in path.read_text(encoding="utf-8"), path.name


def test_titles_are_russian() -> None:
    for chart in CHARTS.values():
        assert re.search(r"[а-яё]", chart["slice_name"], flags=re.IGNORECASE), chart["slice_name"]
    for dashboard in DASHBOARDS.values():
        assert re.search(r"[а-яё]", dashboard["dashboard_title"], flags=re.IGNORECASE)
