"""Static checks of the DAG files. Airflow is not installed in the project's environment (it lives in
the image, airflow/Dockerfile), so the files are read as syntax trees; the real parse runs inside
the container in test_dag_parse.py.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from radar.stream.producer import SOURCES

ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "airflow" / "dags"

ACTIVE = {"radar_collect", "radar_load", "radar_dbt", "radar_sla"}
SKELETONS = {"radar_ml_label", "radar_retrain", "radar_publish"}
# The scheduler and the DAG processor import every file often: only Airflow and the standard
# library at the top, the project's code runs in tasks under its own environment.
ALLOWED_IMPORT_ROOTS = {"airflow", "datetime"}


def dag_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DAG"
    ]


def keyword(call: ast.Call, name: str) -> object:
    for kw in call.keywords:
        if kw.arg == name:
            return ast.literal_eval(kw.value)
    raise KeyError(name)


def assigned(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise KeyError(name)


def constant(tree: ast.Module, name: str) -> object:
    return ast.literal_eval(assigned(tree, name))


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


FILES = sorted(DAGS.glob("*.py"))


def test_one_file_per_expected_dag() -> None:
    assert {p.stem for p in FILES} == ACTIVE | SKELETONS


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_file_defines_its_dag(path: Path) -> None:
    calls = dag_calls(parse(path))
    assert len(calls) == 1, path.name
    assert keyword(calls[0], "dag_id") == path.stem


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_only_light_imports_at_top_level(path: Path) -> None:
    tree = parse(path)
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert roots <= ALLOWED_IMPORT_ROOTS, f"{path.name}: import of {roots}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_skeletons_paused_and_daily_dags_active(path: Path) -> None:
    (call,) = dag_calls(parse(path))
    assert keyword(call, "is_paused_upon_creation") is (path.stem in SKELETONS)


def test_load_and_collect_cover_every_raw_topic() -> None:
    assert constant(parse(DAGS / "radar_load.py"), "SOURCES") == SOURCES
    assert constant(parse(DAGS / "radar_collect.py"), "RAW_SOURCES") == SOURCES


def test_load_dag_skip_code_matches_the_task_script() -> None:
    spec = importlib.util.spec_from_file_location("radar_load_task", ROOT / "airflow" / "include" / "radar_load.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert constant(parse(DAGS / "radar_load.py"), "SKIP_EXIT_CODE") == module.SKIP_EXIT_CODE


def test_collect_takes_no_manual_runs() -> None:
    # A manual run at a date the schedule has not reached blocks the scheduler (see the DAG).
    (call,) = dag_calls(parse(DAGS / "radar_collect.py"))
    assert keyword(call, "allowed_run_types") == ["scheduled", "backfill"]


def test_collect_triggers_load_for_the_same_date() -> None:
    text = (DAGS / "radar_collect.py").read_text(encoding="utf-8")
    assert 'trigger_dag_id="radar_load"' in text
    assert 'logical_date="{{ ds }}"' in text


def test_dbt_is_scheduled_by_the_asset_load_emits() -> None:
    load, dbt = parse(DAGS / "radar_load.py"), parse(DAGS / "radar_dbt.py")
    assert ast.dump(assigned(load, "RAW_VACANCIES")) == ast.dump(assigned(dbt, "RAW_VACANCIES"))
    (call,) = dag_calls(dbt)
    schedule = next(kw.value for kw in call.keywords if kw.arg == "schedule")
    assert ast.unparse(schedule) == "[RAW_VACANCIES]"
    assert "outlets=[RAW_VACANCIES]" in (DAGS / "radar_load.py").read_text(encoding="utf-8")
