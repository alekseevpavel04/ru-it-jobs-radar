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

ACTIVE = {"radar_collect", "radar_load", "radar_dbt"}
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


def constant(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


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


def test_load_dag_covers_every_raw_topic() -> None:
    assert constant(parse(DAGS / "radar_load.py"), "SOURCES") == SOURCES


def test_load_dag_skip_code_matches_the_task_script() -> None:
    spec = importlib.util.spec_from_file_location("radar_load_task", ROOT / "airflow" / "include" / "radar_load.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert constant(parse(DAGS / "radar_load.py"), "SKIP_EXIT_CODE") == module.SKIP_EXIT_CODE


def test_chain_triggers_the_next_dag_for_the_same_date() -> None:
    for path, target in ((DAGS / "radar_collect.py", "radar_load"), (DAGS / "radar_load.py", "radar_dbt")):
        text = path.read_text(encoding="utf-8")
        assert f'trigger_dag_id="{target}"' in text
        assert 'logical_date="{{ ds }}"' in text
