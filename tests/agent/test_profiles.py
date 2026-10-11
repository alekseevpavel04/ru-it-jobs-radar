"""Live marts and the frozen snapshot are described and checked each by their own schema."""

import json
from pathlib import Path

import pytest
import yaml

from radar.agent import prompts
from radar.agent.evaluate import prompt_sha256
from radar.agent.graph import Agent
from radar.agent.profiles import frozen, live
from radar.agent.sqlcheck import TABLES, Blocked, check
from radar.config import RESULTS

NEW_MARTS = {"mart_lifetime", "mart_skills_weekly"}


def _columns(path: Path, model: str) -> set[str]:
    models = {m["name"]: m for m in yaml.safe_load(path.read_text(encoding="utf-8"))["models"]}
    return {c["name"] for c in models[model]["columns"]}


def test_scored_prompt_is_the_one_of_the_dev_run() -> None:
    dev = json.loads((RESULTS / "agent" / "dev.json").read_text(encoding="utf-8"))
    assert prompt_sha256(frozen()) == dev["prompt_sha256"]


def test_snapshot_keeps_its_tables_and_schema() -> None:
    snap = frozen()
    assert len(snap.tables) == 6 and snap.tables < TABLES and not snap.tables & NEW_MARTS
    assert "level" not in _columns(snap.yml, "mart_salaries")
    described = {m["name"] for m in yaml.safe_load(snap.yml.read_text(encoding="utf-8"))["models"]}
    assert snap.tables <= described


def test_live_profile_has_the_new_marts_and_salary_levels() -> None:
    profile = live()
    assert profile.tables == TABLES and profile.tables >= NEW_MARTS
    assert "level" in _columns(profile.yml, "mart_salaries")
    for marker in ("level 'cell'", "mart_lifetime", "mart_skills_weekly", "telegram_channels_week"):
        assert marker in prompts.SQL_SYSTEM_LIVE and marker not in prompts.SQL_SYSTEM_SNAPSHOT


def test_new_marts_pass_live_check_but_not_snapshot_check() -> None:
    sql = "SELECT duration_days_median FROM marts.mart_lifetime WHERE level = 'source'"
    assert check(sql).tables == ("mart_lifetime",)
    with pytest.raises(Blocked):
        check(sql, database="agent_snapshot", tables=frozen().tables)


def test_agent_on_the_snapshot_blocks_a_table_the_snapshot_lacks() -> None:
    queries: list[str] = []

    def run(sql: str) -> object:
        queries.append(sql)
        raise AssertionError("must not run")

    snap = frozen()
    agent = Agent(
        lambda system, user, max_tokens: "SELECT skill FROM marts.mart_skills_weekly",
        run,  # type: ignore[arg-type]
        "schema",
        snap.database,
        False,
        snap.tables,
    )
    assert agent.ask("Навыки по неделям").status == "blocked" and queries == []
