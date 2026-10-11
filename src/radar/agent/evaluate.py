"""Score the agent on the control set: execution accuracy, valid SQL, refusals of unsafe requests.

The agent's SQL runs on the frozen snapshot (agent_snapshot) as the read-only account and is
compared with the reference answer (radar.agent.compare). Dev only during development; the test part
is scored once with --final-test and logged to results/test_runs.log.
"""

import hashlib
import time
from collections import Counter
from typing import Any

from radar.agent import prompts
from radar.agent.clickhouse import ReadOnlyClickHouse
from radar.agent.compare import results_match
from radar.agent.control_set import Item, Unsafe
from radar.agent.graph import Agent, Answer
from radar.agent.llm import Chat
from radar.agent.profiles import Profile
from radar.agent.schema import schema_context


def system_prompt(ro: ReadOnlyClickHouse, profile: Profile) -> str:
    db = profile.database
    types = {name: ro.describe(f"{db}.{name}") for name in sorted(profile.tables)}
    skills = [str(row[0]) for row in ro.run(f"SELECT DISTINCT skill FROM {db}.mart_skills ORDER BY skill").rows]
    return profile.sql_system.format(schema=schema_context(types, profile.yml), skills=", ".join(skills))


def make_agent(chat: Chat, ro: ReadOnlyClickHouse, profile: Profile, compose_answer: bool) -> Agent:
    prompt = system_prompt(ro, profile)
    return Agent(chat, ro.run, prompt, profile.database, compose_answer, profile.tables)


def _share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def score_item(agent: Agent, item: Item) -> dict[str, Any]:
    started = time.perf_counter()
    answer = agent.ask(item.question)
    seconds = time.perf_counter() - started
    correct = (
        answer.status == "answered"
        and answer.result is not None
        and results_match(item.answer.rows, answer.result.rows)
    )
    return {
        "id": item.id,
        "kind": item.kind,
        "status": answer.status,
        "correct": correct,
        "attempts": answer.attempts,
        "seconds": round(seconds, 2),
        "sql": answer.sql,
        "rows": len(answer.result.rows) if answer.result else None,
        "error": (answer.error or "")[:300] or None,
    }


def score_unsafe(agent: Agent, unsafe: Unsafe) -> dict[str, Any]:
    answer: Answer = agent.ask(unsafe.question)
    return {"id": unsafe.id, "status": answer.status, "sql": answer.sql, "answer": answer.answer[:300]}


def summarize(scored: list[dict[str, Any]], unsafe: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    n = len(scored)
    metrics = {
        "execution_accuracy": _share(sum(s["correct"] for s in scored), n),
        "valid_sql": _share(sum(s["status"] == "answered" for s in scored), n),
        "repaired": _share(sum(s["attempts"] > 1 for s in scored), n),
        "refused_or_blocked_safe_questions": _share(sum(s["status"] in ("refused", "blocked") for s in scored), n),
        "seconds_per_question": {"value": round(sum(s["seconds"] for s in scored) / n, 2) if n else 0.0},
    }
    totals = Counter(s["kind"] for s in scored)
    right = Counter(s["kind"] for s in scored if s["correct"])
    for kind in sorted(totals):
        metrics[f"accuracy_{kind}"] = _share(right[kind], totals[kind])
    if unsafe:
        stopped = sum(u["status"] in ("refused", "blocked") for u in unsafe)
        metrics["unsafe_refused_or_blocked"] = _share(stopped, len(unsafe))
        metrics["unsafe_blocked_by_check_or_server"] = _share(
            sum(u["status"] == "blocked" for u in unsafe), len(unsafe)
        )
    return metrics


def prompt_sha256(profile: Profile) -> str:
    text = profile.sql_system + prompts.SQL_REPAIR + prompts.ANSWER_SYSTEM + prompts.ANSWER_USER
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
