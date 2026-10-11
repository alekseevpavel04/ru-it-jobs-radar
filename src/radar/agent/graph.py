"""The analyst agent: question in Russian, SQL over the marts, answer with the numbers and the SQL.

LangGraph graph: generate SQL -> check (radar.agent.sqlcheck) -> execute as the read-only account ->
answer. A query that fails to parse or to run goes back to generation once with the error; a query
the check or the server refuses is never repaired, the agent refuses. The answer step has no path
back to execution, so text inside the query result cannot make the agent run anything.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from radar.agent import prompts
from radar.agent.clickhouse import SERVER_REFUSALS, QueryError, Result
from radar.agent.llm import Chat
from radar.agent.sqlcheck import DATABASE, Blocked, Unparsable, check

MAX_ATTEMPTS = 2  # the first query and one repair
ANSWER_ROWS = 30
_FENCED = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_REFUSE = re.compile(r"^\s*REFUSE\s*:?\s*(.*)$", re.DOTALL | re.IGNORECASE)

Status = Literal["answered", "refused", "blocked", "failed"]


class State(TypedDict, total=False):
    question: str
    attempts: int
    sql: str
    checked_sql: str
    error: str
    status: Status
    reason: str
    result: Result
    answer: str
    trace: list[dict[str, str]]


@dataclass
class Answer:
    question: str
    status: Status
    answer: str
    sql: str | None
    result: Result | None
    attempts: int
    error: str | None = None
    trace: list[dict[str, str]] = field(default_factory=list)


def extract_sql(text: str) -> str:
    fenced = _FENCED.search(text)
    return (fenced.group(1) if fenced else text).strip()


def table_text(result: Result, limit: int = ANSWER_ROWS) -> str:
    lines = ["\t".join(result.columns)]
    lines += ["\t".join("NULL" if v is None else str(v) for v in row) for row in result.rows[:limit]]
    return "\n".join(lines)


class Agent:
    def __init__(
        self,
        chat: Chat,
        run_sql: Callable[[str], Result],
        system_prompt: str,
        database: str = DATABASE,
        compose_answer: bool = True,
    ) -> None:
        self.chat = chat
        self.run_sql = run_sql
        self.system_prompt = system_prompt
        self.database = database
        self.compose_answer = compose_answer
        self.graph = self._build()

    def _build(self) -> Any:
        g = StateGraph(State)
        g.add_node("generate", self._generate)
        g.add_node("check", self._check)
        g.add_node("execute", self._execute)
        g.add_node("answer", self._answer)
        g.add_edge(START, "generate")
        g.add_conditional_edges("generate", self._after_generate, ["check", "answer"])
        g.add_conditional_edges("check", self._after_check, ["execute", "generate", "answer"])
        g.add_conditional_edges("execute", self._after_execute, ["generate", "answer"])
        g.add_edge("answer", END)
        return g.compile()

    def _generate(self, state: State) -> State:
        attempts = state.get("attempts", 0) + 1
        user = state["question"]
        if state.get("error"):
            user += "\n\n" + prompts.SQL_REPAIR.format(sql=state.get("sql", ""), error=state["error"][:800])
        text = self.chat(self.system_prompt, user, max_tokens=700)
        trace = [*state.get("trace", []), {"step": "generate", "output": text}]
        refused = _REFUSE.match(text.strip())
        if refused:
            return {"attempts": attempts, "status": "refused", "reason": refused.group(1).strip(), "trace": trace}
        return {"attempts": attempts, "sql": extract_sql(text), "checked_sql": "", "error": "", "trace": trace}

    def _check(self, state: State) -> State:
        try:
            checked = check(state["sql"], database=self.database)
        except Blocked as e:
            return {
                "status": "blocked",
                "reason": str(e),
                "trace": [*state["trace"], {"step": "check", "blocked": str(e)}],
            }
        except Unparsable as e:
            return {"error": f"syntax: {e}", "trace": [*state["trace"], {"step": "check", "error": str(e)}]}
        return {"checked_sql": checked.sql}

    def _execute(self, state: State) -> State:
        try:
            result = self.run_sql(state["checked_sql"])
        except QueryError as e:
            entry = {"step": "execute", "error": str(e)[:500]}
            if e.code in SERVER_REFUSALS:
                return {
                    "status": "blocked",
                    "reason": f"ClickHouse refused: {str(e)[:200]}",
                    "trace": [*state["trace"], entry],
                }
            return {"error": str(e), "trace": [*state["trace"], entry]}
        return {"result": result, "status": "answered"}

    def _after_generate(self, state: State) -> str:
        return "answer" if state.get("status") == "refused" else "check"

    def _retry_or_give_up(self, state: State) -> str:
        return "generate" if state["attempts"] < MAX_ATTEMPTS else "answer"

    def _after_check(self, state: State) -> str:
        if state.get("status") == "blocked":
            return "answer"
        return self._retry_or_give_up(state) if state.get("error") else "execute"

    def _after_execute(self, state: State) -> str:
        if state.get("status") in ("blocked", "answered"):
            return "answer"
        return self._retry_or_give_up(state)

    def _answer(self, state: State) -> State:
        status = state.get("status")
        if status in ("refused", "blocked"):
            return {"answer": prompts.REFUSAL.format(reason=state.get("reason", "")).strip()}
        if status != "answered":
            return {"status": "failed", "answer": prompts.FAILURE.format(error=state.get("error", "")[:300])}
        result = state["result"]
        table = table_text(result)
        if not self.compose_answer:
            return {"answer": table}
        user = prompts.ANSWER_USER.format(
            question=state["question"],
            sql=state["checked_sql"],
            shown=min(len(result.rows), ANSWER_ROWS),
            total=len(result.rows),
            table=table,
        )
        return {"answer": self.chat(prompts.ANSWER_SYSTEM, user, max_tokens=400).strip()}

    def ask(self, question: str) -> Answer:
        state: State = self.graph.invoke({"question": question, "attempts": 0, "trace": []})
        return Answer(
            question=question,
            status=state["status"],
            answer=state["answer"],
            sql=state.get("checked_sql") or state.get("sql"),
            result=state.get("result"),
            attempts=state["attempts"],
            error=state.get("error") or None,
            trace=state.get("trace", []),
        )
