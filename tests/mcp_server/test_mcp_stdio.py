"""`python -m radar.mcp` as a desktop client runs it: a subprocess spoken to over stdio by the SDK's
client. The salary model is the stub of conftest; no tool that reaches the network is called."""

import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters


def test_stdio_server_lists_its_tools_and_estimates(model_file: Path, expected: dict[str, Any]) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "radar.mcp"],
        env={"SALARY_MODEL_PATH": str(model_file)},
    )

    async def session() -> tuple[set[str], Any]:
        async with Client(params, read_timeout_seconds=120) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("estimate_salary", {"role": "backend", "grade": "middle"})
            return tools, result

    tools, result = anyio.run(session)
    assert tools == {"ask_market", "estimate_salary", "find_vacancies"}
    assert not result.is_error
    assert result.structured_content == {
        **expected,
        "n_similar": 0,
        "source": "trudvsem",
        "currency": "RUB",
        "period": "month",
        "basis": "as_stated",
        "model": "salary-stub",
    }
