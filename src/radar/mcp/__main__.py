"""Run the MCP server, or call one of its tools as a client.

    python -m radar.mcp                                  serve over stdio (a desktop MCP client starts this)
    python -m radar.mcp --http [--host H] [--port P]     serve over streamable HTTP at /mcp
    python -m radar.mcp call TOOL '{"arg": 1}' [--url U] call a tool, print its JSON result; exit 1 on a tool
                                                         error. Without --url the server runs over stdio.

Settings from the environment and the repo-root .env: CLICKHOUSE_URL and the read-only account for
the marts, LLM_BASE_URL for the analyst, SALARY_MODEL_PATH for the salary model.
"""

import argparse
import json
import sys
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent

from radar.api.settings import salary_model_path
from radar.config import load_dotenv
from radar.mcp.server import build_server, live_backends

LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


def host_check(extra_hosts: list[str]) -> TransportSecuritySettings:
    """Host and Origin headers allowed on HTTP: loopback, plus the names other containers use. The SDK
    turns the check on by itself only when bound to loopback; in a container it binds 0.0.0.0."""
    hosts = [*LOOPBACK, *extra_hosts]
    return TransportSecuritySettings(
        allowed_hosts=[f"{h}:*" for h in hosts],
        allowed_origins=[f"http://{h}:*" for h in hosts],
    )


def serve(args: argparse.Namespace) -> int:
    server = build_server(live_backends(salary_model_path()))
    if args.http:
        server.run(
            "streamable-http",
            host=args.host,
            port=args.port,
            stateless_http=True,
            transport_security=host_check(args.allow_host),
        )
    else:
        server.run("stdio")
    return 0


async def call_tool(target: str | StdioServerParameters, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    async with Client(target, read_timeout_seconds=300) as client:
        return await client.call_tool(tool, arguments)


def result_payload(result: CallToolResult) -> Any:
    if result.structured_content is not None:
        return result.structured_content
    return [c.text for c in result.content if isinstance(c, TextContent)]


def call(args: argparse.Namespace) -> int:
    target: str | StdioServerParameters = args.url or StdioServerParameters(
        command=sys.executable, args=["-m", "radar.mcp"]
    )
    result = anyio.run(call_tool, target, args.tool, json.loads(args.arguments))
    print(json.dumps(result_payload(result), ensure_ascii=False, indent=2))
    return 1 if result.is_error else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m radar.mcp", description=__doc__.splitlines()[0])
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--allow-host", action="append", default=[], help="another Host name to accept over HTTP")
    sub = ap.add_subparsers(dest="command")
    c = sub.add_parser("call", help="call one tool as a client")
    c.add_argument("tool")
    c.add_argument("arguments", nargs="?", default="{}", help="tool arguments as a JSON object")
    c.add_argument("--url", help="streamable HTTP endpoint, e.g. http://127.0.0.1:8001/mcp")
    args = ap.parse_args(argv)
    return call(args) if args.command == "call" else serve(args)


if __name__ == "__main__":
    sys.exit(main())
