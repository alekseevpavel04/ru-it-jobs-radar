"""Live checks of the stack: scripts/up.sh brings up core and ml, then each service is called.

Needs Docker (inside the WSL distro when run from Windows), so it is marked slow and stays out of
scripts/check.sh: `pytest -m slow tests/infra`. Variables come from $RADAR_ENV_FILE or the repo-root
.env. The stack is left running.
"""

import os
import re
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(os.environ.get("RADAR_ENV_FILE") or ROOT / ".env")
ON_WINDOWS = os.name == "nt"
RAW_TOPICS = {"raw.trudvsem", "raw.trudvsem_snapshot", "raw.habr", "raw.telegram", "raw.superjob"}
REFUSED = re.compile(r"Code: (164|497)\.")  # READONLY, ACCESS_DENIED


def linux(*cmd: str) -> list[str]:
    if ON_WINDOWS:
        return ["wsl.exe", "-d", os.environ.get("RADAR_WSL_DISTRO", "radar"), "--exec", *cmd]
    return list(cmd)


def run(*cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        linux(*cmd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False
    )


def linux_path(path: Path) -> str:
    return run("wslpath", "-u", path.as_posix()).stdout.strip() if ON_WINDOWS else str(path)


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    if not ENV_FILE.exists():
        pytest.fail(f"{ENV_FILE} not found: copy .env.example to .env or point RADAR_ENV_FILE at an env file")
    return read_env(ENV_FILE)


@pytest.fixture(scope="module")
def stack(env: dict[str, str]) -> None:
    result = run(
        "env", f"RADAR_ENV_FILE={linux_path(ENV_FILE)}", "bash", f"{linux_path(ROOT)}/scripts/up.sh", "core", "ml"
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def clickhouse(env: dict[str, str], query: str, *, readonly: bool = False, params: str = "") -> httpx.Response:
    prefix = "CLICKHOUSE_RO_" if readonly else "CLICKHOUSE_"
    return httpx.post(
        env["CLICKHOUSE_URL"] + params,
        content=query,
        headers={"X-ClickHouse-User": env[prefix + "USER"], "X-ClickHouse-Key": env[prefix + "PASSWORD"]},
        timeout=30,
    )


@pytest.fixture
def probe_tables(env: dict[str, str], stack: None) -> Iterator[None]:
    for db in ("marts", "raw"):
        clickhouse(env, f"CREATE TABLE IF NOT EXISTS {db}._ro_probe (x UInt8) ENGINE = Memory").raise_for_status()
    yield
    for db in ("marts", "raw"):
        clickhouse(env, f"DROP TABLE IF EXISTS {db}._ro_probe").raise_for_status()


def test_all_containers_healthy(stack: None) -> None:
    result = run(
        "docker", "ps", "--filter", "label=com.docker.compose.project=radar", "--format", "{{.Names}} {{.Status}}"
    )
    lines = result.stdout.strip().splitlines()
    assert len(lines) >= 6, result.stdout
    for line in lines:
        assert "(healthy)" in line, line


def test_clickhouse_answers_and_has_layers(env: dict[str, str], stack: None) -> None:
    assert clickhouse(env, "SELECT 1").text == "1\n"
    databases = set(clickhouse(env, "SHOW DATABASES").text.split())
    assert {"raw", "staging", "marts"} <= databases


def test_readonly_user_is_refused_by_server(env: dict[str, str], probe_tables: None) -> None:
    assert clickhouse(env, "SELECT count() FROM marts._ro_probe", readonly=True).text == "0\n"

    attempts = {
        "create": clickhouse(env, "CREATE TABLE marts._ro_new (x UInt8) ENGINE = Memory", readonly=True),
        "insert": clickhouse(env, "INSERT INTO marts._ro_probe VALUES (1)", readonly=True),
        "drop": clickhouse(env, "DROP TABLE marts._ro_probe", readonly=True),
        "select outside marts": clickhouse(env, "SELECT count() FROM raw._ro_probe", readonly=True),
        "lift readonly": clickhouse(env, "SELECT 1", readonly=True, params="?readonly=0"),
    }
    for what, response in attempts.items():
        assert response.status_code != 200 and REFUSED.search(response.text), f"{what}: {response.text[:300]}"
    assert clickhouse(env, "SELECT count() FROM marts._ro_probe").text == "0\n"
    assert clickhouse(env, "EXISTS TABLE marts._ro_new").text == "0\n"


def test_redpanda_has_raw_topics(stack: None) -> None:
    result = run("docker", "exec", "radar-redpanda-1", "rpk", "topic", "list")
    assert result.returncode == 0, result.stderr
    topics = {line.split()[0] for line in result.stdout.splitlines()[1:] if line.strip()}
    assert topics >= RAW_TOPICS


def test_redis_answers_ping(env: dict[str, str], stack: None) -> None:
    url = urlsplit(env["REDIS_URL"])
    with socket.create_connection((url.hostname or "localhost", url.port or 6379), timeout=5) as conn:
        conn.sendall(b"PING\r\n")
        assert conn.recv(64) == b"+PONG\r\n"


def test_mlflow_healthy_with_database_backend(env: dict[str, str], stack: None) -> None:
    base = env["MLFLOW_TRACKING_URI"].rstrip("/")
    health = httpx.get(f"{base}/health", timeout=10)
    assert health.status_code == 200 and health.text == "OK"
    search = httpx.post(f"{base}/api/2.0/mlflow/experiments/search", json={"max_results": 1}, timeout=10)
    assert search.status_code == 200, search.text


def test_empty_required_variable_stops_compose(tmp_path: Path) -> None:
    compose_file = ROOT / "infra" / "compose" / "docker-compose.yml"
    names = re.findall(r"\$\{(\w+):\?", compose_file.read_text(encoding="utf-8"))
    env_file = tmp_path / "missing.env"
    env_file.write_text("".join(f"{n}={'' if n == 'CLICKHOUSE_PASSWORD' else 'x'}\n" for n in names), encoding="utf-8")
    result = run(
        "docker", "compose", "-f", linux_path(compose_file), "--env-file", linux_path(env_file), "config", "--quiet"
    )
    assert result.returncode != 0
    assert "required variable CLICKHOUSE_PASSWORD is missing a value" in result.stderr
