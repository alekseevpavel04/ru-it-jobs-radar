"""Static checks of the compose stack. No Docker needed, so they run in scripts/check.sh and CI."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "infra" / "compose"
COMPOSE_TEXT = (COMPOSE_DIR / "docker-compose.yml").read_text(encoding="utf-8")
COMPOSE: dict[str, Any] = yaml.safe_load(COMPOSE_TEXT)
SERVICES: dict[str, dict[str, Any]] = COMPOSE["services"]

EXPECTED_PROFILES = {"core", "ml", "orchestration", "bi", "mlops", "api"}
RAW_TOPICS = {"raw.trudvsem", "raw.trudvsem_snapshot", "raw.habr", "raw.telegram", "raw.superjob"}

LIB = (COMPOSE_DIR / "lib.sh").read_text(encoding="utf-8")
# Set by lib.sh in front of every compose call (`NAME=${NAME:-default} docker compose`), not by .env.
FROM_LIB = set(re.findall(r"\b(\w+)=\$\{\1:-", LIB))
# Secrets of one profile: interpolated with :- and checked by scripts/up.sh for that profile only.
PROFILE_SECRETS = set(" ".join(re.findall(r'^  \[\w+\]="(.+)"$', LIB, flags=re.MULTILINE)).split())
# Bind mounts a service writes to, with the reason.
WRITABLE_BINDS = {
    # The sync from the collection server adds partitions to the raw layer.
    ("airflow-scheduler", "../../data/raw"),
    # The weekly drift report (radar_drift): its summary and the local Evidently HTML.
    ("airflow-scheduler", "../../results/monitor"),
    ("airflow-scheduler", "../../reports/evidently"),
    # The weekly digest writes its pages for the site build.
    ("n8n", "../../site/src/digest"),
}


def is_oneshot(name: str) -> bool:
    # *-init runs after the stack is healthy; *-migrate before the services that depend on it.
    return name.endswith(("-init", "-migrate"))


def profile_services(profile: str) -> set[str]:
    return {name for name, svc in SERVICES.items() if profile in svc.get("profiles", [])}


def test_every_service_belongs_to_one_spec_profile() -> None:
    for name, svc in SERVICES.items():
        profiles = svc.get("profiles", [])
        assert len(profiles) == 1 and profiles[0] in EXPECTED_PROFILES, f"{name}: {profiles}"


def test_core_and_ml_profiles_hold_spec_services() -> None:
    assert {"clickhouse", "redpanda", "console", "redis"} <= profile_services("core")
    assert "mlflow" in profile_services("ml")


def test_orchestration_profile_holds_airflow() -> None:
    assert profile_services("orchestration") == {
        "airflow-db",
        "airflow-migrate",
        "airflow-api-server",
        "airflow-scheduler",
        "airflow-dag-processor",
        "statsd-exporter",
    }


def test_host_ports_are_unique_and_leave_llama_server_its_port() -> None:
    host_ports = [p.split(":")[1] for svc in SERVICES.values() for p in svc.get("ports", [])]
    assert len(host_ports) == len(set(host_ports)), host_ports
    assert "8080" not in host_ports


def test_gpu_llama_server_stays_outside_compose() -> None:
    # The teachers' and the scoring llama-server run on Windows with the GPU (scripts/llm_server.sh).
    # The one llama.cpp server in compose is the student behind /extract: the CPU build, no device.
    llama = {name for name, svc in SERVICES.items() if "llama" in name or "llama" in svc.get("image", "")}
    assert llama <= {"student"}
    for name in llama:
        svc = SERVICES[name]
        assert svc["profiles"] == ["api"] and ":server-b" in svc["image"], name
        assert not {"deploy", "devices", "gpus", "runtime"} & set(svc), name


def assert_pinned(image: str, where: str) -> None:
    repo, _, tag = image.rpartition(":")
    assert repo and "/" not in tag, f"{where}: {image} has no tag"
    assert tag != "latest" and re.search(r"\d", tag), f"{where}: {image} is not a version tag"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_image_has_exact_version_tag(name: str) -> None:
    svc = SERVICES[name]
    assert_pinned(svc["image"], name)
    if "build" in svc:
        build = svc["build"] if isinstance(svc["build"], dict) else {"context": svc["build"]}
        dockerfile_path = COMPOSE_DIR / build["context"] / build.get("dockerfile", "Dockerfile")
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        for base in re.findall(r"^FROM\s+(\S+)", dockerfile, flags=re.MULTILINE):
            assert_pinned(base, f"{name} Dockerfile")


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_long_running_services_have_healthcheck(name: str) -> None:
    svc = SERVICES[name]
    if is_oneshot(name):
        # scripts/up.sh runs *-init services after the rest is healthy and checks their exit code.
        assert svc["restart"] == "no"
        assert "healthcheck" not in svc
    else:
        assert svc["restart"] == "unless-stopped"
        assert svc["healthcheck"]["test"], name


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_ports_listen_on_loopback_only(name: str) -> None:
    for port in SERVICES[name].get("ports", []):
        assert isinstance(port, str) and port.startswith("127.0.0.1:"), f"{name}: {port}"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_volumes_are_named_or_readonly_config(name: str) -> None:
    declared = set(COMPOSE.get("volumes") or {})
    for volume in SERVICES[name].get("volumes", []):
        source = volume.split(":", 1)[0] if not volume.startswith("${") else volume.split("}", 1)[0] + "}"
        if source.startswith((".", "/", "${")):
            writable = (name, source) in WRITABLE_BINDS
            assert writable or volume.endswith(":ro"), f"{name}: bind mount {volume} must be read-only"
        else:
            assert source in declared, f"{name}: volume {source} is not declared"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_memory_is_capped(name: str) -> None:
    assert SERVICES[name].get("mem_limit"), name


def test_variables_are_required_and_documented() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example, flags=re.MULTILINE))
    references = re.findall(r"\$\{(\w+)([^}]*)\}", COMPOSE_TEXT)
    assert references
    for var, rest in references:
        if var in PROFILE_SECRETS:
            assert rest == ":-", f"${{{var}{rest}}}: profile secrets are checked by scripts/up.sh"
        else:
            assert rest.startswith(":?"), f"${{{var}{rest}}} must fail on an empty value"
        assert var in documented or var in FROM_LIB, f"{var} is missing from .env.example"
    assert documented >= PROFILE_SECRETS


def test_topic_init_creates_spec_topics() -> None:
    script = (COMPOSE_DIR / "redpanda" / "create-topics.sh").read_text(encoding="utf-8")
    loop = re.search(r"^for topic in (.+); do$", script, flags=re.MULTILINE)
    assert loop is not None
    assert set(loop.group(1).split()) == RAW_TOPICS


def test_clickhouse_init_creates_layers_and_applies_the_agent_account_file() -> None:
    init_dir = ROOT / "infra" / "clickhouse"
    script = (init_dir / "init.sh").read_text(encoding="utf-8")
    created = set(re.findall(r"CREATE DATABASE IF NOT EXISTS (\w+);", script))
    assert created == {"raw", "staging", "marts"}
    # One definition of the read-only account: init.sh must not keep a second, diverging copy.
    account_sql = r"^\s*(GRANT|REVOKE|CREATE SETTINGS PROFILE|CREATE USER|ALTER USER)\b"
    assert not re.findall(account_sql, script, flags=re.MULTILINE)
    assert '"$here/agent/readonly_user.sql" | client' in script
    assert "../clickhouse:/init:ro" in SERVICES["clickhouse-init"]["volumes"]

    # Render the file the way the sed call of init.sh does and check what reaches the server.
    substitutions = re.findall(r'-e "s\|(\{ro_\w+\})\|\$\{(\w+)\}\|g"', script)
    sql = (init_dir / "agent" / "readonly_user.sql").read_text(encoding="utf-8")
    assert {placeholder for placeholder, _ in substitutions} == set(re.findall(r"\{ro_\w+\}", sql))
    values = {"CLICKHOUSE_RO_USER": "agent_ro", "CLICKHOUSE_RO_PASSWORD": "s3cret/pass="}
    for placeholder, var in substitutions:
        sql = sql.replace(placeholder, values[var])
    grants = re.findall(r"^GRANT .+$", sql, flags=re.MULTILINE)
    assert grants == ["GRANT SELECT ON marts.* TO agent_ro;", "GRANT SELECT ON agent_snapshot.* TO agent_ro;"]
    assert "REVOKE ALL ON *.* FROM agent_ro;" in sql
    assert re.search(r"^\s+readonly = 1,$", sql, flags=re.MULTILINE)
