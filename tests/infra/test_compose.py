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


def is_oneshot(name: str) -> bool:
    return name.endswith("-init")


def profile_services(profile: str) -> set[str]:
    return {name for name, svc in SERVICES.items() if profile in svc.get("profiles", [])}


def test_every_service_belongs_to_one_spec_profile() -> None:
    for name, svc in SERVICES.items():
        profiles = svc.get("profiles", [])
        assert len(profiles) == 1 and profiles[0] in EXPECTED_PROFILES, f"{name}: {profiles}"


def test_core_and_ml_profiles_hold_spec_services() -> None:
    assert {"clickhouse", "redpanda", "console", "redis"} <= profile_services("core")
    assert "mlflow" in profile_services("ml")


def test_llama_server_stays_outside_compose() -> None:
    for name, svc in SERVICES.items():
        assert "llama" not in name and "llama" not in svc.get("image", ""), name


def assert_pinned(image: str, where: str) -> None:
    repo, _, tag = image.rpartition(":")
    assert repo and "/" not in tag, f"{where}: {image} has no tag"
    assert tag != "latest" and re.search(r"\d", tag), f"{where}: {image} is not a version tag"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_image_has_exact_version_tag(name: str) -> None:
    svc = SERVICES[name]
    assert_pinned(svc["image"], name)
    if "build" in svc:
        dockerfile = (COMPOSE_DIR / svc["build"] / "Dockerfile").read_text(encoding="utf-8")
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
        source = volume.split(":", 1)[0]
        if source.startswith((".", "/")):
            assert volume.endswith(":ro"), f"{name}: bind mount {volume} must be read-only"
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
        assert rest.startswith(":?"), f"${{{var}{rest}}} must fail on an empty value"
        assert var in documented, f"{var} is missing from .env.example"


def test_topic_init_creates_spec_topics() -> None:
    script = (COMPOSE_DIR / "redpanda" / "create-topics.sh").read_text(encoding="utf-8")
    loop = re.search(r"^for topic in (.+); do$", script, flags=re.MULTILINE)
    assert loop is not None
    assert set(loop.group(1).split()) == RAW_TOPICS


def test_clickhouse_init_creates_layers_and_select_only_grant() -> None:
    script = (ROOT / "infra" / "clickhouse" / "init.sh").read_text(encoding="utf-8")
    created = set(re.findall(r"CREATE DATABASE IF NOT EXISTS (\w+);", script))
    assert created == {"raw", "staging", "marts"}
    grants = re.findall(r"^GRANT .+$", script, flags=re.MULTILINE)
    assert grants == ["GRANT SELECT ON marts.* TO ${CLICKHOUSE_RO_USER};"]
    assert "readonly = 2 CONST" in script
