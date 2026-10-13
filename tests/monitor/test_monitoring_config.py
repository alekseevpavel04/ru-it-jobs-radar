"""Static checks of Prometheus, its rules and the Grafana dashboards: they parse, point at compose
services, and every project metric they read is one the code exports or a rule records."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROMETHEUS = yaml.safe_load((ROOT / "infra" / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
RULES = yaml.safe_load((ROOT / "infra" / "prometheus" / "rules.yml").read_text(encoding="utf-8"))
COMPOSE = yaml.safe_load((ROOT / "infra" / "compose" / "docker-compose.yml").read_text(encoding="utf-8"))
DASHBOARDS = sorted((ROOT / "infra" / "grafana" / "dashboards").glob("*.json"))
DATASOURCES = yaml.safe_load(
    (ROOT / "infra" / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text(encoding="utf-8")
)
EXPORTING_CODE = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "radar" / "monitor").glob("*.py"))
SUFFIXES = ("_bucket", "_count", "_sum", "_total")


def rule_exprs() -> list[str]:
    return [r["expr"] for g in RULES["groups"] for r in g["rules"]]


def recorded() -> set[str]:
    return {r["record"] for g in RULES["groups"] for r in g["rules"] if "record" in r}


def panel_exprs(dashboard: dict[str, Any]) -> list[str]:
    return [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]


def test_scrape_targets_are_compose_services() -> None:
    services = set(COMPOSE["services"])
    for job in PROMETHEUS["scrape_configs"]:
        for target in (t for sc in job["static_configs"] for t in sc["targets"]):
            host, port = target.rsplit(":", 1)
            assert host == "localhost" or host in services, f"{job['job_name']}: {target}"
            command = " ".join(COMPOSE["services"].get(host, {}).get("command", []))
            if "--port" in command:
                assert f"--port {port}" in command, f"{job['job_name']}: {target} vs {command}"


def test_prometheus_mounts_its_rules() -> None:
    svc = COMPOSE["services"]["prometheus"]
    assert "../prometheus:/etc/prometheus:ro" in svc["volumes"]
    assert PROMETHEUS["rule_files"] == ["/etc/prometheus/rules.yml"]


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboard_reads_provisioned_prometheus(path: Path) -> None:
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    uid = DATASOURCES["datasources"][0]["uid"]
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids))
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == uid, panel["title"]
        assert panel["targets"], panel["title"]


def test_dashboard_uids_are_unique() -> None:
    uids = [json.loads(p.read_text(encoding="utf-8"))["uid"] for p in DASHBOARDS]
    assert len(uids) == len(set(uids)) == 3


def project_metrics(expr: str) -> set[str]:
    return set(re.findall(r"\bradar[_:][a-zA-Z0-9_:]+", expr))


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboards_read_metrics_the_code_exports(path: Path) -> None:
    exprs = panel_exprs(json.loads(path.read_text(encoding="utf-8")))
    check_metrics(exprs)


def test_rules_read_metrics_the_code_exports() -> None:
    check_metrics(rule_exprs())


def check_metrics(exprs: list[str]) -> None:
    for expr in exprs:
        for name in project_metrics(expr):
            if ":" in name:
                assert name in recorded(), f"{name} is not a recording rule"
                continue
            base = next((name.removesuffix(s) for s in SUFFIXES if name.endswith(s)), name)
            assert f'"{base}' in EXPORTING_CODE, f"{name} is not exported by src/radar/monitor"
