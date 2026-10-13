"""The Helm chart: values that every container needs, entry points that exist, and `helm lint` plus
`helm template` when helm is on PATH (CI installs it; locally it lives in the WSL distro)."""

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "helm" / "radar"
VALUES: dict[str, Any] = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
TEMPLATES = "\n".join(p.read_text(encoding="utf-8") for p in sorted((CHART / "templates").iterdir()))
COMPONENTS = ("api", "mcp", "exporter")
HELM = shutil.which("helm")


def test_image_is_the_api_image_with_a_version_tag() -> None:
    compose = yaml.safe_load((ROOT / "infra" / "compose" / "docker-compose.yml").read_text(encoding="utf-8"))
    assert f"{VALUES['image']['repository']}:{VALUES['image']['tag']}" == compose["services"]["api"]["image"]


@pytest.mark.parametrize("component", COMPONENTS)
def test_every_component_has_probes_and_a_memory_limit(component: str) -> None:
    values = VALUES[component]
    assert values["resources"]["limits"]["memory"]
    assert values["resources"]["requests"]["cpu"]
    assert values["readinessProbe"] and values["livenessProbe"]


def test_entry_points_of_the_containers_exist() -> None:
    modules = set(re.findall(r'"-m", "([\w.]+)"', TEMPLATES)) | set(re.findall(r"- -m\n\s+- ([\w.]+)", TEMPLATES))
    assert modules == {"radar.api", "radar.mcp", "radar.monitor.exporter", "radar.monitor.fetch_model"}
    for module in modules:
        assert importlib.util.find_spec(module) is not None, module


def test_secret_keys_are_the_ones_the_smoke_script_creates() -> None:
    keys = set(re.findall(r"key: (CLICKHOUSE_\w+)", TEMPLATES))
    smoke = (ROOT / "scripts" / "k8s_smoke.sh").read_text(encoding="utf-8")
    created = re.search(r"--from-env-file=<\(\n\s+for name in ([\w ]+); do", smoke)
    assert created is not None
    assert keys == set(created.group(1).split())


def test_smoke_script_pins_its_cluster() -> None:
    smoke = (ROOT / "scripts" / "k8s_smoke.sh").read_text(encoding="utf-8")
    assert re.search(r"^CONTEXT=kind-radar$", smoke, flags=re.MULTILINE)
    calls = re.findall(r"^\s*(?:kubectl|helm) .*$", smoke, flags=re.MULTILINE)
    assert len(calls) >= 3
    for line in calls:
        assert '--context "$CONTEXT"' in line or '--kube-context "$CONTEXT"' in line, line


def helm(*args: str) -> str:
    assert HELM is not None
    result = subprocess.run([HELM, *args], capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


@pytest.mark.skipif(HELM is None, reason="helm is not on PATH")
def test_helm_lint_strict() -> None:
    helm("lint", "--strict", str(CHART))


@pytest.mark.skipif(HELM is None, reason="helm is not on PATH")
@pytest.mark.parametrize("flags", [[], ["--set", "mcp.enabled=false,exporter.enabled=false,salaryModel.enabled=false"]])
def test_helm_template_renders_what_values_enable(flags: list[str]) -> None:
    docs = [d for d in yaml.safe_load_all(helm("template", "radar", str(CHART), *flags)) if d]
    kinds = sorted((d["kind"], d["metadata"]["name"]) for d in docs)
    if flags:
        assert kinds == [("Deployment", "radar-api"), ("Service", "radar-api")]
        (api,) = [d for d in docs if d["kind"] == "Deployment"]
        assert "initContainers" not in api["spec"]["template"]["spec"]
    else:
        assert len(kinds) == 6
        for d in (d for d in docs if d["kind"] == "Deployment"):
            for c in d["spec"]["template"]["spec"]["containers"]:
                assert c["securityContext"]["readOnlyRootFilesystem"] is True
