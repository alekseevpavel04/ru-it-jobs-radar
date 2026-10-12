import importlib.util
import json
from pathlib import Path

import pytest

from radar import config, results

spec = importlib.util.spec_from_file_location("verify_numbers", config.ROOT / "scripts" / "verify_numbers.py")
assert spec is not None and spec.loader is not None
verify_numbers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_numbers)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verify_numbers, "ROOT", tmp_path)
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    results.write_result(
        "extract/dev.json",
        {"f1": {"value": 0.8123}, "valid_json": {"value": 0.97, "n": 194, "denominator": 200}},
        source_script="tests",
    )
    return tmp_path


def run(repo: Path, readme: str, *flags: str) -> int:
    (repo / "README.md").write_text(readme, encoding="utf-8")
    return int(verify_numbers.main(["README.md", "--results", str(repo / "results"), *flags]))


def test_matching_numbers_pass(repo: Path) -> None:
    readme = (
        "F1 <!--n:extract/dev.json:metrics.f1.value:.3f|ru-->0,812<!--/n-->, "
        "JSON <!--n:extract/dev.json:metrics.valid_json.n:d-->194<!--/n--> из "
        "<!--n:extract/dev.json:metrics.valid_json.denominator:d-->200<!--/n--> "
        "(<!--n:extract/dev.json:metrics.valid_json.value:.0f|pct-->97<!--/n-->%)"
    )
    assert run(repo, readme) == 0


def test_mismatch_fails(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(repo, "F1 <!--n:extract/dev.json:metrics.f1.value:.3f|ru-->0,815<!--/n-->") == 1
    assert "0,812" in capsys.readouterr().err


def test_missing_file_and_key_fail(repo: Path) -> None:
    assert run(repo, "<!--n:nope.json:metrics.f1.value:.3f-->1<!--/n-->") == 1
    assert run(repo, "<!--n:extract/dev.json:metrics.nope.value:.3f-->1<!--/n-->") == 1


def test_write_fixes_text(repo: Path) -> None:
    assert run(repo, "<!--n:extract/dev.json:metrics.f1.value:.2f-->?<!--/n-->", "--write") == 0
    assert "-->0.81<!--/n-->" in (repo / "README.md").read_text(encoding="utf-8")
    assert run(repo, (repo / "README.md").read_text(encoding="utf-8")) == 0


def test_result_convention_enforced(repo: Path) -> None:
    bad = {"source_script": "x", "command": "x", "commit": "x", "created_at": "x", "metrics": {"a": {"n": 1}}}
    (repo / "results" / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    assert run(repo, "") == 1


def test_ru_thousands() -> None:
    assert verify_numbers.render(499817, ",d|ru") == "499 817"
    assert verify_numbers.render(1234.5, ",.1f|ru") == "1 234,5"


def test_lookup_dotted_keys() -> None:
    doc = {
        "metrics": {
            "leak.test": {"value": 0, "n": 0, "denominator": 440},
            "staged.records": {"value": 10},
            "staged.records.habr": {"value": 3},
        },
        "plain": {"a": [{"b": 7}]},
    }
    assert verify_numbers.lookup(doc, "metrics.leak.test.denominator") == 440
    assert verify_numbers.lookup(doc, "metrics.staged.records.value") == 10
    assert verify_numbers.lookup(doc, "metrics.staged.records.habr.value") == 3
    assert verify_numbers.lookup(doc, "plain.a.0.b") == 7
    with pytest.raises(KeyError):
        verify_numbers.lookup(doc, "metrics.leak.nope.value")
    with pytest.raises(KeyError):
        verify_numbers.lookup(doc, "plain.a.1.b")


def test_lookup_falls_back_to_shorter_key() -> None:
    doc = {"a.b": {"x": 1}, "a": {"b": {"y": 2}}}
    assert verify_numbers.lookup(doc, "a.b.x") == 1
    assert verify_numbers.lookup(doc, "a.b.y") == 2


def test_lookup_ambiguous_fails() -> None:
    doc = {"a.b": {"c": 1}, "a": {"b.c": 2}}
    with pytest.raises(KeyError, match="ambiguous"):
        verify_numbers.lookup(doc, "a.b.c")


def test_dotted_metric_marker(repo: Path) -> None:
    results.write_result(
        "corpus/refresh.json",
        {"leak.test_with_train_neighbour": {"value": 0.0, "n": 0, "denominator": 440}},
        source_script="tests",
    )
    readme = (
        "<!--n:corpus/refresh.json:metrics.leak.test_with_train_neighbour.n:d-->?<!--/n--> из "
        "<!--n:corpus/refresh.json:metrics.leak.test_with_train_neighbour.denominator:d-->?<!--/n-->"
    )
    assert run(repo, readme, "--write") == 0
    assert "-->0<!--/n--> из <!--n:" in (repo / "README.md").read_text(encoding="utf-8")
    assert "-->440<!--/n-->" in (repo / "README.md").read_text(encoding="utf-8")
