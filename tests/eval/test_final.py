import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil
import pytest

from radar import config, results
from radar.eval import final
from radar.eval.cli import SYSTEM_NAME
from radar.eval.final import Context, System


def fields(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "role": None,
        "grade": None,
        "skills": [],
        "salary": None,
        "work_format": None,
        "cities": [],
        "experience_years": None,
        "employment_type": None,
        "lang": "ru",
    }
    return base | kw


ROLES = ("qa", "backend", "devops", "data_analyst", "frontend")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    gold = tmp_path / "data" / "gold"
    for split, n, kept in (("test", 10, 10), ("tg_test", 6, 5)):
        ids = [f"{split}:{i}" for i in range(n)]
        write_jsonl(
            gold / f"{split}_inputs.jsonl",
            [{"id": i, "source": "habr", "input": f"Тестировщик, Москва, удалённо {i}"} for i in ids],
        )
        rows = [{"id": i, "fields": fields(role=ROLES[k % len(ROLES)])} for k, i in enumerate(ids[:kept])]
        write_jsonl(gold / f"{split}.jsonl", rows)
    return tmp_path


CALLS: Counter[str] = Counter()


def fake(system: System, splits: Sequence[str], ctx: Context) -> None:
    """A perfect system ("good") and one that always answers backend ("bad")."""
    for split in splits:
        CALLS[f"{system.name}/{split}"] += 1
        inputs = final.read_jsonl(final.inputs_path(split))
        gold = {r["id"]: r["fields"] for r in final.read_jsonl(final.data_dir() / "gold" / f"{split}.jsonl")}
        rows = []
        for r in inputs:
            answer = gold.get(r["id"], fields()) if system.name == "good" else fields(role="backend")
            rows.append({"id": r["id"], "output": json.dumps(answer)})
        write_jsonl(final.prediction_path(split, system.name), rows)
        final.write_json(final.sidecar_path(split, system.name), {"system": system.name, "split": split})


FAKE = (System("regex", "regex", gpu=False), System("good", "fake"), System("bad", "fake"))


@pytest.fixture
def harness(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    CALLS.clear()
    monkeypatch.setitem(final.KINDS, "fake", fake)
    monkeypatch.setattr(final, "COMPARISONS", (("good", "regex"), ("bad", "good"), ("good", "absent")))
    return repo


def test_runs_every_system_and_scores_once_with_paired_comparisons(harness: Path) -> None:
    assert final.run(FAKE, Context()) == 0
    doc = json.loads((harness / "results" / "extract" / "final_test.json").read_text(encoding="utf-8"))
    m = doc["metrics"]
    assert m["test/good/macro_f1"]["value"] == 1.0
    assert {"ci_low", "ci_high"} <= set(m["test/good/macro_f1"])
    assert m["tg_test/good/valid_json"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0, "n": 5, "denominator": 5}
    assert m["test/bad_minus_good/role_f1"]["value"] < 0
    assert doc["comparisons"] == ["good - regex", "bad - good"]
    assert set(doc["predictions_sha256"]["tg_test"]) == {"regex", "good", "bad"}
    for split in ("test", "tg_test"):
        assert (harness / "results" / "extract" / f"{split}_good_vs_regex.json").exists()
        assert (harness / "results" / "extract" / f"{split}_regex.json").exists()
    log = (harness / "results" / "test_runs.log").read_text(encoding="utf-8").splitlines()
    assert sum("\tpredict:test,tg_test\t" in line for line in log) == 3
    # one eval call per system without comparisons, one per comparison: regex, good vs regex, bad vs good
    assert sum(line.split("\t")[1] in ("regex", "good", "bad") and "predict" not in line for line in log) == 6
    assert log[-1].split("\t")[1:3] == ["final", "scored"]


def test_a_second_run_skips_done_systems_and_does_not_rescore(harness: Path) -> None:
    final.run(FAKE, Context())
    summary = harness / "results" / "extract" / "final_test.json"
    before = summary.read_text(encoding="utf-8")
    final.run(FAKE, Context())
    assert CALLS == {"good/test": 1, "good/tg_test": 1, "bad/test": 1, "bad/tg_test": 1}
    assert summary.read_text(encoding="utf-8") == before
    final.run(FAKE, Context(), rescore="metric bug fixed")
    log = (harness / "results" / "test_runs.log").read_text(encoding="utf-8")
    assert "rescore: metric bug fixed" in log
    assert json.loads(summary.read_text(encoding="utf-8"))["rescore_reason"] == "metric bug fixed"


def test_an_incomplete_file_is_predicted_again(harness: Path) -> None:
    final.run(FAKE, Context())
    path = final.prediction_path("test", "bad")
    path.write_text("".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:3]), encoding="utf-8")
    assert final.todo(FAKE[2]) == ["test"]
    final.run(FAKE, Context(), rescore="test")
    assert CALLS["bad/test"] == 2 and CALLS["bad/tg_test"] == 1


def test_scoring_waits_for_every_system(harness: Path) -> None:
    final.run(FAKE, Context(), only=["good"])
    assert not (harness / "results" / "extract" / "final_test.json").exists()
    assert not (harness / "results" / "extract" / "test_good.json").exists()
    with pytest.raises(SystemExit, match="unknown systems"):
        final.run(FAKE, Context(), only=["nope"])


def test_a_run_that_leaves_ids_out_stops(harness: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def partial(system: System, splits: Sequence[str], ctx: Context) -> None:
        for split in splits:
            write_jsonl(final.prediction_path(split, system.name), [{"id": "x", "output": None}])
            final.write_json(final.sidecar_path(split, system.name), {})

    monkeypatch.setitem(final.KINDS, "fake", partial)
    with pytest.raises(SystemExit, match="incomplete"):
        final.run(FAKE, Context())


def test_status_needs_no_flag_and_scoring_needs_final_test(harness: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert final.main(["--status"]) == 0
    assert "teacher_qwen" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        final.main([])


class RecordingContext(Context):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def wsl(self, args: Sequence[str], log_name: str) -> None:
        self.calls.append(list(args))

    def require_free_gpu(self, wait_s: float = 90.0) -> None:
        self.calls.append(["gpu-check"])


def test_transformers_systems_reuse_the_dev_settings(repo: Path) -> None:
    system = System("sft_x", "hf", {"adapter": "models/sft/x/adapter"})
    ctx = RecordingContext()
    with pytest.raises(SystemExit, match="no dev run"):
        final.predict_hf(system, ["test"], ctx)
    final.write_json(final.sidecar_path("dev", "sft_x"), {"batch": 4, "backend": "peft"})
    final.predict_hf(system, ["test"], ctx)
    assert ctx.calls[0] == ["gpu-check"]
    assert ctx.calls[1] == [
        "bash",
        "scripts/train_run.sh",
        "predict",
        "--adapter",
        "models/sft/x/adapter",
        "--split",
        "test",
        "--system",
        "sft_x",
        "--batch",
        "4",
        "--backend",
        "peft",
        "--final-test",
    ]


def test_served_systems_keep_workers_and_schema_of_dev(repo: Path) -> None:
    final.write_json(final.sidecar_path("tg_dev", "awq"), {"workers": 4, "decoding": {"json_schema_constrained": True}})
    args = final.served_args(System("awq", "vllm"), "tg_test", "http://127.0.0.1:8095", "vllm")
    assert args[-2:] == ["--final-test", "--schema"]
    assert args[args.index("--workers") + 1] == "4"


def test_teacher_serves_test_inputs_only_for_the_final_test(repo: Path) -> None:
    from radar.extract.teacher import select

    assert len(select("test", None, [], seed=1, final_test=True)) == 10
    with pytest.raises(ValueError, match="not served"):
        select("test", None, [], seed=1)


def test_teacher_prompt_is_found_by_its_hash() -> None:
    v4, v5 = b"prompt v4\n", b"prompt v5\n"
    import hashlib

    assert final.prompt_by_sha(hashlib.sha256(v4).hexdigest(), [v5, v4]) == v4
    with pytest.raises(SystemExit, match="no version"):
        final.prompt_by_sha("0" * 64, [v5, v4])


def test_registry_is_consistent() -> None:
    names = [s.name for s in final.systems(with_dpo=True)]
    assert len(names) == len(set(names))
    assert all(SYSTEM_NAME.match(n) for n in names)
    assert {n for pair in final.COMPARISONS for n in pair} <= set(names)
    assert (final.DPO.name, final.STUDENT) not in final.comparisons([s.name for s in final.systems(with_dpo=False)])
    assert set(final.KINDS) >= {s.kind for s in final.systems(with_dpo=True)}


def test_wsl_paths() -> None:
    assert (
        final.to_wsl(Path("D:/VScode_projects/ru-it-jobs-radar/logs")) == "/mnt/d/VScode_projects/ru-it-jobs-radar/logs"
    )


def test_stop_tree_stops_children_by_pid() -> None:
    code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    deadline = time.monotonic() + 20
    while not psutil.Process(proc.pid).children() and time.monotonic() < deadline:
        time.sleep(0.2)
    child = psutil.Process(proc.pid).children(recursive=True)[0]
    final.stop_tree(proc, timeout=10)
    assert proc.poll() is not None
    assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE


def test_prediction_tools_refuse_test_splits_without_the_flag() -> None:
    from radar.extract import student_server, teacher
    from radar.train import predict
    from radar.train.quantize import check_prompt_splits

    with pytest.raises(SystemExit, match="refused"):
        predict.main(["--model", "m", "--split", "test", "--system", "s"])
    with pytest.raises(SystemExit, match="only goes with a test split"):
        predict.main(["--model", "m", "--split", "dev", "--system", "s", "--final-test"])
    with pytest.raises(SystemExit, match="--out"):
        predict.main(["--model", "m", "--inputs", "x.jsonl", "--system", "s"])
    with pytest.raises(SystemExit, match="refused"):
        student_server.main(["--name", "n", "--split", "tg_test", "--system", "s"])
    with pytest.raises(SystemExit, match="refused"):
        teacher.main(["--teacher", "qwen", "--split", "test"])
    with pytest.raises(SystemExit, match="--teacher"):
        teacher.main(["--agreement", "--split", "test", "--final-test"])
    with pytest.raises(SystemExit, match="refused"):
        check_prompt_splits(["dev", "test"], final_test=False)
    with pytest.raises(SystemExit, match="only goes with test"):
        check_prompt_splits(["dev"], final_test=True)
    check_prompt_splits(["test", "tg_test"], final_test=True)
