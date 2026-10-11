from pathlib import Path

import pytest

from radar.config import ROOT
from radar.train import dataset
from radar.train.config import LORA_TARGETS, load_run

RUNS = sorted((ROOT / "configs" / "sft").glob("qwen3_*.toml"))


def test_both_students_have_a_config() -> None:
    assert {load_run(p).model for p in RUNS} == {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"}


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.stem)
def test_project_runs_match_the_dataset(path: Path) -> None:
    run = load_run(path)
    assert run.train.max_seq_len == dataset.load_config().max_seq_len
    assert run.train.batch * run.train.grad_accum == 16
    assert run.lora.targets == LORA_TARGETS
    assert run.output_dir == ROOT / "models" / "sft" / run.name
    assert run.train_file == ROOT / "data" / "sft" / "train.jsonl"


def test_params_are_flat_for_the_tracker() -> None:
    params = load_run(RUNS[0]).params()
    assert params["lora.r"] == 16
    assert params["lora.targets"] == ",".join(LORA_TARGETS)
    assert "guard.vram_margin_mib" in params
    assert all(not isinstance(v, dict | tuple | list) for v in params.values())


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text('name = "x"\nmodel = "Qwen/Qwen3-1.7B"\n' + text, encoding="utf-8")
    return path


def test_defaults_fill_missing_sections(tmp_path: Path) -> None:
    run = load_run(write(tmp_path, ""))
    assert run.backend == "unsloth"
    assert run.guard.step_time_factor > 1


@pytest.mark.parametrize(
    "text",
    [
        'backend = "deepspeed"\n',
        "[train]\nbatch = 0\n",
        "[train]\nlr = 0.5\n",
        "[train]\nlearning_rate = 1e-4\n",
        "[guard]\nstep_time_factor = 1.0\n",
        '[lora]\ntargets = ["lm_head"]\n',
        "[extra]\nx = 1\n",
    ],
)
def test_bad_configs_are_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_run(write(tmp_path, text))
