"""Run configuration of a QLoRA SFT run, configs/sft/<run>.toml, and of a DPO run on top
of an SFT adapter, configs/dpo/<run>.toml.

Plain data with validation, no torch: the configs are checked on Windows by the test suite, and the
training script turns them into trainer arguments inside WSL.
"""

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from radar.config import ROOT

BACKENDS = ("unsloth", "peft")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Lora:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    targets: tuple[str, ...] = LORA_TARGETS


@dataclass(frozen=True)
class Train:
    max_seq_len: int = 2048
    epochs: float = 2.0
    lr: float = 2e-4
    batch: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"
    seed: int = 20261008
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 100
    save_total_limit: int = 2


@dataclass(frozen=True)
class Guard:
    """Stop when torch's reserved memory passes the card minus `vram_margin_mib` (what other
    processes held at start is subtracted too), or when the time per token of `patience` steps in a
    row is `step_time_factor` times the median of the first `reference_steps` steps after the first."""

    vram_margin_mib: int = 512
    step_time_factor: float = 3.0
    reference_steps: int = 5
    patience: int = 3


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    backend: str
    train_file: Path
    val_file: Path
    dataset_meta: Path
    output_dir: Path
    lora: Lora = field(default_factory=Lora)
    train: Train = field(default_factory=Train)
    guard: Guard = field(default_factory=Guard)

    def params(self) -> dict[str, Any]:
        """Flat key-value view for the tracker."""
        flat: dict[str, Any] = {"name": self.name, "model": self.model, "backend": self.backend}
        for prefix, part in (("lora", self.lora), ("train", self.train), ("guard", self.guard)):
            for k, v in asdict(part).items():
                flat[f"{prefix}.{k}"] = ",".join(v) if isinstance(v, tuple) else v
        return flat


def _section(cls: type, raw: dict[str, Any], name: str) -> Any:
    unknown = sorted(set(raw) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(f"[{name}]: unknown keys {unknown}")
    if "targets" in raw:
        raw = {**raw, "targets": tuple(raw["targets"])}
    return cls(**raw)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_run(path: Path) -> RunConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    top = {"name", "model", "backend", "data", "output_dir", "lora", "train", "guard"}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    data = raw.get("data", {})
    cfg = RunConfig(
        name=str(raw["name"]),
        model=str(raw["model"]),
        backend=str(raw.get("backend", "unsloth")),
        train_file=_path(data.get("train", "data/sft/train.jsonl")),
        val_file=_path(data.get("val", "data/sft/val.jsonl")),
        dataset_meta=_path(data.get("meta", "data/sft/meta.json")),
        output_dir=_path(raw.get("output_dir", f"models/sft/{raw['name']}")),
        lora=_section(Lora, raw.get("lora", {}), "lora"),
        train=_section(Train, raw.get("train", {}), "train"),
        guard=_section(Guard, raw.get("guard", {}), "guard"),
    )
    validate(cfg)
    return cfg


def _check_steps(t: "Train | DpoTrain") -> None:
    if min(t.batch, t.grad_accum, t.max_seq_len, t.logging_steps, t.eval_steps, t.save_steps) < 1:
        raise ValueError("batch, grad_accum, max_seq_len and step intervals are positive")
    if not 0 < t.lr < 1e-2:
        raise ValueError(f"lr {t.lr} is outside (0, 0.01)")
    if t.epochs <= 0:
        raise ValueError("epochs must be positive")


def _check_guard(g: Guard) -> None:
    if g.step_time_factor <= 1 or g.reference_steps < 1 or g.patience < 1 or g.vram_margin_mib < 0:
        raise ValueError("guard: factor above 1, reference_steps and patience at least 1, margin not negative")


def validate(cfg: RunConfig) -> None:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend {cfg.backend!r}: one of {BACKENDS}")
    if not set(cfg.lora.targets) <= set(LORA_TARGETS):
        raise ValueError(f"lora targets outside {LORA_TARGETS}")
    _check_steps(cfg.train)
    _check_guard(cfg.guard)


# --- DPO -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pairs:
    """Where the preference pairs come from (radar.train.dpo_pairs). Every source row is a row of
    data/sft: silver labels of train_pool groups, never a dev or test record."""

    train_sample: int = 2000  # SFT train examples the student answers, half from each source family
    with_val: bool = True  # and every SFT val example, which the student did not train on
    holdout_share: float = 0.1  # share of groups whose pairs only measure the DPO eval loss
    seed: int = 20261008
    # corrupted copies of silver answers, per kind
    work_format: int = 200
    role: int = 200
    drop_field: int = 50
    broken_json: int = 50

    def corrupted(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in CORRUPTIONS}


CORRUPTIONS = ("work_format", "role", "drop_field", "broken_json")


@dataclass(frozen=True)
class DpoTrain:
    max_seq_len: int = 2048
    epochs: float = 2.0
    lr: float = 2e-5
    beta: float = 0.1
    # weight of the NLL term on the chosen answer next to the DPO loss (TRL loss "sft"); 0 is plain DPO
    sft_weight: float = 1.0
    batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"
    seed: int = 20261008
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2


@dataclass(frozen=True)
class DpoConfig:
    name: str
    sft_adapter: Path
    sft_dir: Path  # data/sft: train.jsonl, val.jsonl, meta.json
    pairs_dir: Path  # data/dpo: student inputs and answers, pairs
    output_dir: Path
    pairs: Pairs = field(default_factory=Pairs)
    train: DpoTrain = field(default_factory=DpoTrain)
    guard: Guard = field(default_factory=Guard)

    def params(self) -> dict[str, Any]:
        flat: dict[str, Any] = {"name": self.name, "sft_adapter": _shown(self.sft_adapter)}
        for prefix, part in (("pairs", self.pairs), ("train", self.train), ("guard", self.guard)):
            for k, v in asdict(part).items():
                flat[f"{prefix}.{k}"] = v
        return flat


def _shown(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def load_dpo(path: Path) -> DpoConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    top = {"name", "sft_adapter", "data", "output_dir", "pairs", "train", "guard"}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    data = raw.get("data", {})
    unknown = sorted(set(data) - {"sft", "pairs"})
    if unknown:
        raise ValueError(f"[data]: unknown keys {unknown}")
    cfg = DpoConfig(
        name=str(raw["name"]),
        sft_adapter=_path(raw["sft_adapter"]),
        sft_dir=_path(data.get("sft", "data/sft")),
        pairs_dir=_path(data.get("pairs", "data/dpo")),
        output_dir=_path(raw.get("output_dir", f"models/dpo/{raw['name']}")),
        pairs=_section(Pairs, raw.get("pairs", {}), "pairs"),
        train=_section(DpoTrain, raw.get("train", {}), "train"),
        guard=_section(Guard, raw.get("guard", {}), "guard"),
    )
    _check_steps(cfg.train)
    _check_guard(cfg.guard)
    p = cfg.pairs
    if not 0 <= p.holdout_share < 0.5:
        raise ValueError("pairs.holdout_share: a share of groups below 0.5")
    if min(p.train_sample, *p.corrupted().values()) < 0:
        raise ValueError("pairs: counts are not negative")
    if cfg.train.beta <= 0 or cfg.train.sft_weight < 0:
        raise ValueError("train.beta is positive, train.sft_weight not negative")
    return cfg
