"""configs/dedup.toml as typed settings; the file is the only place the numbers live."""

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "dedup.toml"


@dataclass(frozen=True)
class Params:
    """Candidate generation, score bands and the labeling sample."""

    shingle_k: int
    num_perm: int
    lsh_threshold: float
    knn_k: int
    knn_min_cosine: float
    embedding_model: str
    jaccard_high: float
    jaccard_border: float
    cosine_high: float
    cosine_border: float
    allocation: dict[str, int]
    spill: tuple[str, ...]
    tuning_share: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    """Link two groups when a score clears its threshold and the stated conditions hold.

    A threshold of None means that method takes no part in the decision.
    """

    jaccard: float | None
    cosine: float | None
    same_place: bool
    same_title: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Settings:
    params: Params
    grid_jaccard: tuple[float, ...]
    grid_cosine: tuple[float, ...]
    rule: Rule
    text_cluster_jaccard: float
    max_component: int


def load(path: Path = CONFIG) -> Settings:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    sample = raw["sample"]
    params = Params(
        **raw["candidates"],
        **raw["bands"],
        allocation=dict(sample["allocation"]),
        spill=tuple(sample["spill"]),
        tuning_share=sample["tuning_share"],
    )
    return Settings(
        params=params,
        grid_jaccard=tuple(raw["selection"]["jaccard"]),
        grid_cosine=tuple(raw["selection"]["cosine"]),
        rule=Rule(
            jaccard=raw["rule"].get("jaccard"),
            cosine=raw["rule"].get("cosine"),
            same_place=raw["rule"]["same_place"],
            same_title=raw["rule"]["same_title"],
        ),
        text_cluster_jaccard=raw["text_cluster"]["jaccard"],
        max_component=raw["entities"]["max_component"],
    )
