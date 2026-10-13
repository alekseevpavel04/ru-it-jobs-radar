"""API settings from the environment: a required variable that is empty stops the start."""

from dataclasses import dataclass
from pathlib import Path

from radar.api.salary import DEFAULT_MODEL_PATH
from radar.config import ROOT, ConfigError, optional, require
from radar.models import matching


def _number(name: str, default: float) -> float:
    raw = optional(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a number") from e
    if value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    redis_url: str
    # The fine-tuned student behind llama-server, its root URL without /v1; None: regex only.
    student_url: str | None
    # A run's adapter directory with dataset_meta.json; None: the prompt and input cap of data/sft/meta.json.
    student_adapter: Path | None
    student_timeout_s: float
    # The pickled SalaryModel (radar.models.salary); a relative path is taken from the repo root.
    salary_model: Path
    # The pickled MatchIndex (radar.models.matching) behind /match, relative paths from the repo root.
    match_index: Path
    stats_ttl_s: int
    extract_ttl_s: int


def salary_model_path() -> Path:
    """SALARY_MODEL_PATH, relative to the repo root, or the artifact `python -m radar.models.salary` saves."""
    path = Path(optional("SALARY_MODEL_PATH") or DEFAULT_MODEL_PATH)
    return path if path.is_absolute() else ROOT / path


def match_index_path() -> Path:
    """MATCH_INDEX_PATH, relative to the repo root, or the index `python -m radar.models.matching evaluate` saves."""
    path = Path(optional("MATCH_INDEX_PATH") or matching.MODEL_DIR / matching.ARTIFACT)
    return path if path.is_absolute() else ROOT / path


def from_env() -> Settings:
    adapter = optional("STUDENT_ADAPTER_DIR")
    for name in ("CLICKHOUSE_URL", "CLICKHOUSE_RO_USER", "CLICKHOUSE_RO_PASSWORD"):
        require(name)  # read again by ReadOnlyClickHouse; checked here so a missing one stops the start
    return Settings(
        redis_url=require("REDIS_URL"),
        student_url=optional("STUDENT_LLM_URL"),
        student_adapter=Path(adapter) if adapter else None,
        student_timeout_s=_number("STUDENT_TIMEOUT_S", 60.0),
        salary_model=salary_model_path(),
        match_index=match_index_path(),
        stats_ttl_s=int(_number("API_STATS_TTL_S", 3600)),
        extract_ttl_s=int(_number("API_EXTRACT_TTL_S", 86400)),
    )
