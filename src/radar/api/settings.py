"""API settings from the environment: a required variable that is empty stops the start."""

from dataclasses import dataclass
from pathlib import Path

from radar.config import ConfigError, optional, require


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
    stats_ttl_s: int
    extract_ttl_s: int


def from_env() -> Settings:
    adapter = optional("STUDENT_ADAPTER_DIR")
    for name in ("CLICKHOUSE_URL", "CLICKHOUSE_RO_USER", "CLICKHOUSE_RO_PASSWORD"):
        require(name)  # read again by ReadOnlyClickHouse; checked here so a missing one stops the start
    return Settings(
        redis_url=require("REDIS_URL"),
        student_url=optional("STUDENT_LLM_URL"),
        student_adapter=Path(adapter) if adapter else None,
        student_timeout_s=_number("STUDENT_TIMEOUT_S", 60.0),
        stats_ttl_s=int(_number("API_STATS_TTL_S", 3600)),
        extract_ttl_s=int(_number("API_EXTRACT_TTL_S", 86400)),
    )
