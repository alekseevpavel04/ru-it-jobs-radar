"""Configuration comes only from the environment (and `.env` at the repo root).

A required variable that is missing or empty stops the run with a message naming it: no silent
fallbacks for anything that changes where data goes or which service is called.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


class ConfigError(RuntimeError):
    pass


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set; see .env.example")
    return value


def optional(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def data_dir() -> Path:
    path = Path(os.environ.get("RADAR_DATA_DIR", "").strip() or ROOT / "data")
    return path if path.is_absolute() else ROOT / path


def user_agent() -> str:
    return require("RADAR_USER_AGENT")
