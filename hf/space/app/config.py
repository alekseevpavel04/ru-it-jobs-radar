import json
import os
from pathlib import Path
from typing import Any

ASSETS = Path(os.environ.get("SPACE_ASSETS", Path(__file__).resolve().parents[1] / "assets"))
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8081")


def space_info() -> dict[str, Any]:
    """What radar.site.hf_space wrote when it assembled the Space: names, versions, dev numbers."""
    info: dict[str, Any] = json.loads((ASSETS / "space.json").read_text(encoding="utf-8"))
    return info
