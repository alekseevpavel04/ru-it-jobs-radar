"""Whole-card memory readings from nvidia-smi, the number that decides whether something fits in 8 GB.

torch.cuda.max_memory_allocated misses the CUDA context, cuBLAS workspaces and anything outside PyTorch
(llama-server, vLLM's cache pool), so every smoke step also samples nvidia-smi in a background thread.
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

PARTS_DIR = Path(__file__).resolve().parents[1] / "logs" / "gpu_smoke"


def used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.split()[0])


class VramPoller:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, used_mib())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VramPoller":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def write_part(name: str, payload: dict[str, Any]) -> Path:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
