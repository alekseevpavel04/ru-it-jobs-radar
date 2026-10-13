"""Shared parts of the Hugging Face assemblers (radar.site.hf_model, hf_dataset, hf_space).

They lay out under storage/hf/ exactly the files an upload would send, and upload nothing. Names of repos
and of the published weights live in hf/hf.toml. A card is a template under hf/ with the number
markers of scripts/verify_numbers.py; a number whose result file does not exist yet (the final test)
is written as

    <!--pending:extract/final_test.json:metrics.test.macro_f1.value:.3f|ru-->ждёт финального теста<!--/pending-->

and becomes an ordinary marker, filled from results/, as soon as that file exists. Assembling a card
with a pending number left in it is allowed and reported, so the review before an upload sees what is still open.

Templates are hf/<repo>/README.md.in (not .md: GitHub would render their tables broken, see
Card.published). The card written to storage/hf carries the values only; every assembler run fills it
again from results/, so a value cannot drift from its result file.
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from radar.config import RESULTS, ROOT

HF = ROOT / "hf"
CONFIG = HF / "hf.toml"
# Upstream license texts the model repo and the Space carry next to the weights they serve.
LICENSES = HF / "licenses"
STORAGE = ROOT / "storage" / "hf"
PENDING = re.compile(r"<!--pending:(?P<path>[^:]+):(?P<key>[^:]+):(?P<fmt>[^>]*?)-->(?P<text>.*?)<!--/pending-->", re.S)


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    config: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    return config


@lru_cache(maxsize=1)
def verify_numbers() -> ModuleType:
    """scripts/verify_numbers.py, the one renderer of published numbers."""
    spec = importlib.util.spec_from_file_location("verify_numbers", ROOT / "scripts" / "verify_numbers.py")
    if spec is None or spec.loader is None:
        raise ImportError("scripts/verify_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Card:
    text: str
    numbers: int
    pending: list[str]

    @property
    def published(self) -> str:
        """The card as Hugging Face shows it: values without their markers. A marker's format spec
        carries `|` (`.3f|ru`), which a Markdown table reads as a cell border."""
        out = verify_numbers().MARKER.sub(lambda m: m["text"], self.text)
        return PENDING.sub(lambda m: m["text"], out)


def render_card(template: str, results: Path = RESULTS) -> Card:
    """Pending markers whose result file exists turn into number markers; every number marker is
    filled from results/. A missing key in an existing file is an error, not a pending number."""
    vn = verify_numbers()
    pending: list[str] = []

    def promote(m: re.Match[str]) -> str:
        if (results / m["path"]).exists():
            return f"<!--n:{m['path']}:{m['key']}:{m['fmt']}-->{m['text']}<!--/n-->"
        pending.append(f"{m['path']}:{m['key']}")
        return m.group(0)

    text = PENDING.sub(promote, template)
    cache: dict[str, Any] = {}
    count = 0

    def fill(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        source = results / m["path"]
        if not source.exists():
            raise FileNotFoundError(f"{m['path']}: no such result file for {m['key']}")
        if m["path"] not in cache:
            cache[m["path"]] = json.loads(source.read_text(encoding="utf-8"))
        value = vn.render(vn.lookup(cache[m["path"]], m["key"]), m["fmt"])
        return f"<!--n:{m['path']}:{m['key']}:{m['fmt']}-->{value}<!--/n-->"

    text = vn.MARKER.sub(fill, text)
    return Card(text, count, pending)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path) -> None:
    """dst holds the bytes of src: a hard link on the same volume (weights take gigabytes), a copy otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def manifest(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for path in sorted(files):
        out.append(
            {
                "file": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "size_mib": round(path.stat().st_size / 2**20, 1),
                "sha256": sha256_file(path),
            }
        )
    return out


def tree(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def shown(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def results_doc(relpath: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((RESULTS / relpath).read_text(encoding="utf-8"))
    return doc
