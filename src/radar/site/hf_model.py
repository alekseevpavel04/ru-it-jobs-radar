"""The model repo for Hugging Face, laid out under storage/hf/model; nothing is uploaded.

    python -m radar.site.hf_model

Layout, so that `from_pretrained(repo)` loads the merged weights and the rest sits beside them:

    README.md                 card, hf/model/README.md.in with numbers filled from results/
    config.json, generation_config.json, model.safetensors, tokenizer.json, tokenizer_config.json,
    chat_template.jinja       merged bf16 weights (models/merged/<name>)
    dataset_meta.json         system prompt, max_input_tokens, hashes of the training data
    adapter/                  the LoRA adapter (models/sft/<name>/adapter), for PeftModel on the base
    gguf/<name>-Q4_K_M.gguf, gguf/<name>-Q8_0.gguf     llama.cpp weights (models/gguf/<name>)
    LICENSE.md, licenses/     the mixed license (hf/model/LICENSE.md) and the upstream texts (hf/licenses)

The merged directory also holds prompts/ with the rendered dev prompts, which carry gold texts of
Habr Career and Telegram: those are never copied (DECISIONS 2). Weights are hard-linked, not copied.
Writes results/public/hf_model.json: every file with its size and sha256, and the pending numbers.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from radar.config import ROOT
from radar.results import write_result
from radar.site.hf_common import HF, LICENSES, STORAGE, load_config, manifest, place, render_card, shown, tree
from radar.site.hf_space import gguf_repo_path

log = logging.getLogger("radar.site.hf_model")

OUT = STORAGE / "model"
RESULT = "public/hf_model.json"
MERGED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "dataset_meta.json",
)
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "dataset_meta.json",
)
GGUF_TYPES = ("Q4_K_M", "Q8_0")


def plan(name: str, models: Path = ROOT / "models", out: Path = OUT) -> dict[Path, Path]:
    """Destination in the repo -> source file."""
    merged = models / "merged" / name
    adapter = models / "sft" / name / "adapter"
    files = {out / f: merged / f for f in MERGED_FILES}
    files |= {out / "adapter" / f: adapter / f for f in ADAPTER_FILES}
    for quant in GGUF_TYPES:
        files[out / gguf_repo_path(name, quant)] = models / "gguf" / name / f"{name}-{quant}.gguf"
    files[out / "LICENSE.md"] = HF / "model" / "LICENSE.md"
    files |= {out / "licenses" / f.name: f for f in sorted(LICENSES.iterdir())}
    missing = [shown(src) for src in files.values() if not src.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)}")
    return files


def assemble() -> dict[str, Any]:
    cfg = load_config()
    name = cfg["model"]["name"]
    files = plan(name)
    for stale in tree(OUT) if OUT.exists() else []:
        if stale not in files:
            stale.unlink()
    for dst, src in files.items():
        place(src, dst)
    weights = manifest(OUT, [p for p in tree(OUT) if p.name != "README.md"])
    by_file = {w["file"]: w for w in weights}
    metrics: dict[str, dict[str, Any]] = {
        "files": {"value": len(weights)},
        "total_mib": {"value": round(sum(w["size_bytes"] for w in weights) / 2**20, 1)},
    }
    for label, rel in (
        ("merged_safetensors", "model.safetensors"),
        ("adapter_safetensors", "adapter/adapter_model.safetensors"),
        *((f"gguf_{q.lower()}", gguf_repo_path(name, q)) for q in GGUF_TYPES),
    ):
        metrics[f"{label}_mib"] = {"value": by_file[rel]["size_mib"]}
    template = (HF / "model" / "README.md.in").read_text(encoding="utf-8")
    # The card reads the sizes written here, so the result goes first and the card second.
    write_result(RESULT, metrics, source_script="src/radar/site/hf_model.py", model=name, files=weights, pending=[])
    card = render_card(template)
    (OUT / "README.md").write_text(card.published, encoding="utf-8", newline="\n")
    write_result(
        RESULT, metrics, source_script="src/radar/site/hf_model.py", model=name, files=weights, pending=card.pending
    )
    log.info(
        "model -> %s: %d files, %.0f MiB, %d numbers in the card, pending %s",
        shown(OUT),
        len(weights),
        metrics["total_mib"]["value"],
        card.numbers,
        card.pending or "none",
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.site.hf_model", description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
