"""Student for serving: adapter merged into the base, GGUF, llama.cpp quantization.

    bash scripts/train_run.sh quantize merge --adapter models/sft/qwen3_1p7b_r16_clean/adapter     # WSL
    bash scripts/train_run.sh quantize convert --name qwen3_1p7b_r16_clean                         # WSL
    python -m radar.train.quantize quantize --name qwen3_1p7b_r16_clean                           # Windows
    bash scripts/train_run.sh quantize prompts --name qwen3_1p7b_r16_clean --splits test tg_test --final-test
    python -m radar.train.quantize summary --name qwen3_1p7b_r16_clean --reference sft_1p7b_r16_clean \
        --systems sft_1p7b_clean_f16 sft_1p7b_clean_q8_0 sft_1p7b_clean_q4_k_m sft_1p7b_clean_awq \
        --bench data/inference/bench/*.meta.json

The serving runs in between are radar.extract.student_server; the AWQ checkpoint is radar.train.awq.

merge (training venv) loads the base in bf16 on the CPU, applies the adapter and folds it into the
weights, and writes models/merged/<name>/: safetensors, the adapter's tokenizer and chat template,
dataset_meta.json and merge.json. The adapter was trained against the base quantized to NF4
(QLoRA), and the merge adds it to the bf16 base: the scores of the merged model are measured, not
assumed equal to those of radar.train.predict.

merge also renders the prompts of the dev splits into models/merged/<name>/prompts/<split>.jsonl
with the adapter's tokenizer and dataset_meta.json, through the same functions as
radar.train.predict (system prompt, chat template with thinking off, input cut to max_input_tokens
at a token boundary). radar.extract.student_server sends them to llama-server as text, so every
quantized variant reads the prompt the student was trained on; the Windows venv has no tokenizer.
prompts renders them again for the given splits from the merged directory (its tokenizer and
dataset_meta.json); a test split only with --final-test, which the final test passes.

convert runs llama.cpp's convert_hf_to_gguf.py of the same build as the binaries (LLAMA_CPP_SRC,
the source tree of b11052) to an F16 GGUF; quantize runs llama-quantize.exe (LLAMA_DIR) from it to
each type, without an importance matrix, and writes models/gguf/<name>/manifest.json with the size
and sha256 of every file. summary collects sizes, speed, valid JSON and macro F1 with intervals of
the served variants into results/extract/quant_summary.json; the scores come from
`python -m radar.eval` result files, never computed here.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar.config import RESULTS, ROOT, data_dir
from radar.eval.cli import TEST_SPLITS
from radar.results import git_commit, write_result

log = logging.getLogger("radar.train.quantize")

SPLITS = ("dev", "tg_dev")
TYPES = ("Q8_0", "Q4_K_M")
MERGED = ROOT / "models" / "merged"
GGUF = ROOT / "models" / "gguf"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MIB = 1024 * 1024


def default_llama_src() -> Path:
    if os.name == "nt":
        return Path("D:/VScode_projects/assets/tools/llama.cpp/src-b11052")
    return Path("/mnt/d/VScode_projects/assets/tools/llama.cpp/src-b11052")


def llama_src() -> Path:
    return Path(os.environ.get("LLAMA_CPP_SRC", "").strip() or default_llama_src())


def llama_bin() -> Path:
    return Path(os.environ.get("LLAMA_DIR", "").strip() or "D:/VScode_projects/assets/tools/llama.cpp/b11052")


def run_name(adapter: Path) -> str:
    """models/sft/<run>/adapter -> <run>; an epoch-<k>/ adapter keeps its epoch in the name."""
    adapter = adapter.resolve()
    return adapter.parent.name if adapter.name == "adapter" else f"{adapter.parent.name}_{adapter.name}"


def gguf_path(name: str, qtype: str) -> Path:
    return GGUF / name / f"{name}-{qtype}.gguf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(MIB), b""):
            h.update(block)
    return h.hexdigest()


def shown(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def size_mib(paths: Sequence[Path]) -> float:
    return round(sum(p.stat().st_size for p in paths) / MIB, 1)


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --- merge and prompts (training venv) ----------------------------------------------------------


def prompt_rows(records: Sequence[Mapping[str, Any]], tok: Any, meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    from radar.train.dataset import cap_input, render_prompt, token_spans

    rows = []
    for r in records:
        user, cut = cap_input(r["input"], token_spans(tok, r["input"]), meta["max_input_tokens"])
        prompt = render_prompt(tok, meta["system_prompt"], user)
        rows.append(
            {
                "id": r["id"],
                "system": meta["system_prompt"],
                "user": user,
                "prompt": prompt,
                "prompt_tokens": len(tok(prompt, add_special_tokens=False).input_ids),
                "truncated_input": cut,
            }
        )
    return rows


def write_prompts(adapter: Path, out: Path, splits: Sequence[str] = SPLITS) -> None:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    from radar.train.dataset import read_jsonl, write_jsonl
    from radar.train.predict import dataset_meta

    meta = dataset_meta(adapter)
    tok = AutoTokenizer.from_pretrained(str(adapter))
    for split in splits:
        records = read_jsonl(data_dir() / "gold" / f"{split}_inputs.jsonl")
        rows = prompt_rows(records, tok, meta)
        write_jsonl(out / "prompts" / f"{split}.jsonl", rows)
        log.info("%s: %d prompts, %d inputs cut", split, len(rows), sum(r["truncated_input"] for r in rows))


def cmd_merge(adapter: Path, name: str | None) -> int:
    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM

    from radar.train.hf import adapter_base
    from radar.train.predict import adapter_sha256

    adapter = adapter.resolve()
    out = MERGED / (name or run_name(adapter))
    base = adapter_base(adapter)
    log.info("merging %s into %s (bf16, CPU) -> %s", shown(adapter), base, shown(out))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(out))
    for file in TOKENIZER_FILES:
        shutil.copy2(adapter / file, out / file)
    shutil.copy2(adapter / "dataset_meta.json", out / "dataset_meta.json")
    write_prompts(adapter, out)
    write_json(
        out / "merge.json",
        {
            "adapter": shown(adapter),
            "adapter_sha256": adapter_sha256(adapter),
            "base": base,
            "dtype": "bfloat16",
            "weights_mib": size_mib(sorted(out.glob("*.safetensors"))),
            "commit": git_commit(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return 0


# --- GGUF ---------------------------------------------------------------------------------------


def check_prompt_splits(splits: Sequence[str], final_test: bool) -> None:
    unknown = sorted(set(splits) - {*SPLITS, *TEST_SPLITS})
    if unknown:
        raise SystemExit(f"unknown splits {unknown}")
    tests = [s for s in splits if s in TEST_SPLITS]
    if tests and not final_test:
        raise SystemExit("refused: test prompts are rendered once, by the final test, with --final-test")
    if final_test and len(tests) != len(splits):
        raise SystemExit("--final-test only goes with test splits (test, tg_test)")


def cmd_prompts(name: str, splits: Sequence[str], final_test: bool) -> int:
    check_prompt_splits(splits, final_test)
    write_prompts(MERGED / name, MERGED / name, splits)
    return 0


def cmd_convert(name: str) -> int:
    merged = MERGED / name
    out = gguf_path(name, "F16")
    out.parent.mkdir(parents=True, exist_ok=True)
    script = llama_src() / "convert_hf_to_gguf.py"
    cmd = [sys.executable, str(script), str(merged), "--outtype", "f16", "--outfile", str(out)]
    log.info("%s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


def manifest_entry(path: Path, qtype: str, source: Path | None) -> dict[str, Any]:
    return {
        "file": shown(path),
        "type": qtype,
        "size_bytes": path.stat().st_size,
        "size_mib": size_mib([path]),
        "sha256": sha256_file(path),
        "from": shown(source) if source else None,
    }


def cmd_quantize(name: str, types: Sequence[str]) -> int:
    f16 = gguf_path(name, "F16")
    if not f16.exists():
        raise SystemExit(f"{f16} is missing: run `quantize convert --name {name}` in WSL first")
    exe = llama_bin() / "llama-quantize.exe"
    entries = [manifest_entry(f16, "F16", None)]
    for qtype in types:
        out = gguf_path(name, qtype)
        cmd = [str(exe), str(f16), str(out), qtype]
        log.info("%s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        entries.append(manifest_entry(out, qtype, f16))
    merge = json.loads((MERGED / name / "merge.json").read_text(encoding="utf-8"))
    write_json(
        GGUF / name / "manifest.json",
        {
            "name": name,
            "merged": shown(MERGED / name),
            "adapter": merge["adapter"],
            "adapter_sha256": merge["adapter_sha256"],
            "llama_cpp": llama_bin().name,
            "imatrix": None,
            "files": entries,
            "commit": git_commit(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return 0


# --- summary ------------------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def interval(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {k: metric[k] for k in ("value", "ci_low", "ci_high", "n", "denominator") if k in metric}


def split_metrics(split: str, system: str, reference: str | None) -> dict[str, dict[str, Any]]:
    """macro F1 and valid JSON of `system` on `split`, and the paired difference to `reference`."""
    scored = read_json(RESULTS / "extract" / f"{split}_{system}.json")["metrics"]
    out = {
        f"{split}/{system}/macro_f1": interval(scored["macro_f1"]),
        f"{split}/{system}/valid_json": interval(scored["valid_json"]),
    }
    if reference and reference != system:
        diff = read_json(RESULTS / "extract" / f"{split}_{system}_vs_{reference}.json")["metrics"]
        out[f"{split}/{system}/macro_f1_minus_{reference}"] = interval(diff["macro_f1"])
    return out


def speed_metrics(prefix: str, sidecar: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {
        f"{prefix}/records_per_s": {"value": sidecar["records_per_s"], "workers": sidecar["workers"]},
        f"{prefix}/latency_p50_s": {"value": sidecar["latency_s"]["p50"]},
        f"{prefix}/latency_p95_s": {"value": sidecar["latency_s"]["p95"]},
    }
    if sidecar.get("vram_peak_mib") is not None:
        out[f"{prefix}/vram_peak_mib"] = {"value": sidecar["vram_peak_mib"], "what": "nvidia-smi, whole card"}
    return out


def read_outputs(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r["output"] for r in rows}


def same_answers(system: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, int]:
    """Records where both answers parse to the same JSON value, over the records of `system`."""

    def value(text: str | None) -> Any:
        try:
            return json.loads(text) if text is not None else None
        except json.JSONDecodeError:
            return None

    same = sum(value(out) is not None and value(out) == value(reference.get(rid)) for rid, out in system.items())
    return {"value": round(same / len(system), 4), "n": same, "denominator": len(system)}


def summarize(
    name: str, reference: str, systems: Sequence[str], bench: Sequence[Path]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    served: dict[str, Any] = {}
    for split in SPLITS:
        metrics |= split_metrics(split, reference, None)
        ref = read_json(data_dir() / "predictions" / split / f"{reference}.meta.json")
        metrics[f"{split}/{reference}/records_per_s"] = {
            "value": round(ref["records"] / ref["wall_seconds"], 3),
            "batch": ref["batch"],
            "backend": ref["backend"],
        }
        served.setdefault(reference, {k: ref.get(k) for k in ("adapter", "adapter_sha256", "backend", "batch")})
        for system in systems:
            metrics |= split_metrics(split, system, reference)
            sidecar = read_json(data_dir() / "predictions" / split / f"{system}.meta.json")
            metrics |= speed_metrics(f"{split}/{system}", sidecar)
            outputs = read_outputs(data_dir() / "predictions" / split / f"{system}.jsonl")
            reference_outputs = read_outputs(data_dir() / "predictions" / split / f"{reference}.jsonl")
            metrics[f"{split}/{system}/same_answer_as_{reference}"] = same_answers(outputs, reference_outputs)
            keys = ("backend", "weights", "decoding", "server", "template_match", "prompt_token_mismatch")
            served.setdefault(system, {k: sidecar[k] for k in keys})
    metrics["size_mib/merged_bf16"] = {"value": size_mib(sorted((MERGED / name).glob("*.safetensors")))}
    for system in systems:
        metrics[f"size_mib/{system}"] = {"value": served[system]["weights"]["size_mib"]}
    for path in bench:
        sidecar = read_json(path)
        metrics |= speed_metrics(f"bench/{sidecar['system']}/workers_{sidecar['workers']}", sidecar)
    return metrics, {"served": served, "gguf_manifest": read_json(GGUF / name / "manifest.json")}


def cmd_summary(name: str, reference: str, systems: Sequence[str], bench: Sequence[Path]) -> int:
    metrics, extra = summarize(name, reference, systems, bench)
    path = write_result(
        "extract/quant_summary.json",
        metrics,
        source_script="src/radar/train/quantize.py",
        name=name,
        reference=reference,
        systems=list(systems),
        bench_files=[shown(p) for p in bench],
        **extra,
        method="macro F1 and valid JSON from results/extract/<split>_<system>.json (python -m radar.eval), the "
        "difference to the reference from <split>_<system>_vs_<reference>.json (paired bootstrap on the same gold "
        "rows); same answer: both parse to an equal JSON value, over every predicted record of the split, gold "
        "or not; records per second over the wall time of a run with `workers` requests in flight (llama-server "
        "with at least as many slots, vLLM with max-num-seqs 8); latency per record measured by the client; peak "
        "memory of the whole card by nvidia-smi, vLLM reserving 0.85 of it up front; sizes from the files",
    )
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.train.quantize", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="adapter into the bf16 base, plus the dev prompts (training venv)")
    m.add_argument("--adapter", type=Path, required=True)
    m.add_argument("--name", help="default: the run directory name")
    pr = sub.add_parser("prompts", help="render the prompts of the given splits again (training venv)")
    pr.add_argument("--name", required=True)
    pr.add_argument("--splits", nargs="+", required=True)
    pr.add_argument("--final-test", action="store_true", help="test splits, final test only")
    c = sub.add_parser("convert", help="merged model to an F16 GGUF (training venv)")
    c.add_argument("--name", required=True)
    q = sub.add_parser("quantize", help="F16 GGUF to each type with llama-quantize (Windows)")
    q.add_argument("--name", required=True)
    q.add_argument("--types", nargs="+", default=list(TYPES))
    s = sub.add_parser("summary", help="results/extract/quant_summary.json")
    s.add_argument("--name", required=True)
    s.add_argument("--reference", required=True, help="system of the student before merging (radar.train.predict)")
    s.add_argument("--systems", nargs="+", required=True)
    s.add_argument("--bench", nargs="*", type=Path, default=[], help="sidecars of speed runs (.meta.json)")
    args = ap.parse_args(argv)

    if args.cmd == "merge":
        return cmd_merge(args.adapter, args.name)
    if args.cmd == "prompts":
        return cmd_prompts(args.name, args.splits, args.final_test)
    if args.cmd == "convert":
        return cmd_convert(args.name)
    if args.cmd == "quantize":
        return cmd_quantize(args.name, args.types)
    return cmd_summary(args.name, args.reference, args.systems, args.bench)


if __name__ == "__main__":
    sys.exit(main())
