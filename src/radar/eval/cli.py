"""Score an extraction system on a gold split.

    python -m radar.eval --system regex --split dev
    python -m radar.eval --system qwen3_4b_sft --split dev --compare qwen3_8b

Writes results/extract/<split>_<system>.json (and <split>_<system>_vs_<other>.json with --compare):
every metric with a 95% bootstrap interval, shares with numerator and denominator. The test split is
refused without --final-test; a test run is appended to results/test_runs.log before scoring.
"""

import argparse
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar import config, results
from radar.config import ROOT, data_dir
from radar.eval.bootstrap import Interval, bootstrap, paired_bootstrap, resample_weights
from radar.eval.data import Aligned, align, load_gold, load_predictions, sha256
from radar.eval.metrics import Stat, normalize_item, score
from radar.extract.skills import eval_key

CONFIG = ROOT / "configs" / "eval.toml"
SYSTEM_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SKILL_NORMALIZERS: dict[str, Callable[[str], str]] = {"casefold": normalize_item, "dictionary": eval_key}


@dataclass(frozen=True)
class EvalConfig:
    n_resamples: int
    seed: int
    confidence: float
    salary_rel_tol: float
    macro_fields: tuple[str, ...]
    skill_normalization: str = "casefold"


def load_config(path: Path = CONFIG) -> EvalConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = EvalConfig(
        n_resamples=int(raw["bootstrap"]["n_resamples"]),
        seed=int(raw["bootstrap"]["seed"]),
        confidence=float(raw["bootstrap"]["confidence"]),
        salary_rel_tol=float(raw["salary"]["rel_tol"]),
        macro_fields=tuple(raw["macro"]["fields"]),
        skill_normalization=str(raw.get("skills", {}).get("normalization", "casefold")),
    )
    if cfg.n_resamples < 1000:
        raise ValueError("EXPERIMENTS.md asks for 1000 bootstrap resamples or more")
    if cfg.skill_normalization not in SKILL_NORMALIZERS:
        raise ValueError(f"[skills] normalization: one of {sorted(SKILL_NORMALIZERS)}")
    return cfg


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _r(x: float) -> float:
    return round(x, 6)


def metrics_block(stats: Mapping[str, Stat], intervals: Mapping[str, Interval | None]) -> dict[str, dict[str, Any]]:
    block: dict[str, dict[str, Any]] = {}
    for name, stat in stats.items():
        iv = intervals[name]
        if iv is None:
            continue
        entry: dict[str, Any] = {"value": _r(iv.value), "ci_low": _r(iv.low), "ci_high": _r(iv.high)}
        if stat.share:
            entry["n"], entry["denominator"] = stat.numerator_denominator()
        block[name] = entry
    return block


def score_aligned(aligned: Aligned, cfg: EvalConfig) -> dict[str, Stat]:
    return score(
        [g.fields for g in aligned.gold],
        aligned.parsed,
        cfg.salary_rel_tol,
        macro_fields=cfg.macro_fields,
        normalize_skill=SKILL_NORMALIZERS[cfg.skill_normalization],
    )


def by_source(aligned: Aligned, cfg: EvalConfig) -> dict[str, dict[str, dict[str, Any]]]:
    sources = sorted({g.source for g in aligned.gold if g.source})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for src in sources:
        idx = [i for i, g in enumerate(aligned.gold) if g.source == src]
        sub = Aligned(
            gold=[aligned.gold[i] for i in idx],
            outputs=[aligned.outputs[i] for i in idx],
            parsed=[aligned.parsed[i] for i in idx],
            missing=0,
            extra=0,
        )
        stats = score_aligned(sub, cfg)
        weights = resample_weights(len(idx), cfg.n_resamples, cfg.seed)
        out[src] = metrics_block(stats, bootstrap(stats, weights, cfg.confidence))
    return out


def log_test_run(system: str, compare: str | None) -> None:
    log = config.RESULTS / "test_runs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    command = results.invocation()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{system}\t{compare or '-'}\t{results.git_commit()}\t{command}\n")


def _print_table(title: str, block: Mapping[str, Mapping[str, Any]]) -> None:
    print(title)
    print(f"{'metric':34} {'value':>8} {'95% CI':>19}  n/denominator")
    for name, m in block.items():
        share = f"{m['n']}/{m['denominator']}" if "denominator" in m else ""
        print(f"{name:34} {m['value']:8.3f} [{m['ci_low']:7.3f}, {m['ci_high']:7.3f}]  {share}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radar.eval", description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, help="system name; predictions file name without .jsonl")
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument(
        "--compare", help="second system for a paired bootstrap of the difference; its file sits next to --predictions"
    )
    ap.add_argument("--predictions", type=Path, help="default: data/predictions/<split>/<system>.jsonl")
    ap.add_argument("--gold", type=Path, help="default: data/gold/<split>.jsonl")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--final-test", action="store_true", help="the one scoring of the test split")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.final_test:
        print("refused: the test split is scored once, in the final test, with --final-test", file=sys.stderr)
        return 2
    if args.final_test and args.split != "test":
        print("--final-test only goes with --split test", file=sys.stderr)
        return 2
    for name in filter(None, (args.system, args.compare)):
        if not SYSTEM_NAME.match(name):
            print(f"system name {name!r}: use lowercase letters, digits, _ and -", file=sys.stderr)
            return 2

    cfg = load_config(args.config)
    if args.final_test:
        log_test_run(args.system, args.compare)
    gold_path = args.gold or data_dir() / "gold" / f"{args.split}.jsonl"
    pred_path = args.predictions or data_dir() / "predictions" / args.split / f"{args.system}.jsonl"
    gold = load_gold(gold_path)
    aligned = align(gold, load_predictions(pred_path))
    stats = score_aligned(aligned, cfg)
    weights = resample_weights(len(gold), cfg.n_resamples, cfg.seed)
    intervals = bootstrap(stats, weights, cfg.confidence)
    block = metrics_block(stats, intervals)
    common: dict[str, Any] = {
        "split": args.split,
        "gold_file": _shown(gold_path),
        "gold_sha256": sha256(gold_path),
        "n_examples": len(gold),
        "bootstrap": {"n_resamples": cfg.n_resamples, "seed": cfg.seed, "confidence": cfg.confidence},
        "salary_rel_tol": cfg.salary_rel_tol,
        "macro_fields": list(cfg.macro_fields),
        "skill_normalization": cfg.skill_normalization,
    }
    path = results.write_result(
        f"extract/{args.split}_{args.system}.json",
        block,
        source_script="src/radar/eval/cli.py",
        system=args.system,
        predictions_file=_shown(pred_path),
        predictions_sha256=sha256(pred_path),
        n_missing_predictions=aligned.missing,
        n_extra_predictions_ignored=aligned.extra,
        undefined_metrics=sorted(set(stats) - set(block)),
        by_source=by_source(aligned, cfg),
        **common,
    )
    _print_table(f"{args.system} on {args.split}: {len(gold)} examples -> {_shown(path)}", block)

    if args.compare:
        other_path = pred_path.parent / f"{args.compare}.jsonl"
        other = align(gold, load_predictions(other_path))
        other_stats = score_aligned(other, cfg)
        diff = paired_bootstrap(stats, other_stats, weights, cfg.confidence)
        diff_block: dict[str, dict[str, Any]] = {}
        for name, iv in diff.items():
            if iv is None:
                continue
            diff_block[name] = {
                "value": _r(iv.value),
                "ci_low": _r(iv.low),
                "ci_high": _r(iv.high),
                "a": _r(stats[name].value()),
                "b": _r(other_stats[name].value()),
            }
        path = results.write_result(
            f"extract/{args.split}_{args.system}_vs_{args.compare}.json",
            diff_block,
            source_script="src/radar/eval/cli.py",
            system=args.system,
            compare=args.compare,
            difference="a - b, a = system, b = compare, paired bootstrap on the same gold rows",
            predictions_sha256={args.system: sha256(pred_path), args.compare: sha256(other_path)},
            n_missing_predictions={args.system: aligned.missing, args.compare: other.missing},
            **common,
        )
        _print_table(f"{args.system} - {args.compare} on {args.split} -> {_shown(path)}", diff_block)
    return 0
