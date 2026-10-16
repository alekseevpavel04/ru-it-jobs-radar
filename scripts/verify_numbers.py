"""Recompute every published number from results/ and fail on any mismatch.

A number in README, docs or site pages is written between markers:

    <!--n:extract/final_test.json:metrics.f1_macro.value:.3f|ru-->0,812<!--/n-->

The path is relative to results/, the key is dotted, the format is a Python format spec followed by
optional transforms: `pct` multiplies by 100, `ru` prints a decimal comma and space-separated
thousands. `--write` rewrites the text between markers instead of checking it, so a number is
never typed by hand in two places.

Result files use keys that contain dots themselves (`metrics["leak.test_with_train_neighbour"]`), so a
dotted key is resolved level by level: at a dict, the longest run of the remaining parts that is a
literal key is tried first, then shorter runs down to a single part; at a list, the part is an index.
`metrics.heldout.entity.raw.tp.value` thus reaches `metrics["heldout.entity.raw.tp"]["value"]`. If
the key can be resolved along more than one path, the marker fails as ambiguous instead of picking one.

Also validates every results/**/*.json that has a `metrics` block against the result convention.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radar.config import RESULTS, ROOT
from radar.results import ResultFile

MARKER = re.compile(r"<!--n:(?P<path>[^:]+):(?P<key>[^:]+):(?P<fmt>[^>]*?)-->(?P<text>.*?)<!--/n-->", re.S)
# Hugging Face card templates by name: hf/space also holds node_modules.
DEFAULT_TARGETS = [
    "README.md",
    "docs",
    "site/src",
    "hf/model/README.md.in",
    "hf/dataset/README.md.in",
    "hf/space/README.md.in",
    "hf/space/index.html",
]
TEXT_SUFFIXES = {".md", ".html", ".svelte", ".js", ".ts"}


def _resolve(node: Any, parts: list[str]) -> Iterator[Any]:
    if not parts:
        yield node
    elif isinstance(node, list):
        try:
            child = node[int(parts[0])]
        except (ValueError, IndexError):
            return
        yield from _resolve(child, parts[1:])
    elif isinstance(node, dict):
        for end in range(len(parts), 0, -1):
            key = ".".join(parts[:end])
            if key in node:
                yield from _resolve(node[key], parts[end:])


def lookup(doc: Any, dotted: str) -> Any:
    found = list(islice(_resolve(doc, dotted.split(".")), 2))
    if not found:
        raise KeyError(dotted)
    if len(found) > 1:
        raise KeyError(f"{dotted} is ambiguous: it resolves along more than one path")
    return found[0]


def render(value: Any, fmt: str) -> str:
    # Inside a Markdown table row the separator is written as "\|", or GitHub splits the cell on it.
    spec, *transforms = fmt.replace("\\|", "|").split("|")
    if "pct" in transforms:
        value = value * 100
    out = format(value, spec)
    if "ru" in transforms:
        out = out.replace(",", "\x00").replace(".", ",").replace("\x00", " ")
    return out


def target_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = ROOT / target
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.suffix in TEXT_SUFFIXES and p.is_file())
    return files


def check_file(path: Path, results_dir: Path, write: bool) -> tuple[int, list[str]]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    cache: dict[str, Any] = {}
    count = 0

    def replace(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        where = f"{path.relative_to(ROOT)}: {m['path']}:{m['key']}"
        source = results_dir / m["path"]
        if not source.exists():
            errors.append(f"{where}: no such result file")
            return m.group(0)
        if m["path"] not in cache:
            cache[m["path"]] = json.loads(source.read_text(encoding="utf-8"))
        try:
            expected = render(lookup(cache[m["path"]], m["key"]), m["fmt"])
        except (KeyError, IndexError, ValueError, TypeError) as e:
            errors.append(f"{where}: cannot render ({e!r})")
            return m.group(0)
        if write:
            return m.group(0).replace(f"-->{m['text']}<!--/n-->", f"-->{expected}<!--/n-->")
        if m["text"] != expected:
            errors.append(f"{where}: text {m['text']!r}, results give {expected!r}")
        return m.group(0)

    new_text = MARKER.sub(replace, text)
    if write and new_text != text:
        path.write_text(new_text, encoding="utf-8", newline="")
    return count, errors


def check_result_files(results_dir: Path) -> list[str]:
    errors = []
    for path in sorted(results_dir.rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and "metrics" in doc:
            try:
                ResultFile.model_validate(doc)
            except ValidationError as e:
                errors.append(f"{path.relative_to(results_dir)}: {e.error_count()} convention errors\n{e}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS, help="files or folders, relative to repo root")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--write", action="store_true", help="rewrite numbers from results instead of checking")
    args = parser.parse_args(argv)

    errors = check_result_files(args.results)
    total = 0
    for path in target_files(args.targets):
        count, file_errors = check_file(path, args.results, args.write)
        total += count
        errors.extend(file_errors)

    for line in errors:
        print(line, file=sys.stderr)
    verb = "rewritten" if args.write else "checked"
    print(f"{total} numbers {verb}, {len(errors)} mismatches")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
