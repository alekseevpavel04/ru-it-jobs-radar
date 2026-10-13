"""Scan what an upload would send (storage/hf) for personal data; must find nothing.

    python -m radar.site.hf_scan

Every row of every Parquet file goes through radar.pii.records.scan_record (contacts, INN and OGRNIP
in any field, person employers, then every string through the free-text scanner), and so does every
JSON file. Identifier columns are checked against their exact shape instead (a UUID reads as a phone
number to the text scanner): a vacancy id is a UUID, a link is a trudvsem card of an organization
with a 13-digit OGRN, an 11-digit accreditation number of a foreign company's branch or a UUID code,
never a 15-digit OGRNIP or a 12-digit personal INN. Cards and licenses are scanned as text. Weights,
the tokenizer vocabulary and code (the Space page included: what it shows comes from space.json) are
not data and are skipped by suffix and name. Writes results/public/hf_pii_scan.json and exits 1 on
any finding, naming the file and the kind, never the value.
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from radar.pii.records import scan_record
from radar.pii.text import scan_text
from radar.results import write_result
from radar.site.hf_common import STORAGE, shown, tree

log = logging.getLogger("radar.site.hf_scan")

RESULT = "public/hf_pii_scan.json"
TEXT_SUFFIXES = {".md", ".txt"}
# The Space page is code; the data it shows comes from assets/space.json, which is scanned.
SKIPPED_SUFFIXES = {".gguf", ".safetensors", ".pkl", ".py", ".jinja", ".sh", ".toml", ".html"}
SKIPPED_NAMES = {"Dockerfile", "requirements.txt", "tokenizer.json", "tokenizer_config.json"}
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
SHAPES = {
    "vacancy_id": re.compile(rf"^{_UUID}$"),
    "url": re.compile(rf"^https://trudvsem\.ru/vacancy/card/(?:\d{{11}}|\d{{13}}|{_UUID})/{_UUID}$"),
}


def scan_row(row: dict[str, Any]) -> Counter[str]:
    found: Counter[str] = Counter()
    rest = dict(row)
    for column, shape in SHAPES.items():
        value = rest.pop(column, None)
        if value is not None and not shape.match(value):
            found[f"{column}_shape"] += 1
    found.update(scan_record(rest))
    return found


def scan_file(path: Path) -> tuple[str, int, Counter[str]]:
    """(what was done, records or texts scanned, findings by kind)."""
    if path.suffix == ".parquet":
        found: Counter[str] = Counter()
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            found.update(scan_row(row))
        return "rows", len(rows), found
    if path.suffix in (".json", ".jsonl"):
        text = path.read_text(encoding="utf-8")
        docs: list[Any] = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [json.loads(text)]
        )
        found = Counter()
        for doc in docs:
            found.update(scan_record(doc))
        return "json", len(docs), found
    text = path.read_text(encoding="utf-8")
    return "text", 1, Counter(f.kind for f in scan_text(text))


def scan(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    skipped: list[str] = []
    total: Counter[str] = Counter()
    scanned = 0
    for path in tree(root):
        if path.suffix in SKIPPED_SUFFIXES or path.name in SKIPPED_NAMES:
            skipped.append(shown(path))
            continue
        if path.suffix not in TEXT_SUFFIXES | {".parquet", ".json", ".jsonl"}:
            raise SystemExit(f"{shown(path)}: no rule for this kind of file; add it to the scan or the skip list")
        how, n, found = scan_file(path)
        scanned += n
        total.update(found)
        files.append({"file": shown(path), "scanned_as": how, "items": n, "findings": dict(found)})
    return {"files": files, "skipped": skipped, "total": total, "items": scanned}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.site.hf_scan", description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", type=Path, default=STORAGE)
    args = ap.parse_args(argv)
    report = scan(args.root)
    findings = sum(report["total"].values())
    write_result(
        RESULT,
        {
            "files_scanned": {"value": len(report["files"])},
            "items_scanned": {"value": report["items"]},
            "findings": {"value": findings, "by_kind": dict(report["total"])},
        },
        source_script="src/radar/site/hf_scan.py",
        root=shown(args.root),
        files=report["files"],
        skipped=report["skipped"],
    )
    for f in report["files"]:
        if f["findings"]:
            log.error("%s: %s", f["file"], f["findings"])
    log.info("%d files, %d items scanned, %d findings", len(report["files"]), report["items"], findings)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
