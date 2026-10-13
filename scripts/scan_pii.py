"""Personal data in what the project publishes: the git tree, the built site and the Hugging Face folders.

Detectors are those of the raw-layer cleaner (radar.pii.scan_text: emails, phones, labelled INN and
OGRNIP, person names by patronymic, initials, contact labels and first name next to a surname or in a
signature line, @handles that are not channels or bots, links to personal profiles) plus two that free
text in code and data needs: an unlabelled 12-digit number with a valid INN checksum (an individual's
INN; a company's has 10 digits) and a link to a GitHub or Hugging Face account root.

What is read: text files whole; JSON and JSONL by their string values (escaped Cyrillic included);
Parquet by its string cells. Raster images are not read: there is no OCR in the environment, so they are
listed for a look by eye. Model weights, archives and vendored third-party code are skipped and counted.

Values are never printed unless --show-values is given, and never written to the report: a finding is
path, line (or JSON path or Parquet row and column), kind, rule and the first 12 hex digits of sha256 of
the value. Known false positives go to the allowlist (scripts/pii_allowlist.tsv), one per line, four
fields separated by tabs: a path glob (optionally "::" and a glob of the place, "row * vacancy_id"), the
kind or *, the value fingerprint or *, and the reason. A line without a reason is an error.
Exit 1 when a finding is not allowlisted.

    python scripts/scan_pii.py                         # git-tracked files, site/dist, storage/hf
    python scripts/scan_pii.py storage/hf/dataset      # only these paths
    python scripts/scan_pii.py --report logs/pii_scan.json --show-values
"""

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radar.config import ROOT
from radar.pii import DEFAULT_KEEP_HANDLES, scan_text

ALLOWLIST = ROOT / "scripts" / "pii_allowlist.tsv"
EXTRA_ROOTS = ("site/dist", "storage/hf")
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}
BINARY = {
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".pdf",
    ".woff",
    ".woff2",
}
# Third-party code and lock files: their authors' addresses are not ours to publish or to remove.
SKIP_GLOBS = (
    "*/node_modules/*",
    "*/.observablehq/cache/*",
    "site/dist/_npm/*",
    "site/dist/_observablehq/*",
    "*package-lock.json",
    "uv.lock",
    "*/tokenizer.json",
)
MAX_TEXT_BYTES = 50 * 2**20
PROJECT_HANDLES = {"alekseevpavel04"}

_INN12 = re.compile(r"(?<![\w.\-])\d{12}(?![\w.\-])")
_INN_W11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_W12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_ACCOUNT = re.compile(
    r"(?:https?://)?(?:www\.)?(?P<host>github\.com|huggingface\.co)/(?P<user>[A-Za-z0-9][A-Za-z0-9_.\-]{0,38})"
    r"(?P<rest>/[A-Za-z0-9_.\-]+)?"
)
CODE = {".py", ".js", ".mjs", ".ts", ".css", ".scss", ".html", ".sql", ".sh", ".jinja", ".tpl"}
# CSS at-rules and JS doc tags read like @handles anywhere in code; decorators are caught by position.
_AT_WORDS = {
    "media", "supports", "keyframes", "font", "layer", "page", "import", "container", "charset", "namespace",
    "param", "returns", "type", "typedef", "example", "deprecated", "license", "preserve",
}  # fmt: skip
# t.me/s (the public preview prefix) and t.me/robots.txt name no account.
_PREVIEW = re.compile(r"t\.me/(?:s|robots\.txt)$", re.IGNORECASE)
_ACCOUNT_SERVICE = {
    "settings", "login", "orgs", "topics", "features", "about", "pricing", "docs", "datasets", "spaces",
    "models", "papers", "blog", "collections", "join", "new", "marketplace", "sponsors", "apps", "enterprise",
}  # fmt: skip


@dataclass(frozen=True)
class Hit:
    path: str
    where: str
    kind: str
    rule: str
    fingerprint: str
    value: str


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def is_person_inn(digits: str) -> bool:
    d = [int(c) for c in digits]
    n11 = sum(w * x for w, x in zip(_INN_W11, d, strict=False)) % 11 % 10
    n12 = sum(w * x for w, x in zip(_INN_W12, d, strict=False)) % 11 % 10
    return d[10] == n11 and d[11] == n12


def keep_handles() -> frozenset[str]:
    channels = tomllib.loads((ROOT / "configs" / "telegram_channels.toml").read_text(encoding="utf-8"))["channel"]
    return frozenset(DEFAULT_KEEP_HANDLES | {str(c["name"]).lower() for c in channels} | PROJECT_HANDLES)


def _code_noise(text: str, start: int, end: int) -> bool:
    """A decorator (@ first on its line) or an at-rule, not a handle."""
    line_start = text.rfind(chr(10), 0, start) + 1
    return not text[line_start:start].strip() or text[start + 1 : end].lower() in _AT_WORDS


def _decimal(text: str, start: int) -> bool:
    """Digits after a decimal point (0.3333333333) are a fraction, not a phone."""
    return start > 0 and text[start - 1] in ".,"


def scan_string(text: str, keep: frozenset[str], code: bool = False) -> Iterator[tuple[int, str, str, str]]:
    """(offset, kind, rule, value) of every finding in one string; `code` for source files."""
    for f in scan_text(text, keep):
        if f.kind == "nick" and code and _code_noise(text, f.start, f.end):
            continue
        if f.kind == "phone" and _decimal(text, f.start):
            continue
        if f.kind == "profile_link" and _PREVIEW.search(text[f.start : f.end]):
            continue
        yield f.start, f.kind, f.rule, text[f.start : f.end]
    if any(c.isdigit() for c in text):
        for m in _INN12.finditer(text):
            if is_person_inn(m.group()):
                yield m.start(), "inn_unlabelled", "checksum", m.group()
    if "github.com/" in text or "huggingface.co/" in text:
        for m in _ACCOUNT.finditer(text):
            user = m.group("user")
            if m.group("rest") is None and user.lower() not in _ACCOUNT_SERVICE and user.lower() not in keep:
                yield m.start(), "account_link", m.group("host"), m.group()


def _strings(node: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(k, f"{path}.<key>")
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _strings(v, f"{path}[{i}]")


def _line(text: str, offset: int) -> str:
    return f"line {text.count(chr(10), 0, offset) + 1}"


def scan_file(path: Path, rel: str, keep: frozenset[str]) -> Iterator[Hit]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        for name in table.column_names:
            for i, cell in enumerate(table.column(name).to_pylist()):
                for where, value in _strings(cell, f"row {i} {name}"):
                    for _, kind, rule, found in scan_string(value, keep):
                        yield Hit(rel, where, kind, rule, fingerprint(found), found)
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".json", ".jsonl", ".ipynb"):
        try:
            docs = [json.loads(text)] if suffix != ".jsonl" else [json.loads(x) for x in text.splitlines() if x]
        except json.JSONDecodeError:
            docs = None
        if docs is not None:
            for n, doc in enumerate(docs):
                prefix = f"line {n + 1} " if suffix == ".jsonl" else ""
                for where, value in _strings(doc):
                    for _, kind, rule, found in scan_string(value, keep):
                        yield Hit(rel, prefix + where, kind, rule, fingerprint(found), found)
            return
    for offset, kind, rule, found in scan_string(text, keep, code=suffix in CODE):
        yield Hit(rel, _line(text, offset), kind, rule, fingerprint(found), found)


@dataclass(frozen=True)
class Allow:
    glob: str
    kind: str
    fingerprint: str
    reason: str

    def covers(self, hit: Hit) -> bool:
        path_glob, _, where_glob = self.glob.partition("::")
        return (
            fnmatch.fnmatch(hit.path, path_glob)
            and (not where_glob or fnmatch.fnmatch(hit.where, where_glob))
            and self.kind in ("*", hit.kind)
            and self.fingerprint in ("*", hit.fingerprint)
        )


def load_allowlist(path: Path) -> list[Allow]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4 or not parts[3].strip():
            raise SystemExit(f"{path.name}:{n}: want <path glob>\\t<kind>\\t<fingerprint>\\t<reason>")
        out.append(Allow(*(p.strip() for p in parts)))
    return out


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def _rel(path: Path) -> str:
    """Relative to the repo when inside it, else as given."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def files_under(paths: Iterable[str]) -> list[str]:
    found: list[str] = []
    for p in paths:
        full = (ROOT / p).resolve()
        if full.is_file():
            found.append(_rel(full))
        elif full.is_dir():
            found.extend(_rel(f) for f in sorted(full.rglob("*")) if f.is_file())
    return found


def scan(rels: Iterable[str], keep: frozenset[str]) -> tuple[list[Hit], Counter[str], list[str]]:
    hits: list[Hit] = []
    skipped: Counter[str] = Counter()
    images: list[str] = []
    for rel in sorted(set(rels)):
        path = ROOT / rel
        suffix = path.suffix.lower()
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(f"/{rel}", g) for g in SKIP_GLOBS):
            skipped["third_party_or_vocabulary"] += 1
        elif suffix in IMAGES:
            images.append(rel)
        elif suffix in BINARY:
            skipped["binary"] += 1
        elif path.stat().st_size > MAX_TEXT_BYTES and suffix != ".parquet":
            skipped["too_large"] += 1
        elif suffix != ".parquet" and b"\0" in path.read_bytes()[:8192]:
            skipped["binary"] += 1
        else:
            hits.extend(scan_file(path, rel, keep))
    return hits, skipped, images


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "paths", nargs="*", help="files or folders relative to the repo; default: git tree, site/dist, storage/hf"
    )
    ap.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    ap.add_argument("--report", type=Path, help="JSON with every finding (no values) and the counts")
    ap.add_argument("--show-values", action="store_true", help="print the found values (terminal only)")
    args = ap.parse_args(argv)

    rels = files_under(args.paths) if args.paths else tracked_files() + files_under(EXTRA_ROOTS)
    allow = load_allowlist(args.allowlist)
    hits, skipped, images = scan(rels, keep_handles())
    open_hits = [h for h in hits if not any(a.covers(h) for a in allow)]
    allowed = len(hits) - len(open_hits)

    for h in open_hits:
        shown = f"  {h.value!r}" if args.show_values else ""
        print(f"{h.path}: {h.where}: {h.kind} ({h.rule or '-'}) {h.fingerprint}{shown}")
    by_kind = Counter(h.kind for h in open_hits)
    print(
        f"files {len(set(rels))}; findings {len(open_hits)} open, {allowed} allowlisted; open by kind {dict(by_kind)}"
    )
    print(f"skipped {dict(skipped)}; images not read (no OCR), look by eye: {len(images)}")
    for rel in images:
        print(f"  image {rel}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "files": len(set(rels)),
            "open_by_kind": dict(by_kind),
            "allowlisted": allowed,
            "skipped": dict(skipped),
            "images_unread": images,
            "open": [{k: v for k, v in asdict(h).items() if k != "value"} for h in open_hits],
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if open_hits else 0


if __name__ == "__main__":
    sys.exit(main())
