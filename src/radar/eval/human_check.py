"""Manual check of 50 gold test records against the vacancy text and docs/annotation_guide.md.

Gold inputs are vacancy texts from Habr and Telegram that are not published, so the sample, the
page and the answers live in data/review/ (not in git). Only counts, ids and field names go to
results/gold/human_check.json.

    python -m radar.eval.human_check sample    # draw the sample once; serve draws it if missing
    python -m radar.eval.human_check serve     # http://127.0.0.1:8765, answers saved on every click
    python -m radar.eval.human_check report    # results/gold/human_check.json

The sample is stratified by source over the gold test splits: proportional, but no source gets fewer
than MIN_PER_SOURCE records, so Habr and Telegram are not two or three examples each.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from radar.config import ROOT, data_dir
from radar.eval.cli import load_config
from radar.eval.data import load_gold, sha256
from radar.extract.schema import VacancyFields
from radar.results import write_result

FIELDS: tuple[str, ...] = tuple(VacancyFields.model_fields)
SPLITS = ("test", "tg_test")
SAMPLE_SIZE = 50
MIN_PER_SOURCE = 8
PAGE = Path(__file__).with_name("human_check.html")
GUIDE = ROOT / "docs" / "annotation_guide.md"
SAMPLE_FILE = "human_check_sample.json"
ANSWERS_FILE = "human_check_answers.jsonl"
DEFAULT_PORT = 8765
MAX_BODY = 64_000

Mark = Literal["ok", "wrong"]
Verdict = Literal["agree", "disagree", "unsure"]


class Answer(BaseModel):
    """One save of one record. The answers file is append-only; the last line of an id wins."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    fields_sha: str = Field(min_length=64, max_length=64)
    marks: dict[str, Mark] = {}
    comments: dict[str, str] = {}
    cases: dict[str, list[int]] = {}
    verdict: Verdict | None = None
    note: str = ""
    answered_at: str | None = None

    def complete(self) -> bool:
        return self.verdict is not None and all(f in self.marks for f in FIELDS)


@dataclass(frozen=True)
class Item:
    id: str
    source: str
    split: str
    input: str
    fields: dict[str, Any]

    @property
    def fields_sha(self) -> str:
        return fields_sha(self.fields)


def fields_sha(fields: dict[str, Any]) -> str:
    """Identifies the gold label a person saw, so an answer to a label rebuilt since is not counted."""
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def allocate(sizes: dict[str, int], n: int, minimum: int) -> dict[str, int]:
    """Proportional allocation with a floor: strata whose share falls below `minimum` get `minimum`
    (or all they have), the rest is split in proportion to size by largest remainder."""
    if n > sum(sizes.values()):
        raise ValueError(f"sample of {n} from {sum(sizes.values())} records")
    fixed: dict[str, int] = {}
    while True:
        free = {s: k for s, k in sizes.items() if s not in fixed}
        rest = n - sum(fixed.values())
        total = sum(free.values())
        if not free or total == 0:
            quotas: dict[str, float] = {}
            break
        quotas = {s: rest * k / total for s, k in free.items()}
        low = {s for s, q in quotas.items() if q < min(minimum, free[s])}
        if not low:
            break
        fixed.update({s: min(minimum, sizes[s]) for s in low})
    counts = {s: int(q) for s, q in quotas.items()}
    by_remainder = sorted(quotas, key=lambda s: (counts[s] - quotas[s], s))
    for s in by_remainder[: rest - sum(counts.values())]:
        counts[s] += 1
    out = {s: fixed.get(s, counts.get(s, 0)) for s in sizes}
    if sum(out.values()) != n or any(out[s] > sizes[s] for s in sizes):
        raise ValueError(f"cannot allocate {n} records over {sizes} with at least {minimum} each")
    return out


def draw(gold_dir: Path, seed: int) -> dict[str, Any]:
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    gold_sha: dict[str, str] = {}
    for split in SPLITS:
        path = gold_dir / f"{split}.jsonl"
        gold_sha[split] = sha256(path)
        for rec in load_gold(path):
            by_source[rec.source or "unknown"].append((rec.id, split))
    sizes = {s: len(v) for s, v in sorted(by_source.items())}
    quotas = allocate(sizes, SAMPLE_SIZE, MIN_PER_SOURCE)
    rng = random.Random(seed)
    picked: list[dict[str, str]] = []
    for source in sizes:
        for rid, split in rng.sample(sorted(by_source[source]), quotas[source]):
            picked.append({"id": rid, "source": source, "split": split})
    # Sources interleaved, so the page does not show all Telegram posts in a row.
    rng.shuffle(picked)
    ids = [p["id"] for p in picked]
    return {
        "seed": seed,
        "size": SAMPLE_SIZE,
        "min_per_source": MIN_PER_SOURCE,
        "population": sizes,
        "composition": quotas,
        "gold_sha256": gold_sha,
        "ids_sha256": ids_sha(ids),
        "records": picked,
    }


def load_sample(review_dir: Path, gold_dir: Path, seed: int, create: bool) -> dict[str, Any]:
    path = review_dir / SAMPLE_FILE
    if path.exists():
        sample: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if ids_sha([r["id"] for r in sample["records"]]) != sample["ids_sha256"]:
            raise ValueError(f"{path}: ids do not match ids_sha256; the sample was edited")
        return sample
    if not create:
        raise FileNotFoundError(f"{path}: no sample yet, run `python -m radar.eval.human_check sample`")
    sample = draw(gold_dir, seed)
    review_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sample


def load_items(sample: dict[str, Any], gold_dir: Path) -> list[Item]:
    gold: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    for split in SPLITS:
        for rec in load_gold(gold_dir / f"{split}.jsonl"):
            gold[rec.id] = rec.fields.model_dump(mode="json", by_alias=True)
        for line in (gold_dir / f"{split}_inputs.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                texts[row["id"]] = row["input"]
    missing = [r["id"] for r in sample["records"] if r["id"] not in gold or r["id"] not in texts]
    if missing:
        raise ValueError(f"sample ids missing from the gold set: {missing[:5]}")
    return [Item(r["id"], r["source"], r["split"], texts[r["id"]], gold[r["id"]]) for r in sample["records"]]


def load_answers(path: Path) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    if not path.exists():
        return answers
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            answer = Answer.model_validate_json(line)
        except ValidationError as e:
            raise ValueError(f"{path}:{no}: not an answer\n{e}") from e
        answers[answer.id] = answer
    return answers


def current(items: list[Item], answers: dict[str, Answer]) -> tuple[dict[str, Answer], list[str]]:
    """Answers given to the gold label that is in the gold set now, and ids whose label changed since."""
    valid: dict[str, Answer] = {}
    stale: list[str] = []
    for item in items:
        answer = answers.get(item.id)
        if answer is None:
            continue
        if answer.fields_sha == item.fields_sha:
            valid[item.id] = answer
        else:
            stale.append(item.id)
    return valid, stale


def summarize(items: list[Item], answers: dict[str, Answer]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    valid, stale = current(items, answers)
    checked = [item for item in items if item.id in valid and valid[item.id].complete()]
    # A reviewer may give the overall verdict without marking the fields one by one: such a record
    # counts for the verdict and not for the per-field shares.
    judged = [item for item in items if item.id in valid and valid[item.id].verdict is not None]
    if not judged:
        raise ValueError("no record is judged yet: give an overall verdict at least")

    def share(k: int, d: int) -> dict[str, Any]:
        return {"value": round(k / d, 4) if d else None, "n": k, "denominator": d}

    n = len(checked)
    done = {item.id for item in judged}
    all_ok = {item.id for item in checked if all(valid[item.id].marks[f] == "ok" for f in FIELDS)}
    metrics: dict[str, dict[str, Any]] = {
        "records_judged": share(len(judged), len(items)),
        "records_checked": share(n, len(items)),
        "records_all_fields_correct": share(len(all_ok), n),
    }
    for field in FIELDS:
        metrics[f"field_{field}_correct"] = share(sum(valid[i.id].marks[field] == "ok" for i in checked), n)
    verdicts = Counter(valid[i.id].verdict for i in judged)
    for verdict in ("agree", "disagree", "unsure"):
        metrics[f"verdict_{verdict}"] = share(verdicts[verdict], len(judged))

    by_source: dict[str, dict[str, int]] = {}
    for item in checked:
        row = by_source.setdefault(item.source, {"checked": 0, "all_fields_correct": 0})
        row["checked"] += 1
        row["all_fields_correct"] += item.id in all_ok
    # Ids and field names only: comments may quote the vacancy text and stay in data/review/.
    disagreements = [
        {"id": i.id, "source": i.source, "field": f, "guide_cases": valid[i.id].cases.get(f, [])}
        for i in checked
        for f in FIELDS
        if valid[i.id].marks[f] == "wrong"
    ]
    extra = {
        "by_source": dict(sorted(by_source.items())),
        "disagreements": disagreements,
        "verdict_disagree_ids": [i.id for i in checked if valid[i.id].verdict == "disagree"],
        "unchecked_ids": [i.id for i in items if i.id not in done],
        "stale_answers": stale,
    }
    return metrics, extra


# Server


class ReviewServer(HTTPServer):
    def __init__(self, port: int, items: list[Item], answers_path: Path) -> None:
        super().__init__(("127.0.0.1", port), ReviewHandler)
        self.items = {item.id: item for item in items}
        self.order = [item.id for item in items]
        self.answers_path = answers_path

    @property
    def hosts(self) -> set[str]:
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def state(self) -> dict[str, Any]:
        valid, _ = current(list(self.items.values()), load_answers(self.answers_path))
        return {
            "items": [
                {
                    "id": i.id,
                    "source": i.source,
                    "split": i.split,
                    "input": i.input,
                    "fields": i.fields,
                    "fields_sha": i.fields_sha,
                }
                for i in (self.items[rid] for rid in self.order)
            ],
            "fields": list(FIELDS),
            "answers": {rid: a.model_dump() for rid, a in valid.items()},
        }

    def save(self, answer: Answer) -> Answer:
        item = self.items.get(answer.id)
        if item is None:
            raise LookupError(f"{answer.id} is not in the sample")
        if answer.fields_sha != item.fields_sha:
            raise LookupError(f"{answer.id}: the gold label changed since the page was loaded, reload it")
        unknown = (set(answer.marks) | set(answer.comments) | set(answer.cases)) - set(FIELDS)
        if unknown:
            raise LookupError(f"unknown fields {sorted(unknown)}")
        saved = answer.model_copy(update={"answered_at": datetime.now(UTC).isoformat(timespec="seconds")})
        self.answers_path.parent.mkdir(parents=True, exist_ok=True)
        with self.answers_path.open("a", encoding="utf-8") as f:
            f.write(saved.model_dump_json() + "\n")
        return saved


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self) -> bool:
        # The page serves unpublished texts: refuse requests addressed to any other host name.
        if self.headers.get("Host") in self.server.hosts:
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "wrong host"})
        return False

    def do_GET(self) -> None:
        if not self._host_ok():
            return
        if self.path == "/":
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(HTTPStatus.OK, self.server.state())
        elif self.path == "/guide":
            self._send(HTTPStatus.OK, GUIDE.read_bytes(), "text/plain; charset=utf-8")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._host_ok():
            return
        if self.path != "/api/answer":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "answer too large"})
            return
        try:
            answer = Answer.model_validate_json(self.rfile.read(length))
            saved = self.server.save(answer)
        except (ValidationError, LookupError) as e:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        self._json(HTTPStatus.OK, saved.model_dump())


# CLI


def _paths() -> tuple[Path, Path]:
    return data_dir() / "gold", data_dir() / "review"


def cmd_sample(_: argparse.Namespace) -> int:
    gold_dir, review_dir = _paths()
    sample = load_sample(review_dir, gold_dir, load_config().seed, create=True)
    print(json.dumps({k: sample[k] for k in ("composition", "ids_sha256")}, ensure_ascii=False))
    print(review_dir / SAMPLE_FILE)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    gold_dir, review_dir = _paths()
    sample = load_sample(review_dir, gold_dir, load_config().seed, create=True)
    server = ReviewServer(args.port, load_items(sample, gold_dir), review_dir / ANSWERS_FILE)
    print(f"http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_report(_: argparse.Namespace) -> int:
    gold_dir, review_dir = _paths()
    sample = load_sample(review_dir, gold_dir, load_config().seed, create=False)
    items = load_items(sample, gold_dir)
    metrics, extra = summarize(items, load_answers(review_dir / ANSWERS_FILE))
    path = write_result(
        "gold/human_check.json",
        metrics,
        source_script="src/radar/eval/human_check.py",
        method=(
            "the project author read each sampled test record next to its gold label and gave an overall verdict "
            "(agree, disagree, unsure) against the vacancy text and docs/annotation_guide.md; records_judged "
            "counts the verdicts; where the author also marked every field as correct or wrong the record counts "
            "as checked, and the per-field shares are over checked records only"
        ),
        sample={k: sample[k] for k in ("seed", "size", "min_per_source", "population", "composition")}
        | {"gold_sha256": sample["gold_sha256"], "ids_sha256": sample["ids_sha256"]}
        | {"ids": [r["id"] for r in sample["records"]]},
        **extra,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k.startswith("records_")}, ensure_ascii=False))
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m radar.eval.human_check", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample", help="draw the 50 records into data/review/").set_defaults(func=cmd_sample)
    serve = sub.add_parser("serve", help="local page on 127.0.0.1, answers to data/review/")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.set_defaults(func=cmd_serve)
    sub.add_parser("report", help="write results/gold/human_check.json").set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
