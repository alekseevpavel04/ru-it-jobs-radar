"""Latency of POST /extract as a client sees it, on trudvsem vacancies outside the gold sets.

    python -m radar.api.latency                       20 records against http://127.0.0.1:8000
    python -m radar.api.latency --n 20 --salt s2      another sample, e.g. while the first is cached

The records come from the extraction corpus (data/corpus): trudvsem and trudvsem snapshot inputs of
groups in train_pool, so nothing of gold dev or test (or the withheld groups around them), one record
per group, ordered by a salted hash. Each goes to the API as the text a client would send, one
request at a time, after one warm-up request that is not counted. An answer from the Redis cache or
from the regex fallback would not time the student, so either stops the run. Writes
results/api/extract_latency.json with the compose settings of the student service the API called.
"""

import argparse
import hashlib
import logging
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from radar.config import ROOT, data_dir
from radar.results import write_result

log = logging.getLogger("radar.api.latency")

RESULT = "api/extract_latency.json"
SOURCES = ("trudvsem", "trudvsem_snapshot")
COMPOSE = ROOT / "infra" / "compose" / "docker-compose.yml"


class NotTimed(RuntimeError):
    pass


def nearest_rank(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def sample(corpus: pd.DataFrame, split: pd.DataFrame, n: int, salt: str) -> pd.DataFrame:
    pool = set(split.loc[split["split"] == "train_pool", "group_id"])
    rows = corpus[corpus["source"].isin(SOURCES) & corpus["group_id"].isin(pool)].copy()
    rows["order"] = [hashlib.sha256(f"{salt}|{i}".encode()).hexdigest() for i in rows["id"]]
    rows = rows.sort_values("order").drop_duplicates("group_id")
    return rows.head(n).reset_index(drop=True)


def docker_host(*command: str) -> str:
    """Output of a command on the machine Docker runs on: the WSL distro of scripts/up.sh when this
    runs on Windows. The student shares that machine's CPUs with everything else on it, so its load
    goes into the result next to the latency."""
    if sys.platform == "win32":
        command = ("wsl.exe", "-d", os.environ.get("RADAR_WSL_DISTRO", "radar"), "--exec", *command)
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def vm_loadavg() -> str:
    return docker_host("cat", "/proc/loadavg")


def student_service(path: Path = COMPOSE) -> dict[str, Any]:
    service = yaml.safe_load(path.read_text(encoding="utf-8"))["services"]["student"]
    command = [str(c) for c in service["command"]]
    flag = {command[i]: command[i + 1] for i in range(len(command) - 1) if command[i].startswith("--")}
    return {
        "image": service["image"],
        "cpus": service.get("cpus"),
        "mem_limit": service.get("mem_limit"),
        "threads": int(flag["--threads"]),
        "parallel": int(flag["--parallel"]),
        "alias": flag["--alias"],
    }


def timed_extract(client: httpx.Client, text: str) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    response = client.post("/extract", json={"text": text})
    seconds = time.perf_counter() - started
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    if body["cached"]:
        raise NotTimed("the answer came from the cache; run with another --salt")
    if body["backend"] != "student":
        raise NotTimed(f"the regex answered for the student: {body['fallback_reason']}")
    return seconds, body


def measure(client: httpx.Client, records: pd.DataFrame) -> list[dict[str, Any]]:
    """The first record warms the server up and is not returned."""
    calls = []
    for i, row in enumerate(records.itertuples()):
        seconds, body = timed_extract(client, str(row.input))
        if i == 0:
            continue
        calls.append(
            {
                "id": row.id,
                "input_chars": len(str(row.input)),
                "truncated_input": body["truncated_input"],
                "seconds": round(seconds, 3),
                "model": body["model"],
            }
        )
        log.info("%d/%d %s %.2f s", i, len(records) - 1, row.id, seconds)
    return calls


def summary(calls: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    seconds = [c["seconds"] for c in calls]
    n = len(calls)
    truncated = sum(c["truncated_input"] for c in calls)
    return {
        "extract_latency_p50_s": {"value": round(nearest_rank(seconds, 0.5), 3)},
        "extract_latency_p95_s": {"value": round(nearest_rank(seconds, 0.95), 3)},
        "extract_latency_max_s": {"value": round(max(seconds), 3)},
        "extract_latency_mean_s": {"value": round(statistics.fmean(seconds), 3)},
        "requests": {"value": n},
        "truncated_inputs": {"value": round(truncated / n, 4), "n": truncated, "denominator": n},
        "input_chars_median": {"value": int(statistics.median(c["input_chars"] for c in calls))},
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.api.latency", description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the API root")
    ap.add_argument("--n", type=int, default=20, help="timed requests (one more warms up)")
    ap.add_argument("--salt", default="api-latency-v1", help="orders the sample; change it to draw another")
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(data_dir() / "corpus" / "corpus.parquet", columns=["id", "source", "group_id", "input"])
    split = pd.read_parquet(data_dir() / "corpus" / "split.parquet", columns=["group_id", "split"])
    records = sample(corpus, split, args.n + 1, args.salt)
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(180.0, connect=5.0)) as client:
        health = client.get("/health").json()
        load_before = vm_loadavg()
        try:
            calls = measure(client, records)
            load_after = vm_loadavg()
        except NotTimed as e:
            log.error("%s", e)
            return 1
    path = write_result(
        RESULT,
        summary(calls),
        "src/radar/api/latency.py",
        api_url=args.url,
        salt=args.salt,
        sample="trudvsem and trudvsem_snapshot inputs of data/corpus groups in train_pool, one per group",
        warmup_requests=1,
        concurrency=1,
        quantiles="nearest rank over the timed requests",
        api_health=health,
        docker_host_loadavg={"before": load_before, "after": load_after, "cpus": int(docker_host("nproc"))},
        student_service=student_service(),
        calls=calls,
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
