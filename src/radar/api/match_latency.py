"""Latency of POST /match as a client sees it and the memory of the api container serving it.

    python -m radar.api.match_latency                  10 dev resumes against http://127.0.0.1:8000
    python -m radar.api.match_latency --n 10 --salt s2

The resumes are the synthetic dev resumes of the matching evaluation (data/matching/queries.parquet),
ordered by a salted hash of their id; one more goes first as a warm-up and is not counted. /match is
not cached, so every request encodes the resume, scores the pool and reads the open vacancies from
ClickHouse. Memory is read from the container's cgroup after the run: memory.current is what
mem_limit is enforced against, memory.peak the most it has held since it started. Writes
results/api/match_latency.json.
"""

import argparse
import hashlib
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from typing import Any

import httpx
import pandas as pd

from radar.api.latency import docker_host, nearest_rank, vm_loadavg
from radar.models.matching_data import DEV
from radar.models.matching_embed import work_dir
from radar.results import write_result

log = logging.getLogger("radar.api.match_latency")

RESULT = "api/match_latency.json"
SERVICE = "api"
IMAGE = "radar/api:0.1.0"
MIB = 1 << 20


def sample(queries: pd.DataFrame, n: int, salt: str) -> pd.DataFrame:
    rows = queries[queries["split"] == DEV].copy()
    rows["order"] = [hashlib.sha256(f"{salt}|{q}".encode()).hexdigest() for q in rows["query_id"]]
    return rows.sort_values("order").head(n).reset_index(drop=True)


def measure(client: httpx.Client, resumes: pd.DataFrame, k: int) -> list[dict[str, Any]]:
    """The first resume warms the server up and is not returned."""
    calls = []
    for i, row in enumerate(resumes.itertuples()):
        started = time.perf_counter()
        response = client.post("/match", json={"resume_text": str(row.resume), "k": k})
        seconds = time.perf_counter() - started
        response.raise_for_status()
        if i == 0:
            continue
        items = response.json()["items"]
        calls.append(
            {
                "query_id": row.query_id,
                "resume_chars": len(str(row.resume)),
                "items": len(items),
                "ms": round(seconds * 1000, 1),
            }
        )
        log.info("%d/%d %s %.0f ms", i, len(resumes) - 1, row.query_id, seconds * 1000)
    return calls


def cgroup_memory(container: str) -> dict[str, int]:
    """Bytes of the container's cgroup v2: current, peak, limit and anonymous (heap) memory."""
    out = docker_host(
        "docker", "exec", container, "cat",
        "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.stat",
    )  # fmt: skip
    current, peak, limit, *stat = out.splitlines()
    fields = dict(line.split(" ", 1) for line in stat)
    return {"current": int(current), "peak": int(peak), "limit": int(limit), "anon": int(fields["anon"])}


def summary(calls: Sequence[dict[str, Any]], memory: dict[str, int], image_bytes: int) -> dict[str, dict[str, Any]]:
    ms = [c["ms"] for c in calls]
    return {
        "match_latency_p50_ms": {"value": round(nearest_rank(ms, 0.5), 1)},
        "match_latency_p95_ms": {"value": round(nearest_rank(ms, 0.95), 1)},
        "match_latency_max_ms": {"value": round(max(ms), 1)},
        "match_latency_mean_ms": {"value": round(statistics.fmean(ms), 1)},
        "requests": {"value": len(calls)},
        "container_memory_current_mib": {"value": round(memory["current"] / MIB, 1)},
        "container_memory_peak_mib": {"value": round(memory["peak"] / MIB, 1)},
        "container_memory_anon_mib": {"value": round(memory["anon"] / MIB, 1)},
        "container_memory_limit_mib": {"value": round(memory["limit"] / MIB, 1)},
        "image_size_mb": {"value": round(image_bytes / 1e6, 1)},
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m radar.api.match_latency", description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the API root")
    ap.add_argument("--n", type=int, default=10, help="timed requests (one more warms up)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--salt", default="match-latency-v1", help="orders the sample; change it to draw another")
    args = ap.parse_args(argv)

    queries = pd.read_parquet(work_dir() / "queries.parquet", columns=["query_id", "split", "resume"])
    resumes = sample(queries, args.n + 1, args.salt)
    container = docker_host("docker", "ps", "-q", "--filter", f"label=com.docker.compose.service={SERVICE}")
    if not container:
        log.error("no running %s container", SERVICE)
        return 1
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        health = client.get("/health").json()
        load_before = vm_loadavg()
        calls = measure(client, resumes, args.k)
        load_after = vm_loadavg()
    memory = cgroup_memory(container)
    image_bytes = int(docker_host("docker", "image", "inspect", IMAGE, "--format", "{{.Size}}"))
    path = write_result(
        RESULT,
        summary(calls, memory, image_bytes),
        "src/radar/api/match_latency.py",
        api_url=args.url,
        salt=args.salt,
        k=args.k,
        sample="synthetic dev resumes of data/matching/queries.parquet",
        warmup_requests=1,
        concurrency=1,
        quantiles="nearest rank over the timed requests",
        memory_source="cgroup v2 of the api container, read after the timed requests",
        image=IMAGE,
        api_health=health,
        docker_host_loadavg={"before": load_before, "after": load_after, "cpus": int(docker_host("nproc"))},
        calls=calls,
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
