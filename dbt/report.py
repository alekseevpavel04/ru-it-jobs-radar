"""results/dbt/build.json: what the last `dbt build` did and what staging holds.

Reads dbt/target/build_results.json (kept by scripts/dbt.sh after each build) and, when present,
dbt/target/sources.json of the last freshness check; counts rows in ClickHouse. Run it through the
wrapper right after a build:

    bash scripts/dbt.sh build && bash scripts/dbt.sh source freshness && bash scripts/dbt.sh report
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from radar.config import ROOT, load_dotenv
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse

TARGET = ROOT / "dbt" / "target"
STAGING = ("stg_trudvsem_api", "stg_trudvsem_snapshot", "stg_habr", "stg_vacancies")


def share(n: int, denominator: int) -> dict[str, Any]:
    return {"value": round(n / denominator, 4) if denominator else 0, "n": n, "denominator": denominator}


def rows(ch: ClickHouse, sql: str) -> list[list[str]]:
    return [line.split("\t") for line in ch.query(sql + " FORMAT TSV").splitlines() if line]


def build_metrics(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kinds: dict[str, Counter[str]] = {"model": Counter(), "seed": Counter(), "test": Counter()}
    for result in run["results"]:
        kind = result["unique_id"].split(".", 1)[0]
        if kind in kinds:
            kinds[kind][result["status"]] += 1
    models, seeds, tests = kinds["model"], kinds["seed"], kinds["test"]
    return {
        "models_ok": share(models["success"], sum(models.values())),
        "seeds_ok": share(seeds["success"], sum(seeds.values())),
        "tests_passed": share(tests["pass"], sum(tests.values())),
        "tests_warned": {"value": tests["warn"], "denominator": sum(tests.values()), "n": tests["warn"]},
        "tests_failed": {
            "value": tests["fail"] + tests["error"],
            "n": tests["fail"] + tests["error"],
            "denominator": sum(tests.values()),
        },
        "build_seconds": {"value": round(run["elapsed_time"], 1)},
    }


def freshness_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for result in doc["results"]:
        name = result["unique_id"].rsplit(".", 1)[-1]
        out[f"freshness_{name}"] = {
            "value": result["status"],
            "hours_since_last_fetch": round(result.get("max_loaded_at_time_ago_in_s", 0) / 3600, 1),
            "checked_at": doc["metadata"]["generated_at"],
        }
    return out


def staging_metrics(ch: ClickHouse) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model in STAGING:
        for source, n in rows(ch, f"SELECT source, count() FROM staging.{model} GROUP BY source ORDER BY source"):
            out[f"rows_{model}_{source}"] = {"value": int(n)}
    for dt, it, total in rows(
        ch,
        "SELECT dt, countIf(is_it_candidate), count() FROM staging.stg_trudvsem_snapshot GROUP BY dt ORDER BY dt",
    ):
        out[f"it_candidates_trudvsem_snapshot_{dt}"] = share(int(it), int(total))
    for dt, it, total in rows(
        ch, "SELECT dt, countIf(is_it_candidate), count() FROM staging.stg_trudvsem_api GROUP BY dt ORDER BY dt"
    ):
        out[f"it_candidates_trudvsem_api_{dt}"] = share(int(it), int(total))
    for dt, future, total in rows(
        ch,
        "SELECT dt, countIf(has_future_published_at), countIf(has_card) FROM staging.stg_habr GROUP BY dt ORDER BY dt",
    ):
        out[f"habr_future_date_posted_{dt}"] = share(int(future), int(total))
    return out


def main() -> int:
    build_path = TARGET / "build_results.json"
    if not build_path.exists():
        print("no dbt/target/build_results.json; run `bash scripts/dbt.sh build` first", file=sys.stderr)
        return 1
    run = json.loads(build_path.read_text(encoding="utf-8"))
    load_dotenv()
    metrics = build_metrics(run) | freshness_metrics(TARGET / "sources.json")
    with ClickHouse() as ch:
        metrics |= staging_metrics(ch)
    path = write_result(
        "dbt/build.json",
        metrics,
        source_script="dbt/report.py",
        dbt_version=run["metadata"]["dbt_version"],
        build_finished_at=run["metadata"]["generated_at"],
        notes={
            "it_candidates": "wide IT filter is_it_candidate, not measured against labels yet",
            "habr_future_date_posted": "cards whose datePosted is later than the fetch, of cards read that day",
        },
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
