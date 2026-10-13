"""Spark against ClickHouse on one history computation of the project.

The trudvsem snapshot history (history.py) runs twice from the same raw data:
  Spark       local mode over the Parquet lake data/raw/trudvsem_snapshot (one file set per day)
  ClickHouse  the SQL dbt compiles for the same models (dbt/target/compiled), over raw.vacancies:
              stg_trudvsem_snapshot + hist_snapshot_days one snapshot day per query, as dbt runs
              them, then hist_vacancy_scd2 and hist_vacancy_lifetime for the snapshot source
Both write their outputs to Parquet in the work directory; Spark then compares them row by row
(exceptAll both ways) and against the tables dbt built (staging.hist_*), so equality is measured,
not assumed. Wall time per stage and peak memory go to results/spark/bench.json (probe.json for a
run on part of the days): resident memory of the Spark driver process tree sampled every 0.2 s,
memory_usage of each ClickHouse query from system.query_log.

Runs in WSL with Java and the spark venv (docs/methods/spark.md):
  PYTHONPATH=src ~/venvs/spark/bin/python jobs/spark/bench.py --days 1     # probe on the first day
  PYTHONPATH=src ~/venvs/spark/bin/python jobs/spark/bench.py              # every snapshot day
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from base64 import b64encode
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import history
import psutil
import pyspark
import rules
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from radar.config import load_dotenv, require
from radar.results import write_result

ROOT = rules.ROOT
COMPILED = ROOT / "dbt" / "target" / "compiled" / "radar" / "models"
LAKE = ROOT / "data" / "raw" / "trudvsem_snapshot"
# Right-only rows whose vacancies are listed and looked up in raw; more means a real disagreement.
MAX_LISTED = 1000
# Macro files the compared models expand (dbt/macros); the materialization is not part of the SQL.
MODEL_MACROS = ("history", "parsing", "staging")
SNAPSHOT_FILTER = f"(SELECT * FROM {{table}} WHERE source = '{rules.SOURCE}')"

OBSERVATION_COLUMNS = [
    "source_id",
    "dt",
    "content_hash",
    "is_closed_state",
    "title",
    "title_norm",
    "employer_name",
    "employer_key",
    "location_key",
    "salary_from",
    "salary_to",
    "salary_currency_code",
    "schedule",
    "employment",
    "status",
    "skills",
    "published_at",
]
SCD2_COLUMNS = [
    "source_id",
    "version_no",
    "valid_from",
    "valid_to",
    "is_current",
    "change_reason",
    "ended_by",
    "closed_in_gap",
    "changed_fields",
    "last_seen",
    "days_seen",
    "content_hash",
    "title",
    "employer_name",
    "salary_from",
    "salary_to",
    "salary_currency_code",
    "schedule",
    "employment",
    "status",
    "skills",
]
LIFETIME_COLUMNS = [
    "source_id",
    "first_seen",
    "last_seen",
    "is_closed",
    "closed_in_gap",
    "closed_at",
    "is_censored",
    "is_left_censored",
    "duration_days",
    "n_versions",
    "n_reopened",
    "n_salary_changes",
]


class ClickHouse:
    """HTTP client with a query_id per statement, so its memory can be read from system.query_log."""

    def __init__(self) -> None:
        self.url = require("CLICKHOUSE_URL").rstrip("/") + "/"
        token = b64encode(f"{require('CLICKHOUSE_USER')}:{require('CLICKHOUSE_PASSWORD')}".encode()).decode()
        self.auth = f"Basic {token}"
        self.query_ids: dict[str, list[str]] = {}

    def _open(self, sql: str, settings: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.url + "?" + urllib.parse.urlencode({k: str(v) for k, v in settings.items()}),
            data=sql.encode("utf-8"),
            headers={"Authorization": self.auth},
        )
        try:
            return urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ClickHouse {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}") from e

    def text(self, sql: str) -> str:
        with self._open(sql, {}) as response:
            return str(response.read().decode("utf-8"))

    def to_file(self, sql: str, path: Path, stage: str | None = None, **settings: Any) -> None:
        """Runs one statement with FORMAT Parquet into path; a staged one is counted in stage's memory."""
        query_id = f"spark-bench-{uuid.uuid4().hex}"
        if stage:
            self.query_ids.setdefault(stage, []).append(query_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            self._open(sql + "\nFORMAT Parquet", {"query_id": query_id, **settings}) as response,
            path.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, 1 << 20)

    def peak_memory(self, stage: str) -> dict[str, int]:
        """Largest memory_usage and summed server time of the stage's queries, once query_log has them all."""
        ids = self.query_ids[stage]
        listed = ", ".join(f"'{i}'" for i in ids)
        sql = (
            "SELECT count(), max(memory_usage), sum(query_duration_ms) FROM system.query_log "
            f"WHERE type = 'QueryFinish' AND query_id IN ({listed}) FORMAT TSV"
        )
        for _ in range(60):
            found, memory, duration = (int(v) for v in self.text(sql).split())
            if found == len(ids):
                return {"queries": found, "peak_memory_bytes": memory, "server_time_ms": duration}
            time.sleep(1)
        raise RuntimeError(f"system.query_log has {found} of {len(ids)} queries of {stage} after a minute")


def compiled(model: str) -> str:
    """SQL dbt compiled for a model; stale against the model or its macros means `dbt compile` first."""
    folder = "staging" if model.startswith("stg_") else "history"
    path = COMPILED / folder / f"{model}.sql"
    sources = [ROOT / "dbt" / "models" / folder / f"{model}.sql"]
    sources += [ROOT / "dbt" / "macros" / f"{name}.sql" for name in MODEL_MACROS]
    newest = max(p.stat().st_mtime for p in sources)
    if not path.exists() or path.stat().st_mtime < newest:
        raise SystemExit(f"{path} is missing or older than its model or macros: run `dbt compile` in dbt/")
    return path.read_text(encoding="utf-8")


def snapshot_only(sql: str, *tables: str) -> str:
    """Restrict the inputs of a compiled history model to the snapshot source."""
    for table in tables:
        quoted = f"`staging`.`{table}`"
        if quoted not in sql:
            raise ValueError(f"{quoted} is not referenced by the compiled SQL")
        sql = sql.replace(quoted, SNAPSHOT_FILTER.format(table=quoted))
    return sql


class MemorySampler:
    """Peak resident memory of this process and its children (the JVM), sampled in a thread."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()

    def _rss(self) -> int:
        total = 0
        for p in [self.process, *self.process.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval_s)

    @contextmanager
    def measure(self) -> Iterator["MemorySampler"]:
        self.peak = self._rss()
        self._stop.clear()
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            self._stop.set()
            thread.join()


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 2)


def snapshot_days(lake: Path, limit: int | None) -> list[str]:
    days = sorted(p.name.removeprefix("dt=") for p in lake.glob("dt=*") if p.is_dir())
    return days[:limit] if limit else days


def spark_session(cores: int, driver_memory: str, workdir: Path) -> SparkSession:
    return (
        SparkSession.builder.master(f"local[{cores}]")
        .appName("radar-history-bench")
        .config("spark.driver.memory", driver_memory)
        .config("spark.local.dir", str(workdir / "spark-tmp"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(cores * 4))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def diff(left: DataFrame, right: DataFrame, columns: list[str]) -> dict[str, Any]:
    """Rows of each side and the rows (with multiplicity) that the other side does not have.

    The vacancies of the right-only rows are listed when there are few, so they can be looked up
    in the raw layer (compare does).
    """
    types = dict(left.select(*columns).dtypes)
    a = left.select(*columns)
    b = right.select(*[F.col(c).cast(types[c]).alias(c) for c in columns])
    a.cache()
    b.cache()
    only_right = b.exceptAll(a)
    out: dict[str, Any] = {
        "rows_left": a.count(),
        "rows_right": b.count(),
        "only_left": a.exceptAll(b).count(),
        "only_right": only_right.count(),
    }
    if 0 < out["only_right"] <= MAX_LISTED:
        out["only_right_source_ids"] = [r.source_id for r in only_right.select("source_id").collect()]
    a.unpersist()
    b.unpersist()
    return out


def with_has_text(df: DataFrame) -> DataFrame:
    """text_hash differs by hash function between the engines; whether there is a text does not."""
    return df.withColumn("has_text", F.col("text_hash").isNotNull())


def run_spark(spark: SparkSession, lake: Path, days: list[str], all_days: bool, out: Path) -> dict[str, Any]:
    text = rules.macros()
    pattern, history_rules, positions = (
        rules.it_title_pattern(text),
        rules.history_rules(text),
        rules.it_typical_positions(),
    )
    sampler = MemorySampler()
    raw = history.read_snapshot(spark, lake, None if all_days else days)

    with sampler.measure() as m:
        _, obs_s = timed(
            lambda: (
                history.observations(raw, pattern, positions).write.mode("overwrite").parquet(str(out / "observations"))
            )
        )
    obs_peak = m.peak
    with sampler.measure() as m:
        obs = spark.read.parquet(str(out / "observations"))
        cal = history.calendar(obs, history_rules)

        def build() -> None:
            versions = history.scd2(obs, cal, history_rules)
            versions.write.mode("overwrite").parquet(str(out / "scd2"))
            history.lifetime(spark.read.parquet(str(out / "scd2")), cal).write.mode("overwrite").parquet(
                str(out / "lifetime")
            )

        _, hist_s = timed(build)
    hist_peak = m.peak

    # Untimed: the volume the observations step read, and how many vacancy-days its dedup resolved.
    volume = (
        raw.groupBy("dt", "source_id")
        .count()
        .agg(F.sum("count").alias("rows"), F.count(F.when(F.col("count") > 1, 1)).alias("dup_keys"))
        .first()
    )
    assert volume is not None
    return {
        "rules": history_rules,
        "raw_rows": int(volume["rows"]),
        "duplicate_vacancy_days": int(volume["dup_keys"]),
        "observations_s": obs_s,
        "history_s": hist_s,
        "observations_peak_rss_bytes": obs_peak,
        "history_peak_rss_bytes": hist_peak,
    }


def run_clickhouse(ch: ClickHouse, days: list[str], all_days: bool, out: Path, threads: int) -> dict[str, Any]:
    stg = compiled("stg_trudvsem_snapshot")
    per_day = compiled("hist_snapshot_days").replace("`staging`.`stg_trudvsem_snapshot`", f"(\n{stg}\n)")
    # max_block_size as in the dbt config of the snapshot models: it keeps a day's dedup under the
    # server budget; the thread count is the same as Spark's cores.
    settings = {"max_threads": threads, "max_block_size": 8192}

    def observations() -> None:
        for day in days:
            ch.to_file(
                per_day.replace("__staging_day__", day),
                out / "observations" / f"{day}.parquet",
                "observations",
                **settings,
            )

    _, obs_s = timed(observations)
    result: dict[str, Any] = {"observations_s": obs_s, "observations_queries": ch.peak_memory("observations")}
    if all_days:
        scd2 = snapshot_only(compiled("hist_vacancy_scd2"), "hist_observations", "hist_calendar")
        lifetime = snapshot_only(
            compiled("hist_vacancy_lifetime"), "hist_calendar", "hist_vacancy_scd2", "hist_reposts"
        )

        def build() -> None:
            ch.to_file(scd2, out / "scd2" / "part.parquet", "history", **settings)
            ch.to_file(lifetime, out / "lifetime" / "part.parquet", "history", **settings)

        _, hist_s = timed(build)
        result |= {"history_s": hist_s, "history_queries": ch.peak_memory("history")}

    # The tables dbt built, for a third side of the comparison (not timed).
    listed = ", ".join(f"'{d}'" for d in days)
    ch.to_file(
        f"SELECT * EXCEPT ingested_at FROM staging.hist_snapshot_days WHERE dt IN ({listed})",
        out / "dbt_observations" / "part.parquet",
    )
    if all_days:
        for table in ("hist_vacancy_scd2", "hist_vacancy_lifetime"):
            ch.to_file(
                f"SELECT * FROM staging.{table} WHERE source = '{rules.SOURCE}'",
                out / f"dbt_{table.removeprefix('hist_vacancy_')}" / "part.parquet",
            )
    result["version"] = ch.text("SELECT version()").strip()
    return result


def compare(spark: SparkSession, work: Path, raw: DataFrame, all_days: bool) -> dict[str, dict[str, Any]]:
    def read(side: str, name: str) -> DataFrame:
        return spark.read.parquet(str(work / side / name))

    obs_cols = [*OBSERVATION_COLUMNS, "has_text"]
    spark_obs = with_has_text(read("spark", "observations"))
    out = {
        "observations_spark_vs_clickhouse": diff(
            spark_obs, with_has_text(read("clickhouse", "observations")), obs_cols
        ),
        "observations_spark_vs_dbt": diff(spark_obs, with_has_text(read("clickhouse", "dbt_observations")), obs_cols),
    }
    if all_days:
        out |= {
            "scd2_spark_vs_clickhouse": diff(read("spark", "scd2"), read("clickhouse", "scd2"), SCD2_COLUMNS),
            "scd2_clickhouse_vs_dbt": diff(read("clickhouse", "scd2"), read("clickhouse", "dbt_scd2"), SCD2_COLUMNS),
            "lifetime_spark_vs_clickhouse": diff(
                read("spark", "lifetime"), read("clickhouse", "lifetime"), LIFETIME_COLUMNS
            ),
        }
    # A right-only row of a vacancy that the raw layer never had is not an engine difference: the
    # right side read something that is not in raw (the dbt tables can hold rows raw lost since).
    listed = sorted({i for d in out.values() for i in d.get("only_right_source_ids", [])})
    in_raw = {r.source_id for r in raw.select("source_id").where(F.col("source_id").isin(listed)).distinct().collect()}
    for d in out.values():
        if "only_right_source_ids" in d:
            d["only_right_not_in_raw"] = sum(i not in in_raw for i in d["only_right_source_ids"])
    return out


def metrics(spark_run: dict[str, Any], ch_run: dict[str, Any], diffs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mib = 1 << 20
    out: dict[str, Any] = {
        "raw_rows": {"value": spark_run["raw_rows"]},
        "duplicate_vacancy_days": {"value": spark_run["duplicate_vacancy_days"]},
        "spark_observations_s": {"value": spark_run["observations_s"]},
        "spark_history_s": {"value": spark_run["history_s"]},
        "spark_peak_rss_mib": {
            "value": round(max(spark_run["observations_peak_rss_bytes"], spark_run["history_peak_rss_bytes"]) / mib)
        },
        "clickhouse_observations_s": {"value": ch_run["observations_s"]},
        "clickhouse_peak_query_memory_mib": {
            "value": round(
                max(ch_run[k]["peak_memory_bytes"] for k in ("observations_queries", "history_queries") if k in ch_run)
                / mib
            )
        },
    }
    if "history_s" in ch_run:
        out["clickhouse_history_s"] = {"value": ch_run["history_s"]}
    for name, d in diffs.items():
        matched = d["rows_left"] - d["only_left"]
        out[f"{name}.rows_left"] = {"value": d["rows_left"]}
        out[f"{name}.rows_right"] = {"value": d["rows_right"]}
        out[f"{name}.matched"] = {
            "value": round(matched / d["rows_right"], 6) if d["rows_right"] else 0.0,
            "n": matched,
            "denominator": d["rows_right"],
        }
        out[f"{name}.only_left"] = {"value": d["only_left"]}
        out[f"{name}.only_right"] = {"value": d["only_right"]}
        if "only_right_not_in_raw" in d:
            out[f"{name}.only_right_not_in_raw"] = {"value": d["only_right_not_in_raw"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, help="first N snapshot days only (a probe); default every day")
    parser.add_argument("--cores", type=int, default=4, help="Spark local cores and ClickHouse max_threads")
    parser.add_argument("--driver-memory", default="2g")
    parser.add_argument("--workdir", type=Path, default=Path.home() / "spark-work" / "history")
    parser.add_argument("--keep", action="store_true", help="keep the Parquet outputs of both engines")
    parser.add_argument("--lake", type=Path, default=LAKE, help="snapshot Parquet root with dt=... folders")
    parser.add_argument("--result", help="file name under results/spark/; default bench.json, probe.json with --days")
    args = parser.parse_args()

    load_dotenv()
    days = snapshot_days(args.lake, args.days)
    all_days = args.days is None
    if args.workdir.exists():
        shutil.rmtree(args.workdir)

    ch = ClickHouse()
    ch_run = run_clickhouse(ch, days, all_days, args.workdir / "clickhouse", args.cores)
    print(json.dumps(ch_run, indent=1), flush=True)

    spark = spark_session(args.cores, args.driver_memory, args.workdir)
    spark_run = run_spark(spark, args.lake, days, all_days, args.workdir / "spark")
    print(json.dumps(spark_run, indent=1), flush=True)
    raw = history.read_snapshot(spark, args.lake, None if all_days else days)
    diffs = compare(spark, args.workdir, raw, all_days)
    print(json.dumps(diffs, indent=1), flush=True)
    spark.stop()

    raw_bytes = sum(f.stat().st_size for d in days for f in (args.lake / f"dt={d}").glob("*.parquet"))
    name = args.result or ("bench.json" if all_days else "probe.json")
    path = write_result(
        f"spark/{name}",
        metrics(spark_run, ch_run, diffs),
        source_script="jobs/spark/bench.py",
        setup={
            "snapshot_days": len(days),
            "first_day": days[0],
            "last_day": days[-1],
            "raw_parquet_bytes": raw_bytes,
            "lake": str(args.lake),
            "cores": args.cores,
            "spark_driver_memory": args.driver_memory,
            "spark": pyspark.__version__,
            "java": os.environ.get("JAVA_HOME", ""),
            "clickhouse": ch_run["version"],
            "history_rules": spark_run["rules"],
        },
        clickhouse=ch_run,
        spark=spark_run,
        diffs=diffs,
        notes=(
            "Spark reads Parquet from the Windows disk through WSL (/mnt/d); ClickHouse reads its own "
            "MergeTree on the WSL disk. Memory: Spark is the peak resident set of the driver process "
            "tree, ClickHouse the largest memory_usage of one query in system.query_log. text_hash is "
            "compared as presence only (xxhash64 in Spark, cityHash64 in ClickHouse)."
        ),
    )
    if not args.keep:
        shutil.rmtree(args.workdir)
    print(path, file=sys.stderr)


if __name__ == "__main__":
    main()
