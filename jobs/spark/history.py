"""The trudvsem snapshot history of dbt (models/history) in PySpark, for the benchmark in bench.py.

Three steps mirror three dbt models for the source trudvsem_snapshot:
  observations  stg_trudvsem_snapshot + hist_snapshot_days: the latest row of each vacancy per
                snapshot day, its payload parsed, IT candidates only
  scd2          hist_calendar + hist_vacancy_scd2: versions with validity, reasons and closures
  lifetime      hist_vacancy_lifetime without the repost columns (hist_reposts is not mirrored)
The rules (grace days, complete-day ratio, IT filter) are read from the dbt macros where they are
data, so a change there reaches this file; the SQL itself is translated by hand, which is why
bench.py compares every output row with what ClickHouse computes.

One deliberate difference: text_hash is xxhash64 here and cityHash64 in ClickHouse (Spark has no
cityHash64). The value is only compared for equality between versions, so changed_fields agree.
"""

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from rules import SALARY_FIELDS, SOURCE, TRACKED_FIELDS

PAYLOAD = T.StructType(
    [
        T.StructField("codeProfessionalSphere", T.StringType()),
        T.StructField("typicalPosition", T.StringType()),
        T.StructField("fullCompanyName", T.StringType()),
        T.StructField("company", T.StructType([T.StructField("inn", T.StringType())])),
        T.StructField("regionName", T.StringType()),
        T.StructField("salaryMin", T.StringType()),
        T.StructField("salaryMax", T.StringType()),
        T.StructField("scheduleType", T.StringType()),
        T.StructField("busyType", T.StringType()),
        T.StructField("status", T.StringType()),
        T.StructField("deleted", T.StringType()),
        T.StructField("skills", T.ArrayType(T.StringType())),
    ]
)


def read_snapshot(spark: SparkSession, root: Path, days: list[str] | None = None) -> DataFrame:
    """Raw snapshot rows from the Parquet lake (data/raw/trudvsem_snapshot/dt=...), all days or some."""
    # The root itself for every day: partition discovery skips _meta and _rejects.
    paths = [str(root / f"dt={d}") for d in days] if days else [str(root)]
    return (
        spark.read.option("basePath", str(root))
        .parquet(*paths)
        .where(F.col("source") == SOURCE)
        .select("source_id", "dt", "fetched_at", "content_hash", "title", "text", "payload", "published_at")
    )


def _json_string(field: Column) -> Column:
    return F.when(field != "", field)


# Macro stated_amount: 0 means "not stated", so it becomes NULL like an unparsable amount.
def _stated_amount(field: Column) -> Column:
    value = field.try_cast("double")
    return F.when(value != 0, value)


def observations(raw: DataFrame, title_pattern: str, positions: list[str]) -> DataFrame:
    """hist_snapshot_days: one row per (source_id, dt) of the IT candidates, the latest row of the day.

    ClickHouse keeps the latest fetched_at per source_id and day (FINAL, then LIMIT 1 BY). Finding
    the latest needs every row of the day, so that part runs on a narrow projection (a 64-bit key
    and fetched_at); the wide parsed columns travel only for the IT candidates, about 2% of rows.
    """
    p = F.from_json("payload", PAYLOAD)
    rows = raw.withColumn("key", F.xxhash64("source_id")).withColumn("p", p)
    sphere = _json_string(F.col("p.codeProfessionalSphere"))
    position = _json_string(F.col("p.typicalPosition"))
    is_it = F.coalesce(
        (sphere == "InformationTechnology")
        | position.isin(positions)
        | F.lower(F.coalesce("title", F.lit(""))).rlike(title_pattern),
        F.lit(False),
    )

    latest = (
        raw.select("dt", F.xxhash64("source_id").alias("key"), "fetched_at")
        .groupBy("dt", "key")
        .agg(F.count(F.lit(1)).alias("n"), F.max("fetched_at").alias("max_fetched_at"))
        .where(F.col("n") > 1)
    )

    employer_name = _json_string(F.col("p.fullCompanyName"))
    status = _json_string(F.col("p.status"))
    salary_from = _stated_amount(F.col("p.salaryMin"))
    salary_to = _stated_amount(F.col("p.salaryMax"))
    text = F.col("text")
    candidates = rows.where(is_it).select(
        "key",
        "source_id",
        "dt",
        "fetched_at",
        "content_hash",
        (
            F.coalesce(F.col("p.deleted") == "true", F.lit(False))
            | F.coalesce(status.isin("Не опубликовано", "Не одобрено"), F.lit(False))
        ).alias("is_closed_state"),
        "title",
        F.trim(
            F.regexp_replace(
                F.replace(F.lower(F.coalesce("title", F.lit(""))), F.lit("ё"), F.lit("е")), "[^a-zа-я0-9+#]+", " "
            )
        ).alias("title_norm"),
        employer_name.alias("employer_name"),
        F.coalesce(_json_string(F.col("p.company.inn")), F.when(employer_name != "ИП", employer_name)).alias(
            "employer_key"
        ),
        _json_string(F.col("p.regionName")).alias("location_key"),
        salary_from.alias("salary_from"),
        salary_to.alias("salary_to"),
        F.when(salary_from.isNotNull() | salary_to.isNotNull(), F.lit("RUB")).alias("salary_currency_code"),
        _json_string(F.col("p.scheduleType")).alias("schedule"),
        _json_string(F.col("p.busyType")).alias("employment"),
        F.lit(None).cast("boolean").alias("remote_work"),
        status.alias("status"),
        F.array_sort(F.coalesce("p.skills", F.array().cast("array<string>"))).alias("skills"),
        F.when(
            F.length(F.trim(F.coalesce(text, F.lit("")))) > 0,
            F.xxhash64(F.trim(F.regexp_replace(F.lower(text), r"\s+", " "))),
        ).alias("text_hash"),
        "published_at",
    )
    # A vacancy with several rows on one day counts only through its latest one; when two rows share
    # the latest fetched_at, ClickHouse keeps either, here the larger content_hash.
    tie = Window.partitionBy("dt", "key").orderBy(F.col("fetched_at").desc(), F.col("content_hash").desc())
    return (
        candidates.join(F.broadcast(latest), ["dt", "key"], "left")
        .where(F.col("n").isNull() | (F.col("fetched_at") == F.col("max_fetched_at")))
        .withColumn("_rank", F.row_number().over(tie))
        .where(F.col("_rank") == 1)
        .drop("key", "n", "max_fetched_at", "_rank", "fetched_at")
    )


def calendar(obs: DataFrame, rules: dict[str, float]) -> DataFrame:
    """hist_calendar for the snapshot source: next complete day and the day a closure is confirmed."""
    days = obs.groupBy("dt").agg(F.count(F.lit(1)).alias("n_rows"))
    median = days.agg(F.percentile("n_rows", 0.5).alias("median_rows"))
    flagged = days.crossJoin(median).withColumn(
        "is_complete", F.col("n_rows") >= F.lit(rules["complete_day_ratio"]) * F.col("median_rows")
    )
    complete = flagged.where("is_complete").agg(F.array_sort(F.collect_list("dt")).alias("complete_days"))
    absent = int(rules["absent_days_to_close"])
    later = F.filter("complete_days", lambda d: d > F.col("dt"))
    return (
        flagged.crossJoin(complete)
        .withColumn("later_complete", later)
        .select(
            "dt",
            "n_rows",
            "median_rows",
            "is_complete",
            F.when(F.size("later_complete") >= 1, F.element_at("later_complete", 1)).alias("next_complete_dt"),
            F.when(F.size("later_complete") >= absent, F.element_at("later_complete", absent)).alias(
                "closure_confirmed_dt"
            ),
        )
    )


def scd2(obs: DataFrame, cal: DataFrame, rules: dict[str, float]) -> DataFrame:
    """hist_vacancy_scd2: one row per version of a vacancy, valid_to exclusive."""
    by_day = Window.partitionBy("source_id").orderBy("dt")
    sightings = obs.select("source_id", "dt", "content_hash", "is_closed_state").join(
        F.broadcast(cal.select("dt", "next_complete_dt", "closure_confirmed_dt")), "dt"
    )
    sequenced = sightings.withColumn("next_dt", F.lead("dt").over(by_day)).withColumn(
        "next_is_closed_state", F.lead("is_closed_state").over(by_day)
    )
    open_sightings = sequenced.where(~F.col("is_closed_state")).withColumn("next_open_dt", F.lead("dt").over(by_day))

    next_open, confirmed = F.col("next_open_dt"), F.col("closure_confirmed_dt")
    closed_at = (
        F.when(next_open.isNotNull() & (confirmed.isNull() | (next_open <= confirmed)), F.lit(None))
        .when(
            F.coalesce("next_is_closed_state", F.lit(False)),
            F.least("next_dt", F.coalesce("next_complete_dt", "next_dt")),
        )
        .when(confirmed.isNotNull(), F.col("next_complete_dt"))
        .cast("date")
    )
    active = open_sightings.withColumn("closed_at", closed_at).withColumn(
        "closed_in_gap",
        F.col("closed_at").isNotNull() & (F.datediff("closed_at", "dt") > F.lit(int(rules["max_step_days"]))),
    )
    reasoned = active.withColumn(
        "change_reason",
        F.when(F.row_number().over(by_day) == 1, "new")
        .when(F.lag("closed_at").over(by_day).isNotNull(), "reopened")
        .when(F.col("content_hash") != F.lag("content_hash").over(by_day), "changed")
        .otherwise(""),
    )
    running = by_day.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    numbered = reasoned.withColumn(
        "version_no", F.sum((F.col("change_reason") != "").cast("int")).over(running).cast("int")
    )
    # The first row of a version names its reason and only the last can carry a closure.
    spans = numbered.groupBy("source_id", "version_no").agg(
        F.min("dt").alias("valid_from"),
        F.max("dt").alias("last_seen"),
        F.count(F.lit(1)).cast("int").alias("days_seen"),
        F.min(F.struct("dt", "change_reason")).change_reason.alias("change_reason"),
        F.max(F.struct("dt", "closed_at", "closed_in_gap")).alias("_last"),
    )
    attributes = obs.select("source_id", F.col("dt").alias("valid_from"), "content_hash", *TRACKED_FIELDS)
    versions = spans.join(attributes, ["source_id", "valid_from"]).select(
        "*", F.col("_last.closed_at").alias("closed_at"), F.col("_last.closed_in_gap").alias("closed_in_gap")
    )

    by_version = Window.partitionBy("source_id").orderBy("version_no")
    prev = F.lag(F.struct("content_hash", *TRACKED_FIELDS)).over(by_version)
    neighbours = versions.withColumn("next_valid_from", F.lead("valid_from").over(by_version)).withColumn("prev", prev)
    named = F.array_compact(
        F.array(*[F.when(~F.col(f).eqNullSafe(F.col(f"prev.{f}")), F.lit(f)) for f in TRACKED_FIELDS])
    )
    changes = F.when(F.col("version_no") == 1, F.array().cast("array<string>")).otherwise(named)
    closed = F.col("closed_at")
    return neighbours.withColumn("named_changes", changes).select(
        "source_id",
        "version_no",
        "valid_from",
        F.coalesce(closed, F.col("next_valid_from")).alias("valid_to"),
        F.coalesce(closed, F.col("next_valid_from")).isNull().alias("is_current"),
        "change_reason",
        F.when(closed.isNotNull(), "closed").when(F.col("next_valid_from").isNotNull(), "changed").alias("ended_by"),
        "closed_in_gap",
        F.when(
            (F.col("version_no") > 1)
            & (F.size("named_changes") == 0)
            & (F.col("content_hash") != F.col("prev.content_hash")),
            F.array(F.lit("other")),
        )
        .otherwise(F.col("named_changes"))
        .alias("changed_fields"),
        "last_seen",
        "days_seen",
        "content_hash",
        *TRACKED_FIELDS,
    )


def lifetime(versions: DataFrame, cal: DataFrame) -> DataFrame:
    """hist_vacancy_lifetime for the snapshot source, without the repost columns."""
    first_collected = cal.agg(F.min("dt").alias("first_collected_dt"))
    salary_changed = F.arrays_overlap("changed_fields", F.array(*[F.lit(f) for f in SALARY_FIELDS]))
    per_vacancy = versions.groupBy("source_id").agg(
        F.min("valid_from").alias("first_seen"),
        F.max("last_seen").alias("last_seen"),
        (F.count(F.when(F.col("is_current"), 1)) == 0).alias("is_closed"),
        F.max(F.struct("version_no", "valid_to", "closed_in_gap")).alias("_last"),
        F.count(F.lit(1)).cast("int").alias("n_versions"),
        F.count(F.when(F.col("change_reason") == "reopened", 1)).cast("int").alias("n_reopened"),
        F.count(F.when(salary_changed, 1)).cast("int").alias("n_salary_changes"),
    )
    is_closed, gap = F.col("is_closed"), F.col("_last.closed_in_gap")
    dated = is_closed & ~gap
    return per_vacancy.crossJoin(first_collected).select(
        "source_id",
        "first_seen",
        "last_seen",
        "is_closed",
        (is_closed & gap).alias("closed_in_gap"),
        F.when(dated, F.col("_last.valid_to")).alias("closed_at"),
        (~is_closed | gap).alias("is_censored"),
        (F.col("first_seen") == F.col("first_collected_dt")).alias("is_left_censored"),
        F.when(dated, F.datediff("_last.valid_to", "first_seen"))
        .otherwise(F.datediff("last_seen", "first_seen") + 1)
        .cast("int")
        .alias("duration_days"),
        "n_versions",
        "n_reopened",
        "n_salary_changes",
    )
