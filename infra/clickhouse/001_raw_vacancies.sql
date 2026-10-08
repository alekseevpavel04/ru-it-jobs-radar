-- Raw layer in ClickHouse: the Redpanda topics raw.<source> land in raw.vacancies.
--
-- Applied by init.sh on `up` and by `python -m radar.stream.clickhouse` on a running server. Every
-- statement is idempotent; a later schema change goes into a new numbered file, not into this one.
--
-- One table for all sources: they share the raw-layer envelope, and staging filters by `source`.
-- A row is one record of one raw partition `dt` (the day a collector saw it, or the snapshot date):
-- the same vacancy unchanged on ten days is ten rows, as in Parquet, because the history of
-- sightings is what SCD2 and survival analysis are built on. A replay delivers the same rows again;
-- ReplacingMergeTree keeps one per (source, dt, source_id, content_hash), so read with FINAL or
-- group by the key until background merges catch up.

CREATE TABLE IF NOT EXISTS raw.vacancies
(
    source LowCardinality(String),
    dt Date COMMENT 'raw partition the record came from (Kafka header dt)',
    source_id String,
    content_hash String,
    schema_version UInt8,
    fetched_at DateTime64(6, 'UTC'),
    published_at Nullable(DateTime64(6, 'UTC')),
    modified_at Nullable(DateTime64(6, 'UTC')),
    url Nullable(String),
    title Nullable(String) CODEC(ZSTD(3)),
    text Nullable(String) CODEC(ZSTD(3)),
    payload String CODEC(ZSTD(3)) COMMENT 'JSON object of source fields after PII removal',
    pii_removed Map(LowCardinality(String), UInt32),
    is_vacancy Nullable(Bool),
    multi_vacancy Nullable(Bool),
    topic LowCardinality(String),
    kafka_offset UInt64,
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(dt)
-- Days of one vacancy sit next to each other, so its repeated text compresses to almost nothing.
ORDER BY (source, source_id, content_hash, dt);

-- Messages that could not be parsed: kept with the error instead of blocking the queue or vanishing.
CREATE TABLE IF NOT EXISTS raw.vacancies_errors
(
    topic LowCardinality(String),
    kafka_partition UInt64,
    kafka_offset UInt64,
    message_key String,
    error String,
    raw_message String CODEC(ZSTD(3)),
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY (topic, kafka_offset);

-- The consumer. ClickHouse commits the offsets only after a block is written through the views
-- below, so delivery is at least once and the ReplacingMergeTree absorbs the repeats.
CREATE TABLE IF NOT EXISTS raw.vacancies_queue
(
    schema_version UInt8,
    source String,
    source_id String,
    fetched_at DateTime64(6, 'UTC'),
    published_at Nullable(DateTime64(6, 'UTC')),
    modified_at Nullable(DateTime64(6, 'UTC')),
    url Nullable(String),
    title Nullable(String),
    text Nullable(String),
    payload String,
    content_hash String,
    pii_removed Map(String, UInt32),
    is_vacancy Nullable(Bool),
    multi_vacancy Nullable(Bool)
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'redpanda:9092',
    kafka_topic_list = 'raw.trudvsem,raw.trudvsem_snapshot,raw.habr,raw.telegram,raw.superjob',
    kafka_group_name = 'clickhouse_raw',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 1,
    -- A snapshot message is about 4.5 KB; 65536 of them keep a block well inside the server's memory.
    kafka_max_block_size = 65536,
    kafka_handle_error_mode = 'stream',
    -- The envelope writes times as ISO 8601 with a zone (2026-10-08T10:56:05Z).
    date_time_input_format = 'best_effort',
    -- `payload` is a JSON object in the message and a JSON string in the table.
    input_format_json_read_objects_as_strings = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS raw.vacancies_mv TO raw.vacancies AS
SELECT
    source,
    assumeNotNull(toDateOrNull(_headers.value[indexOf(_headers.name, 'dt')])) AS dt,
    source_id,
    content_hash,
    schema_version,
    fetched_at,
    published_at,
    modified_at,
    url,
    title,
    text,
    payload,
    pii_removed,
    is_vacancy,
    multi_vacancy,
    _topic AS topic,
    _offset AS kafka_offset
FROM raw.vacancies_queue
WHERE length(_error) = 0 AND isNotNull(toDateOrNull(_headers.value[indexOf(_headers.name, 'dt')]));

CREATE MATERIALIZED VIEW IF NOT EXISTS raw.vacancies_errors_mv TO raw.vacancies_errors AS
SELECT
    _topic AS topic,
    _partition AS kafka_partition,
    _offset AS kafka_offset,
    _key AS message_key,
    if(length(_error) > 0, _error, 'no valid dt header') AS error,
    _raw_message AS raw_message
FROM raw.vacancies_queue
WHERE length(_error) > 0 OR isNull(toDateOrNull(_headers.value[indexOf(_headers.name, 'dt')]));
