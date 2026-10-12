-- Fields extracted from vacancy texts, written by `python -m radar.extract.corpus_run`.
--
-- Applied by init.sh on `up`, by `python -m radar.stream.clickhouse` on a running server and by the
-- runner itself before its first insert. Every statement is idempotent.
--
-- One row per vacancy version and extractor: (source, source_id, content_hash) is the version of
-- the raw layer, model_id names the weights and prompt together, so a new model or a new prompt is a
-- new model_id and the old rows stay for comparison. A rerun that writes the same key again is
-- folded by the ReplacingMergeTree (the later insert wins); read with FINAL.
--
-- The enumerations repeat radar.extract.schema (tests/extract/test_corpus_run.py checks they match):
-- a value outside the schema fails the insert instead of landing in a mart.

CREATE DATABASE IF NOT EXISTS extract;

CREATE TABLE IF NOT EXISTS extract.vacancy_fields
(
    source LowCardinality(String),
    source_id String,
    content_hash String,
    model_id LowCardinality(String) COMMENT 'extractor: weights, quantization and prompt as one name',
    prompt_sha256 Nullable(String) COMMENT 'sha256 of the system prompt; NULL for an extractor without one',
    schema_version UInt8,
    dt Date COMMENT 'raw partition of the version that was extracted',
    input_sha256 String COMMENT 'sha256 of the extractor input (radar.extract.inputs)',
    input_truncated Nullable(Bool) COMMENT 'the input was cut to the model context; NULL if the extractor does not say',
    valid Bool COMMENT 'the answer passed the schema; an invalid answer has every field empty',
    finish_reason LowCardinality(Nullable(String)),
    reused Bool COMMENT 'answer of an identical input extracted earlier by the same model_id, not a request of its own',
    role Nullable(Enum8(
        'ml_engineer' = 1, 'data_scientist' = 2, 'mlops' = 3, 'data_engineer' = 4, 'data_analyst' = 5,
        'business_analyst' = 6, 'system_analyst' = 7, 'backend' = 8, 'frontend' = 9, 'fullstack' = 10,
        'mobile' = 11, 'software_engineer' = 12, 'developer_1c' = 13, 'embedded' = 14, 'devops' = 15,
        'qa' = 16, 'sysadmin' = 17, 'support' = 18, 'security' = 19, 'designer' = 20, 'manager' = 21,
        'other_it' = 22, 'non_it' = 23
    )),
    grade Nullable(Enum8('intern' = 1, 'junior' = 2, 'middle' = 3, 'senior' = 4, 'lead' = 5)),
    skills Array(String),
    salary_from Nullable(UInt64),
    salary_to Nullable(UInt64),
    salary_currency Nullable(Enum8(
        'RUB' = 1, 'USD' = 2, 'EUR' = 3, 'KZT' = 4, 'BYN' = 5, 'UZS' = 6, 'CNY' = 7, 'GBP' = 8
    )),
    salary_gross_net Nullable(Enum8('gross' = 1, 'net' = 2)),
    salary_period Nullable(Enum8('month' = 1, 'year' = 2, 'day' = 3, 'hour' = 4)),
    work_format Nullable(Enum8('office' = 1, 'hybrid' = 2, 'remote' = 3)),
    cities Array(String),
    experience_min Nullable(Float64),
    experience_max Nullable(Float64),
    employment_type Nullable(Enum8('labor_contract' = 1, 'gph' = 2, 'self_employed' = 3, 'internship' = 4)),
    lang Nullable(Enum8('ru' = 1, 'en' = 2, 'other' = 3)) COMMENT 'NULL only on an invalid answer',
    run_id String COMMENT 'the runner invocation that wrote the row (state in data/corpus_run/<model_id>/meta.json)',
    created_at DateTime64(3, 'UTC') COMMENT 'when the runner received the answer, or copied it for a reused row',
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
ORDER BY (source, source_id, content_hash, model_id);
