{{ config(enabled=var('extraction_model') != '') }}

-- Fields extracted by radar.extract.corpus_run, one row per vacancy version and extractor. The
-- runner writes a key again on a rerun, and the ReplacingMergeTree folds it only on merge, hence
-- FINAL. Rows of every extractor stay, so two models can be compared on the same versions;
-- int_vacancy_fields picks the one the marts read.
SELECT
    source,
    source_id,
    content_hash,
    model_id,
    prompt_sha256,
    schema_version,
    dt,
    input_sha256,
    input_truncated,
    valid,
    finish_reason,
    reused,
    role,
    grade,
    skills,
    salary_from,
    salary_to,
    salary_currency,
    salary_gross_net,
    salary_period,
    work_format,
    cities,
    experience_min,
    experience_max,
    employment_type,
    lang,
    run_id,
    created_at
FROM {{ source('extract', 'vacancy_fields') }} FINAL
