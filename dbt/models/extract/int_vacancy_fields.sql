{{ config(enabled=var('extraction_model') != '') }}

-- One row per vacancy record (source, source_id) at the version radar.extract.corpus_run extracts,
-- with the fields that var('extraction_model') gave that version. Portal records (trudvsem API and
-- snapshots, Habr, SuperJob, all through stg_vacancies): the latest staged version of an IT
-- candidate, as the runner reads it from the per-source staging tables. Telegram: the latest
-- stored version of a post stg_telegram counts as a single vacancy, ordered as the runner orders
-- them (fetched_at, dt, content_hash); stg_telegram itself keeps no content_hash. A record joins
-- int_vacancy_records and stg_telegram on (source, source_id).
WITH portal AS (
    SELECT source, source_id, dt, content_hash
    FROM {{ ref('stg_vacancies') }}
    WHERE is_it_candidate
    ORDER BY dt DESC
    LIMIT 1 BY source, source_id
),

telegram AS (
    SELECT 'telegram' AS source, source_id, latest.1 AS dt, latest.2 AS content_hash
    FROM (
        SELECT source_id, argMax((dt, content_hash), (fetched_at, dt, content_hash)) AS latest
        FROM {{ source('raw', 'telegram') }} FINAL
        WHERE source = 'telegram'
            AND source_id IN (SELECT source_id FROM {{ ref('stg_telegram') }} WHERE NOT is_multi_vacancy)
        GROUP BY source_id
    )
),

versions AS (
    SELECT source, source_id, dt, content_hash FROM portal
    UNION ALL
    SELECT source, source_id, dt, content_hash FROM telegram
),

fields AS (
    SELECT *
    FROM {{ ref('stg_vacancy_fields') }}
    WHERE model_id = '{{ var("extraction_model") }}'
)

-- Without join_use_nulls an unmatched model_id is '' and an unmatched valid is false, which is what
-- is_extracted and valid should say; the Nullable fields come as NULL and the lists as [].
SELECT
    v.source AS source,
    v.source_id AS source_id,
    v.dt AS dt,
    v.content_hash AS content_hash,
    CAST(f.model_id != '' AS Bool) AS is_extracted,
    f.valid AS valid,
    f.role AS role,
    f.grade AS grade,
    f.skills AS skills,
    f.salary_from AS salary_from,
    f.salary_to AS salary_to,
    f.salary_currency AS salary_currency,
    f.salary_gross_net AS salary_gross_net,
    f.salary_period AS salary_period,
    f.work_format AS work_format,
    f.cities AS cities,
    f.experience_min AS experience_min,
    f.experience_max AS experience_max,
    f.employment_type AS employment_type,
    f.lang AS lang
FROM versions AS v
LEFT JOIN fields AS f
    ON v.source = f.source AND v.source_id = f.source_id AND v.content_hash = f.content_hash
