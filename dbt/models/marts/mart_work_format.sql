{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, role_bucket, work_format)',
    query_settings={'max_threads': 2}
) }}

-- Work format x role bucket x source, of the vacancies (Telegram: posts) in that bucket. The format
-- rule (macros/marts.sql, work_format) reads source fields and narrow phrases of the text and says
-- unknown rather than guess; on trudvsem unknown is the large majority, because the portal has a
-- remote flag and nothing for office or hybrid.
WITH units AS (
    SELECT family AS source, role_bucket, work_format
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'

    UNION ALL

    SELECT 'telegram', role_bucket, work_format
    FROM {{ ref('stg_telegram') }}
    WHERE role_bucket != 'not_it' AND NOT is_week_duplicate
        AND published_at > (SELECT max(published_at) FROM {{ ref('stg_telegram') }}) - INTERVAL {{ telegram_window_days() }} DAY
),

cells AS (
    SELECT source, role_bucket, work_format, count() AS vacancies
    FROM units
    GROUP BY source, role_bucket, work_format
)

SELECT
    CAST(source AS LowCardinality(String)) AS source,
    CAST(role_bucket AS LowCardinality(String)) AS role_bucket,
    CAST(work_format AS LowCardinality(String)) AS work_format,
    vacancies,
    sum(vacancies) OVER (PARTITION BY source, role_bucket) AS vacancies_in_bucket
FROM cells
