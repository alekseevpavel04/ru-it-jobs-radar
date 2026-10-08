{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, role_bucket)',
    query_settings={'max_threads': 2}
) }}

-- The junior slice per source and role bucket: vacancies open to a beginner, that is graded intern
-- or junior (Habr: the grade the employer states, else the title; trudvsem: the title, else
-- «Стажировка» employment; Telegram: the first line and hashtags) or saying no experience is needed
-- (trudvsem: required experience 0 years, the portal's «без опыта»; Habr and Telegram, which have no
-- such field: a phrase of the text). Every count has the bucket's vacancies as its denominator.
WITH units AS (
    SELECT family AS source, role_bucket, grade, no_experience, experience_min_years IS NOT NULL AS experience_stated
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'

    UNION ALL

    SELECT 'telegram', role_bucket, grade, no_experience_stated, false
    FROM {{ ref('stg_telegram') }}
    WHERE role_bucket != 'not_it' AND NOT is_week_duplicate
        AND published_at > (SELECT max(published_at) FROM {{ ref('stg_telegram') }}) - INTERVAL {{ telegram_window_days() }} DAY
)

SELECT
    CAST(source AS LowCardinality(String)) AS source,
    CAST(role_bucket AS LowCardinality(String)) AS role_bucket,
    count() AS vacancies,
    countIf(grade IN ('intern', 'junior') OR coalesce(no_experience, false)) AS open_to_junior,
    countIf(grade = 'intern') AS graded_intern,
    countIf(grade = 'junior') AS graded_junior,
    countIf(coalesce(no_experience, false)) AS no_experience_required,
    countIf(grade != 'unknown') AS grade_known,
    -- only trudvsem has an experience field; elsewhere the share of it cannot be measured
    if(source = 'trudvsem', countIf(experience_stated), NULL) AS experience_field_stated
FROM units
GROUP BY source, role_bucket
