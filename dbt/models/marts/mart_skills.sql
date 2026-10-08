{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, role_bucket, grade, skill)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_group_by': 268435456}
) }}

-- Skill x role bucket x grade per source: vacancies (Telegram: posts) whose title, text or source
-- skill tags mention the skill by the dictionary (seed skill_dictionary, v0), of the vacancies in
-- that bucket and grade. A vacancy has one bucket and one grade, so rows of a skill add up across
-- grades and buckets within a source. Telegram takes the posts of the last telegram_window_days
-- before its newest post, after the weekly cross-post dedup, multi-vacancy posts in their own
-- bucket.
WITH units AS (
    SELECT family AS source, role_bucket, grade, skills
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'

    UNION ALL

    SELECT 'telegram', role_bucket, grade, skills
    FROM {{ ref('stg_telegram') }}
    WHERE role_bucket != 'not_it' AND NOT is_week_duplicate
        AND published_at > (SELECT max(published_at) FROM {{ ref('stg_telegram') }}) - INTERVAL {{ telegram_window_days() }} DAY
),

cells AS (
    SELECT source, role_bucket, grade, count() AS vacancies_in_cell
    FROM units
    GROUP BY source, role_bucket, grade
),

mentions AS (
    SELECT source, role_bucket, grade, arrayJoin(skills) AS skill, count() AS vacancies_mentioning
    FROM units
    GROUP BY source, role_bucket, grade, skill
),

periods AS (
    SELECT family AS source, min(first_seen) AS period_from, max(last_seen) AS period_to
    FROM {{ ref('int_vacancy_entities') }}
    GROUP BY family

    UNION ALL

    SELECT 'telegram', toDate(max(published_at) - INTERVAL {{ telegram_window_days() }} DAY) + 1, toDate(max(published_at))
    FROM {{ ref('stg_telegram') }}
)

SELECT
    CAST(m.source AS LowCardinality(String)) AS source,
    CAST(m.role_bucket AS LowCardinality(String)) AS role_bucket,
    CAST(m.grade AS LowCardinality(String)) AS grade,
    m.skill AS skill,
    CAST(d.category AS LowCardinality(String)) AS skill_category,
    m.vacancies_mentioning AS vacancies_mentioning,
    c.vacancies_in_cell AS vacancies_in_cell,
    p.period_from AS period_from,
    p.period_to AS period_to
FROM mentions AS m
INNER JOIN cells AS c ON m.source = c.source AND m.role_bucket = c.role_bucket AND m.grade = c.grade
INNER JOIN {{ ref('skill_dictionary') }} AS d ON m.skill = d.skill
INNER JOIN periods AS p ON m.source = p.source
