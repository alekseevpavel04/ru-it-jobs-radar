{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, week, skill)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_group_by': 268435456}
) }}

-- Skill x ISO week (its Monday) x source, for the weekly digest: vacancies of the week that mention
-- the skill by the dictionary (seed skill_dictionary), of all vacancies of the week. The units are
-- those of mart_weekly_dynamics, so vacancies_in_week equals the sum of its rows of the week:
-- trudvsem_snapshot, trudvsem (API) and habr count dedup entities seen open on a collected day of
-- the week (vacancies_active), with the skills of the entity's latest record; Telegram counts the
-- vacancy posts published that week after the cross-post dedup (telegram_posts). IT roles only.
-- Only skills with a mention get a row.
WITH entities AS (
    SELECT family, entity_id, skills
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'
),

active AS (
    SELECT DISTINCT r.source AS source, toMonday(o.dt) AS week, r.family AS family, r.entity_id AS entity_id
    FROM {{ ref('hist_observations') }} AS o
    INNER JOIN {{ ref('int_vacancy_records') }} AS r ON o.source = r.source AND o.source_id = r.source_id
    WHERE NOT o.is_closed_state
),

units AS (
    SELECT a.source AS source, a.week AS week, e.skills AS skills
    FROM active AS a
    INNER JOIN entities AS e ON a.family = e.family AND a.entity_id = e.entity_id

    UNION ALL

    SELECT 'telegram', published_week, skills
    FROM {{ ref('stg_telegram') }}
    WHERE role_bucket != 'not_it' AND NOT is_week_duplicate
),

weeks AS (
    SELECT source, week, count() AS vacancies_in_week
    FROM units
    GROUP BY source, week
),

mentions AS (
    SELECT source, week, arrayJoin(skills) AS skill, count() AS vacancies_mentioning
    FROM units
    GROUP BY source, week, skill
)

SELECT
    m.week AS week,
    CAST(m.source AS LowCardinality(String)) AS source,
    m.skill AS skill,
    CAST(d.category AS LowCardinality(String)) AS skill_category,
    m.vacancies_mentioning AS vacancies_mentioning,
    w.vacancies_in_week AS vacancies_in_week
FROM mentions AS m
INNER JOIN weeks AS w ON m.source = w.source AND m.week = w.week
INNER JOIN {{ ref('skill_dictionary') }} AS d ON m.skill = d.skill
