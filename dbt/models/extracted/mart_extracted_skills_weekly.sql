{{ config(
    enabled=var('extraction_model') != '',
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, week, skill)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_group_by': 268435456}
) }}

-- Skill x ISO week x source on the extracted skills (dictionary v1): IT vacancies of the week whose
-- extracted skills name the skill, of the IT vacancies of the week with fields (vacancies_it). Units
-- are those of int_extracted_weekly_units. A share is vacancies_mentioning / vacancies_it; a week
-- that the corpus run has covered only in part (vacancies_with_fields below vacancies_in_week) gives
-- a share of the units it covered.
WITH units AS (
    SELECT source, week, is_it, valid, skills
    FROM {{ ref('int_extracted_weekly_units') }}
),

weeks AS (
    SELECT source, week, count() AS vacancies_in_week, countIf(valid) AS vacancies_with_fields, countIf(is_it) AS vacancies_it
    FROM units
    GROUP BY source, week
),

mentions AS (
    SELECT source, week, arrayJoin(skills) AS skill, count() AS vacancies_mentioning
    FROM units
    WHERE is_it
    GROUP BY source, week, skill
),

skills AS (
    SELECT skill, any(category) AS category, any(parent) AS parent
    FROM {{ ref('skill_dictionary_v1') }}
    GROUP BY skill
)

SELECT
    m.week AS week,
    CAST(m.source AS LowCardinality(String)) AS source,
    m.skill AS skill,
    CAST(s.category AS LowCardinality(String)) AS skill_category,
    nullIf(s.parent, '') AS skill_parent,
    m.vacancies_mentioning AS vacancies_mentioning,
    w.vacancies_it AS vacancies_it,
    w.vacancies_with_fields AS vacancies_with_fields,
    w.vacancies_in_week AS vacancies_in_week
FROM mentions AS m
INNER JOIN weeks AS w ON m.source = w.source AND m.week = w.week
INNER JOIN skills AS s ON m.skill = s.skill
