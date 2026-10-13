{{ config(
    enabled=var('extraction_model') != '',
    schema='staging',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, week, unit_id)',
    query_settings={
        'max_threads': 2,
        'max_bytes_before_external_sort': 268435456,
        'max_bytes_before_external_group_by': 268435456,
        'join_use_nulls': 1
    }
) }}

-- The units the weekly marts on extracted fields count, one row per source, week and unit. They are
-- the units of mart_weekly_dynamics and mart_skills_weekly, so the rule and the extracted marts can be
-- read side by side:
-- - trudvsem_snapshot, trudvsem (API), habr, superjob: a vacancy (dedup entity within its source
--   family) seen open on a collected day of the week in that source, with the fields of the family
--   entity (int_extracted_entities). The API and a snapshot are separate rows, as in the rule marts.
-- - all_portals: a vacancy seen open in any portal source that week, counted once across all of them
--   (scope all_portals: the dedup entity across families, so a vacancy on trudvsem and on Habr is one).
-- - telegram: a single-vacancy post published that week, after the weekly cross-post dedup of
--   stg_telegram (is_week_duplicate). Multi-vacancy posts are left out: the corpus does not extract
--   them. Telegram is never merged into all_portals, also not the Habr Career reposts
--   (links_to_habr), which are counted here and shown in mart_extraction_coverage.
-- Every unit has a row whether or not it has fields yet, so coverage is units with fields of units.
WITH active AS (
    SELECT DISTINCT r.source AS source, toMonday(o.dt) AS week, r.family AS family, r.entity_id AS entity_id
    FROM {{ ref('hist_observations') }} AS o
    INNER JOIN {{ ref('int_vacancy_records') }} AS r ON o.source = r.source AND o.source_id = r.source_id
    WHERE NOT o.is_closed_state
),

portal_units AS (
    SELECT source, week, family AS scope, entity_id FROM active

    UNION ALL

    SELECT DISTINCT 'all_portals', week, 'all_portals', entity_id FROM active
),

entities AS (
    SELECT * FROM {{ ref('int_extracted_entities') }}
),

telegram AS (
    SELECT source_id, published_week, links_to_habr
    FROM {{ ref('stg_telegram') }}
    WHERE NOT is_multi_vacancy AND NOT is_week_duplicate
),

telegram_fields AS (
    SELECT * FROM {{ ref('int_extracted_fields') }} WHERE source = 'telegram'
)

SELECT
    CAST(u.source AS LowCardinality(String)) AS source,
    u.week AS week,
    u.entity_id AS unit_id,
    CAST(e.record_source AS LowCardinality(String)) AS record_source,
    e.record_source_id AS record_source_id,
    false AS links_to_habr,
    e.has_fields AS has_fields,
    e.fields_from_older_version AS fields_from_older_version,
    false AS not_extractable,
    e.valid AS valid,
    e.is_it AS is_it,
    e.role AS role,
    e.grade AS grade,
    e.work_format AS work_format,
    e.employment_type AS employment_type,
    e.experience_band AS experience_band,
    e.skills AS skills,
    e.skills_written_n AS skills_written_n,
    e.skills_by_spelling_n AS skills_by_spelling_n,
    e.skills_in_dictionary_n AS skills_in_dictionary_n,
    e.skills_non_skill_n AS skills_non_skill_n,
    e.skills_unchecked_n AS skills_unchecked_n
FROM portal_units AS u
INNER JOIN entities AS e ON u.scope = e.scope AND u.entity_id = e.entity_id

UNION ALL

SELECT
    CAST('telegram' AS LowCardinality(String)),
    t.published_week,
    t.source_id,
    CAST('telegram' AS LowCardinality(String)),
    t.source_id,
    t.links_to_habr,
    coalesce(f.has_fields, false),
    coalesce(f.fields_from_older_version, false),
    f.dropped_reason IS NOT NULL AND NOT coalesce(f.has_fields, false),
    coalesce(f.valid, false),
    coalesce(f.is_it, false),
    f.role,
    f.grade,
    f.work_format,
    f.employment_type,
    coalesce(f.experience_band, 'unknown'),
    f.skills,
    coalesce(f.skills_written_n, 0),
    coalesce(f.skills_by_spelling_n, 0),
    coalesce(f.skills_in_dictionary_n, 0),
    coalesce(f.skills_non_skill_n, 0),
    coalesce(f.skills_unchecked_n, 0)
FROM telegram AS t
LEFT JOIN telegram_fields AS f ON t.source_id = f.source_id
