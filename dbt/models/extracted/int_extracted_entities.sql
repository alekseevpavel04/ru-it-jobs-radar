{{ config(
    enabled=var('extraction_model') != '',
    schema='staging',
    materialized='table',
    engine='MergeTree()',
    order_by='(scope, entity_id)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_sort': 268435456, 'join_use_nulls': 1}
) }}

-- One row per portal vacancy (dedup entity) and scope, with the extracted fields of the record that
-- stands for it. Scope trudvsem, habr or superjob is a source family, the vacancy of
-- int_vacancy_entities and its latest record. Scope all_portals is the entity across families: a
-- vacancy the dedup found on trudvsem and on Habr is one row here, with the fields of its latest
-- record in any family (the order of int_vacancy_entities: latest staged day, the API before the
-- snapshot, then the id). Telegram is never in an entity: its posts are units of their own.
WITH family_scope AS (
    SELECT
        family AS scope,
        entity_id,
        latest_source AS record_source,
        latest_source_id AS record_source_id,
        toUInt8(1) AS families,
        first_seen,
        last_seen
    FROM {{ ref('int_vacancy_entities') }}
),

all_portals_latest AS (
    SELECT entity_id, source AS record_source, source_id AS record_source_id
    FROM {{ ref('int_vacancy_records') }}
    ORDER BY entity_id, dt DESC, source = 'trudvsem' DESC, source_id
    LIMIT 1 BY entity_id
),

all_portals_span AS (
    SELECT entity_id, toUInt8(uniqExact(family)) AS families, min(first_seen) AS first_seen, max(last_seen) AS last_seen
    FROM {{ ref('int_vacancy_records') }}
    GROUP BY entity_id
),

scopes AS (
    SELECT scope, entity_id, record_source, record_source_id, families, first_seen, last_seen FROM family_scope

    UNION ALL

    SELECT 'all_portals', l.entity_id, l.record_source, l.record_source_id, s.families, s.first_seen, s.last_seen
    FROM all_portals_latest AS l
    INNER JOIN all_portals_span AS s ON l.entity_id = s.entity_id
)

SELECT
    CAST(s.scope AS LowCardinality(String)) AS scope,
    s.entity_id AS entity_id,
    CAST(s.record_source AS LowCardinality(String)) AS record_source,
    s.record_source_id AS record_source_id,
    s.families AS families,
    s.first_seen AS first_seen,
    s.last_seen AS last_seen,
    coalesce(f.has_fields, false) AS has_fields,
    coalesce(f.fields_from_older_version, false) AS fields_from_older_version,
    coalesce(f.valid, false) AS valid,
    coalesce(f.is_it, false) AS is_it,
    f.role AS role,
    f.grade AS grade,
    f.skills AS skills,
    coalesce(f.skills_written_n, 0) AS skills_written_n,
    coalesce(f.skills_by_spelling_n, 0) AS skills_by_spelling_n,
    coalesce(f.skills_in_dictionary_n, 0) AS skills_in_dictionary_n,
    coalesce(f.skills_non_skill_n, 0) AS skills_non_skill_n,
    coalesce(f.skills_unchecked_n, 0) AS skills_unchecked_n,
    f.salary_from AS salary_from,
    f.salary_to AS salary_to,
    f.salary_currency AS salary_currency,
    f.salary_gross_net AS salary_gross_net,
    f.salary_period AS salary_period,
    f.work_format AS work_format,
    coalesce(f.experience_band, 'unknown') AS experience_band,
    f.employment_type AS employment_type
FROM scopes AS s
LEFT JOIN {{ ref('int_extracted_fields') }} AS f
    ON s.record_source = f.source AND s.record_source_id = f.source_id
