{{ config(
    schema='staging',
    materialized='table',
    engine='MergeTree()',
    order_by='(family, entity_id, source, source_id)',
    query_settings={
        'max_threads': 2,
        'max_block_size': 8192,
        'max_bytes_before_external_sort': 268435456,
        'max_bytes_before_external_group_by': 268435456,
        'join_use_nulls': 1
    }
) }}

-- One row per IT vacancy record within its source, (source, source_id): its latest staged day,
-- the rule fields of the marts (macros/marts.sql) and the dedup entity it belongs to. IT records are
-- those the history tracks: snapshot IT candidates, API IT candidates (the API slice also carries
-- non-IT mass postings, which the dedup leaves out too) and all of Habr.
WITH {{ skill_dictionary_constant() }} AS skill_dict,

snapshot_latest AS (
    SELECT source_id, max(dt) AS dt
    FROM {{ ref('hist_snapshot_days') }}
    GROUP BY source_id
),

records AS (
    SELECT
        source, source_id, dt, title, text, employer_name,
        employer_inn AS employer_key,
        region, CAST([] AS Array(String)) AS cities,
        salary_from, salary_to, salary_currency_code,
        employment, CAST(NULL AS Nullable(Bool)) AS remote_work,
        experience_min_years, CAST(NULL AS Nullable(String)) AS stated_grade, skills
    FROM {{ ref('stg_trudvsem_api') }}
    WHERE is_it_candidate
    ORDER BY source_id, dt DESC
    LIMIT 1 BY source_id

    UNION ALL

    SELECT
        source, source_id, dt, title, text, employer_name,
        employer_inn AS employer_key,
        region, CAST([] AS Array(String)) AS cities,
        salary_from, salary_to, salary_currency_code,
        employment, CAST(NULL AS Nullable(Bool)) AS remote_work,
        experience_min_years, CAST(NULL AS Nullable(String)) AS stated_grade, skills
    FROM {{ ref('stg_trudvsem_snapshot') }}
    WHERE is_it_candidate AND (source_id, dt) IN (SELECT source_id, dt FROM snapshot_latest)

    UNION ALL

    SELECT
        source, source_id, dt, title, text, employer_name,
        CAST(toString(employer_id) AS Nullable(String)) AS employer_key,
        CAST(NULL AS Nullable(String)) AS region, cities,
        salary_from, salary_to, salary_currency_code,
        employment, remote_work,
        CAST(NULL AS Nullable(UInt8)) AS experience_min_years, grade AS stated_grade, skills
    FROM {{ ref('stg_habr') }}
    ORDER BY source_id, dt DESC
    LIMIT 1 BY source_id
),

-- The API and the snapshots share vacancy ids, so the fallback for a record the dedup has not seen
-- yet is keyed by the portal, not by the source: its API and snapshot records still meet.
entities AS (
    SELECT if(source = 'habr', 'habr', 'trudvsem') AS family, source_id, any(vacancy_entity_id) AS vacancy_entity_id
    FROM {{ source('dedup', 'vacancy_entities') }}
    GROUP BY family, source_id
),

ruled AS (
    SELECT
        r.*,
        CAST(if(r.source = 'habr', 'habr', 'trudvsem') AS LowCardinality(String)) AS family,
        {{ rule_text('r.title') }} AS title_rule,
        {{ rule_text("concat(coalesce(r.title, ''), '\\n', coalesce(r.text, ''), '\\n', arrayStringConcat(r.skills, ' | '))") }}
            AS body_rule,
        coalesce(
            r.employer_name = 'ИП' OR startsWith(r.employer_name, 'ИП ')
                OR startsWith(lowerUTF8(r.employer_name), 'индивидуальный предприниматель'),
            false
        ) AS is_sole_proprietor
    FROM records AS r
),

derived AS (
    SELECT
        *,
        {{ work_format(
            'body_rule',
            "if(source = 'habr', remote_work, employment = 'Дистанционная (удаленная) работа')",
            "source = 'habr' AND remote_work = false AND notEmpty(cities)"
        ) }} AS work_format_rule
    FROM ruled
)

SELECT
    d.source AS source,
    d.source_id AS source_id,
    d.family AS family,
    coalesce(e.vacancy_entity_id, concat(d.family, ':', d.source_id)) AS entity_id,
    e.vacancy_entity_id IS NOT NULL AS has_dedup_entity,
    d.dt AS dt,
    l.first_seen AS first_seen,
    l.last_seen AS last_seen,
    coalesce(l.is_closed, false) AS is_closed,
    l.closed_at AS closed_at,
    l.duration_days AS duration_days,
    l.is_left_censored AS is_left_censored,
    d.title AS title,
    if(d.is_sole_proprietor, NULL, d.employer_key) AS employer_key,
    if(d.is_sole_proprietor, 'ИП', d.employer_name) AS employer_name,
    d.is_sole_proprietor AS is_sole_proprietor,
    d.salary_from AS salary_from,
    d.salary_to AS salary_to,
    d.salary_currency_code AS salary_currency_code,
    {{ role_bucket('d.title_rule') }} AS role_bucket,
    multiIf(
        d.stated_grade IS NOT NULL, d.stated_grade,
        {{ grade_from_text('d.title_rule') }} != 'unknown', {{ grade_from_text('d.title_rule') }},
        d.employment = 'Стажировка', 'intern',
        'unknown'
    ) AS grade,
    d.stated_grade IS NOT NULL AS grade_is_stated,
    d.work_format_rule AS work_format,
    {{ region_group(
        'd.work_format_rule',
        "d.region = 'Город Москва' OR has(d.cities, 'Москва')",
        "d.region = 'Город Санкт-Петербург' OR has(d.cities, 'Санкт-Петербург')",
        'd.region IS NOT NULL OR notEmpty(d.cities)'
    ) }} AS region_group,
    d.experience_min_years AS experience_min_years,
    if(d.source = 'habr', {{ no_experience_text('d.body_rule') }}, d.experience_min_years = 0) AS no_experience,
    {{ skills_found('d.body_rule') }} AS skills
FROM derived AS d
LEFT JOIN entities AS e ON d.family = e.family AND d.source_id = e.source_id
LEFT JOIN {{ ref('hist_vacancy_lifetime') }} AS l ON d.source = l.source AND d.source_id = l.source_id
