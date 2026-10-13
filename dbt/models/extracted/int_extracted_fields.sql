{{ config(
    enabled=var('extraction_model') != '',
    schema='staging',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, source_id)',
    query_settings={
        'max_threads': 2,
        'max_block_size': 8192,
        'max_bytes_before_external_sort': 268435456,
        'max_bytes_before_external_group_by': 268435456
    }
) }}

-- One row per vacancy record of int_vacancy_fields (portal IT records, single-vacancy Telegram posts)
-- with the fields the marts read, materialized once: the view behind it reads the whole staging layer.
-- The fields are those of the record's latest version when var('extraction_model') has extracted it;
-- a record changed since the last corpus run keeps the fields of its latest extracted version
-- (fields_from_older_version), so a day of new edits does not empty the marts until the next run.
-- Skills are mapped to the dictionary v1 by their lookup key (seed skill_dictionary_v1, macro skill_key);
-- a spelling outside it is counted in skills_written_n but not in skills.
WITH {{ skill_v1_map() }} AS skill_map,

latest AS (
    SELECT * FROM {{ ref('int_vacancy_fields') }}
),

older AS (
    SELECT *
    FROM {{ ref('stg_vacancy_fields') }}
    WHERE model_id = '{{ var("extraction_model") }}'
    ORDER BY source, source_id, dt DESC, created_at DESC
    LIMIT 1 BY source, source_id
),

-- Without join_use_nulls an unmatched model_id is '', which is what has_older says.
picked AS (
    SELECT
        l.source AS source,
        l.source_id AS source_id,
        l.dt AS dt,
        l.is_extracted AS is_extracted,
        l.is_extracted OR o.model_id != '' AS has_fields,
        NOT l.is_extracted AND o.model_id != '' AS fields_from_older_version,
        if(l.is_extracted, l.valid, o.valid) AS valid,
        CAST(if(l.is_extracted, l.role, o.role) AS Nullable(String)) AS role,
        CAST(if(l.is_extracted, l.grade, o.grade) AS Nullable(String)) AS grade,
        if(l.is_extracted, l.skills, o.skills) AS skills_written,
        if(l.is_extracted, l.salary_from, o.salary_from) AS salary_from,
        if(l.is_extracted, l.salary_to, o.salary_to) AS salary_to,
        CAST(if(l.is_extracted, l.salary_currency, o.salary_currency) AS Nullable(String)) AS salary_currency,
        CAST(if(l.is_extracted, l.salary_gross_net, o.salary_gross_net) AS Nullable(String)) AS salary_gross_net,
        CAST(if(l.is_extracted, l.salary_period, o.salary_period) AS Nullable(String)) AS salary_period,
        CAST(if(l.is_extracted, l.work_format, o.work_format) AS Nullable(String)) AS work_format,
        if(l.is_extracted, l.experience_min, o.experience_min) AS experience_min,
        if(l.is_extracted, l.experience_max, o.experience_max) AS experience_max,
        CAST(if(l.is_extracted, l.employment_type, o.employment_type) AS Nullable(String)) AS employment_type,
        CAST(if(l.is_extracted, l.lang, o.lang) AS Nullable(String)) AS lang
    FROM latest AS l
    LEFT JOIN older AS o ON l.source = o.source AND l.source_id = o.source_id
)

SELECT
    CAST(source AS LowCardinality(String)) AS source,
    source_id,
    dt,
    is_extracted,
    has_fields,
    fields_from_older_version,
    has_fields AND valid AS valid,
    -- The record passed its source's IT filter; the extractor's non_it overrules it, a NULL role does not.
    has_fields AND valid AND coalesce(role, '') != 'non_it' AS is_it,
    CAST(role AS LowCardinality(Nullable(String))) AS role,
    CAST(grade AS LowCardinality(Nullable(String))) AS grade,
    toUInt16(length(skills_written)) AS skills_written_n,
    toUInt16(length(arrayFilter(s -> mapContains(skill_map, {{ skill_key('s') }}), skills_written)))
        AS skills_in_dictionary_n,
    arraySort(arrayDistinct(arrayFilter(
        s -> s != '',
        arrayMap(s -> skill_map[{{ skill_key('s') }}], skills_written)
    ))) AS skills,
    salary_from,
    salary_to,
    CAST(salary_currency AS LowCardinality(Nullable(String))) AS salary_currency,
    CAST(salary_gross_net AS LowCardinality(Nullable(String))) AS salary_gross_net,
    CAST(salary_period AS LowCardinality(Nullable(String))) AS salary_period,
    CAST(work_format AS LowCardinality(Nullable(String))) AS work_format,
    experience_min,
    experience_max,
    CAST({{ experience_band('experience_min', 'experience_max') }} AS LowCardinality(String)) AS experience_band,
    CAST(employment_type AS LowCardinality(Nullable(String))) AS employment_type,
    CAST(lang AS LowCardinality(Nullable(String))) AS lang
FROM picked
