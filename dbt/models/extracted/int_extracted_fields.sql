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
-- Skills are mapped to the dictionary v1 names through the seed skill_dictionary_v1, which holds the
-- dictionary's spellings and every corpus spelling resolved by radar.extract.skills.normalize_list
-- (dbt/skill_seed.py), keyed by the hash of the lookup key (macros skill_key, skill_hash). The counts
-- say how each written spelling fared. A skill of skills_named_outside_links (HTTP, HTTPS) stays only
-- when the text the extractor read names it outside a link: the extractor takes "https://t.me/..."
-- for HTTPS, and the spelling alone cannot tell the two apart. dropped_reason is the rule
-- by which the corpus run never extracts the latest version (extract.vacancy_dropped, its latest check).
{%- set checked = skills_named_outside_links().keys() | list | tojson | replace('"', "'") %}
WITH {{ skill_seed_maps() }},

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

dropped AS (
    SELECT source, source_id, content_hash, reason
    FROM {{ source('extract', 'vacancy_dropped') }} FINAL
    WHERE check_id = (SELECT max(check_id) FROM {{ source('extract', 'vacancy_dropped') }})
),

-- Extracted versions whose skills map to a checked skill, and the checked skills their text names
-- outside links. The text is what the extractor read: title and text, and a portal's skill tags.
link_candidates AS (
    SELECT DISTINCT source, source_id, content_hash
    FROM {{ ref('stg_vacancy_fields') }}
    WHERE model_id = '{{ var("extraction_model") }}'
        AND arrayExists(s -> hasAny(skill_map[{{ skill_hash('s') }}], {{ checked }}), skills)
),

link_texts AS (
    SELECT source AS link_source, source_id AS link_source_id, content_hash AS link_hash,
        {{ named_outside_links(link_free_text('body')) }} AS named
    FROM (
        SELECT source, source_id, content_hash,
            concat(coalesce(title, ''), '\n', coalesce(text, ''), '\n', arrayStringConcat(skills, ' | ')) AS body
        FROM {{ ref('stg_vacancies') }}
        WHERE (source, source_id, content_hash) IN (
            SELECT source, source_id, content_hash FROM link_candidates WHERE source != 'telegram'
        )
        LIMIT 1 BY source, source_id, content_hash

        UNION ALL

        SELECT source, source_id, content_hash, concat(coalesce(title, ''), '\n', coalesce(text, '')) AS body
        FROM {{ source('raw', 'telegram') }}
        WHERE source = 'telegram' AND (source_id, content_hash) IN (
            SELECT source_id, content_hash FROM link_candidates WHERE source = 'telegram'
        )
        LIMIT 1 BY source_id, content_hash
    )
),

-- Without join_use_nulls an unmatched model_id is '' and an unmatched reason too.
picked AS (
    SELECT
        l.source AS source,
        l.source_id AS source_id,
        l.dt AS dt,
        l.is_extracted AS is_extracted,
        l.is_extracted OR o.model_id != '' AS has_fields,
        NOT l.is_extracted AND o.model_id != '' AS fields_from_older_version,
        d.reason AS dropped_reason,
        if(l.is_extracted, l.content_hash, o.content_hash) AS fields_hash,
        if(l.is_extracted, l.valid, o.valid) AS valid,
        CAST(if(l.is_extracted, l.role, o.role) AS Nullable(String)) AS role,
        CAST(if(l.is_extracted, l.grade, o.grade) AS Nullable(String)) AS grade,
        if(l.is_extracted, l.skills, o.skills) AS skills_written,
        arrayMap(s -> {{ skill_hash('s') }}, skills_written) AS written_hashes,
        arrayDistinct(arrayFlatten(arrayMap(h -> skill_map[h], written_hashes))) AS mapped,
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
    LEFT JOIN dropped AS d
        ON l.source = d.source AND l.source_id = d.source_id AND l.content_hash = d.content_hash
)

SELECT
    CAST(source AS LowCardinality(String)) AS source,
    source_id,
    dt,
    is_extracted,
    has_fields,
    fields_from_older_version,
    CAST(nullIf(dropped_reason, '') AS LowCardinality(Nullable(String))) AS dropped_reason,
    has_fields AND valid AS valid,
    -- The record passed its source's IT filter; the extractor's non_it overrules it, a NULL role does not.
    has_fields AND valid AND coalesce(role, '') != 'non_it' AS is_it,
    CAST(role AS LowCardinality(Nullable(String))) AS role,
    CAST(grade AS LowCardinality(Nullable(String))) AS grade,
    toUInt16(length(written_hashes)) AS skills_written_n,
    toUInt16(length(arrayFilter(h -> has(dictionary_keys, h), written_hashes))) AS skills_by_spelling_n,
    toUInt16(length(arrayFilter(h -> notEmpty(skill_map[h]), written_hashes))) AS skills_in_dictionary_n,
    toUInt16(countEqual(arrayMap(h -> skill_status[h], written_hashes), 'non_skill')) AS skills_non_skill_n,
    toUInt16(length(arrayFilter(h -> NOT mapContains(skill_status, h), written_hashes))) AS skills_unchecked_n,
    -- Without join_use_nulls a version without a text row has named [] and loses its checked skills.
    toUInt16(length(arrayFilter(n -> has({{ checked }}, n) AND NOT has(t.named, n), mapped))) AS skills_link_only_n,
    arraySort(arrayFilter(n -> NOT has({{ checked }}, n) OR has(t.named, n), mapped)) AS skills,
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
FROM picked AS p
LEFT JOIN link_texts AS t
    ON p.source = t.link_source AND p.source_id = t.link_source_id AND p.fields_hash = t.link_hash
