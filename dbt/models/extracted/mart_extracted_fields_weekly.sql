{{ config(
    enabled=var('extraction_model') != '',
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, week, field, value)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_group_by': 268435456}
) }}

-- Distributions of extracted fields by ISO week and source, one row per field and value: role over
-- every vacancy of the week with valid fields (non_it is a value of it), grade, work_format,
-- employment_type and experience (bands, macro experience_band) over the IT vacancies among them. A
-- field not stated is the value unknown, so the rows of a field add up to vacancies_base. Units are
-- those of int_extracted_weekly_units.
WITH units AS (
    SELECT * FROM {{ ref('int_extracted_weekly_units') }}
),

bases AS (
    SELECT source, week, count() AS vacancies_in_week, countIf(valid) AS with_fields, countIf(is_it) AS it
    FROM units
    GROUP BY source, week
),

counts AS (
{%- for name, expr, _ in distribution_fields() %}
    SELECT source, week, '{{ name }}' AS field, coalesce({{ expr }}, 'unknown') AS value, count() AS vacancies
    FROM units
    WHERE {{ 'valid' if name == 'role' else 'is_it' }}
    GROUP BY source, week, value
    {{ 'UNION ALL' if not loop.last }}
{%- endfor %}
)

SELECT
    c.week AS week,
    CAST(c.source AS LowCardinality(String)) AS source,
    CAST(c.field AS LowCardinality(String)) AS field,
    CAST(c.value AS LowCardinality(String)) AS value,
    c.vacancies AS vacancies,
    if(c.field = 'role', b.with_fields, b.it) AS vacancies_base,
    b.vacancies_in_week AS vacancies_in_week
FROM counts AS c
INNER JOIN bases AS b ON c.source = b.source AND c.week = b.week
