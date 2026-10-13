{{ config(
    enabled=var('extraction_model') != '',
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, level, week)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_group_by': 268435456}
) }}

-- How much of what the marts on extracted fields count has fields, per source: level week for each
-- ISO week (the units of that week), level source once over the collection period (each unit once,
-- week is the first week of the source). Units are those of int_extracted_weekly_units. The Telegram
-- corpus leaves out digests and posts with almost no text of their own (radar.extract.telegram_corpus
-- drop_reason); they stay units here and never get fields, so Telegram coverage stays below 1 even
-- after the run has reached every week.
WITH units AS (
    SELECT * FROM {{ ref('int_extracted_weekly_units') }}
),

-- A unit's fields do not depend on the week, so over the period each unit is taken once.
period_units AS (
    SELECT
        source,
        unit_id,
        min(week) AS week,
        any(has_fields) AS has_fields,
        any(fields_from_older_version) AS fields_from_older_version,
        any(valid) AS valid,
        any(is_it) AS is_it,
        any(links_to_habr) AS links_to_habr,
        any(skills_written_n) AS skills_written_n,
        any(skills_in_dictionary_n) AS skills_in_dictionary_n
    FROM units
    GROUP BY source, unit_id
),

counted AS (
{%- for level, input in [('week', 'units'), ('source', 'period_units')] %}
    SELECT
        '{{ level }}' AS level,
        source,
        {{ 'week' if level == 'week' else 'min(week)' }} AS week,
        count() AS units,
        countIf(has_fields) AS units_with_fields,
        countIf(fields_from_older_version) AS units_from_older_version,
        countIf(valid) AS units_valid,
        countIf(is_it) AS units_it,
        countIf(links_to_habr) AS units_linking_habr,
        sum(skills_written_n) AS skills_written,
        sum(skills_in_dictionary_n) AS skills_in_dictionary
    FROM {{ input }}
    GROUP BY source{{ ', week' if level == 'week' }}
    {{ 'UNION ALL' if not loop.last }}
{%- endfor %}
)

SELECT
    CAST(level AS LowCardinality(String)) AS level,
    CAST(source AS LowCardinality(String)) AS source,
    week,
    units,
    units_with_fields,
    units_from_older_version,
    units_valid,
    units_it,
    units_linking_habr,
    round(units_with_fields / units, 4) AS coverage,
    if(units_with_fields = 0, NULL, round(units_valid / units_with_fields, 4)) AS valid_share,
    skills_written,
    skills_in_dictionary
FROM counted
