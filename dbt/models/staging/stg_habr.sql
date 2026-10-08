{{ config(
    materialized='incremental_by_day',
    raw_source='habr',
    engine='MergeTree()',
    partition_by='dt',
    order_by='source_id',
    query_settings={'max_threads': 2, 'max_block_size': 8192, 'min_insert_block_size_bytes': 33554432}
) }}

-- A rerun on the same day writes a second row for a vacancy whose list entry moved; the later run
-- wins, and between equal runs the fresher card.
WITH latest AS (
    SELECT
        *,
        parseDateTime64BestEffortOrNull(JSONExtractString(payload, 'card_fetched_at'), 6, 'UTC') AS card_fetched_at
    FROM {{ source('raw', 'habr') }} FINAL
    WHERE source = 'habr' AND dt = {{ staging_day() }}
    ORDER BY source_id, fetched_at DESC, card_fetched_at DESC NULLS LAST
    LIMIT 1 BY source_id
)

SELECT
    source,
    source_id,
    dt,
    content_hash,
    fetched_at,
    published_at,
    -- card datePosted is sometimes ahead of the day the card was read; kept and flagged
    coalesce(published_at > fetched_at, false) AS has_future_published_at,
    modified_at,
    url,
    title,
    text,
    coalesce(length(trimBoth(text)) > 0, false) AS has_text,
    card_fetched_at,
    card_fetched_at IS NOT NULL AS has_card,
    {{ json_dim("'company', 'title'") }} AS employer_name,
    JSONExtract(payload, 'company', 'id', 'Nullable(UInt64)') AS employer_id,
    {{ json_string("'company', 'alias_name'") }} AS employer_alias,
    arrayMap(x -> JSONExtractString(x, 'title'), JSONExtractArrayRaw(payload, 'locations')) AS cities,
    {{ json_string("'salary', 'formatted'") }} AS salary_text,
    {{ stated_amount("JSONExtractRaw(payload, 'salary', 'from')") }} AS salary_from,
    {{ stated_amount("JSONExtractRaw(payload, 'salary', 'to')") }} AS salary_to,
    {{ json_dim("'salary', 'currency'") }} AS salary_currency,
    {{ currency_code("salary_currency") }} AS salary_currency_code,
    {{ stated_amount("JSONExtractRaw(payload, 'predictedSalary', 'from')") }} AS predicted_salary_from,
    {{ stated_amount("JSONExtractRaw(payload, 'predictedSalary', 'to')") }} AS predicted_salary_to,
    {{ json_string("'predictedSalary', 'currency'") }} AS predicted_salary_currency,
    {{ json_dim("'employment'") }} AS employment,
    {{ json_dim("'card', 'employmentType'") }} AS card_employment_type,
    JSONExtract(payload, 'remoteWork', 'Nullable(Bool)') AS remote_work,
    {{ json_dim("'card', 'jobLocationType'") }} AS job_location_type,
    lower({{ json_string("'qualification'") }}) AS grade,
    arrayMap(x -> JSONExtractString(x, 'title'), JSONExtractArrayRaw(payload, 'divisions')) AS divisions,
    arrayMap(x -> JSONExtractString(x, 'title'), JSONExtractArrayRaw(payload, 'skills')) AS skills,
    {{ source_time("JSONExtractString(payload, 'publishedDate', 'date')") }} AS raised_at,
    toDateOrNull(JSONExtractString(payload, 'card', 'validThrough')) AS valid_through,
    JSONExtractBool(payload, 'archived') AS is_archived,
    JSONExtract(payload, '_passes', 'Array(String)') AS collector_passes,
    ingested_at
FROM latest
