{{ config(
    materialized='incremental_by_day',
    raw_source='trudvsem',
    engine='MergeTree()',
    partition_by='dt',
    order_by='source_id',
    query_settings={'max_threads': 2, 'max_block_size': 8192, 'min_insert_block_size_bytes': 33554432}
) }}

WITH latest AS (
    SELECT *
    FROM {{ source('raw', 'trudvsem') }} FINAL
    WHERE source = 'trudvsem' AND dt = {{ staging_day() }}
    ORDER BY source_id, fetched_at DESC
    LIMIT 1 BY source_id
)

SELECT
    source,
    source_id,
    dt,
    content_hash,
    fetched_at,
    published_at,
    modified_at,
    url,
    title,
    text,
    coalesce(length(trimBoth(text)) > 0, false) AS has_text,
    {{ json_dim("'company', 'name'") }} AS employer_name,
    {{ json_string("'company', 'inn'") }} AS employer_inn,
    JSONExtractBool(payload, 'company', 'hr-agency') AS is_hr_agency,
    {{ json_dim("'region', 'name'") }} AS region,
    {{ json_string("'region', 'region_code'") }} AS region_code,
    {{ json_string("'addresses', 'address', 1, 'location'") }} AS address,
    {{ json_string("'salary'") }} AS salary_text,
    {{ stated_amount("JSONExtractRaw(payload, 'salary_min')") }} AS salary_from,
    {{ stated_amount("JSONExtractRaw(payload, 'salary_max')") }} AS salary_to,
    {{ json_dim("'currency'") }} AS salary_currency,
    {{ currency_code("salary_currency") }} AS salary_currency_code,
    {{ json_dim("'schedule'") }} AS schedule,
    {{ json_dim("'employment'") }} AS employment,
    -- experience is a number of years, sometimes the string "от 0"
    toUInt8OrNull(extract(JSONExtractRaw(payload, 'requirement', 'experience'), '\\d+')) AS experience_min_years,
    {{ json_dim("'requirement', 'education'") }} AS education,
    {{ json_dim("'category', 'specialisation'") }} AS sphere_name,
    {{ json_string("'code_profession'") }} AS profession_code,
    {{ json_dim("'typicalPosition'") }} AS typical_position,
    {{ json_dim("'source'") }} AS source_type,
    JSONExtract(payload, 'skills', 'Array(String)') AS skills,
    JSONExtract(payload, '_queries', 'Array(String)') AS collector_queries,
    {{ is_it_candidate(
        "sphere_name = 'Информационные технологии, телекоммуникации, связь'",
        "typical_position",
        "title"
    ) }} AS is_it_candidate,
    ingested_at
FROM latest
