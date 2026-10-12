{{ config(
    materialized='incremental_by_day',
    raw_source='superjob',
    engine='MergeTree()',
    partition_by='dt',
    order_by='source_id',
    query_settings={'max_threads': 2, 'max_block_size': 8192}
) }}

-- A rerun on the same day writes a second raw row for a vacancy that changed in between; the later
-- run wins.
WITH latest AS (
    SELECT
        *,
        arrayMap(x -> JSONExtractUInt(x, 'id'), JSONExtractArrayRaw(payload, 'catalogues')) AS catalogue_ids
    FROM {{ source('raw', 'superjob') }} FINAL
    WHERE source = 'superjob' AND dt = {{ staging_day() }}
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
    {{ json_dim("'firm_name'") }} AS employer_name,
    JSONExtract(payload, 'id_client', 'Nullable(UInt64)') AS employer_id,
    {{ json_dim("'town', 'title'") }} AS town,
    if(town IS NULL, CAST([] AS Array(String)), [CAST(assumeNotNull(town) AS String)]) AS cities,
    {{ json_string("'address'") }} AS address_town_level,
    {{ stated_amount("JSONExtractRaw(payload, 'payment_from')") }} AS salary_from,
    {{ stated_amount("JSONExtractRaw(payload, 'payment_to')") }} AS salary_to,
    {{ json_dim("'currency'") }} AS salary_currency,
    {{ currency_code("salary_currency") }} AS salary_currency_code,
    JSONExtract(payload, 'agreement', 'Nullable(Bool)') AS salary_by_agreement,
    {{ json_dim("'experience', 'title'") }} AS experience,
    JSONExtract(payload, 'experience', 'id', 'Nullable(UInt8)') AS experience_id,
    -- «Без опыта», «От 1 года», «От 3 лет», «От 6 лет»; «Не имеет значения» (0) states no minimum
    CAST(multiIf(experience_id = 1, 0, experience_id = 2, 1, experience_id = 3, 3, experience_id = 4, 6, NULL)
        AS Nullable(UInt8)) AS experience_min_years,
    {{ json_dim("'type_of_work', 'title'") }} AS type_of_work,
    {{ json_dim("'place_of_work', 'title'") }} AS place_of_work,
    JSONExtract(payload, 'place_of_work', 'id', 'Nullable(UInt8)') AS place_of_work_id,
    -- «Не имеет значения» (0, 784 of 787 on 2026-10-12) says nothing about the format: NULL, not false
    CAST(multiIf(place_of_work_id = 2, true, place_of_work_id IN (1, 3), false, NULL) AS Nullable(Bool))
        AS remote_work,
    catalogue_ids,
    arrayFlatten(
        arrayMap(
            c -> arrayMap(p -> JSONExtractString(p, 'title'), JSONExtractArrayRaw(c, 'positions')),
            JSONExtractArrayRaw(payload, 'catalogues')
        )
    ) AS catalogue_positions,
    has(catalogue_ids, 33) AS in_it_catalogue,
    arrayMap(x -> JSONExtractString(x, 'name'), JSONExtractArrayRaw(payload, 'professionalSkills')) AS skills,
    JSONExtract(payload, 'trud_vsem', 'is_uploded', 'Nullable(Bool)') AS uploaded_to_trudvsem,
    toDateTime(nullIf(JSONExtractUInt(payload, 'date_pub_to'), 0), 'UTC') AS published_until,
    coalesce(
        JSONExtract(payload, 'is_archive', 'Nullable(Bool)') OR JSONExtract(payload, 'is_closed', 'Nullable(Bool)'),
        false
    ) AS is_archived,
    JSONExtract(payload, '_queries', 'Array(String)') AS collector_queries,
    coalesce(
        has(catalogue_ids, 33) OR match(lowerUTF8(coalesce(title, '')), '{{ it_title_pattern() }}'),
        false
    ) AS is_it_candidate,
    ingested_at
FROM latest
