{{ config(
    materialized='incremental_by_day',
    raw_source='trudvsem_snapshot',
    skip_unchanged_days=true,
    engine='MergeTree()',
    partition_by='dt',
    order_by='source_id',
    query_settings={'max_threads': 2, 'max_block_size': 8192, 'min_insert_block_size_bytes': 33554432}
) }}

WITH latest AS (
    SELECT *
    FROM {{ source('raw', 'trudvsem_snapshot') }} FINAL
    WHERE source = 'trudvsem_snapshot' AND dt = {{ staging_day() }}
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
    -- Text is kept for IT candidates only: about 98% of snapshot rows are other professions, their
    -- text has no reader downstream and was 1.40 of 2.13 GiB of this table on 5 days. It stays in
    -- raw.vacancies under (source, source_id, dt, content_hash). Written through '' because
    -- if(..., text, NULL) keeps every string under the null mask and saves nothing on disk.
    nullIf(if(is_it_candidate, assumeNotNull(latest.text), ''), '') AS text,
    coalesce(length(trimBoth(latest.text)) > 0, false) AS has_text,
    {{ json_dim("'fullCompanyName'") }} AS employer_name,
    {{ json_string("'company', 'inn'") }} AS employer_inn,
    {{ json_dim("'companyBusinessSize'") }} AS company_size,
    {{ json_dim("'regionName'") }} AS region,
    {{ json_string("'stateRegionCode'") }} AS region_code,
    {{ json_string("'vacancyAddress'") }} AS address,
    {{ json_string("'salary'") }} AS salary_text,
    {{ stated_amount("JSONExtractString(payload, 'salaryMin')") }} AS salary_from,
    {{ stated_amount("JSONExtractString(payload, 'salaryMax')") }} AS salary_to,
    CAST(NULL AS Nullable(String)) AS salary_currency,
    -- The snapshot does not repeat the currency; the same portal's API states «руб.» on every record.
    if(salary_from IS NOT NULL OR salary_to IS NOT NULL, 'RUB', NULL) AS salary_currency_code,
    {{ json_dim("'scheduleType'") }} AS schedule,
    {{ json_dim("'busyType'") }} AS employment,
    toUInt8OrNull(JSONExtractString(payload, 'experienceRequirements')) AS experience_min_years,
    {{ json_dim("'educationRequirements', 'educationType'") }} AS education,
    {{ json_dim("'codeProfessionalSphere'") }} AS sphere_code,
    {{ json_dim("'professionalSphereName'") }} AS sphere_name,
    {{ json_dim("'codeProfession'") }} AS profession_code,
    {{ json_dim("'typicalPosition'") }} AS typical_position,
    {{ json_dim("'sourceType'") }} AS source_type,
    {{ json_dim("'status'") }} AS status,
    JSONExtractString(payload, 'deleted') = 'true' AS is_deleted,
    toUInt16OrNull(JSONExtractString(payload, 'workPlaces')) AS work_places,
    {{ source_time("JSONExtractString(payload, 'datePublished')") }} AS date_published,
    {{ source_time("JSONExtractString(payload, 'changeTime')") }} AS change_time,
    JSONExtract(payload, 'skills', 'Array(String)') AS skills,
    toLowCardinality(JSONExtractString(payload, 'snapshot_version')) AS snapshot_version,
    {{ is_it_candidate("sphere_code = 'InformationTechnology'", "typical_position", "title") }} AS is_it_candidate,
    ingested_at
FROM latest
