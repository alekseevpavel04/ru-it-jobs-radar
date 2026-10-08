{{ config(materialized='view') }}

-- Every sighting of a tracked vacancy: one row per (source, source_id, dt). The snapshot comes
-- from its per-day IT table; Habr is IT as a whole and the API slice is IT by how it is collected.
SELECT
    source, source_id, dt, content_hash, is_closed_state, title, title_norm, employer_name, employer_key,
    location_key, salary_from, salary_to, salary_currency_code, schedule, employment, remote_work, status,
    skills, text_hash, published_at, ingested_at
FROM {{ ref('hist_snapshot_days') }}

UNION ALL

SELECT
    {{ history_observation(
        employer_key="coalesce(toString(employer_id), nullIf(employer_name, 'ИП'))",
        location_key="nullIf(arrayStringConcat(arraySort(cities), '|'), '')",
        is_closed_state='is_archived',
        schedule='NULL',
        remote_work='remote_work',
        status='NULL'
    ) }}
FROM {{ ref('stg_habr') }}

UNION ALL

SELECT
    {{ history_observation(
        employer_key="coalesce(employer_inn, nullIf(employer_name, 'ИП'))",
        location_key='region',
        is_closed_state='false',
        schedule='schedule',
        remote_work='NULL',
        status='NULL'
    ) }}
FROM {{ ref('stg_trudvsem_api') }}
