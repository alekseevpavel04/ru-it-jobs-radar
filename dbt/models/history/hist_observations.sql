{{ config(materialized='view') }}

-- Every sighting of a tracked vacancy: one row per (source, source_id, dt). The snapshot comes
-- from its per-day IT table; Habr is IT as a whole, and the API and SuperJob slices are IT by how they
-- are collected.
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

UNION ALL

-- The whole collected slice, like the API: every run is a full snapshot of it, so a vacancy that
-- leaves the IT filter (its title changed) still reads as present. A person employer has no client
-- id and no key.
SELECT
    {{ history_observation(
        employer_key='toString(employer_id)',
        location_key="nullIf(arrayStringConcat(arraySort(cities), '|'), '')",
        is_closed_state='is_archived',
        schedule='NULL',
        remote_work='remote_work',
        status='NULL',
        employment='type_of_work'
    ) }}
FROM {{ ref('stg_superjob') }}
