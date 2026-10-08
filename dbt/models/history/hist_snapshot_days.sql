{{ config(
    materialized='incremental_by_day',
    raw_source='trudvsem_snapshot',
    engine='MergeTree()',
    partition_by='dt',
    order_by='source_id',
    query_settings={'max_threads': 2, 'max_block_size': 8192}
) }}

-- IT candidates of one snapshot day in the columns the history compares, about 10k of 550k rows.
-- The history reads this table instead of the whole snapshot staging, so its size grows with the
-- IT slice, not with the portal.
SELECT
    {{ history_observation(
        employer_key="coalesce(employer_inn, nullIf(employer_name, 'ИП'))",
        location_key='region',
        is_closed_state="is_deleted OR coalesce(status IN ('Не опубликовано', 'Не одобрено'), false)",
        schedule='schedule',
        remote_work='NULL',
        status='status'
    ) }}
FROM {{ ref('stg_trudvsem_snapshot') }}
WHERE dt = {{ staging_day() }} AND is_it_candidate
