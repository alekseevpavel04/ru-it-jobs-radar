{{ config(
    schema='staging',
    materialized='table',
    engine='MergeTree()',
    order_by='(family, entity_id)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_sort': 268435456}
) }}

-- One row per vacancy within a source family: trudvsem (the API and the snapshots, one portal) or
-- habr. A vacancy is a dedup entity (vacancy_entity_id, else the portal id), so a
-- vacancy seen in the API and in a snapshot, or posted twice, counts once. Its fields are those of
-- its latest record: the latest staged day, the API before the snapshot on the same day (its text
-- is complete for every record). Families are not merged: Habr and trudvsem rows of the marts stay
-- apart, and the entities that span both (1 on 2026-10-08) count once in each.
WITH ranked AS (
    SELECT *
    FROM {{ ref('int_vacancy_records') }}
    ORDER BY family, entity_id, dt DESC, source = 'trudvsem' DESC, source_id
    LIMIT 1 BY family, entity_id
),

spans AS (
    SELECT
        family,
        entity_id,
        toUInt32(count()) AS n_records,
        arraySort(groupUniqArray(source)) AS sources,
        min(first_seen) AS span_first_seen,
        max(last_seen) AS span_last_seen,
        min(is_closed) AS all_closed,
        if(all_closed, max(closed_at), NULL) AS span_closed_at
    FROM {{ ref('int_vacancy_records') }}
    GROUP BY family, entity_id
),

family_days AS (
    SELECT family, max(dt) AS family_last_dt
    FROM {{ ref('int_vacancy_records') }}
    GROUP BY family
)

SELECT
    r.family AS family,
    r.entity_id AS entity_id,
    s.n_records AS n_records,
    s.sources AS sources,
    r.source AS latest_source,
    r.source_id AS latest_source_id,
    r.dt AS latest_dt,
    s.span_first_seen AS first_seen,
    s.span_last_seen AS last_seen,
    s.all_closed AS is_closed,
    s.span_closed_at AS closed_at,
    f.family_last_dt AS family_last_dt,
    NOT s.all_closed AND s.span_last_seen IS NOT NULL
        AND s.span_last_seen >= f.family_last_dt - ({{ open_now_days() }} - 1) AS is_open_now,
    r.employer_key AS employer_key,
    r.employer_name AS employer_name,
    r.is_sole_proprietor AS is_sole_proprietor,
    r.salary_from AS salary_from,
    r.salary_to AS salary_to,
    r.salary_currency_code AS salary_currency_code,
    r.role_bucket AS role_bucket,
    r.grade AS grade,
    r.grade_is_stated AS grade_is_stated,
    r.work_format AS work_format,
    r.region_group AS region_group,
    r.experience_min_years AS experience_min_years,
    r.no_experience AS no_experience,
    r.skills AS skills
FROM ranked AS r
INNER JOIN spans AS s ON r.family = s.family AND r.entity_id = s.entity_id
INNER JOIN family_days AS f ON r.family = f.family
