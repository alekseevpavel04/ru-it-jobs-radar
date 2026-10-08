{{ config(
    materialized='table',
    engine='MergeTree()',
    order_by='(source, old_source_id)',
    query_settings={'max_threads': 2}
) }}

-- A closed vacancy and the vacancy that replaced it under a new source_id within one source: same
-- employer (INN or Habr company id, never the bare "ИП"), same location, the same title up to case
-- and punctuation and the same text up to case and spacing (macros/history.sql). The new one has
-- to appear between the last sighting of the old one and history_repost_scan_days after its
-- closure; is_repost applies the chosen window. Each old vacancy keeps its earliest successor and
-- each new one its latest predecessor. Near-duplicates with edited text are the cross-source dedup's
-- job (3.5), not this model's.
WITH vacancies AS (
    SELECT
        source,
        source_id,
        min(valid_from) AS first_seen,
        max(last_seen) AS last_seen,
        countIf(is_current) = 0 AS is_closed,
        argMax(tuple(valid_to, employer_key, location_key, title_norm, text_hash), version_no) AS last_state,
        argMin(tuple(employer_key, location_key, title_norm, text_hash), version_no) AS first_state
    FROM {{ ref('hist_vacancy_scd2') }}
    GROUP BY source, source_id
),

closed AS (
    SELECT
        source,
        source_id,
        first_seen,
        last_seen,
        assumeNotNull(last_state.1) AS closed_at,
        assumeNotNull(last_state.2) AS employer_key,
        assumeNotNull(last_state.3) AS location_key,
        last_state.4 AS title_norm,
        assumeNotNull(last_state.5) AS text_hash
    FROM vacancies
    WHERE is_closed
        AND last_state.2 IS NOT NULL AND last_state.3 IS NOT NULL AND last_state.5 IS NOT NULL
        AND last_state.4 != ''
),

started AS (
    SELECT
        source,
        source_id,
        first_seen,
        assumeNotNull(first_state.1) AS employer_key,
        assumeNotNull(first_state.2) AS location_key,
        first_state.3 AS title_norm,
        assumeNotNull(first_state.4) AS text_hash
    FROM vacancies
    WHERE first_state.1 IS NOT NULL AND first_state.2 IS NOT NULL AND first_state.4 IS NOT NULL
),

candidates AS (
    SELECT
        o.source AS source,
        o.source_id AS old_source_id,
        n.source_id AS new_source_id,
        o.first_seen AS old_first_seen,
        o.last_seen AS old_last_seen,
        o.closed_at AS old_closed_at,
        n.first_seen AS new_first_seen,
        toInt32(dateDiff('day', o.closed_at, n.first_seen)) AS gap_days
    FROM closed AS o
    INNER JOIN started AS n
        ON o.source = n.source
        AND o.employer_key = n.employer_key
        AND o.location_key = n.location_key
        AND o.title_norm = n.title_norm
        AND o.text_hash = n.text_hash
    WHERE n.source_id != o.source_id
        AND n.first_seen > o.first_seen
        AND n.first_seen >= o.last_seen
        AND n.first_seen <= o.closed_at + {{ history_repost_scan_days() }}
),

earliest_successor AS (
    SELECT *
    FROM candidates
    ORDER BY source, old_source_id, new_first_seen, new_source_id
    LIMIT 1 BY source, old_source_id
)

SELECT
    source,
    old_source_id,
    new_source_id,
    old_first_seen,
    old_last_seen,
    old_closed_at,
    new_first_seen,
    gap_days,
    gap_days <= {{ history_repost_window_days() }} AS is_repost
FROM earliest_successor
ORDER BY source, new_source_id, old_closed_at DESC, old_source_id
LIMIT 1 BY source, new_source_id
