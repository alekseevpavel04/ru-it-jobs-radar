{{ config(
    materialized='table',
    engine='MergeTree()',
    order_by='(source, source_id)',
    query_settings={'max_threads': 2}
) }}

-- One row per vacancy within its source: how long it was seen and whether its end was observed.
-- Survival models need the two censorings right: is_censored (still open, or a source where absence
-- never closes) and is_left_censored (already open on the first collected day, so first_seen is not
-- its start; published_at is the source's own date).
WITH bounds AS (
    SELECT source, min(dt) AS first_collected_dt
    FROM {{ ref('hist_calendar') }}
    GROUP BY source
),

per_vacancy AS (
    SELECT
        source,
        source_id,
        min(valid_from) AS first_seen,
        max(last_seen) AS last_seen,
        countIf(is_current) = 0 AS is_closed,
        argMax(tuple(valid_to), version_no).1 AS last_valid_to,
        toUInt32(count()) AS n_versions,
        toUInt32(countIf(change_reason = 'reopened')) AS n_reopened,
        toUInt32(countIf(hasAny(changed_fields, ['salary_from', 'salary_to', 'salary_currency_code'])))
            AS n_salary_changes,
        argMin(tuple(published_at, salary_from, salary_to, salary_currency_code), version_no) AS first_state,
        argMax(tuple(salary_from, salary_to, salary_currency_code), version_no) AS last_state
    FROM {{ ref('hist_vacancy_scd2') }}
    GROUP BY source, source_id
),

-- Nullable, so that a vacancy without a pair reads NULL from the LEFT JOIN, not ''.
reposts AS (
    SELECT source, toNullable(old_source_id) AS old_source_id, toNullable(new_source_id) AS new_source_id
    FROM {{ ref('hist_reposts') }}
    WHERE is_repost
)

SELECT
    v.source AS source,
    v.source_id AS source_id,
    v.first_seen AS first_seen,
    v.last_seen AS last_seen,
    v.is_closed AS is_closed,
    if(v.is_closed, v.last_valid_to, NULL) AS closed_at,
    NOT v.is_closed AS is_censored,
    v.first_seen = b.first_collected_dt AS is_left_censored,
    toUInt32(if(
        v.is_closed,
        dateDiff('day', v.first_seen, assumeNotNull(v.last_valid_to)),
        dateDiff('day', v.first_seen, v.last_seen) + 1
    )) AS duration_days,
    CAST({{ history_absent_days_to_close('v.source') }} AS Nullable(UInt8)) AS absent_days_to_close,
    v.n_versions AS n_versions,
    v.n_reopened AS n_reopened,
    v.n_salary_changes AS n_salary_changes,
    v.first_state.1 AS published_at,
    v.first_state.2 AS salary_from_first,
    v.first_state.3 AS salary_to_first,
    v.first_state.4 AS salary_currency_code_first,
    v.last_state.1 AS salary_from_last,
    v.last_state.2 AS salary_to_last,
    v.last_state.3 AS salary_currency_code_last,
    prev.old_source_id AS repost_of_source_id,
    next.new_source_id AS reposted_as_source_id,
    next.new_source_id IS NOT NULL AS closed_by_repost
FROM per_vacancy AS v
INNER JOIN bounds AS b ON v.source = b.source
LEFT JOIN reposts AS prev ON v.source = prev.source AND v.source_id = prev.new_source_id
LEFT JOIN reposts AS next ON v.source = next.source AND v.source_id = next.old_source_id
