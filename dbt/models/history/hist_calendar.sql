{{ config(materialized='view') }}

-- Collected days per source and what a vacancy missing after each of them means. A vacancy last
-- seen on dt and not seen again by closure_confirmed_dt is closed from next_complete_dt on, unless
-- the days between dt and next_complete_dt are a gap (no complete collection on more than
-- history_max_step_days - 1 of them): then the closure is known to have happened, but not when.
WITH days AS (
    SELECT source, dt, count() AS n_rows
    FROM {{ ref('hist_observations') }}
    GROUP BY source, dt
),

flagged AS (
    SELECT
        source,
        dt,
        n_rows,
        median(n_rows) OVER (PARTITION BY source) AS median_rows,
        n_rows >= {{ history_complete_day_ratio() }} * median_rows AS is_complete
    FROM days
),

complete AS (
    SELECT source, arraySort(groupArrayIf(dt, is_complete)) AS complete_days
    FROM flagged
    GROUP BY source
),

ahead AS (
    SELECT
        f.source AS source,
        f.dt AS dt,
        f.n_rows AS n_rows,
        f.median_rows AS median_rows,
        f.is_complete AS is_complete,
        CAST({{ history_absent_days_to_close('f.source') }} AS Nullable(UInt8)) AS absent_days_to_close,
        arrayFilter(d -> d > f.dt, c.complete_days) AS later_complete
    FROM flagged AS f
    INNER JOIN complete AS c ON f.source = c.source
)

SELECT
    source,
    dt,
    n_rows,
    median_rows,
    is_complete,
    absent_days_to_close,
    if(length(later_complete) >= 1, later_complete[1], NULL) AS next_complete_dt,
    if(
        absent_days_to_close IS NOT NULL AND length(later_complete) >= absent_days_to_close,
        later_complete[assumeNotNull(absent_days_to_close)],
        NULL
    ) AS closure_confirmed_dt,
    CAST(if(length(later_complete) >= 1, dateDiff('day', dt, later_complete[1]) - 1, NULL) AS Nullable(UInt16))
        AS unknown_days_after,
    CAST(unknown_days_after >= {{ history_max_step_days() }} AS Nullable(Bool)) AS next_complete_after_gap
FROM ahead
