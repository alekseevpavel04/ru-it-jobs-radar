{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, week, role_bucket)',
    query_settings={'max_threads': 2, 'max_bytes_before_external_group_by': 268435456}
) }}

-- Week x source x role bucket. Vacancy sources (trudvsem_snapshot, trudvsem API, habr) count dedup
-- entities within the source: active = seen open on at least one collected day of the week, new =
-- first seen that week (never on the source's first collected day, when everything is already
-- there), closed = closure confirmed by the history (hist_vacancy_lifetime) that week. Telegram
-- counts posts, a separate measure in separate columns: a post is not a vacancy and is never added
-- to the vacancy counts. Every vacancy and post has one bucket, so rows of a week add up across
-- buckets.
WITH entities AS (
    SELECT family, entity_id, role_bucket
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'
),

records AS (
    SELECT r.source AS source, r.source_id AS source_id, r.entity_id AS entity_id, e.role_bucket AS role_bucket,
        r.first_seen AS first_seen, r.is_closed AS is_closed, r.closed_at AS closed_at
    FROM {{ ref('int_vacancy_records') }} AS r
    INNER JOIN entities AS e ON r.family = e.family AND r.entity_id = e.entity_id
),

calendar AS (
    SELECT
        source,
        toMonday(dt) AS week,
        toUInt8(count()) AS days_collected,
        toUInt8(countIf(is_complete)) AS complete_days,
        any(absent_days_to_close) AS absent_days_to_close
    FROM {{ ref('hist_calendar') }}
    GROUP BY source, week
),

first_days AS (
    SELECT source, min(dt) AS first_dt
    FROM {{ ref('hist_calendar') }}
    GROUP BY source
),

active AS (
    SELECT r.source AS source, toMonday(o.dt) AS week, r.role_bucket AS role_bucket, uniqExact(r.entity_id) AS n
    FROM {{ ref('hist_observations') }} AS o
    INNER JOIN records AS r ON o.source = r.source AND o.source_id = r.source_id
    WHERE NOT o.is_closed_state
    GROUP BY source, week, role_bucket
),

-- Within one source an entity is new when its first record appears and closed when its last one
-- closes.
per_entity AS (
    SELECT
        source,
        entity_id,
        any(role_bucket) AS role_bucket,
        min(first_seen) AS entity_first_seen,
        min(is_closed) AS all_closed,
        max(closed_at) AS entity_closed_at
    FROM records
    WHERE first_seen IS NOT NULL
    GROUP BY source, entity_id
),

events AS (
    SELECT p.source AS source, toMonday(assumeNotNull(p.entity_first_seen)) AS week, p.role_bucket AS role_bucket,
        1 AS is_new, 0 AS is_closed
    FROM per_entity AS p
    INNER JOIN first_days AS f ON p.source = f.source
    WHERE p.entity_first_seen > f.first_dt

    UNION ALL

    SELECT source, toMonday(assumeNotNull(entity_closed_at)), role_bucket, 0, 1
    FROM per_entity
    WHERE all_closed AND entity_closed_at IS NOT NULL
),

vacancy_cells AS (
    SELECT source, week, role_bucket, sum(active_n) AS active_n, sum(new_n) AS new_n, sum(closed_n) AS closed_n
    FROM (
        SELECT source, week, role_bucket, n AS active_n, 0 AS new_n, 0 AS closed_n FROM active
        UNION ALL
        SELECT source, week, role_bucket, 0, sum(is_new), sum(is_closed) FROM events GROUP BY source, week, role_bucket
    )
    GROUP BY source, week, role_bucket
),

telegram_cells AS (
    SELECT
        published_week AS week,
        role_bucket,
        countIf(NOT is_week_duplicate) AS posts,
        count() AS posts_before_dedup,
        uniqExact(channel) AS channels
    FROM {{ ref('stg_telegram') }}
    WHERE role_bucket != 'not_it'
    GROUP BY week, role_bucket
)

SELECT
    v.week AS week,
    CAST(v.source AS LowCardinality(String)) AS source,
    CAST(v.role_bucket AS LowCardinality(String)) AS role_bucket,
    CAST(c.days_collected AS Nullable(UInt8)) AS days_collected,
    CAST(c.complete_days AS Nullable(UInt8)) AS complete_days,
    v.week = toMonday(f.first_dt) AS is_first_collected_week,
    CAST(v.active_n AS Nullable(UInt64)) AS vacancies_active,
    CAST(v.new_n AS Nullable(UInt64)) AS vacancies_new,
    -- the API is a slice of records modified since the last run: absence closes nothing there
    CAST(if(c.absent_days_to_close IS NULL, NULL, v.closed_n) AS Nullable(UInt64)) AS vacancies_closed,
    CAST(NULL AS Nullable(UInt64)) AS telegram_posts,
    CAST(NULL AS Nullable(UInt64)) AS telegram_posts_before_dedup,
    CAST(NULL AS Nullable(UInt64)) AS telegram_channels
FROM vacancy_cells AS v
INNER JOIN calendar AS c ON v.source = c.source AND v.week = c.week
INNER JOIN first_days AS f ON v.source = f.source

UNION ALL

SELECT
    week,
    CAST('telegram' AS LowCardinality(String)),
    CAST(role_bucket AS LowCardinality(String)),
    CAST(NULL AS Nullable(UInt8)),
    CAST(NULL AS Nullable(UInt8)),
    false,
    CAST(NULL AS Nullable(UInt64)),
    CAST(NULL AS Nullable(UInt64)),
    CAST(NULL AS Nullable(UInt64)),
    CAST(posts AS Nullable(UInt64)),
    CAST(posts_before_dedup AS Nullable(UInt64)),
    CAST(channels AS Nullable(UInt64))
FROM telegram_cells
