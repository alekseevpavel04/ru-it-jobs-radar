{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, level, role_group)',
    query_settings={'max_threads': 2, 'group_by_use_nulls': 1, 'join_use_nulls': 1}
) }}

-- How long IT vacancies stayed up, per source and role group, from hist_vacancy_lifetime: one row
-- per source vacancy (a repost under a new id is another vacancy here). Only sources whose absence
-- closes a vacancy (the snapshots, Habr); the API never shows a closure.
--
-- The quantiles describe closures seen in the collection window, nothing more: a vacancy still open
-- or closed in a gap has no end date, one open on the first collected day has no start, and long
-- lifetimes have had less time to end, so the quantiles lean short and are not a survival estimate.
-- The Kaplan-Meier and Cox estimates with censoring are results/models/survival.json.
WITH vacancies AS (
    SELECT
        l.source AS source,
        g.role_group AS role_group,
        l.is_closed AS is_closed,
        l.closed_in_gap AS closed_in_gap,
        l.is_censored AS is_censored,
        l.is_left_censored AS is_left_censored,
        l.closed_by_repost AS closed_by_repost,
        l.duration_days AS duration_days,
        l.is_closed AND l.closed_at IS NOT NULL AS closure_dated,
        closure_dated AND NOT l.is_left_censored AND NOT l.closed_by_repost AS in_quantiles
    FROM {{ ref('hist_vacancy_lifetime') }} AS l
    INNER JOIN {{ ref('int_vacancy_records') }} AS r ON l.source = r.source AND l.source_id = r.source_id
    INNER JOIN {{ ref('role_groups') }} AS g ON r.role_bucket = g.role_bucket
    WHERE l.absent_days_to_close IS NOT NULL
),

windows AS (
    SELECT source, min(dt) AS period_from, max(dt) AS period_to
    FROM {{ ref('hist_calendar') }}
    GROUP BY source
),

per_group AS (
    SELECT
        source,
        role_group,
        count() AS vacancies,
        countIf(NOT is_closed) AS still_open,
        countIf(closure_dated) AS closed,
        countIf(closed_in_gap) AS closed_in_gap,
        countIf(is_censored) AS censored,
        countIf(is_left_censored) AS left_censored,
        countIf(closure_dated AND closed_by_repost) AS closed_by_repost,
        countIf(in_quantiles) AS closed_in_quantiles,
        quantilesExactInclusiveIf(0.25, 0.5, 0.75, 0.9)(duration_days, in_quantiles) AS q
    FROM vacancies
    GROUP BY GROUPING SETS ((source, role_group), (source))
)

SELECT
    CAST(assumeNotNull(r.source) AS LowCardinality(String)) AS source,
    CAST(if(r.role_group IS NULL, 'source', 'group') AS LowCardinality(String)) AS level,
    CAST(coalesce(r.role_group, 'all') AS LowCardinality(String)) AS role_group,
    r.vacancies AS vacancies,
    r.still_open AS still_open,
    r.closed AS closed,
    r.closed_in_gap AS closed_in_gap,
    r.censored AS censored,
    r.left_censored AS left_censored,
    r.closed_by_repost AS closed_by_repost,
    r.closed_in_quantiles AS closed_in_quantiles,
    {%- for i, name in [(1, 'p25'), (2, 'median'), (3, 'p75'), (4, 'p90')] %}
    if(r.closed_in_quantiles >= {{ lifetime_min_n() }}, round(r.q[{{ i }}], 1), NULL) AS duration_days_{{ name }},
    {%- endfor %}
    w.period_from AS period_from,
    w.period_to AS period_to,
    toUInt32(dateDiff('day', w.period_from, w.period_to) + 1) AS window_days
FROM per_group AS r
INNER JOIN windows AS w ON r.source = w.source
