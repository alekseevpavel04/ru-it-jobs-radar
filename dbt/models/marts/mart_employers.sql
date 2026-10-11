{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, employer_rank)',
    query_settings={'max_threads': 2, 'join_use_nulls': 1}
) }}

-- Top employers per source family by open IT vacancies: trudvsem by INN, Habr by company id. Open
-- means not closed and seen within open_now_days of the family's latest collected day (as_of). Sole
-- proprietors are left out: their name is personal data and the raw cleaner has already replaced it
-- with «ИП» and removed the INN, so they have no key to group by. Lifetime is the median
-- duration_days of the employer's IT vacancies whose closure was observed and whose start was
-- (not left-censored), per source vacancy, printed from lifetime_min_closed of them.
WITH vacancies AS (
    SELECT *
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it' AND NOT is_sole_proprietor AND employer_key IS NOT NULL
),

per_employer AS (
    SELECT
        family AS source,
        assumeNotNull(employer_key) AS employer_key,
        argMax(employer_name, latest_dt) AS employer_name,
        countIf(is_open_now) AS open_it_vacancies,
        count() AS it_vacancies_seen,
        any(family_last_dt) AS as_of
    FROM vacancies
    GROUP BY source, employer_key
),

lifetimes AS (
    SELECT
        family AS source,
        assumeNotNull(employer_key) AS employer_key,
        count() AS closed_vacancies,
        quantileExactInclusive(0.5)(duration_days) AS median_days
    FROM {{ ref('int_vacancy_records') }}
    WHERE role_bucket != 'not_it' AND NOT is_sole_proprietor AND employer_key IS NOT NULL
        AND is_closed AND closed_at IS NOT NULL AND NOT coalesce(is_left_censored, true)
    GROUP BY source, employer_key
),

ranked AS (
    SELECT
        *,
        row_number() OVER (PARTITION BY source ORDER BY open_it_vacancies DESC, it_vacancies_seen DESC, employer_key)
            AS employer_rank
    FROM per_employer
    WHERE open_it_vacancies > 0
)

SELECT
    CAST(r.source AS LowCardinality(String)) AS source,
    toUInt32(r.employer_rank) AS employer_rank,
    r.employer_key AS employer_key,
    CAST(if(r.source = 'habr', 'habr_company_id', 'inn') AS LowCardinality(String)) AS employer_key_kind,
    r.employer_name AS employer_name,
    r.open_it_vacancies AS open_it_vacancies,
    r.it_vacancies_seen AS it_vacancies_seen,
    coalesce(l.closed_vacancies, 0) AS lifetime_closed_vacancies,
    if(coalesce(l.closed_vacancies, 0) >= {{ lifetime_min_closed() }}, l.median_days, NULL) AS lifetime_median_days,
    r.as_of AS as_of
FROM ranked AS r
LEFT JOIN lifetimes AS l ON r.source = l.source AND r.employer_key = l.employer_key
WHERE r.employer_rank <= {{ employers_top_n() }}
