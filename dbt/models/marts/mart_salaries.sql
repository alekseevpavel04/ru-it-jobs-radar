{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, role_bucket, grade, region_group)',
    query_settings={'max_threads': 2}
) }}

-- Salaries as the source states them: from and to of RUB amounts, each quantile over the vacancies
-- that state that bound. Both sources state monthly amounts without a period field (the trudvsem
-- portal and Habr Career publish a monthly salary), so nothing is converted; amounts in another
-- currency are counted, not converted, and left out of the quantiles. Habr's own estimate
-- (predictedSalary) is never used. One row per source family x role bucket x grade x region group,
-- over every vacancy entity of the collection period (int_vacancy_entities, latest record).
WITH vacancies AS (
    SELECT
        family AS source,
        role_bucket,
        grade,
        region_group,
        salary_from IS NOT NULL OR salary_to IS NOT NULL AS has_salary,
        coalesce(salary_currency_code = 'RUB', false) AS is_rub,
        if(is_rub, salary_from, NULL) AS rub_from,
        if(is_rub, salary_to, NULL) AS rub_to
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'
),

periods AS (
    SELECT family AS source, min(first_seen) AS period_from, max(last_seen) AS period_to
    FROM {{ ref('int_vacancy_entities') }}
    GROUP BY family
),

cells AS (
    SELECT
        source,
        role_bucket,
        grade,
        region_group,
        count() AS vacancies,
        countIf(has_salary AND is_rub) AS with_salary_rub,
        countIf(has_salary AND NOT is_rub) AS with_salary_other_currency,
        count(rub_from) AS with_from,
        count(rub_to) AS with_to,
        quantilesExactInclusive(0.25, 0.5, 0.75)(rub_from) AS from_q,
        quantilesExactInclusive(0.25, 0.5, 0.75)(rub_to) AS to_q
    FROM vacancies
    GROUP BY source, role_bucket, grade, region_group
)

SELECT
    CAST(c.source AS LowCardinality(String)) AS source,
    CAST(c.role_bucket AS LowCardinality(String)) AS role_bucket,
    CAST(c.grade AS LowCardinality(String)) AS grade,
    CAST(c.region_group AS LowCardinality(String)) AS region_group,
    c.vacancies AS vacancies,
    c.with_salary_rub AS with_salary_rub,
    c.with_salary_other_currency AS with_salary_other_currency,
    c.with_from AS with_from,
    c.with_to AS with_to,
    {%- for bound in ['from', 'to'] %}
    {%- for q, name in [(1, 'p25'), (2, 'median'), (3, 'p75')] %}
    if(c.with_{{ bound }} >= {{ salary_min_cell() }}, round(c.{{ bound }}_q[{{ q }}]), NULL) AS salary_{{ bound }}_{{ name }},
    {%- endfor %}
    {%- endfor %}
    p.period_from AS period_from,
    p.period_to AS period_to
FROM cells AS c
INNER JOIN periods AS p ON c.source = p.source
