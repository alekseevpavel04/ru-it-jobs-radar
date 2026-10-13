{{ config(
    enabled=var('extraction_model') != '',
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, level, role, grade)',
    query_settings={'max_threads': 2, 'group_by_use_nulls': 1}
) }}

-- Salaries from the extracted salary fields, by extracted role and grade; mart_salaries keeps the
-- salaries the sources state in their own fields, with the rule role buckets. IT vacancies with valid
-- fields only (role not non_it).
-- Units: portal vacancies over the whole collection period per source family (trudvsem, habr,
-- superjob), and all_portals with a vacancy on several portals counted once (int_extracted_entities);
-- Telegram single-vacancy posts after the weekly cross-post dedup, of the last
-- telegram_window_days days before the newest post, as in the other cross-sectional marts.
-- The quantiles take amounts in RUB per month: currency RUB and period month, or no period, the
-- default the annotation guide leaves to normalization. Nothing is converted: another currency or
-- period is counted and left out; net and gross stay as stated, their counts are in the row.
-- Levels (GROUPING SETS, each over the vacancies of its group): cell (role x grade), role, grade,
-- source; a dimension a level does not split by reads 'all'.
{%- set levels = [
    ('cell', ['role', 'grade']),
    ('role', ['role']),
    ('grade', ['grade']),
    ('source', [])
] %}
{%- set dims = ['role', 'grade'] %}

WITH portal AS (
    SELECT
        scope AS source, role, grade, salary_from, salary_to, salary_currency, salary_gross_net, salary_period,
        first_seen AS seen_from, last_seen AS seen_to
    FROM {{ ref('int_extracted_entities') }}
    WHERE is_it
),

telegram_posts AS (
    SELECT source_id, toDate(published_at) AS published
    FROM {{ ref('stg_telegram') }}
    WHERE NOT is_multi_vacancy AND NOT is_week_duplicate
        AND published_at >= (SELECT max(published_at) FROM {{ ref('stg_telegram') }}) - INTERVAL {{ telegram_window_days() }} DAY
),

telegram AS (
    SELECT
        'telegram' AS source, f.role AS role, f.grade AS grade, f.salary_from AS salary_from, f.salary_to AS salary_to,
        f.salary_currency AS salary_currency, f.salary_gross_net AS salary_gross_net, f.salary_period AS salary_period,
        CAST(p.published AS Nullable(Date)) AS seen_from, CAST(p.published AS Nullable(Date)) AS seen_to
    FROM telegram_posts AS p
    INNER JOIN {{ ref('int_extracted_fields') }} AS f ON f.source = 'telegram' AND p.source_id = f.source_id
    WHERE f.is_it
),

vacancies AS (
    SELECT
        source,
        coalesce(role, 'unknown') AS role,
        coalesce(grade, 'unknown') AS grade,
        salary_from IS NOT NULL OR salary_to IS NOT NULL AS has_salary,
        coalesce(salary_currency = 'RUB', false) AS is_rub,
        coalesce(salary_period, 'month') = 'month' AS is_month,
        has_salary AND is_rub AND is_month AS in_quantiles,
        if(in_quantiles, salary_from, NULL) AS rub_from,
        if(in_quantiles, salary_to, NULL) AS rub_to,
        salary_gross_net,
        seen_from,
        seen_to
    FROM (
        SELECT * FROM portal
        UNION ALL
        SELECT * FROM telegram
    )
),

periods AS (
    SELECT source, min(seen_from) AS period_from, max(seen_to) AS period_to
    FROM vacancies
    GROUP BY source
),

cells AS (
    SELECT
        source,
        role,
        grade,
        count() AS vacancies,
        countIf(has_salary) AS with_salary,
        countIf(in_quantiles) AS with_salary_rub_month,
        countIf(has_salary AND NOT is_rub) AS with_salary_other_currency,
        countIf(has_salary AND is_rub AND NOT is_month) AS with_salary_other_period,
        countIf(in_quantiles AND salary_gross_net = 'net') AS with_net,
        countIf(in_quantiles AND salary_gross_net = 'gross') AS with_gross,
        count(rub_from) AS with_from,
        count(rub_to) AS with_to,
        quantilesExactInclusive(0.25, 0.5, 0.75)(rub_from) AS from_q,
        quantilesExactInclusive(0.25, 0.5, 0.75)(rub_to) AS to_q
    FROM vacancies
    GROUP BY GROUPING SETS (
    {%- for name, keys in levels %}
        ({{ (['source'] + keys) | join(', ') }}){{ ',' if not loop.last }}
    {%- endfor %}
    )
)

SELECT
    CAST(assumeNotNull(c.source) AS LowCardinality(String)) AS source,
    CAST(multiIf(
    {%- for name, keys in levels %}
        {%- if not loop.last %}
        {% for d in dims %}c.{{ d }} IS {{ 'NOT ' if d in keys else '' }}NULL{{ ' AND ' if not loop.last }}{% endfor %}, '{{ name }}',
        {%- else %}
        '{{ name }}'
        {%- endif %}
    {%- endfor %}
    ) AS LowCardinality(String)) AS level,
    {%- for d in dims %}
    CAST(coalesce(c.{{ d }}, 'all') AS LowCardinality(String)) AS {{ d }},
    {%- endfor %}
    c.vacancies AS vacancies,
    c.with_salary AS with_salary,
    c.with_salary_rub_month AS with_salary_rub_month,
    c.with_salary_other_currency AS with_salary_other_currency,
    c.with_salary_other_period AS with_salary_other_period,
    c.with_net AS with_net,
    c.with_gross AS with_gross,
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
