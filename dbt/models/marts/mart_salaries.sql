{{ config(
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, level, role_group, role_bucket, grade, region_group)',
    query_settings={'max_threads': 2, 'group_by_use_nulls': 1}
) }}

-- Salaries as the source states them: from and to of RUB amounts, each quantile over the vacancies
-- that state that bound. The sources state monthly amounts without a period field (the trudvsem
-- portal, Habr Career and SuperJob publish a monthly salary), so nothing is converted; amounts in another
-- currency are counted, not converted, and left out of the quantiles. Habr's own estimate
-- (predictedSalary) is never used. Over every vacancy entity of the collection period
-- (int_vacancy_entities, latest record), per source family.
--
-- Rows come at several levels (GROUPING SETS), so a group median is computed over the vacancies of
-- the group, not stitched from cell medians: cell (role bucket x grade x region group), then the
-- totals listed in levels below. A dimension a level does not split by reads 'all'. Every level
-- covers each vacancy of the source exactly once, so rows of one level add up to the source total.
{%- set levels = [
    ('cell', ['role_group', 'role_bucket', 'grade', 'region_group']),
    ('bucket_grade', ['role_group', 'role_bucket', 'grade']),
    ('bucket', ['role_group', 'role_bucket']),
    ('group_grade', ['role_group', 'grade']),
    ('group', ['role_group']),
    ('grade', ['grade']),
    ('region', ['region_group']),
    ('source', [])
] %}
{%- set dims = ['role_group', 'role_bucket', 'grade', 'region_group'] %}

WITH vacancies AS (
    SELECT
        e.family AS source,
        g.role_group AS role_group,
        e.role_bucket AS role_bucket,
        e.grade AS grade,
        e.region_group AS region_group,
        e.salary_from IS NOT NULL OR e.salary_to IS NOT NULL AS has_salary,
        coalesce(e.salary_currency_code = 'RUB', false) AS is_rub,
        if(is_rub, e.salary_from, NULL) AS rub_from,
        if(is_rub, e.salary_to, NULL) AS rub_to
    FROM {{ ref('int_vacancy_entities') }} AS e
    INNER JOIN {{ ref('role_groups') }} AS g ON e.role_bucket = g.role_bucket
    WHERE e.role_bucket != 'not_it'
),

periods AS (
    SELECT family AS source, min(first_seen) AS period_from, max(last_seen) AS period_to
    FROM {{ ref('int_vacancy_entities') }}
    GROUP BY family
),

-- group_by_use_nulls: a dimension outside the grouping set reads NULL here (never a data value: the
-- dimensions are never NULL in vacancies), which tells the level apart.
cells AS (
    SELECT
        source,
        role_group,
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
