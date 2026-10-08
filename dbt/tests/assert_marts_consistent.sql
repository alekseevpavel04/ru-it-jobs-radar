-- The marts agree with each other and with their base: every IT vacancy of a source lands in exactly
-- one cell of the salary, format and junior marts; multi-vacancy posts exist only for Telegram; the
-- weekly rows fill either the vacancy or the post columns; no sole proprietor is named; every bucket
-- is one the rules can produce. Each failing check returns a row naming it.
{%- set buckets = [] -%}
{%- for name in role_bucket_values() if name != 'not_it' %}{% do buckets.append("'" ~ name ~ "'") %}{% endfor %}

WITH base AS (
    SELECT family AS source, count() AS n
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'
    GROUP BY family
),

totals AS (
    SELECT 'mart_salaries' AS mart, source, sum(vacancies) AS n FROM {{ ref('mart_salaries') }} GROUP BY source
    UNION ALL
    SELECT 'mart_work_format', source, sum(vacancies) FROM {{ ref('mart_work_format') }} WHERE source != 'telegram' GROUP BY source
    UNION ALL
    SELECT 'mart_junior', source, sum(vacancies) FROM {{ ref('mart_junior') }} WHERE source != 'telegram' GROUP BY source
),

telegram_totals AS (
    SELECT
        (SELECT sum(vacancies) FROM {{ ref('mart_work_format') }} WHERE source = 'telegram') AS in_format,
        (SELECT sum(vacancies) FROM {{ ref('mart_junior') }} WHERE source = 'telegram') AS in_junior
)

SELECT concat(t.mart, ' total differs from int_vacancy_entities for ', t.source) AS failed_check
FROM totals AS t
LEFT JOIN base AS b ON t.source = b.source
WHERE t.n != b.n

UNION ALL

SELECT 'telegram totals of mart_work_format and mart_junior differ'
FROM telegram_totals
WHERE in_format != in_junior

UNION ALL

SELECT concat(mart, ': multi_post outside telegram')
FROM (
    SELECT 'mart_weekly_dynamics' AS mart, source, role_bucket FROM {{ ref('mart_weekly_dynamics') }}
    UNION ALL SELECT 'mart_skills', source, role_bucket FROM {{ ref('mart_skills') }}
    UNION ALL SELECT 'mart_work_format', source, role_bucket FROM {{ ref('mart_work_format') }}
    UNION ALL SELECT 'mart_junior', source, role_bucket FROM {{ ref('mart_junior') }}
    UNION ALL SELECT 'mart_salaries', source, role_bucket FROM {{ ref('mart_salaries') }}
)
WHERE (role_bucket = 'multi_post' AND source != 'telegram')
    OR role_bucket NOT IN ({{ buckets | join(', ') }}, 'multi_post')

UNION ALL

SELECT 'mart_weekly_dynamics: vacancy and post columns mixed'
FROM {{ ref('mart_weekly_dynamics') }}
WHERE (source = 'telegram') != (vacancies_active IS NULL AND vacancies_new IS NULL AND vacancies_closed IS NULL)
    OR (source = 'telegram') != (telegram_posts IS NOT NULL)

UNION ALL

SELECT 'mart_employers: sole proprietor named'
FROM {{ ref('mart_employers') }}
WHERE employer_name = 'ИП' OR startsWith(employer_name, 'ИП ')
    OR startsWith(lowerUTF8(employer_name), 'индивидуальный предприниматель')
