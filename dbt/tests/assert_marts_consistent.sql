-- The marts agree with each other and with their base: every IT vacancy of a source lands in exactly
-- one row of each salary level and in one row of the format and junior marts; multi-vacancy posts
-- exist only for Telegram; the weekly rows fill either the vacancy or the post columns; the weekly
-- skill denominators are the weekly dynamics totals; the lifetime groups add up to their source; no
-- sole proprietor is named; every bucket is one the rules can produce. Each failing check returns a
-- row naming it.
{%- set buckets = [] -%}
{%- for name in role_bucket_values() if name != 'not_it' %}{% do buckets.append("'" ~ name ~ "'") %}{% endfor %}

WITH base AS (
    SELECT family AS source, count() AS n
    FROM {{ ref('int_vacancy_entities') }}
    WHERE role_bucket != 'not_it'
    GROUP BY family
),

totals AS (
    SELECT concat('mart_salaries level ', level) AS mart, source, sum(vacancies) AS n
    FROM {{ ref('mart_salaries') }}
    GROUP BY source, level
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
    UNION ALL SELECT 'mart_salaries', source, role_bucket FROM {{ ref('mart_salaries') }} WHERE role_bucket != 'all'
)
WHERE (role_bucket = 'multi_post' AND source != 'telegram')
    OR role_bucket NOT IN ({{ buckets | join(', ') }}, 'multi_post')

UNION ALL

SELECT DISTINCT concat('mart_salaries: level ', level, ' does not split by what its name says')
FROM {{ ref('mart_salaries') }}
WHERE (role_group = 'all') != (level IN ('grade', 'region', 'source'))
    OR (role_bucket = 'all') != (level NOT IN ('cell', 'bucket_grade', 'bucket'))
    OR (grade = 'all') != (level NOT IN ('cell', 'bucket_grade', 'group_grade', 'grade'))
    OR (region_group = 'all') != (level NOT IN ('cell', 'region'))

UNION ALL

SELECT concat('mart_skills_weekly: vacancies_in_week differs from mart_weekly_dynamics for ', s.source, ' ', toString(s.week))
FROM (
    SELECT DISTINCT source, week, vacancies_in_week FROM {{ ref('mart_skills_weekly') }}
) AS s
LEFT JOIN (
    SELECT source, week, sum(if(source = 'telegram', telegram_posts, vacancies_active)) AS n
    FROM {{ ref('mart_weekly_dynamics') }}
    GROUP BY source, week
) AS w ON s.source = w.source AND s.week = w.week
WHERE s.vacancies_in_week != w.n

UNION ALL

SELECT concat('mart_lifetime: groups do not add up to the source row for ', g.source)
FROM (
    SELECT source, sum(vacancies) AS n, sum(closed) AS closed, sum(censored) AS censored
    FROM {{ ref('mart_lifetime') }} WHERE level = 'group' GROUP BY source
) AS g
INNER JOIN (
    SELECT source, vacancies AS n, closed, censored FROM {{ ref('mart_lifetime') }} WHERE level = 'source'
) AS t ON g.source = t.source
WHERE g.n != t.n OR g.closed != t.closed OR g.censored != t.censored

UNION ALL

SELECT 'mart_weekly_dynamics: weekly Telegram channels differ between rows of a week'
FROM {{ ref('mart_weekly_dynamics') }}
WHERE source = 'telegram'
GROUP BY week
HAVING uniqExact(telegram_channels_week) != 1 OR max(telegram_channels) > any(telegram_channels_week)

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
