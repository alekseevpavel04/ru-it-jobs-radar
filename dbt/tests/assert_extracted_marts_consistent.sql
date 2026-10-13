{{ config(enabled=var('extraction_model') != '') }}

-- The marts on extracted fields agree with each other and with their units: the weekly denominators
-- of the skills and field marts are the week's units of the coverage mart; the rows of each field add
-- up to their base; all_portals counts a week's vacancy once, so it is never above the sum of the
-- portal sources and never below the largest of them; every salary level covers each vacancy of the
-- source once. Each failing check returns a row naming it.
WITH coverage AS (
    SELECT source, assumeNotNull(week) AS week, units, units_valid, units_it
    FROM {{ ref('mart_extraction_coverage') }}
    WHERE level = 'week'
),

skills_weeks AS (
    SELECT DISTINCT source, week, vacancies_in_week, vacancies_with_fields, vacancies_it
    FROM {{ ref('mart_extracted_skills_weekly') }}
),

field_sums AS (
    SELECT source, week, field, sum(vacancies) AS n, any(vacancies_base) AS base, any(vacancies_in_week) AS in_week
    FROM {{ ref('mart_extracted_fields_weekly') }}
    GROUP BY source, week, field
),

portal_weeks AS (
    SELECT week, sum(units) AS summed, max(units) AS largest
    FROM coverage
    WHERE source IN ('trudvsem_snapshot', 'trudvsem', 'habr', 'superjob')
    GROUP BY week
),

salary_levels AS (
    SELECT source, level, sum(vacancies) AS n
    FROM {{ ref('mart_extracted_salaries') }}
    GROUP BY source, level
)

SELECT 'skills denominators differ from coverage' AS check_name, s.source AS source, toString(s.week) AS detail
FROM skills_weeks AS s
LEFT JOIN coverage AS c ON s.source = c.source AND s.week = c.week
WHERE s.vacancies_in_week != c.units OR s.vacancies_with_fields != c.units_valid OR s.vacancies_it != c.units_it

UNION ALL

SELECT 'field rows do not add up to their base', f.source, concat(toString(f.week), ' ', f.field)
FROM field_sums AS f
LEFT JOIN coverage AS c ON f.source = c.source AND f.week = c.week
WHERE f.n != f.base OR f.in_week != c.units OR f.base != if(f.field = 'role', c.units_valid, c.units_it)

UNION ALL

SELECT 'all_portals outside its bounds', 'all_portals', toString(c.week)
FROM coverage AS c
INNER JOIN portal_weeks AS p ON c.week = p.week
WHERE c.source = 'all_portals' AND (c.units > p.summed OR c.units < p.largest)

UNION ALL

SELECT 'salary level does not cover the source once', l.source, l.level
FROM salary_levels AS l
INNER JOIN salary_levels AS t ON l.source = t.source AND t.level = 'source'
WHERE l.n != t.n
