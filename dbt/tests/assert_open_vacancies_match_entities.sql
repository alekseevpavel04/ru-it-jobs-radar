{{ config(enabled=var('extraction_model') != '') }}

-- mart_open_vacancies holds every open IT entity of int_vacancy_entities once and nothing else, and
-- what it publishes stays within the rules: no sole proprietor's name after «ИП» and no trudvsem
-- link of a sole proprietor (its 15-digit OGRNIP in the card path; the cleaner removes those links).
-- Each failing check returns a row naming it.
WITH expected AS (
    SELECT DISTINCT entity_id
    FROM {{ ref('int_vacancy_entities') }}
    WHERE is_open_now AND role_bucket != 'not_it'
),

listed AS (
    SELECT vacancy_entity_id AS entity_id, employer_name, url
    FROM {{ ref('mart_open_vacancies') }}
)

SELECT 'open IT entity missing from the mart' AS check_name, e.entity_id AS detail
FROM expected AS e
LEFT ANTI JOIN listed AS l ON e.entity_id = l.entity_id

UNION ALL

SELECT 'mart row that is not an open IT entity', l.entity_id
FROM listed AS l
LEFT ANTI JOIN expected AS e ON l.entity_id = e.entity_id

UNION ALL

SELECT 'person name after ИП', entity_id
FROM listed
WHERE match(coalesce(employer_name, ''), '(?i)^(ИП|индивидуальный предприниматель)\\s+\\S')

UNION ALL

SELECT 'sole proprietor link', entity_id
FROM listed
WHERE match(coalesce(url, ''), 'trudvsem\\.ru/vacancy/card/\\d{15}/')
