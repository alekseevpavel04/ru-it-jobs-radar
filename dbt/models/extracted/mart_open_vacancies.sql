{{ config(
    enabled=var('extraction_model') != '',
    schema='marts',
    materialized='table',
    engine='MergeTree()',
    order_by='(source, role_bucket, vacancy_entity_id)',
    query_settings={'max_threads': 2, 'join_use_nulls': 1}
) }}

-- One row per portal vacancy (dedup entity) open now and IT by the rule buckets, the rows the MCP tool
-- find_vacancies lists. Open and IT are those of the rule marts (int_vacancy_entities is_open_now,
-- role_bucket != 'not_it'), so the count of a filter agrees with mart_employers and the weekly marts.
-- An entity open in two families is one row, taken from trudvsem when it is open there (its open data
-- may be published), else from the family seen last. Telegram is never a row: its posts are not
-- vacancies of an entity and are published only as weekly aggregates (DECISIONS 2 and 10).
-- Field precedence (models/extracted/_extracted.yml): a value the source states in its own field,
-- then the student's extracted field, then the rule on the title and text.
WITH open_it AS (
    SELECT *
    FROM {{ ref('int_vacancy_entities') }}
    WHERE is_open_now AND role_bucket != 'not_it'
),

picked AS (
    SELECT *
    FROM open_it
    ORDER BY entity_id, family = 'trudvsem' DESC, last_seen DESC, family
    LIMIT 1 BY entity_id
),

spans AS (
    SELECT entity_id, min(first_seen) AS first_seen, max(last_seen) AS last_seen
    FROM open_it
    GROUP BY entity_id
),

fields AS (
    SELECT
        p.family AS source,
        p.entity_id AS entity_id,
        p.employer_name AS employer_name,
        p.role_bucket AS role_bucket,
        r.title AS title,
        r.url AS url,
        r.region AS region,
        r.cities AS cities,
        multiIf(
            p.grade_is_stated, p.grade,
            x.valid AND x.grade IS NOT NULL, x.grade,
            p.grade
        ) AS grade_final,
        if(x.valid AND x.work_format IS NOT NULL, x.work_format, p.work_format) AS work_format_final,
        if(x.valid, x.skills, p.skills) AS skills_final,
        p.salary_from IS NOT NULL OR p.salary_to IS NOT NULL AS source_states_salary,
        p.salary_currency_code = 'RUB' AS source_salary_rub,
        x.valid AND (x.salary_from IS NOT NULL OR x.salary_to IS NOT NULL)
            AND x.salary_currency = 'RUB' AND coalesce(x.salary_period, 'month') = 'month' AS student_salary_rub_month,
        multiIf(source_states_salary, if(source_salary_rub, p.salary_from, NULL),
            student_salary_rub_month, x.salary_from, NULL) AS salary_from_final,
        multiIf(source_states_salary, if(source_salary_rub, p.salary_to, NULL),
            student_salary_rub_month, x.salary_to, NULL) AS salary_to_final,
        s.first_seen AS first_seen,
        s.last_seen AS last_seen
    FROM picked AS p
    INNER JOIN spans AS s ON p.entity_id = s.entity_id
    INNER JOIN {{ ref('int_vacancy_records') }} AS r
        ON p.latest_source = r.source AND p.latest_source_id = r.source_id
    LEFT JOIN {{ ref('int_extracted_entities') }} AS x
        ON x.scope = p.family AND x.entity_id = p.entity_id
)

SELECT
    entity_id AS vacancy_entity_id,
    CAST(source AS LowCardinality(String)) AS source,
    -- Publication rules of docs/sources.md: titles and employer names of the open data only; a link
    -- to a Habr vacancy is a pointer to the source, SuperJob allows no links to single vacancies.
    if(source = 'trudvsem', title, NULL) AS title,
    if(source = 'trudvsem', employer_name, NULL) AS employer_name,
    CAST(role_bucket AS LowCardinality(String)) AS role_bucket,
    CAST(grade_final AS LowCardinality(String)) AS grade,
    CAST({{ region_group(
        'work_format_final',
        "region = 'Город Москва' OR has(cities, 'Москва')",
        "region = 'Город Санкт-Петербург' OR has(cities, 'Санкт-Петербург')",
        'region IS NOT NULL OR notEmpty(cities)'
    ) }} AS LowCardinality(String)) AS region_group,
    CAST(work_format_final AS LowCardinality(String)) AS work_format,
    skills_final AS skills,
    CAST(if(salary_from_final > 0, toUInt32(round(salary_from_final)), NULL) AS Nullable(UInt32)) AS salary_from,
    CAST(if(salary_to_final > 0, toUInt32(round(salary_to_final)), NULL) AS Nullable(UInt32)) AS salary_to,
    if(source = 'superjob', NULL, url) AS url,
    assumeNotNull(first_seen) AS first_seen,
    assumeNotNull(last_seen) AS last_seen
FROM fields
