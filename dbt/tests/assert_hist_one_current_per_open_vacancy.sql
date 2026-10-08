-- An open vacancy has exactly one current version, a closed one none, and only the last version
-- can be current. Every vacancy of the SCD2 table has its lifetime row.
WITH per_vacancy AS (
    SELECT
        source,
        source_id,
        countIf(is_current) AS n_current,
        max(version_no) AS last_version,
        maxIf(version_no, is_current) AS current_version
    FROM {{ ref('hist_vacancy_scd2') }}
    GROUP BY source, source_id
)

SELECT p.source, p.source_id, p.n_current, l.is_closed, l.source_id = '' AS missing_lifetime
FROM per_vacancy AS p
LEFT JOIN {{ ref('hist_vacancy_lifetime') }} AS l ON p.source = l.source AND p.source_id = l.source_id
WHERE l.source_id = ''
    OR (NOT l.is_closed AND p.n_current != 1)
    OR (l.is_closed AND p.n_current != 0)
    OR (p.n_current = 1 AND p.current_version != p.last_version)
SETTINGS join_use_nulls = 0
