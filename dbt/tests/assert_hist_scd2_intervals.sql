-- Versions of one vacancy are ordered, non-empty and do not overlap: valid_from < valid_to, a
-- version's sightings fall inside it, is_current is exactly "valid_to is NULL", and a version ends
-- no later than the next one starts.
WITH ordered AS (
    SELECT
        source,
        source_id,
        version_no,
        valid_from,
        valid_to,
        is_current,
        last_seen,
        leadInFrame(toNullable(valid_from)) OVER (
            PARTITION BY source, source_id ORDER BY version_no ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING
        ) AS next_valid_from
    FROM {{ ref('hist_vacancy_scd2') }}
)

SELECT *
FROM ordered
WHERE (valid_to IS NOT NULL AND valid_to <= valid_from)
    OR is_current != (valid_to IS NULL)
    OR last_seen < valid_from
    OR (valid_to IS NOT NULL AND last_seen >= valid_to)
    OR (next_valid_from IS NOT NULL AND (valid_to IS NULL OR valid_to > next_valid_from))
