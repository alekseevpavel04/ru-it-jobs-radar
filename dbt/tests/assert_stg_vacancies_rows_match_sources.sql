-- The union loses or repeats no rows: per source, stg_vacancies holds exactly the rows of its model,
-- and the total equals the sum of the per-source models. A source missing on one side of the join reads as 0.
WITH staged AS (
    SELECT 'trudvsem' AS source, count() AS n FROM {{ ref('stg_trudvsem_api') }}
    UNION ALL
    SELECT 'trudvsem_snapshot', count() FROM {{ ref('stg_trudvsem_snapshot') }}
    UNION ALL
    SELECT 'habr', count() FROM {{ ref('stg_habr') }}
    UNION ALL
    SELECT 'superjob', count() FROM {{ ref('stg_superjob') }}
),

unioned AS (
    SELECT toString(source) AS source, count() AS n
    FROM {{ ref('stg_vacancies') }}
    GROUP BY source
)

SELECT
    if(staged.source != '', staged.source, unioned.source) AS source,
    staged.n AS staged_rows,
    unioned.n AS union_rows
FROM staged
FULL OUTER JOIN unioned ON staged.source = unioned.source
WHERE staged.n != unioned.n
    OR (SELECT sum(n) FROM staged) != (SELECT count() FROM {{ ref('stg_vacancies') }})
SETTINGS join_use_nulls = 0
