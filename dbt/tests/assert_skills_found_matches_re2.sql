-- The skill search of the marts (macro skills_found: Hyperscan shortlists, RE2 decides) finds exactly
-- what RE2 alone finds, pattern by pattern: on texts where Hyperscan alone was wrong (Node.js inside
-- «законодательства», Unity inside «коммьюнити», a skill before a «) and on the latest texts of up
-- to 2000 Habr and 2000 trudvsem API vacancies. A failing row names the text (its id for real ones).
WITH {{ skill_dictionary_constant() }} AS skill_dict,

texts AS (
    SELECT 'fixture' AS kind, arrayJoin([
        'знание трудового законодательства',
        'профессиональное коммьюнити',
        'стек «python», c++ и c#',
        'бэкенд на node.js, мастер-нода kubernetes',
        'написание тз и технических заданий'
    ]) AS id, id AS body

    UNION ALL

    SELECT 'habr', source_id, body
    FROM (
        SELECT source_id, {{ rule_text("concat(coalesce(title, ''), '\\n', coalesce(text, ''), '\\n', arrayStringConcat(skills, ' | '))") }} AS body
        FROM {{ ref('stg_habr') }}
        ORDER BY source_id, dt DESC
        LIMIT 1 BY source_id
        LIMIT 2000
    )

    UNION ALL

    SELECT 'trudvsem', source_id, body
    FROM (
        SELECT source_id, {{ rule_text("concat(coalesce(title, ''), '\\n', coalesce(text, ''), '\\n', arrayStringConcat(skills, ' | '))") }} AS body
        FROM {{ ref('stg_trudvsem_api') }}
        ORDER BY source_id, dt DESC
        LIMIT 1 BY source_id
        LIMIT 2000
    )
)

SELECT kind, id, found, re2
FROM (
    SELECT kind, id, {{ skills_found('body') }} AS found, arraySort({{ skills_found_re2('body') }}) AS re2
    FROM texts
)
WHERE found != re2
