{{ config(
    materialized='table',
    engine='MergeTree()',
    order_by='(published_week, source_id)',
    query_settings={
        'max_threads': 2,
        'max_block_size': 8192,
        'max_bytes_before_external_sort': 268435456,
        'max_bytes_before_external_group_by': 268435456
    }
) }}

-- Vacancy posts of the Telegram channels (raw is_vacancy, rules of radar.collect.telegram_rules),
-- one row per post: an edited post is collected again under a new content_hash, the latest fetch
-- wins. The text stays in raw (DECISIONS 2 and 10); this table keeps what the weekly aggregates
-- need, all of it by dictionary and rules.
WITH {{ skill_dictionary_constant() }} AS skill_dict,

latest AS (
    SELECT
        source_id,
        dt,
        fetched_at,
        published_at,
        text,
        multi_vacancy,
        JSONExtractString(payload, 'channel') AS channel,
        JSONExtract(payload, 'vacancies_in_post', 'UInt16') AS vacancies_in_post,
        JSONExtract(payload, 'job_links', 'Array(String)') AS job_links,
        ingested_at
    FROM {{ source('raw', 'telegram') }} FINAL
    WHERE source = 'telegram' AND is_vacancy
    ORDER BY source_id, fetched_at DESC
    LIMIT 1 BY source_id
),

channels AS (
    SELECT channel, if(coalesce(network, '') = '', channel, network) AS network, topic
    FROM {{ ref('telegram_channels') }}
),

ruled AS (
    SELECT
        l.source_id AS source_id,
        l.channel AS channel,
        -- a channel missing from the seed is a network of its own
        if(c.channel = '', l.channel, c.network) AS network,
        nullIf(c.topic, '') AS channel_topic,
        l.dt AS dt,
        l.fetched_at AS fetched_at,
        assumeNotNull(l.published_at) AS published_at,
        toMonday(toTimeZone(assumeNotNull(l.published_at), 'Europe/Moscow')) AS published_week,
        coalesce(l.multi_vacancy, false) AS is_multi_vacancy,
        l.vacancies_in_post AS vacancies_in_post,
        arrayExists(x -> match(x, 'career[.]habr[.]com|u[.]habr[.]com'), l.job_links) AS links_to_habr,
        {{ telegram_text_hash('l.text') }} AS text_hash,
        {{ rule_text(telegram_title_line('l.text')) }} AS title_line,
        {{ rule_text("arrayStringConcat(extractAll(coalesce(l.text, ''), '#[^ \\n#]+'), ' ')") }} AS hashtags,
        {{ rule_text('l.text') }} AS body,
        l.ingested_at AS ingested_at
    FROM latest AS l
    LEFT JOIN channels AS c ON l.channel = c.channel
    WHERE l.published_at IS NOT NULL
),

bucketed AS (
    SELECT
        *,
        {{ role_bucket('title_line') }} AS title_bucket,
        {{ role_bucket('hashtags') }} AS hashtag_bucket,
        {{ role_bucket('substringUTF8(body, 1, 300)') }} AS head_bucket
    FROM ruled
)

SELECT
    source_id,
    channel,
    network,
    channel_topic,
    dt,
    fetched_at,
    published_at,
    published_week,
    is_multi_vacancy,
    vacancies_in_post,
    links_to_habr,
    text_hash,
    -- The same text in one week is one post, whichever channel carried it: copies run inside a
    -- publisher network and also across networks and within one channel; the earliest copy counts.
    row_number() OVER (PARTITION BY published_week, text_hash ORDER BY published_at, source_id) > 1
        AS is_week_duplicate,
    -- A post with several vacancies is one row of its own, not split and not given a role.
    -- A first line without a role (a company name, "Всем привет!") falls back to the hashtags, then
    -- to the head of the post, where "Позиция: ..." usually sits. The first line itself is not kept:
    -- it is free text of the post (DECISIONS 2) and sometimes a person's name.
    multiIf(
        is_multi_vacancy, 'multi_post',
        title_bucket != 'not_it', title_bucket,
        hashtag_bucket != 'not_it', hashtag_bucket,
        head_bucket
    ) AS role_bucket,
    {{ grade_from_text("concat(title_line, ' ', hashtags)") }} AS grade,
    {{ work_format('body', 'NULL', 'NULL') }} AS work_format,
    {{ no_experience_text('body') }} AS no_experience_stated,
    {{ skills_found('body') }} AS skills,
    ingested_at
FROM bucketed
