{{ config(
    materialized='table',
    engine='MergeTree()',
    order_by='(source, source_id, valid_from)',
    query_settings={
        'max_threads': 2,
        'max_bytes_before_external_sort': 268435456,
        'max_bytes_before_external_group_by': 268435456
    }
) }}

{%- set fields = history_tracked_fields() -%}
{%- set attributes = ['content_hash', 'title', 'title_norm', 'employer_name', 'employer_key', 'location_key',
                      'salary_from', 'salary_to', 'salary_currency_code', 'schedule', 'employment', 'remote_work',
                      'status', 'skills', 'text_hash', 'published_at'] -%}

-- One row per version of a vacancy within its source. A version is valid from the day it was first
-- seen to the day the next version was first seen or the day the vacancy closed (valid_to is
-- exclusive); a vacancy seen again after closing opens a new version.
--
-- Memory: the window steps over all sightings carry only the id, the day, a 64-bit digest of
-- content_hash and the closure dates; the fields of a version are joined back from its first
-- sighting, and versions are compared with each other only after that, at one row per version.
-- Memory grows with the number of IT sightings (about 10k per snapshot day); past 256 MiB the sorts
-- and the grouping spill to disk instead of growing further.
WITH sightings AS (
    SELECT
        o.source AS source,
        o.source_id AS source_id,
        o.dt AS dt,
        cityHash64(o.content_hash) AS digest,
        o.is_closed_state AS is_closed_state,
        c.next_complete_dt AS next_complete_dt,
        c.closure_confirmed_dt AS closure_confirmed_dt
    FROM {{ ref('hist_observations') }} AS o
    INNER JOIN {{ ref('hist_calendar') }} AS c ON o.source = c.source AND o.dt = c.dt
),

sequenced AS (
    SELECT
        *,
        leadInFrame(toNullable(dt)) OVER w AS next_dt,
        leadInFrame(toNullable(is_closed_state)) OVER w AS next_is_closed_state
    FROM sightings
    WINDOW w AS (PARTITION BY source, source_id ORDER BY dt ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING)
),

-- Rows in a closed state (deleted, unpublished, archived) are not sightings of an open vacancy;
-- they only close the one seen before them, as of the first complete day it was not seen open.
-- Otherwise the vacancy closes when it is missing for the source's number of complete days, as of
-- the first of them (hist_calendar).
active AS (
    SELECT
        source,
        source_id,
        dt,
        digest,
        multiIf(
            coalesce(next_is_closed_state, false),
            least(assumeNotNull(next_dt), ifNull(next_complete_dt, assumeNotNull(next_dt))),
            closure_confirmed_dt IS NOT NULL AND (next_dt IS NULL OR next_dt > closure_confirmed_dt),
            next_complete_dt,
            NULL
        ) AS closed_at
    FROM sequenced
    WHERE NOT is_closed_state
),

marked AS (
    SELECT
        *,
        row_number() OVER v AS seq,
        lagInFrame(digest) OVER v AS prev_digest,
        lagInFrame(closed_at) OVER v AS prev_closed_at
    FROM active
    WINDOW v AS (PARTITION BY source, source_id ORDER BY dt ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
),

reasoned AS (
    SELECT
        source,
        source_id,
        dt,
        closed_at,
        multiIf(seq = 1, 'new', prev_closed_at IS NOT NULL, 'reopened', digest != prev_digest, 'changed', '')
            AS change_reason
    FROM marked
),

numbered AS (
    SELECT
        *,
        sum(change_reason != '') OVER (PARTITION BY source, source_id ORDER BY dt ROWS UNBOUNDED PRECEDING)
            AS version_no
    FROM reasoned
),

-- Only the first row of a version names its reason and only the last can carry a closure; the
-- tuple keeps argMax from skipping a NULL closure.
spans AS (
    SELECT
        source,
        source_id,
        toUInt32(version_no) AS version_no,
        min(dt) AS valid_from,
        max(dt) AS last_seen,
        toUInt32(count()) AS days_seen,
        argMin(change_reason, dt) AS change_reason,
        argMax(tuple(closed_at), dt).1 AS closed_at
    FROM numbered
    GROUP BY source, source_id, version_no
),

-- Rows of one version share content_hash, so its fields are those of its first sighting.
versions AS (
    SELECT
        s.source AS source,
        s.source_id AS source_id,
        s.version_no AS version_no,
        s.valid_from AS valid_from,
        s.last_seen AS last_seen,
        s.days_seen AS days_seen,
        s.change_reason AS change_reason,
        s.closed_at AS closed_at,
        {%- for name in attributes %}
        o.{{ name }} AS {{ name }}{{ ',' if not loop.last }}
        {%- endfor %}
    FROM {{ ref('hist_observations') }} AS o
    INNER JOIN spans AS s ON o.source = s.source AND o.source_id = s.source_id AND o.dt = s.valid_from
),

neighbours AS (
    SELECT
        *,
        leadInFrame(toNullable(valid_from)) OVER w AS next_valid_from,
        lagInFrame(tuple(content_hash, {{ fields | join(', ') }})) OVER w AS prev
    FROM versions
    WINDOW w AS (PARTITION BY source, source_id ORDER BY version_no ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
),

compared AS (
    SELECT
        *,
        if(
            version_no = 1,
            CAST([] AS Array(String)),
            arrayFilter(f -> f != '', [
                {%- for name in fields %}
                if(isNotDistinctFrom({{ name }}, prev.{{ loop.index + 1 }}), '', '{{ name }}'){{ ',' if not loop.last }}
                {%- endfor %}
            ])
        ) AS named_changes
    FROM neighbours
)

SELECT
    source,
    source_id,
    version_no,
    valid_from,
    if(closed_at IS NOT NULL, closed_at, next_valid_from) AS valid_to,
    valid_to IS NULL AS is_current,
    CAST(change_reason AS LowCardinality(String)) AS change_reason,
    CAST(multiIf(closed_at IS NOT NULL, 'closed', next_valid_from IS NOT NULL, 'changed', NULL)
        AS Nullable(String)) AS ended_by,
    if(version_no > 1 AND empty(named_changes) AND content_hash != prev.1, ['other'], named_changes)
        AS changed_fields,
    last_seen,
    days_seen,
    {%- for name in attributes %}
    {{ name }}{{ ',' if not loop.last }}
    {%- endfor %}
FROM compared
