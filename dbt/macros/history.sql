{#- Rules of the vacancy history (models/history). Each is a choice with a reason; the numbers that
    back them are in results/dbt/history.json. -#}

{#- A collected day counts as complete when it holds at least this share of the source's median
    rows per day. A day still streaming into ClickHouse, or a run cut short by a block, is not
    evidence that the vacancies missing from it closed; a vacancy seen on it still counts as seen. -#}
{% macro history_complete_day_ratio() -%}
    0.9
{%- endmacro %}


{#- How many consecutive complete days a vacancy has to be missing before it counts as closed;
    the closing day is then the first of them. NULL: absence never closes it.

    trudvsem_snapshot, var history_grace_days (default 7): every snapshot is the full set of the
    portal, but the portal hides open vacancies for a few days and shows them again. Of the
    closures that one missing snapshot would call on 2026-06-30..09-10, 2 692 of 7 139 came back;
    with 7 days 897 of 4 794 (grace in results/models/survival.json). A vacancy back within the
    window is the same open vacancy, one version, and the survival analysis (7.3) uses the same
    rule, so its lifetimes and these agree. The price is that a closure is confirmed 7 complete days
    later; until then the vacancy is open and censored at its last sighting.
    habr, 3: the list returned 1 050 of the 1 227 vacancies of the site counter (2026-10-08,
    last_run_coverage_of_site_counter in results/collect/habr.json), so a
    vacancy can miss a day while still open. If that miss were independent from day to day with
    p = 1 - 1 050 / 1 227 = 0.144, a false closure would need 0.144^3 = 0.003 against 0.021 for two
    days. The price is that a closure is confirmed two days later; until then the vacancy stays
    open and censored at its last sighting.
    trudvsem (API), NULL: the daily run asks for records modified since the last run, so absence
    says nothing; the API has no status or deleted field either. -#}
{% macro history_absent_days_to_close(source_expr) -%}
    multiIf({{ source_expr }} = 'trudvsem_snapshot', {{ history_grace_days() }}, {{ source_expr }} = 'habr', 3, NULL)
{%- endmacro %}

{% macro history_grace_days() -%}
    {{ var('history_grace_days', 7) }}
{%- endmacro %}


{#- Days without a complete collection are unknown, not evidence. A vacancy last seen on dt whose
    first complete day without it comes more than this many days later disappeared somewhere in
    between: the history does not date that closure. It still ends the vacancy (it is gone), but
    hist_vacancy_lifetime has no closed_at for it and censors it at its last sighting. 2, as
    MAX_STEP_DAYS in radar.models.survival: one missing day dates a closure at most a day late; the
    snapshot hole 2026-09-10..10-09 would otherwise close every vacancy that left in it on 10-09. -#}
{% macro history_max_step_days() -%}
    {{ var('history_max_step_days', 2) }}
{%- endmacro %}


{#- Repost: a new source_id of the same employer, location, title and text that appears from the
    last day the old vacancy was seen up to this many days after it closed. Candidate pairs are kept
    up to history_repost_scan_days so the choice of the window can be checked on the data
    (repost_candidates_*_gap_le_* in results/dbt/history.json): on 22 snapshot days the pairs level
    off by two weeks, so a wider window would mostly add coincidences. -#}
{% macro history_repost_window_days() -%}
    14
{%- endmacro %}

{% macro history_repost_scan_days() -%}
    30
{%- endmacro %}


{#- Fields whose change is named in changed_fields of the SCD2 rows. A new version starts on any
    change of content_hash (the whole record without bookkeeping fields); a change of
    content_hash that touches none of these is named "other". -#}
{% macro history_tracked_fields() -%}
    {{ return(['title', 'employer_name', 'salary_from', 'salary_to', 'salary_currency_code', 'schedule',
               'employment', 'remote_work', 'status', 'skills', 'text_hash']) }}
{%- endmacro %}


{#- Title as compared for reposts: case, ё, punctuation and spacing do not count. -#}
{% macro history_title_norm(expr) -%}
    trimBoth(replaceRegexpAll(replaceAll(lowerUTF8(coalesce({{ expr }}, '')), 'ё', 'е'), '[^a-zа-я0-9+#]+', ' '))
{%- endmacro %}


{#- Text as compared between versions and for reposts: case and spacing do not count; no text is NULL. -#}
{% macro history_text_hash(expr) -%}
    if(
        length(trimBoth(coalesce({{ expr }}, ''))) = 0,
        NULL,
        cityHash64(trimBoth(replaceRegexpAll(lowerUTF8({{ expr }}), '\\s+', ' ')))
    )
{%- endmacro %}


{#- One observation of a vacancy on one collected day, the same columns for every source. -#}
{% macro history_observation(employer_key, location_key, is_closed_state, schedule, remote_work, status) -%}
    CAST(source AS LowCardinality(String)) AS source,
    source_id,
    dt,
    content_hash,
    CAST({{ is_closed_state }} AS Bool) AS is_closed_state,
    title,
    {{ history_title_norm('title') }} AS title_norm,
    CAST(employer_name AS Nullable(String)) AS employer_name,
    CAST({{ employer_key }} AS Nullable(String)) AS employer_key,
    CAST({{ location_key }} AS Nullable(String)) AS location_key,
    salary_from,
    salary_to,
    salary_currency_code,
    CAST({{ schedule }} AS Nullable(String)) AS schedule,
    CAST(employment AS Nullable(String)) AS employment,
    CAST({{ remote_work }} AS Nullable(Bool)) AS remote_work,
    CAST({{ status }} AS Nullable(String)) AS status,
    arraySort(skills) AS skills,
    {{ history_text_hash('text') }} AS text_hash,
    published_at,
    ingested_at
{%- endmacro %}
