{#- Timestamps inside payloads, written as sources write them: ISO with an offset, a bare date (read as
    midnight Moscow time, as radar.collect.envelope.parse_source_time does) or Unix milliseconds. -#}
{% macro source_time(expr) -%}
    if(
        match({{ expr }}, '^\\d{13}$'),
        toDateTime64(fromUnixTimestamp64Milli(toInt64OrZero({{ expr }}), 'UTC'), 6, 'UTC'),
        toTimeZone(parseDateTime64BestEffortOrNull({{ expr }}, 6, 'Europe/Moscow'), 'UTC')
    )
{%- endmacro %}


{% macro json_string(path) -%}
    nullIf(JSONExtractString(payload, {{ path }}), '')
{%- endmacro %}


{#- Fields with few distinct values (dictionaries, employer names) repeat across 550k rows a day. -#}
{% macro json_dim(path) -%}
    toLowCardinality({{ json_string(path) }})
{%- endmacro %}
