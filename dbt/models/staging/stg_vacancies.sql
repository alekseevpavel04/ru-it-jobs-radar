{%- set common = [
    ('source', 'LowCardinality(String)'),
    ('source_id', 'String'),
    ('dt', 'Date'),
    ('content_hash', 'String'),
    ('fetched_at', "DateTime64(6, 'UTC')"),
    ('published_at', "Nullable(DateTime64(6, 'UTC'))"),
    ('modified_at', "Nullable(DateTime64(6, 'UTC'))"),
    ('url', 'Nullable(String)'),
    ('title', 'Nullable(String)'),
    ('text', 'Nullable(String)'),
    ('has_text', 'Bool'),
    ('employer_name', 'Nullable(String)'),
    ('region', 'Nullable(String)'),
    ('cities', 'Array(String)'),
    ('salary_text', 'Nullable(String)'),
    ('salary_from', 'Nullable(Float64)'),
    ('salary_to', 'Nullable(Float64)'),
    ('salary_currency', 'Nullable(String)'),
    ('salary_currency_code', 'Nullable(String)'),
    ('schedule', 'Nullable(String)'),
    ('employment', 'Nullable(String)'),
    ('remote_work', 'Nullable(Bool)'),
    ('experience_min_years', 'Nullable(UInt8)'),
    ('grade', 'Nullable(String)'),
    ('source_type', 'Nullable(String)'),
    ('skills', 'Array(String)'),
    ('is_it_candidate', 'Bool'),
    ('ingested_at', "DateTime64(3, 'UTC')"),
] -%}

{#- Columns a source does not have come as NULL or an empty array; Habr is an IT site, so all of it
    is an IT candidate. SuperJob names its employment type type_of_work and has no region field (the
    town is in cities) and no salary string. -#}
{%- set absent = {
    'stg_trudvsem_api': {'cities': '[]', 'remote_work': 'NULL', 'grade': 'NULL'},
    'stg_trudvsem_snapshot': {'cities': '[]', 'remote_work': 'NULL', 'grade': 'NULL'},
    'stg_habr': {
        'region': 'NULL', 'schedule': 'NULL', 'experience_min_years': 'NULL', 'source_type': 'NULL',
        'is_it_candidate': 'true'
    },
    'stg_superjob': {
        'region': 'NULL', 'salary_text': 'NULL', 'schedule': 'NULL', 'employment': 'type_of_work',
        'grade': 'NULL', 'source_type': 'NULL'
    },
} -%}

{%- for model_name, missing in absent.items() %}
SELECT
    {%- for name, type in common %}
    CAST({{ missing.get(name, name) }} AS {{ type }}) AS {{ name }}{{ ',' if not loop.last }}
    {%- endfor %}
FROM {{ ref(model_name) }}
{% if not loop.last %}UNION ALL{% endif %}
{%- endfor %}
