{#- Rules of the marts on extracted fields (models/extracted). -#}

{#- Enumerations of the extraction schema (radar.extract.schema; tests/dbt checks they match).
    The marts write a NULL field as 'unknown'. -#}
{% macro extraction_enums() -%}
    {{ return({
        'role': ['ml_engineer', 'data_scientist', 'mlops', 'data_engineer', 'data_analyst', 'business_analyst',
            'system_analyst', 'backend', 'frontend', 'fullstack', 'mobile', 'software_engineer', 'developer_1c',
            'embedded', 'devops', 'qa', 'sysadmin', 'support', 'security', 'designer', 'manager', 'other_it',
            'non_it'],
        'grade': ['intern', 'junior', 'middle', 'senior', 'lead'],
        'work_format': ['office', 'hybrid', 'remote'],
        'employment_type': ['labor_contract', 'gph', 'self_employed', 'internship'],
        'currency': ['RUB', 'USD', 'EUR', 'KZT', 'BYN', 'UZS', 'CNY', 'GBP'],
        'gross_net': ['gross', 'net'],
        'period': ['month', 'year', 'day', 'hour'],
        'lang': ['ru', 'en', 'other']
    }) }}
{%- endmacro %}


{#- Bands of the required experience by its lower bound: none is min 0 ("без опыта", the guide's
    {min: 0, max: null}) or an upper bound alone, which sets no minimum; under_1 is 0 < min < 1
    ("от полугода"); unknown is nothing stated. -#}
{% macro experience_bands() -%}
    {{ return(['none', 'under_1', '1_3', '3_6', '6_plus', 'unknown']) }}
{%- endmacro %}


{% macro experience_band(min_expr, max_expr) -%}
    multiIf(
        {{ min_expr }} IS NULL AND {{ max_expr }} IS NULL, 'unknown',
        coalesce({{ min_expr }}, 0) = 0, 'none',
        {{ min_expr }} < 1, 'under_1',
        {{ min_expr }} < 3, '1_3',
        {{ min_expr }} < 6, '3_6',
        '6_plus'
    )
{%- endmacro %}


{#- Fields of the distribution mart: name, expression over int_extracted_entities or the weekly units,
    allowed values. role is counted over every valid unit, the others over the IT units. -#}
{% macro distribution_fields() -%}
    {%- set enums = extraction_enums() -%}
    {{ return([
        ('role', 'role', enums['role']),
        ('grade', 'grade', enums['grade']),
        ('work_format', 'work_format', enums['work_format']),
        ('employment_type', 'employment_type', enums['employment_type']),
        ('experience', 'experience_band', experience_bands())
    ]) }}
{%- endmacro %}


{#- Lookup key of a skill spelling, the folding of radar.extract.skills.key: edge punctuation trimmed,
    lowercase, ё as е, whitespace, hyphens, underscores and dots removed. -#}
{% macro skill_key(expr) -%}
    replaceRegexpAll(
        replaceAll(lowerUTF8(replaceRegexpAll({{ expr }}, '^[ \\t\\n"\'«»“”„`,;:()\\[\\]{}*•·]+|[ \\t\\n"\'«»“”„`,;:()\\[\\]{}*•·]+$', '')), 'ё', 'е'),
        '[\\s_.-]+', ''
    )
{%- endmacro %}


{#- The hash the seed skill_dictionary_v1 keys a spelling by (dbt/skill_seed.py key_hash). -#}
{% macro skill_hash(expr) -%}
    substring(lower(hex(SHA256({{ skill_key(expr) }}))), 1, 16)
{%- endmacro %}


{#- WITH expressions over the seed: skill_map (key hash -> dictionary names, [] for a non-skill or an
    unknown spelling), skill_status (key hash -> known, non_skill, unknown) and dictionary_keys (hashes of
    the dictionary's own names and aliases). A hash in neither map is a spelling the seed has not seen. -#}
{% macro skill_seed_maps() -%}
    (
        SELECT mapFromArrays(groupArray(key_hash), groupArray(names))
        FROM (
            SELECT key_hash, arraySort(arrayFilter(x -> x != '', groupArray(skill))) AS names
            FROM {{ ref('skill_dictionary_v1') }}
            GROUP BY key_hash
        )
    ) AS skill_map,
    (
        SELECT mapFromArrays(groupArray(key_hash), groupArray(status))
        FROM (SELECT key_hash, any(status) AS status FROM {{ ref('skill_dictionary_v1') }} GROUP BY key_hash)
    ) AS skill_status,
    (SELECT groupArray(key_hash) FROM {{ ref('skill_dictionary_v1') }} WHERE origin = 'dictionary') AS dictionary_keys
{%- endmacro %}
