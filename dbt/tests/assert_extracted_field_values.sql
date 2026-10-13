{{ config(enabled=var('extraction_model') != '') }}

-- Every value of mart_extracted_fields_weekly is one the extraction schema allows for its field
-- (macro extraction_enums; experience in the bands of experience_band) or unknown.
SELECT field, value, count() AS rows
FROM {{ ref('mart_extracted_fields_weekly') }}
WHERE (field, value) NOT IN (
{%- for name, _, values in distribution_fields() %}
    {%- for v in values + ['unknown'] %}
    ('{{ name }}', '{{ v }}'){{ ',' if not (loop.last) }}
    {%- endfor %}{{ ',' if not loop.last }}
{%- endfor %}
)
GROUP BY field, value
