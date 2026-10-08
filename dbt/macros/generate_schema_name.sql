{#- ClickHouse databases are the layers themselves (raw, staging, marts), so a model's +schema is
    used as is instead of dbt's default "<target schema>_<custom schema>". -#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ (custom_schema_name or target.schema) | trim }}
{%- endmacro %}
