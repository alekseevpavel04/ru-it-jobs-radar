{#- Staging over raw.vacancies is rebuilt one raw day (dt) at a time.

    Why not dbt's incremental: one statement over many days does not fit the server. The 2.2b
    backfill lands about 100 snapshot days (550k rows each) within a day, and reading them in key
    order for the dedup grew by about 70-100 MiB per day on ClickHouse with a 3.6 GiB budget
    (measured: 1 day 169 MiB, 4 days 439 MiB). Per day it stays flat.

    Which days: on the first build or --full-refresh every dt of the source; otherwise each dt that
    got a raw row after the newest row already staged (a new day, a replay, the late part of a
    snapshot still streaming in), with an hour of overlap for blocks that commit out of order.

    How: the model selects one day where it says `dt = {{ staging_day() }}`; each day is inserted
    into a scratch table and swapped in with REPLACE PARTITION, so readers never see a day half
    built, and a day that vanished from raw is emptied. A full refresh builds a new table and
    exchanges it atomically; a change of the model's columns needs one (--full-refresh). -#}

{% macro staging_day() -%}
    toDate('__staging_day__')
{%- endmacro %}


{% materialization incremental_by_day, adapter='clickhouse' %}
  {%- set raw_source = config.require('raw_source') -%}
  {%- set raw_relation = source('raw', raw_source) -%}
  {%- set target_relation = this.incorporate(type='table') -%}
  {%- set existing_relation = load_cached_relation(this) -%}
  {%- set rebuild = existing_relation is none or existing_relation.is_view or should_full_refresh() -%}
  {%- set build_relation = make_intermediate_relation(target_relation) if rebuild else target_relation -%}
  {%- set day_relation = target_relation.incorporate(path={'identifier': target_relation.identifier ~ '__dbt_day'}) -%}

  {{ run_hooks(pre_hooks, inside_transaction=False) }}

  {% call statement('drop_scratch') %}
    DROP TABLE IF EXISTS {{ day_relation }}
  {% endcall %}
  {% if rebuild %}
    {% call statement('drop_intermediate') %}
      DROP TABLE IF EXISTS {{ build_relation }}
    {% endcall %}
    {% do clickhouse__create_empty_table(False, build_relation, sql | replace('__staging_day__', '1970-01-01'), False) %}
  {% endif %}

  {% set days_query %}
    SELECT DISTINCT toString(dt)
    FROM {{ raw_relation }}
    WHERE source = '{{ raw_source }}'
    {%- if not rebuild %}
      AND ingested_at > (SELECT max(ingested_at) FROM {{ target_relation }}) - INTERVAL 1 HOUR
    {%- endif %}
    ORDER BY 1
  {% endset %}
  {% set days = run_query(days_query).columns[0].values() if execute else [] %}
  {{ log(this.identifier ~ ': ' ~ days | length ~ ' day(s) to build', info=True) }}

  {% for day in days %}
    {% call statement('day_create') %}
      CREATE TABLE {{ day_relation }} AS {{ build_relation }}
    {% endcall %}
    {% call statement('day_insert') %}
      {{ clickhouse__insert_into(day_relation, sql | replace('__staging_day__', day), False) }}
    {% endcall %}
    {% call statement('day_replace') %}
      ALTER TABLE {{ build_relation }} REPLACE PARTITION ID '{{ day | replace('-', '') }}' FROM {{ day_relation }}
      SETTINGS allow_replace_partition_from_empty_source = 1
    {% endcall %}
    {% call statement('day_drop') %}
      DROP TABLE {{ day_relation }}
    {% endcall %}
  {% endfor %}

  {% if rebuild and existing_relation is not none %}
    {% if existing_relation.is_view %}
      {% do adapter.drop_relation(existing_relation) %}
      {% do adapter.rename_relation(build_relation, target_relation) %}
    {% else %}
      {% do exchange_tables_atomic(build_relation, target_relation) %}
      {% do adapter.drop_relation(build_relation) %}
    {% endif %}
  {% elif rebuild %}
    {% do adapter.rename_relation(build_relation, target_relation) %}
  {% endif %}

  {% call statement('main') %}
    SELECT {{ days | length }} AS days_built
  {% endcall %}

  {% do persist_docs(target_relation, model) %}
  {{ run_hooks(post_hooks, inside_transaction=False) }}
  {% do adapter.commit() %}

  {{ return({'relations': [target_relation]}) }}
{% endmaterialization %}
