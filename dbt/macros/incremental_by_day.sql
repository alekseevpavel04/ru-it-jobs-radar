{#- Staging over raw.vacancies is rebuilt one raw day (dt) at a time.

    Why not dbt's incremental: one statement over many days does not fit the server. The 2.2b
    backfill lands about 100 snapshot days (550k rows each) within a day, and reading them in key
    order for the dedup grew by about 70-100 MiB per day on ClickHouse with a 3.6 GiB budget
    (measured: 1 day 169 MiB, 4 days 439 MiB). Per day it stays flat.

    Which days: on the first build or --full-refresh every dt of the source; otherwise each dt that
    got a raw row after the newest row already staged (a new day, a replay, the late part of a
    snapshot still streaming in), with an hour of overlap for blocks that commit out of order.

    With skip_unchanged_days=true a model that keeps exactly one row per raw key of its day
    (source_id, content_hash) skips the days whose key set is already staged: a replay or a
    redelivery of the same rows, and the days the hour of overlap brings back. The key sets are
    compared by count and a 64-bit XOR of cityHash64(source_id, content_hash), raw read with FINAL,
    about 0.5 s a snapshot day against 22 s for building it. A day whose raw holds two versions of
    one source_id never matches and is rebuilt every time, as before.

    How: the model selects one day where it says `dt = {{ staging_day() }}`; each day is inserted
    into a scratch table and swapped in with REPLACE PARTITION, so readers never see a day half
    built, and a day that vanished from raw is emptied. A full refresh builds a new table and
    exchanges it atomically; a change of the model's columns needs one (--full-refresh), and so does
    a change of its logic when skip_unchanged_days is on, since unchanged days are not rebuilt. -#}

{% macro first_column(result) -%}
  {{ return(result.columns[0].values() if result.columns | length else []) }}
{%- endmacro %}


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
  {#- an empty result comes back from the ClickHouse adapter without columns, not with an empty one -#}
  {% set days = first_column(run_query(days_query)) if execute else [] %}
  {% if days and not rebuild and config.get('skip_unchanged_days', false) %}
    {% set day_list %}{% for day in days %}'{{ day }}'{{ ',' if not loop.last }}{% endfor %}{% endset %}
    {% set changed_query %}
      SELECT toString(r.dt)
      FROM (
          SELECT dt, count() AS n, groupBitXor(cityHash64(source_id, content_hash)) AS x
          FROM {{ raw_relation }} FINAL
          WHERE source = '{{ raw_source }}' AND dt IN ({{ day_list }})
          GROUP BY dt
      ) AS r
      LEFT JOIN (
          SELECT dt, count() AS n, groupBitXor(cityHash64(source_id, content_hash)) AS x
          FROM {{ target_relation }}
          WHERE dt IN ({{ day_list }})
          GROUP BY dt
      ) AS t ON r.dt = t.dt
      WHERE r.n != t.n OR r.x != t.x
      ORDER BY 1
      SETTINGS join_use_nulls = 0, max_threads = 2
    {% endset %}
    {% set changed = first_column(run_query(changed_query)) %}
    {{ log(this.identifier ~ ': ' ~ (days | length - changed | length) ~ ' of ' ~ days | length
           ~ ' day(s) with new rows already staged, skipped', info=True) }}
    {% set days = changed %}
  {% endif %}
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
