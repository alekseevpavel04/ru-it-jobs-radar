{#- Uniqueness of a column set. On staging tables partitioned by dt the grouping runs one partition
    at a time, so a 55M-row snapshot table is checked without holding every key in memory. -#}
{% test unique_combination(model, columns) %}
SELECT {{ columns | join(', ') }}, count() AS n
FROM {{ model }}
GROUP BY {{ columns | join(', ') }}
HAVING n > 1
SETTINGS allow_aggregate_partitions_independently = 1
{% endtest %}
