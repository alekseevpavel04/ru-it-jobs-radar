{#- Generic tests of the marts. -#}

{#- The model has at least one row: an empty mart means a broken upstream, not an empty market. -#}
{% test has_rows(model) %}
SELECT 1 AS empty_model
WHERE (SELECT count() FROM {{ model }}) = 0
{% endtest %}


{#- A count never exceeds its denominator, so "N of M" can always be printed. NULLs (a measure that
    does not apply to the row) pass. -#}
{% test not_above(model, column_name, denominator) %}
SELECT {{ column_name }}, {{ denominator }}
FROM {{ model }}
WHERE {{ column_name }} > {{ denominator }}
{% endtest %}
