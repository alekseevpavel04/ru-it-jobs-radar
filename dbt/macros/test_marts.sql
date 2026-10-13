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


{#- A count is at least this share of its denominator: a coverage bound. Rows with a zero denominator
    pass. -#}
{% test share_at_least(model, column_name, denominator, min_share) %}
SELECT {{ column_name }}, {{ denominator }}
FROM {{ model }}
WHERE {{ denominator }} > 0 AND {{ column_name }} < {{ min_share }} * {{ denominator }}
{% endtest %}


{#- The column is empty on the rows of the test's where config: a field the publication rules forbid
    for those sources. -#}
{% test is_null(model, column_name) %}
SELECT {{ column_name }}
FROM {{ model }}
WHERE {{ column_name }} IS NOT NULL
{% endtest %}
