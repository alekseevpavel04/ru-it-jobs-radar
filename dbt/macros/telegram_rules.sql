{# radar.collect.telegram_rules.RESUME_HEAD: a post that opens with a résumé or a digest of résumés. Posts
   collected before rules version 3 carry is_vacancy from the older rules, so staging applies the pattern
   again. tests/collect/test_telegram_rules.py keeps the two patterns equal. #}
{% macro telegram_resume_head(text) -%}
match(lowerUTF8({{ text }}), '^[^a-zа-яё0-9]*(резюме|лучшие резюме|топ резюме|обзор за|ищу работу)')
{%- endmacro %}
