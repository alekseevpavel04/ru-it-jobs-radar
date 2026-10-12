{#- A Telegram post without links, hashtags, @names and PII placeholders, as
    radar.extract.telegram_corpus.own_text cuts them. `expr` is the lowercased text (rule_text), so
    the URL pattern needs no case flag. Python's \s and \w are Unicode and RE2's are ASCII, so the
    classes spell them out: \pL\pN_ is a word character, \p{Z} with the ASCII controls is a space.
    Python's \b before a domain becomes a consumed non-word character here, which is no letter, so
    the letters left are the same. tests/dbt/test_dbt_post_lang.py keeps the patterns in step. -#}
{% macro telegram_own_text(expr) -%}
    {#- a block, not a string: Jinja leaves the backslashes alone, ClickHouse halves them for RE2 -#}
    {%- set space %}\\t\\n\\x0B\\f\\r\\x{1C}-\\x{1F}\\x{85}\\p{Z}{% endset -%}
    replaceRegexpAll(replaceRegexpAll(replaceRegexpAll(replaceRegexpAll(
        {{ expr }},
        '(?:https?://|www[.])[^{{ space }}]+|(?:^|[^\\pL\\pN_])(?:[\\pL\\pN_-]+[.])+[a-z]{2,}/[^{{ space }}]*', ' '),
        '#[^{{ space }}#]+', ' '),
        '@[\\pL\\pN_]+', ' '),
        '\\[(?:{{ telegram_placeholders() | join('|') }})\\]', ' ')
{%- endmacro %}


{#- radar.pii.text.PLACEHOLDER values without brackets. -#}
{% macro telegram_placeholders() -%}
    {{ return(['email', 'link', 'nick', 'inn', 'ogrn', 'phone', 'name']) }}
{%- endmacro %}


{#- Language of a post's own text, as radar.extract.telegram_corpus.post_lang decides it with the
    letter shares of radar.extract.baseline.predict_lang: no letters is 'other', Latin above 60% of
    the letters 'en', Kazakh, Ukrainian and other non-Russian Cyrillic letters above 1% of the
    Cyrillic ones 'other', the rest 'ru'. ё folded to е by rule_text is Cyrillic all the same. -#}
{% macro post_lang(expr) -%}
    {%- set own = telegram_own_text(expr) -%}
    {%- set cyr = "countMatches(" ~ own ~ ", '[а-яё]')" -%}
    {%- set lat = "countMatches(" ~ own ~ ", '[a-z]')" -%}
    multiIf(
        {{ cyr }} + {{ lat }} = 0, 'other',
        {{ lat }} / ({{ cyr }} + {{ lat }}) > 0.6, 'en',
        {{ cyr }} > 0 AND countMatches({{ own }}, '[әғқңөұүһіїєўҳҷ]') / {{ cyr }} > 0.01, 'other',
        'ru'
    )
{%- endmacro %}
