{#- Source amounts use 0 for "not stated" (172 of 11 632 API records say "от 0"); the data rules want null
    there, never zero. -#}
{% macro stated_amount(expr) -%}
    nullIf(toFloat64OrNull(toString({{ expr }})), 0)
{%- endmacro %}


{% macro currency_code(expr) -%}
    multiIf(
        {{ expr }} IN ('«руб.»', 'rur', 'RUR', 'rub', 'RUB'), 'RUB',
        lower({{ expr }}) = 'usd', 'USD',
        lower({{ expr }}) = 'eur', 'EUR',
        lower({{ expr }}) = 'kzt', 'KZT',
        lower({{ expr }}) = 'byn', 'BYN',
        lower({{ expr }}) = 'uzs', 'UZS',
        lower({{ expr }}) = 'cny', 'CNY',
        lower({{ expr }}) = 'gbp', 'GBP',
        NULL
    )
{%- endmacro %}


{#- Wide IT filter of the trudvsem slices: recall first, precision is measured later on labelled
    data. Not measured yet. See the is_it_candidate column of stg_trudvsem_snapshot. A missing
    sphere or position is not IT evidence; coalesce keeps it from turning the flag NULL. -#}
{% macro is_it_candidate(sphere_is_it, typical_position, title) -%}
    coalesce(
        {{ sphere_is_it }}
        OR {{ typical_position }} IN (SELECT typical_position FROM {{ ref('it_typical_positions') }})
        OR match(lowerUTF8(coalesce({{ title }}, '')), '{{ it_title_pattern() }}'),
        false
    )
{%- endmacro %}


{#- RE2 over the lowercased title. \b in RE2 knows only ASCII letters, so Cyrillic tokens are
    bounded by explicit classes. -#}
{% macro it_title_pattern() -%}
    {%- set parts = [
        'программист', 'разработчик', 'developer', 'devops', 'девопс', '\\\\bsre\\\\b',
        'тестировщик', '\\\\bqa\\\\b', 'автотест',
        'аналитик данных', 'системный аналитик', 'бизнес-аналитик', 'бизнес аналитик', 'bi-аналитик',
        'продуктовый аналитик', 'analyst',
        'data', 'machine learning', 'машинного обучения', 'нейросет', '\\\\bml\\\\b', 'mlops', '\\\\bllm\\\\b',
        '\\\\bnlp\\\\b', 'computer vision', 'компьютерного зрения', 'искусственного интеллекта',
        'системный администратор', 'сисадмин', 'администратор баз данных', '\\\\bdba\\\\b',
        'сетевой инженер', 'сетевой администратор',
        'информационной безопасност', 'кибербезопасност', 'пентест',
        'консультант 1с', '1с-консультант', 'аналитик 1с', 'архитектор 1с', 'внедрени[а-яё]* 1с',
        'python', '\\\\bjava\\\\b', 'javascript', 'typescript', 'golang', '\\\\bphp\\\\b', 'c\\\\+\\\\+', 'c#',
        '\\\\.net', 'kotlin', '\\\\bios\\\\b', 'android', 'frontend', 'front-end', 'фронтенд', 'backend',
        'back-end', 'бэкенд', 'бекенд', 'fullstack', 'full-stack', 'фулстек', 'верстальщик', 'web-',
        'веб-разработ', 'веб-программ', 'битрикс', 'bitrix',
        '(^|[^а-яё])ит([^а-яё]|$)', '(^|[^a-z])it([^a-z]|$)',
        'технической поддержк', 'helpdesk', 'service desk',
        'архитектор (решений|программ|информационн|систем|данных|облач|ит|it)', 'solution architect',
        'scrum', 'product owner', 'product manager', 'продакт',
        'ui/ux', 'ux/ui', 'ux-', 'gamedev', '\\\\bunity\\\\b', '\\\\bunreal\\\\b'
    ] -%}
    {{- parts | join('|') -}}
{%- endmacro %}
