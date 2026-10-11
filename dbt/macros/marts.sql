{#- Rules of the marts (models/marts) over source fields. Every rule reads text lowercased with ё as е,
    so the patterns below are written that way. RE2 knows word boundaries only for ASCII letters, so
    a word here is bounded by explicit classes: (^|[^a-zа-я0-9]) before and ([^a-zа-я0-9]|$) after.
    These rules hold until the extracted fields of 6.7 replace them; Telegram keeps them for good
    (DECISIONS 10: dictionary and rules, no model). -#}

{% macro rule_text(expr) -%}
    replaceAll(lowerUTF8(coalesce({{ expr }}, '')), 'ё', 'е')
{%- endmacro %}


{#- One RE2 alternation of whole words; a word may carry its own [а-я]* ending. -#}
{% macro words_pattern(words) -%}
    (^|[^a-zа-я0-9])({{ words | join('|') }})([^a-zа-я0-9]|$)
{%- endmacro %}


{#- Role bucket by title (Telegram: first line of the post). The first matching bucket wins, so the
    order is part of the rule: specific data and ML roles before generic analysts and developers,
    1C before analysts and developers, security before admins, explicit backend and frontend words
    before language names, and the non-IT exclusions only after every IT role had its chance, so
    "Python developer in the sales team" stays backend. not_it is a title with no IT role at all:
    the trudvsem IT candidate filter is wide on purpose (postmen and drivers pass it through the IT
    sphere), Habr lists sales and copywriting too; marts drop not_it and report how many. -#}
{% macro role_buckets() -%}
    {{ return([
        ('ml_ds', [
            'ml', 'mlops', 'cv', 'nlp', 'llm', 'genai', 'ds', 'research engineer', 'data scien[a-z]*', 'datascien[a-z]*', 'machine learning',
            'deep learning', 'computer vision', 'дата[- ]?сайентист[а-я]*', 'машинн[а-я]* обучени[а-я]*',
            'компьютерн[а-я]* зрени[а-я]*', 'исследовател[а-я]* данных', 'ученый по данным', 'research scientist',
            '(инженер|разработчик|developer|engineer|программист)[а-я]*[- ]+(по )?(ии|ai|искусственн[а-я]* интеллект[а-я]*|нейросет[а-я]*|нейронн[а-я]* сет[а-я]*|машинн[а-я]* обучени[а-я]*)',
            '(ии|ai)[- ]+(инженер|разработчик|engineer|developer|researcher|архитектор|исследовател)[а-я]*'
        ]),
        ('data_engineering', [
            'data engineer[a-z]*', 'data-engineer[a-z]*', 'dataengineer', 'инженер[а-я]* (по )?данн[а-я]*',
            'дата[- ]инженер[а-я]*', 'dwh', 'etl', 'elt', 'big ?data', 'hadoop', 'spark',
            'разработчик[а-я]* (баз[а-я]* данных|бд|dwh|etl|хранилищ[а-я]*)', 'database developer',
            'data platform', 'архитектор[а-я]* данных', 'data architect'
        ]),
        ('data_analytics', [
            'аналитик[а-я]* (по )?данн[а-я]*', 'data analyst', 'data analytics', 'bi', 'power bi',
            'продуктов[а-я]* аналитик[а-я]*', 'product analyst', 'веб[- ]аналитик[а-я]*', 'web analyst',
            'маркетингов[а-я]* аналитик[а-я]*', 'marketing analyst', 'дата[- ]аналитик[а-я]*', 'sql[- ]аналитик[а-я]*',
            'аналитик[а-я]* (sql|bi|dwh|big data|больших данных)', 'growth analyst', 'аналитик[а-я]* продукт[а-я]*'
        ]),
        ('developer_1c', ['1[сc]']),
        ('business_system_analysis', [
            'системн[а-я]* аналитик[а-я]*', 'бизнес[- ]аналитик[а-я]*', 'business analyst', 'systems? analyst',
            'аналитик[а-я]* бизнес[- ]процесс[а-я]*', 'аналитик[а-я]* требовани[а-я]*', '(it|ит)[- ]аналитик[а-я]*',
            'functional analyst', 'функциональн[а-я]* аналитик[а-я]*', 'solution analyst', 'аналитик[а-я]* (ис|информационн[а-я]* систем[а-я]*)'
        ]),
        ('security', [
            'информационн[а-я]* безопасност[а-я]*', 'защит[а-я]* информаци[а-я]*', 'кибербезопасност[а-я]*',
            'security', 'пентест[а-я]*', 'pentest[a-z]*', 'appsec', 'devsecops', 'secops', 'soc', 'иб', 'siem',
            'безопасност[а-я]* (информаци|данных|ит|it|приложени|сет|разработк)[а-я]*', 'ddos', 'reverse',
            'fuzzing'
        ]),
        ('devops', [
            'devops', 'девопс[а-я]*', 'sre', 'site reliability', 'platform engineer[a-z]*', 'kubernetes',
            'cloud engineer[a-z]*', 'облачн[а-я]* инженер[а-я]*', 'инженер[а-я]* (облачн[а-я]*|ci/cd)',
            'build engineer', 'release engineer', 'infrastructure engineer'
        ]),
        ('qa', [
            'qa', 'aqa', 'qaa', 'sdet', 'тестировщи[а-я]*', 'тестировани[а-я]*', 'test engineer', 'tester',
            'автотест[а-я]*', 'quality assurance', 'автоматизатор[а-я]*'
        ]),
        ('mobile', [
            'ios', 'android', 'mobile', 'мобильн[а-я]* (разработ|приложени|developer)[а-я]*', 'flutter',
            'react native', 'swift', 'kotlin multiplatform', 'kmp'
        ]),
        ('fullstack', ['full[- ]?stack', 'фулл?[- ]?стек[а-я]*']),
        ('backend', [
            'backend', 'back[- ]end', 'бэкенд[а-я]*', 'бекенд[а-я]*', 'бэк[- ]?энд[а-я]*', 'серверн[а-я]* разработ[а-я]*'
        ]),
        ('frontend', [
            'frontend', 'front[- ]end', 'фронтенд[а-я]*', 'фронт[- ]?энд[а-я]*', 'верстальщик[а-я]*', 'верстк[а-я]*',
            'ui[- ]разработчик[а-я]*', 'react', 'vue', 'vue[.]?js', 'angular', 'next[.]?js'
        ]),
        ('embedded', [
            'embedded', 'встраиваем[а-я]*', 'микроконтроллер[а-я]*', 'stm32', 'плис', 'fpga', 'asic', 'firmware',
            'прошивк[а-я]*', 'rtos', 'verilog', 'vhdl', 'rtl', 'схемотехник[а-я]*'
        ]),
        ('backend', [
            'python', 'питон[а-я]*', 'java', 'golang', 'go', 'php', 'c#', '[.]net', 'dotnet', 'ruby', 'rails',
            'scala', 'rust', 'node[.]?js', 'node', 'django', 'laravel', 'symfony', 'spring', 'kotlin', 'elixir',
            'erlang', 'perl', 'битрикс[а-я0-9]*', 'bitrix[0-9]*', 'api'
        ]),
        ('frontend', ['javascript', 'js', 'typescript']),
        ('sysadmin_support', [
            'системн[а-я]* администратор[а-я]*', 'сисадмин[а-я]*', 'sysadmin', 'system administrator',
            'администратор[а-я]* (сет[а-я]*|вычислительн[а-я]*|баз[а-я]* данных|бд|систем[а-я]*|серверов|linux|windows|информационн[а-я]*|ит|it|субд|active directory)',
            'dba', 'сетев[а-я]* (инженер|администратор)[а-я]*', 'network (engineer|administrator)',
            'системн[а-я]* инженер[а-я]*', 'systems? engineer', 'техническ[а-я]* поддержк[а-я]*', 'техподдержк[а-я]*',
            'поддержк[а-я]* пользовател[а-я]*', 'help ?desk', 'service ?desk', 'support', 'саппорт[а-я]*',
            '(специалист|инженер)[а-я]* (первой|второй|третьей) линии', 'l[123] support', 'devnet', 'linux', 'unix',
            'виртуализаци[а-я]*', 'дежурн[а-я]* инженер[а-я]*', 'инженер[а-я]* (проактивн[а-я]* )?мониторинг[а-я]*',
            'инженер[а-я]* эксплуатаци[а-я]*', 'продуктов[а-я]* поддержк[а-я]*'
        ]),
        ('design', [
            'дизайнер[а-я]*', 'designer', 'ui/ux', 'ux/ui', 'ux', 'ui', 'product design',
            'дизайн[а-я]* (графическ|пользовательск|интерфейс)[а-я]*'
        ]),
        ('management', ['владел[а-я]* продукт[а-я]*', 'технич[а-я]* директор[а-я]*', 'pmo']),
        ('other_it', ['архитектор[а-я]*', 'architect', 'sap', 'пресейл[а-я]*', 'presale', 'erp']),
        ('not_it', [
            'продаж[а-я]*', 'sales', 'клиентск[а-я]* менеджер[а-я]*', 'по работе с клиентами', 'маркетолог[а-я]*',
            'маркетинг[а-я]*', 'marketing', 'smm', 'копирайтер[а-я]*', 'copywriter', 'редактор[а-я]*', 'журналист[а-я]*',
            'корреспондент[а-я]*', 'рекрутер[а-я]*', 'recruiter', 'hr', 'бухгалтер[а-я]*', 'юрист[а-я]*', 'почтальон[а-я]*',
            'водител[а-я]*', 'оператор[а-я]*', 'кассир[а-я]*', 'преподавател[а-я]*', 'учител[а-я]*', 'педагог[а-я]*',
            'электромонтер[а-я]*', 'монтажник[а-я]*', 'электромеханик[а-я]*', 'энергетик[а-я]*', 'кабельщик[а-я]*',
            'диспетчер[а-я]*', 'контент[а-я]*', 'content', 'электроник[а-я]*', 'связист[а-я]*', 'инженер[а-я]* связи'
        ]),
        ('management', [
            'product manager', 'продакт[а-я]*', 'менеджер[а-я]* (по )?(продукт|проект)[а-я]*', 'project manager',
            'руководител[а-я]* (it|ит)[- ]?проект[а-я]*',
            'руководител[а-я]* (проект[а-я]*|разработк[а-я]*|отдела разработк[а-я]*|группы разработк[а-я]*|команды разработк[а-я]*|it|ит|отдела (it|ит|информационн[а-я]*)|направления (разработк[а-я]*|it|ит)|цифров[а-я]*)',
            'начальник[а-я]* (отдела |управления |службы )?(информационн[а-я]* технолог[а-я]*|ит|it|автоматизаци[а-я]*|программн[а-я]*|информатизаци[а-я]*)',
            'директор[а-я]* (по )?(it|ит|информационн[а-я]*|цифров[а-я]*)', 'cto', 'cio', 'cdo', 'head of [a-z ]*',
            'scrum[- ]?master', 'scrum', 'agile coach', 'product owner', 'delivery manager', '(it|ит)[- ]менеджер[а-я]*',
            'engineering manager', 'тимлид[а-я]*', 'team ?lead', 'техлид[а-я]*', 'tech ?lead'
        ]),
        ('software_engineer', [
            'программист[а-я]*', 'разработчик[а-я]*', 'developer', 'programmer', 'software', 'c[+][+]', 'c',
            'gamedev', 'game developer', 'unity', 'unreal', 'инженер[а-я]*-разработчик[а-я]*'
        ]),
        ('other_it', [
            'архитектор[а-я]*', 'architect', 'it', 'ит', 'информационн[а-я]* (технолог|систем|ресурс)[а-я]*',
            'техническ[а-я]* писател[а-я]*', 'technical writer', 'искусственн[а-я]* интеллект[а-я]*', 'нейросет[а-я]*',
            'ии', 'ai', 'аналитик[а-я]*', 'analyst', 'data', 'цифров[а-я]*', 'асу', 'асу тп', 'автоматизированн[а-я]* систем[а-я]*',
            'компьютер[а-я]*', 'эвм', 'web', 'веб', 'сайт[а-я]*', 'интернет', 'web3', 'blockchain', 'блокчейн[а-я]*',
            'администратор[а-я]*', 'инженер[а-я]*-программист[а-я]*'
        ])
    ]) }}
{%- endmacro %}


{% macro role_bucket_values() -%}
    {%- set names = [] -%}
    {%- for name, _ in role_buckets() if name not in names -%}{%- do names.append(name) -%}{%- endfor -%}
    {{ return(names) }}
{%- endmacro %}


{#- Expression of the bucket for a text that is already rule_text(). -#}
{% macro role_bucket(text) -%}
    multiIf(
    {%- for name, words in role_buckets() %}
        match({{ text }}, '{{ words_pattern(words) }}'), '{{ name }}',
    {%- endfor %}
        'not_it'
    )
{%- endmacro %}


{#- Grade named in a title. When a title names several ("Junior/Middle", "Middle+/Senior"), the
    lowest wins: the junior slice asks who may apply, and a range is open from its bottom. ведущий
    and главный (the portal's civil-service style "leading", "chief" specialist) read as senior;
    руководитель and начальник as lead. -#}
{% macro grade_levels() -%}
    {{ return([
        ('intern', ['стажер[а-я]*', 'стажировк[а-я]*', 'intern', 'internship', 'trainee', 'практикант[а-я]*']),
        ('junior', ['junior', 'jun', 'jr', 'джун[а-я]*', 'младш[а-я]*', 'начинающ[а-я]*', 'entry[- ]level']),
        ('middle', ['middle', 'mid', 'мидл[а-я]*']),
        ('senior', ['senior', 'sr', 'сеньор[а-я]*', 'синьор[а-я]*', 'старш[а-я]*', 'ведущ[а-я]*', 'главн[а-я]*', 'expert', 'эксперт[а-я]*']),
        ('lead', [
            'lead', 'teamlead', 'тимлид[а-я]*', 'техлид[а-я]*', 'лид', 'head', 'руководител[а-я]*', 'начальник[а-я]*',
            'principal', 'chief', 'cto', 'директор[а-я]*'
        ])
    ]) }}
{%- endmacro %}


{% macro grade_from_text(text) -%}
    multiIf(
    {%- for name, words in grade_levels() %}
        match({{ text }}, '{{ words_pattern(words) }}'), '{{ name }}',
    {%- endfor %}
        'unknown'
    )
{%- endmacro %}


{#- Work format from what the record states. hybrid only from the text; remote from a source field
    (trudvsem employment «Дистанционная (удаленная) работа», Habr remoteWork) or a remote phrase;
    office from an office phrase, or on Habr when remote work is not allowed and a city is named.
    Phrases are narrow on purpose: «удаленный доступ», «удаленное администрирование» are duties of
    admins, not a format. Anything else is unknown, never office by default. -#}
{% macro work_format(text, remote_flag, office_default) -%}
    multiIf(
        match({{ text }}, '{{ words_pattern(["гибрид[а-я]*", "hybrid", "смешанн[а-я]* (формат|график|режим)[а-я]*", "#гибрид"]) }}'), 'hybrid',
        coalesce({{ remote_flag }}, false)
            OR match({{ text }}, '{{ words_pattern([
                "удаленн[а-я]* (работ|формат|режим|занятост)[а-я]*", "удаленно", "удаленк[а-я]*", "full ?remote",
                "remote", "работ[а-я]* из дома", "дистанционн[а-я]* (работ|формат|режим|занятост)[а-я]*",
                "формат[а-я]* работы:? удаленн[а-я]*", "удаленн[а-я]* или", "из любой точки"
            ]) }}'), 'remote',
        match({{ text }}, '{{ words_pattern([
                "(в|из) офис[а-я]*", "офисн[а-я]* (формат|работ|режим)[а-я]*", "формат[а-я]* работы:? офис[а-я]*",
                "office", "on-?site", "onsite", "#офис"
            ]) }}') OR coalesce({{ office_default }}, false), 'office',
        'unknown'
    )
{%- endmacro %}


{#- Region group. Remote work comes first: a remote vacancy of a Moscow employer is open to anyone,
    which is the point of the group. Moscow and St Petersburg are the cities, not their oblasts. -#}
{% macro region_group(work_format, moscow, spb, any_place) -%}
    multiIf(
        {{ work_format }} = 'remote', 'remote',
        coalesce({{ moscow }}, false), 'moscow',
        coalesce({{ spb }}, false), 'spb',
        coalesce({{ any_place }}, false), 'other',
        'unknown'
    )
{%- endmacro %}


{#- "No experience needed" said in a text (Habr and Telegram, which have no experience field). -#}
{% macro no_experience_text(text) -%}
    match({{ text }}, '{{ words_pattern([
        "без опыта", "опыт[а-я]* (работы )?не (требуется|обязател[а-я]*|нужен|важен)", "no experience",
        "без опыта работы", "можно без опыта", "опыт от 0", "опыт работы от 0"
    ]) }}')
{%- endmacro %}


{#- Text as compared for Telegram cross-posts: links, @names and hashtags are channel furniture
    (each channel signs its copy), so they and punctuation, emoji and spacing do not count. -#}
{% macro telegram_text_hash(expr) -%}
    cityHash64(trimBoth(replaceRegexpAll(
        replaceRegexpAll({{ rule_text(expr) }}, 'https?://[^ \n]+|t[.]me/[^ \n]+|@[a-z0-9_]+|#[^ \n#]+', ' '),
        '[^a-zа-я0-9+]+', ' '
    )))
{%- endmacro %}


{#- First line of a post that is not only hashtags: the post's title, as radar.collect.telegram_rules
    title_line takes it. -#}
{% macro telegram_title_line(expr) -%}
    substringUTF8(trimBoth(arrayFirst(
        l -> trimBoth(l) != '' AND NOT match(l, '^([ \t]*#[^ \t#]+)+[ \t]*$'),
        splitByChar('\n', coalesce({{ expr }}, ''))
    )), 1, 120)
{%- endmacro %}


{#- The skill dictionary (seed skill_dictionary) as one constant: (skill, pattern, prefilter) sorted by
    skill. A synonym is an RE2 fragment matched between non-word characters, where + and # count as
    word characters so that c, c++ and c# stay apart.
    The word boundaries are negated classes that mix Cyrillic and Latin ranges, and Hyperscan
    (multiMatch*, ClickHouse 26.8) gets those wrong: '[^a-zа-я0-9_+#]нода' matches inside
    «законодательства», which made Node.js 690 of 10 506 snapshot vacancies instead of 34. So
    Hyperscan only shortlists the skills by their bare synonyms (positive classes, which it handles),
    and RE2 (match) decides with the boundaries. The seed keeps negated classes out of the synonyms,
    so the shortlist stays a superset; assert_skills_found_matches_re2 checks it on real texts. -#}
{% macro skill_dictionary_constant() -%}
    (
        SELECT arraySort(groupArray((
            skill,
            concat('(^|[^a-zа-я0-9_+#])(', synonyms, ')([^a-zа-я0-9_+#]|$)'),
            concat('(', synonyms, ')')
        )))
        FROM {{ ref('skill_dictionary') }}
    )
{%- endmacro %}


{% macro skills_found(text) -%}
    arraySort(arrayMap(i -> skill_dict[i].1, arrayFilter(
        i -> match({{ text }}, skill_dict[i].2),
        arrayDistinct(multiMatchAllIndices({{ text }}, arrayMap(p -> p.3, skill_dict)))
    )))
{%- endmacro %}


{#- The same skills by RE2 alone, pattern by pattern: the reference the test above compares with. -#}
{% macro skills_found_re2(text) -%}
    arrayMap(p -> p.1, arrayFilter(p -> match({{ text }}, p.2), skill_dict))
{%- endmacro %}


{#- Quantiles of a salary cell are printed only from this many stated amounts; below it they are
    NULL and the count still says how many there were. -#}
{% macro salary_min_cell() -%}
    5
{%- endmacro %}


{#- An entity counts as open now when it is not closed and was seen within this many days of the
    latest collected day of its source family. With the snapshot backfill still in July and the API
    in October, the July snapshot vacancies that were never seen again are not called open today. -#}
{% macro open_now_days() -%}
    7
{%- endmacro %}


{#- Telegram in the cross-sectional marts (skills, format, junior): posts of the last this many days
    before the newest post, the span of the trudvsem snapshot backfill, so that a 2019 post does not
    weigh in a skill share next to 2026 vacancies. Weekly dynamics keeps every week. -#}
{% macro telegram_window_days() -%}
    100
{%- endmacro %}


{% macro employers_top_n() -%}
    100
{%- endmacro %}


{#- A median lifetime is printed for an employer from this many closed vacancies. -#}
{% macro lifetime_min_closed() -%}
    3
{%- endmacro %}


{#- Quantiles of a lifetime row (mart_lifetime) are printed only from this many closures. -#}
{% macro lifetime_min_n() -%}
    5
{%- endmacro %}
