-- The link check of int_extracted_fields (macros link_free_text, named_outside_links) on fixed texts.
-- A failing row is a text whose checked skills differ from the expected ones. tests/dbt reads these
-- cases and runs radar.extract.skills.named_outside_links on them, so RE2 and Python agree.
SELECT body, expected, arrayStringConcat({{ named_outside_links(link_free_text('body')) }}, ',') AS found
FROM (
    SELECT tupleElement(c, 1) AS body, tupleElement(c, 2) AS expected
    FROM (SELECT arrayJoin([
        ('Отклик: https://example.com/jobs', ''),
        ('Понимание протоколов HTTP/HTTPS, TCP', 'HTTP,HTTPS'),
        ('Знание HTTP. Подробнее https://example.com', 'HTTP'),
        ('Сайт www.example.com/https', ''),
        ('HTTP-запросы, анкета https://example.com/a?b=http', 'HTTP'),
        ('Протокол HTTPS', 'HTTPS'),
        ('aiohttp и httpx', ''),
        ('Ссылка http://example.com\nHTTPS обязателен', 'HTTPS'),
        ('ЗНАНИЕ Https', 'HTTPS'),
        ('HTTPS://EXAMPLE.COM/VACANCY', '')
    ]) AS c)
)
WHERE found != expected
