import pytest

from radar.pii import clean_text, scan_text

PHONES = [
    "+7 (900) 000-00-00",
    "+7(900)000-00-01",
    "8 900 000 00 02",
    "89000000003",
    "+79000000004",
    "8-900-000-00-05",
    "+7 900 000 00 06",
    "8 (4912) 00-00-07",
    "(495) 000-00-08",
    "7 900 000-00-09",
    "000-00-10",
]


@pytest.mark.parametrize("phone", PHONES)
def test_phone_forms_are_removed(phone: str) -> None:
    cleaned, counts = clean_text(f"Звоните {phone}, ждём.")
    assert cleaned == "Звоните [phone], ждём."
    assert counts == {"phone": 1}


@pytest.mark.parametrize(
    "text",
    [
        "Зарплата от 50 000 до 70 000 руб.",
        "30 000-45 000 в месяц",
        "Опыт 2020-2024, бюджет 1 000 000 000",
        "ОГРН 1227700702074, ИНН компании 7710538364, КПП 770301001",
        "Код профессии 203191, регион 2400000000000",
        "График 5/2 с 09:00 до 18:00, начало 12.05.2026",
        "Python 3.11, 8 часов, 100%",
    ],
)
def test_numbers_that_are_not_phones_stay(text: str) -> None:
    assert clean_text(text) == (text, {})


def test_emails() -> None:
    cleaned, counts = clean_text("Резюме на hr@example.com или иван.петров@пример.рф.")
    assert cleaned == "Резюме на [email] или [email]."
    assert counts == {"email": 2}


def test_people_handles_go_bots_and_channels_stay() -> None:
    text = "Пишите @ivan_petrov, бот @hr_helper_bot, канал @habr_career"
    cleaned, counts = clean_text(text)
    assert cleaned == "Пишите [nick], бот @hr_helper_bot, канал @habr_career"
    assert counts == {"nick": 1}


def test_handles_allowlist_can_be_replaced() -> None:
    cleaned, _ = clean_text("канал @habr_career", keep=[])
    assert cleaned == "канал [nick]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("t.me/ivan_petrov", "[link]"),
        ("https://t.me/ivan_petrov", "[link]"),
        ("https://t.me/s/geekjobs", "https://t.me/s/geekjobs"),
        ("https://t.me/hr_helper_bot", "https://t.me/hr_helper_bot"),
        ("https://t.me/+AbCdEf123", "https://t.me/+AbCdEf123"),
        ("https://t.me/joinchat/AbCdEf", "https://t.me/joinchat/AbCdEf"),
        ("vk.com/id12345", "[link]"),
        ("https://vk.com/ivanpetrov", "[link]"),
        ("https://vk.com/club123", "https://vk.com/club123"),
        ("wa.me/79000000000", "[link]"),
        ("https://www.linkedin.com/in/ivan-petrov", "[link]"),
        ("https://hh.ru/resume/0123abcd", "[link]"),
        ("https://career.habr.com/vacancies/1000150058", "https://career.habr.com/vacancies/1000150058"),
    ],
)
def test_profile_links(text: str, expected: str) -> None:
    assert clean_text(text)[0] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Контактное лицо: Петрова Мария Сергеевна.",
        "Петрова Мария Сергеевна, менеджер.",
        "Мария Сергеевна ответит.",
        "Обращаться к Иванову Ивану Ивановичу.",
        "Подпись: Иванов И.И.",
        "Подпись: И. И. Иванов",
        "Спросить Ольгу.",
        "HR-менеджер: Анна",
    ],
)
def test_names_in_reliable_shapes(text: str) -> None:
    cleaned, counts = clean_text(text)
    assert "[name]" in cleaned
    assert counts["person_name"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "Технический специалист, логичный подход.",
        "Отдел кадров, Москва Сити.",
        "Работа в компании Яндекс, офис на Красной Пресне.",
    ],
)
def test_ordinary_capitalized_words_stay(text: str) -> None:
    assert clean_text(text) == (text, {})


def test_person_inn_and_ogrnip_in_text() -> None:
    cleaned, counts = clean_text("ИНН 770000000001, ОГРНИП 300000000000001")
    assert cleaned == "[inn], [ogrn]"
    assert counts == {"inn": 1, "ogrn": 1}


def test_scan_finds_positions_and_nothing_after_cleaning() -> None:
    text = "Петрова Мария Сергеевна, +7 900 000-00-00, hr@example.com, @ivan_petrov, t.me/ivan_petrov"
    kinds = sorted(f.kind for f in scan_text(text))
    assert kinds == ["email", "nick", "person_name", "phone", "profile_link"]
    assert all(0 <= f.start < f.end <= len(text) for f in scan_text(text))
    assert scan_text(clean_text(text)[0]) == []


def test_long_text_matches_short_path() -> None:
    body = "Обязанности: разработка. " * 20
    text = body + "Телефон +7 900 000-00-00."
    assert len(text) > 200
    cleaned, counts = clean_text(text)
    assert cleaned.endswith("Телефон [phone].")
    assert counts == {"phone": 1}
