"""Names found by a first name: alone on a line, in a signature line, or next to a surname-shaped word.

Every name here is synthetic.
"""

import pytest

from radar.collect import telegram
from radar.pii import DEFAULT_KEEP_HANDLES, clean_text, scan_record, scan_text


@pytest.mark.parametrize(
    ("text", "expected", "rule"),
    [
        ("Иванова Мария\nPython-разработчик, удалённо", "[name]\nPython-разработчик, удалённо", "name_line"),
        ("👨‍💻 Сидоров Пётр\nML-инженер", "👨‍💻 [name]\nML-инженер", "name_line"),
        ("Вакансия выше.\n— Пётр Сидоров", "Вакансия выше.\n— [name]", "name_line"),
        ("🟠 Анна Ким", "🟠 [name]", "name_line"),
        ("Мария Иванова, HR", "[name], HR", "signature"),
        ("Анна Ким, рекрутер", "[name], рекрутер", "signature"),
        ("Контакт: [nick] (Анна)", "Контакт: [nick] ([name])", "signature"),
        ("Отклики сюда: [phone], Анна", "Отклики сюда: [phone], [name]", "signature"),
        ("Ищет Анна Ким, [nick]", "Ищет [name], [nick]", "signature"),
        ("Maria Ivanova, Talent Acquisition", "[name], Talent Acquisition", "signature"),
        ("Alexey Smirnov, recruiter", "[name], recruiter", "signature"),
        ("Расскажет Иван Петров, основатель сообщества.", "Расскажет [name], основатель сообщества.", "full_name"),
        ("Говорит Роман Ильин, резюме важнее всего.", "Говорит [name], резюме важнее всего.", "full_name"),
    ],
)
def test_names_are_replaced_and_counted(text: str, expected: str, rule: str) -> None:
    cleaned, counts = clean_text(text)
    assert cleaned == expected
    assert counts["person_name"] == 1
    assert [f.rule for f in scan_text(text) if f.kind == "person_name"] == [rule]
    assert scan_text(cleaned) == []


def test_name_with_a_live_contact_on_the_same_line() -> None:
    cleaned, counts = clean_text("Пишите в телеграм @anna_hr_example, Анна")
    assert cleaned == "Пишите в телеграм [nick], [name]"
    assert counts == {"nick": 1, "person_name": 1}


def test_self_introduction() -> None:
    cleaned, counts = clean_text("Привет! Меня зовут Анна, я HR в компании.")
    assert cleaned == "Привет! Меня зовут [name], я HR в компании."
    assert counts == {"person_name": 1}


@pytest.mark.parametrize(
    "text",
    [
        # Companies named after people, or that look like it.
        "Лаборатория Касперского ищет аналитика данных",
        "Работа в «Мария-Ра», сеть «Петрович»",
        "Leroy Merlin и Леруа Мерлен расширяют IT",
        # Job titles and title lines.
        "Python-разработчик (Middle)",
        "Senior Data Scientist",
        "Менеджер по продукту, Альфа-Банк",
        "Руководитель группы ML | LLM/AI Сбер",
        "HR-менеджер в продуктовую команду",
        # Names that are also tools, places, products or words.
        "Языки: Python, Julia, Kotlin",
        "Julia Kotlin Swift",
        "Офис в г. Владимир, менеджер по продажам",
        "Яндекс Алиса и VK Маруся",
        "Иван-чай к обеду",
        "Вера в продукт",
        "Надежда Сбер",
    ],
)
def test_companies_titles_and_words_stay(text: str) -> None:
    assert clean_text(text) == (text, {})


def test_scanner_counts_name_lines() -> None:
    assert scan_record({"text": "Иванова Мария\nДата-инженер"}) == {"person_name": 1}
    assert scan_record({"text": "[name]\nДата-инженер"}) == {}


def test_scanner_keeps_configured_channels_when_given_them() -> None:
    channel = next(c for c in telegram.enabled_channels() if c not in DEFAULT_KEEP_HANDLES)
    record = {"text": f"Больше вакансий в @{channel}"}
    assert scan_record(record) == {"nick": 1}
    assert scan_record(record, telegram.keep_handles()) == {}
