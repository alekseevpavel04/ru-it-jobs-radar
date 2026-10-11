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
        ("Ищет Анна Ким, [nick]", "Ищет [name], [nick]", "seeker"),
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
    ("text", "expected"),
    [
        # Surnames the shape rules do not know, in the credit line one channel puts under every post.
        ("Ищет Дарья Остапенко, её\xa0пост на\xa0LinkedIn.", "Ищет [name], её\xa0пост на\xa0LinkedIn."),
        ("Ищет Олег Ли, его пост.", "Ищет [name], его пост."),
        ("Ищет Анна Смирнова", "Ищет [name]"),
        ("Ищет: Мария Шульц", "Ищет: [name]"),
        ("Вакансия в финтех, ищет Кира Пак.", "Вакансия в финтех, ищет [name]."),
        ("Ищет Елена К., её пост на LinkedIn.", "Ищет [name], её пост на LinkedIn."),
        ("Ищет Ольга Римская-Корсак", "Ищет [name]"),
        ("Ищет Ivan Petrov, his post", "Ищет [name], his post"),
        # First names outside first_names.csv count with the rest of the credit line.
        ("Ищет Айгерим Сапарова, её\xa0пост на\xa0LinkedIn.", "Ищет [name], её\xa0пост на\xa0LinkedIn."),
        ("Ищет Нурдаулет Б., его пост.", "Ищет [name], его пост."),
        ("Ищет Alesia Mazur, подробности в её посте на LinkedIn.", "Ищет [name], подробности в её посте на LinkedIn."),
    ],
)
def test_recruiter_credit_after_the_verb(text: str, expected: str) -> None:
    cleaned, counts = clean_text(text)
    assert cleaned == expected
    assert counts == {"person_name": 1}
    assert [f.rule for f in scan_text(text)] == ["seeker"]
    assert scan_text(cleaned) == []


@pytest.mark.parametrize(
    "text",
    [
        "Компания ищет Python-разработчика",
        "Команда ищет Senior Data Scientist",
        "Банк ищет Владимира в отдел",
        "Ищет Анна",
        "Яндекс ищет Julia Developer",
        "ищет Julia Kotlin",
        "Разыщет Анна Остапенко",
        "Ищет Айгерим Сапарова",
        "Ищет Senior Python, его пост на LinkedIn.",
        "Ищет Data Scientist, её пост на LinkedIn.",
    ],
)
def test_the_verb_alone_does_not_make_a_name(text: str) -> None:
    findings = [f for f in scan_text(text) if f.rule == "seeker"]
    assert findings == []


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
