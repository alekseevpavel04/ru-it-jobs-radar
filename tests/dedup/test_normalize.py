import pytest

from radar.dedup.normalize import jaccard, normalize_employer, normalize_text, normalize_title, shingles


def test_text_drops_urls_and_folds_numbers_and_case() -> None:
    raw = "Зарплата 150 000 руб., подробности на https://example.ru/job?id=7 и ЁЛКА C++ / C#!"
    assert normalize_text(raw) == "зарплата 0 руб подробности на и елка c++ c#"


def test_reposts_that_differ_only_in_amounts_are_equal() -> None:
    a = normalize_text("Оклад от 90000 до 120 000 рублей, опыт 3 года")
    b = normalize_text("Оклад от 95000 до 130 000 рублей, опыт 2 года")
    assert a == b


def test_shingles_are_word_kgrams() -> None:
    assert shingles("Python разработчик в банк", 3) == {"python разработчик в", "разработчик в банк"}


def test_short_text_is_one_shingle_and_empty_is_none() -> None:
    assert shingles("Data Scientist", 3) == {"data scientist"}
    assert shingles(" ,.; ", 3) == set()


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), {"a"}) == 0.0


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ('ПАО "МТС"', "мтс"),
        ("МТС", "мтс"),
        ("Публичное акционерное общество «Сбербанк России»", "сбербанк россии"),
        ('ООО "ЯНДЕКС"', "яндекс"),
        ("ИП", None),
        ("", None),
        (None, None),
    ],
)
def test_employer_key(name: str | None, key: str | None) -> None:
    assert normalize_employer(name) == key


def test_title_key_drops_bracketed_notes() -> None:
    assert normalize_title("Senior RAN инженер [ИРТЕЯ]") == "senior ran инженер"
    assert normalize_title("Python-разработчик (удалённо)") == "python разработчик"
