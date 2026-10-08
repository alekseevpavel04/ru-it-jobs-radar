import pytest

from radar.extract.salary import is_negotiable, parse_salary


def as_tuple(phrase: str) -> tuple[object, ...] | None:
    s = parse_salary(phrase)
    return None if s is None else (s.from_, s.to, s.currency, s.gross_net, s.period)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # The salary kinds the parser knows.
        ("300-400к net", (300000, 400000, None, "net", None)),
        ("до $5000", (None, 5000, "USD", None, None)),
        ("от 150 000 руб. до вычета НДФЛ", (150000, None, "RUB", "gross", None)),
        ("до 300к", (None, 300000, None, None, None)),
        # Ranges and bounds.
        ("от 50 000 до 52 000 руб.", (50000, 52000, "RUB", None, None)),
        ("120 000 – 180 000 ₽ на руки", (120000, 180000, "RUB", "net", None)),
        ("50 000 руб. - 60 000 руб.", (50000, 60000, "RUB", None, None)),
        ("от 100 тыс. руб.", (100000, None, "RUB", None, None)),
        ("200-250 тыс. руб. gross", (200000, 250000, "RUB", "gross", None)),
        ("80-100 т.р.", (80000, 100000, "RUB", None, None)),
        ("не более 90 000 рублей", (None, 90000, "RUB", None, None)),
        # Single amounts fill both bounds.
        ("5000р.", (5000, 5000, "RUB", None, None)),
        ("оклад 45 000 руб., 13% НДФЛ", (45000, 45000, "RUB", None, None)),
        ("ЗП на руки около 50 000, после стажировки до 59 000", (50000, 50000, None, "net", None)),
        # Decimal commas, thousands separators, multipliers.
        ("до 60 000,00 рублей до вычета налогов", (None, 60000, "RUB", "gross", None)),
        ("150,000 USD per year", (150000, 150000, "USD", None, "year")),
        ("от 2,4 до 3 млн руб. в год", (2400000, 3000000, "RUB", None, "year")),
        ("3.6 млн в год", (3600000, 3600000, None, None, "year")),
        ("1.5k$", (1500, 1500, "USD", None, None)),
        # Currencies and periods.
        ("$3000-4000 gross", (3000, 4000, "USD", "gross", None)),
        ("€4k–6k/month", (4000, 6000, "EUR", None, "month")),
        ("100 000 - 150 000 тенге", (100000, 150000, "KZT", None, None)),
        ("2000 BYN", (2000, 2000, "BYN", None, None)),
        ("1500 руб/час", (1500, 1500, "RUB", None, "hour")),
        ("2500 ₽ за смену", (2500, 2500, "RUB", None, "day")),
        ("90 000 руб. в месяц после вычета налогов", (90000, 90000, "RUB", "net", "month")),
        ("Salary: from 4000 to 5000 EUR net", (4000, 5000, "EUR", "net", None)),
    ],
)
def test_phrases(phrase: str, expected: tuple[object, ...]) -> None:
    assert as_tuple(phrase) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "Зарплата по итогам собеседования",
        "Заработная плата: конкурентная, обсуждается по результатам собеседования",
        "по договорённости",
        "от 0",
        "",
        None,
        "150 000 - 100 000 руб.",  # inverted range is not guessed into shape
    ],
)
def test_no_salary_is_none_not_zero(phrase: str | None) -> None:
    assert parse_salary(phrase) is None


def test_negotiable_needs_no_amount() -> None:
    assert is_negotiable("Зарплата по итогам собеседования")
    assert not is_negotiable("от 50 000 руб., далее по итогам собеседования")
    assert not is_negotiable("от 50 000 руб.")
