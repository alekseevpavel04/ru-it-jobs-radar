import pytest

from radar.pii import clean_employer_name, coarse_address, is_person_employer


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ИП Иванов Иван Иванович", "ИП"),
        ("Индивидуальный предприниматель Иванов И.И.", "ИП"),
        ("ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ ИВАНОВ ИВАН ИВАНОВИЧ", "ИП"),
        ("и.п. Иванова", "ИП"),
        ("Глава КФХ Петров П.П.", "Глава КФХ"),
        ("Глава крестьянского (фермерского) хозяйства Петров Петр Петрович", "Глава КФХ"),
        ("Иванова Мария Петровна", "[name]"),
    ],
)
def test_person_names_lose_the_person(name: str, expected: str) -> None:
    assert clean_employer_name(name) == (expected, 1)


@pytest.mark.parametrize("name", ["ООО Ромашка", "ИПОТЕЧНЫЙ ЦЕНТР", 'АО "ИПК"', "ИП-Строй ООО"])
def test_companies_keep_their_names(name: str) -> None:
    assert clean_employer_name(name) == (name, 0)


def test_person_employer_by_name_inn_or_ogrnip() -> None:
    assert is_person_employer("ИП Иванов", "7700000000", "1027700000000")
    assert is_person_employer('ООО "Ромашка"', "770000000001", None)
    assert is_person_employer('ООО "Ромашка"', None, "300000000000001")
    assert not is_person_employer('ООО "Ромашка"', "7700000000", "1027700000000")


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("Красноярский край, г Красноярск, Примерная улица, 1", "Красноярский край, г Красноярск"),
        ("660000, Красноярский край, г Красноярск, ул. Ленина, д.7", "Красноярский край, г Красноярск"),
        ("Орловская область, г. Орел, площадь Каменского д.1", "Орловская область, г. Орел"),
        ("Самарская область, г Тольятти, Поволжское шоссе, 32", "Самарская область, г Тольятти"),
        ("г Москва", "г Москва"),
        ("ул. Ленина, д. 7", None),
        (None, None),
    ],
)
def test_coarse_address_keeps_region_and_city(address: str | None, expected: str | None) -> None:
    assert coarse_address(address) == expected
