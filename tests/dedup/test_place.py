import pytest

from radar.dedup.place import Place, parse_address, place_of, same_place


@pytest.mark.parametrize(
    ("address", "settlement", "street"),
    [
        ("Город Москва, Большая Юшуньская улица, дом: Д. 1А;", "москва", {"большая", "юшуньская"}),
        ("г Москва, Центральный федеральный округ,Москва", "москва", None),
        (
            "Самарская область, Красноярский район, Волжский поселок городского типа, Пионерская улица",
            "волжский",
            {"пионерская"},
        ),
        ("Волгоградская область, г Волгоград, им. Канунникова улица, дом: Д. 11 А;", "волгоград", {"канунникова"}),
        ("Город Санкт-Петербург, Пискарёвский проспект, 2, корп. 2", "санкт петербург", {"пискаревский"}),
        (None, None, None),
    ],
)
def test_parse_address(address: str | None, settlement: str | None, street: set[str] | None) -> None:
    got_settlement, got_street = parse_address(address)
    assert got_settlement == settlement
    assert got_street == (frozenset(street) if street is not None else None)


def trud(region: str, address: str | None, remote: bool | None = False) -> Place:
    return place_of(region, address, remote, "trudvsem")


def test_api_and_snapshot_spellings_of_one_address_agree() -> None:
    a = trud("Город Санкт-Петербург", "г Санкт-Петербург, Пискарёвский проспект, 2к2")
    b = trud("Город Санкт-Петербург", "Город Санкт-Петербург, Пискарёвский проспект, 2, корп. 2")
    assert same_place(a, b)


def test_two_streets_or_two_towns_are_two_places() -> None:
    kaliningrad = trud("Калининградская область", "Калининградская область, г Калининград, А.Невского улица, 214")
    other_street = trud("Калининградская область", "Калининградская область, Город Калининград, Крымская улица, 10")
    engels = trud("Саратовская область", "Саратовская область, г Энгельс, Степная улица, 124")
    rtishchevo = trud("Саратовская область", "Саратовская область, г Ртищево, Железнодорожная улица, 19а")
    assert not same_place(kaliningrad, other_street)
    assert not same_place(engels, rtishchevo)


def test_missing_street_is_unknown_not_different() -> None:
    a = trud("Самарская область", "Самарская область, г Самара, Приволжский федеральный округ")
    b = trud("Самарская область", "Самарская область, г Самара, Алексея Толстого улица, дом: Д.122;")
    assert same_place(a, b)


def test_regions_without_address() -> None:
    assert same_place(trud("Республика Саха (Якутия)", None), trud("Саха (Якутия) республика", None))
    assert not same_place(trud("Рязанская область", None), trud("Ленинградская область", None))


def test_both_remote_ignores_place() -> None:
    assert same_place(trud("Рязанская область", None, remote=True), trud("Ленинградская область", None, remote=True))


def test_habr_cities_meet_trudvsem_settlement() -> None:
    habr = place_of("Нижний Новгород, Москва", None, False, "habr")
    nn = trud("Нижегородская область", "Нижегородская область, Город Нижний Новгород")
    kazan = trud("Татарстан республика", "Татарстан республика, г Казань, Сибирский Тракт улица, 9/5")
    assert same_place(habr, nn)
    assert not same_place(habr, kazan)
    assert same_place(place_of("Москва", None, False, "habr"), place_of("Москва, Санкт-Петербург", None, False, "habr"))
