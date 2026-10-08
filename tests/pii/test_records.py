from radar.pii import clean_trudvsem_api, clean_trudvsem_snapshot


def test_contacts_without_value_are_not_counted() -> None:
    record = {
        "id": "x",
        "contactList": [
            {"contactType": "Телефон", "contactValue": "+7 900 000-00-00"},
            {"contactType": "Скайп", "isPreferred": "false"},
            {"contactType": "Другое", "contactValue": "  "},
        ],
    }
    cleaned, counts = clean_trudvsem_snapshot(record)
    assert "contactList" not in cleaned
    assert counts == {"phone": 1}


def test_api_contacts_without_value_are_not_counted() -> None:
    vacancy = {"id": "x", "contact_list": [{"contact_type": "Эл. почта", "contact_value": ""}], "contact_person": ""}
    cleaned, counts = clean_trudvsem_api(vacancy)
    assert cleaned == {"id": "x"}
    assert counts == {}


def test_company_that_loses_every_field_is_dropped_not_kept() -> None:
    record = {
        "fullCompanyName": "ИП Иванов Иван Иванович",
        "company": {"companyCode": "300000000000001", "inn": "770000000001", "ogrn": "300000000000001"},
    }
    cleaned, counts = clean_trudvsem_snapshot(record)
    assert cleaned == {"fullCompanyName": "ИП"}
    assert counts == {"employer_name": 1, "inn": 1, "ogrn": 1, "company_code": 1}
