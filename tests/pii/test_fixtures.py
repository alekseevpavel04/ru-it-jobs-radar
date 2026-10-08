"""Every recorded fixture carries synthetic personal data; after cleaning the scanner must find none."""

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from radar.collect.contracts import parse_habr_card
from radar.collect.trudvsem_snapshot import iter_vacancies
from radar.pii import (
    clean_habr_card_company,
    clean_habr_vacancy,
    clean_text,
    clean_trudvsem_api,
    clean_trudvsem_snapshot,
    scan_record,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _trudvsem_api() -> list[dict[str, Any]]:
    page = json.loads((FIXTURES / "trudvsem/vacancies_it_page.json").read_text(encoding="utf-8"))
    return [item["vacancy"] for item in page["results"]["vacancies"]]


def _trudvsem_snapshot() -> list[dict[str, Any]]:
    with (FIXTURES / "trudvsem/snapshot_sample.xml").open("rb") as f:
        return list(iter_vacancies(f))


def _habr_list() -> list[dict[str, Any]]:
    page: dict[str, Any] = json.loads((FIXTURES / "habr/vacancies_page.json").read_text(encoding="utf-8"))
    return list(page["list"])


def _telegram_posts() -> list[str]:
    posts = []
    for page in sorted((FIXTURES / "telegram").glob("*_latest.html")):
        raw = page.read_text(encoding="utf-8")
        for body in re.findall(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', raw, re.DOTALL):
            text = re.sub(r"<br\s*/?>", "\n", body)
            text = re.sub(r'<a [^>]*href="([^"]*)"[^>]*>', r" \1 ", text)
            posts.append(html.unescape(re.sub(r"<[^>]+>", "", text)))
    return posts


def _total(records: list[Any]) -> Counter[str]:
    found: Counter[str] = Counter()
    for record in records:
        found.update(scan_record(record))
    return found


def test_trudvsem_api_fixture() -> None:
    raw = _trudvsem_api()
    assert {"contact", "email", "inn", "ogrn", "employer_name", "company_code"} <= set(_total(raw))
    cleaned = [clean_trudvsem_api(v) for v in raw]
    assert _total([c for c, _ in cleaned]) == Counter()
    sole = cleaned[0]
    assert sole[0]["company"] == {"hr-agency": False, "name": "ИП"}
    assert "vac_url" not in sole[0]
    assert sole[0]["addresses"] == {"address": [{"location": "Красноярский край, г Красноярск"}]}
    assert sole[1] == {
        "contact_person": 1,
        "phone": 1,
        "email": 2,
        "employer_name": 1,
        "inn": 1,
        "ogrn": 1,
        "company_code": 1,
        "address": 1,
    }
    company = cleaned[1][0]["company"]
    assert company["inn"] == "9703115178"
    assert company["url"].endswith("/1227700702074")


def test_trudvsem_snapshot_fixture() -> None:
    raw = _trudvsem_snapshot()
    assert {"contact", "email", "inn", "ogrn", "employer_name", "company_code"} <= set(_total(raw))
    cleaned = [clean_trudvsem_snapshot(v)[0] for v in raw]
    assert _total(cleaned) == Counter()
    assert cleaned[0]["codeExternalSystem"] == "example.org"
    assert cleaned[1]["fullCompanyName"] == "ИП"
    assert cleaned[1]["vacancyAddress"] == "Орловская область, г. Орел"
    assert not {"company", "vacancyUrl", "geo", "addressCode", "vacancyAddressHouse"} & set(cleaned[1])


def test_habr_list_fixture() -> None:
    raw = _habr_list()
    assert _total(raw)["employer_name"] == 1
    cleaned = [clean_habr_vacancy(v) for v in raw]
    assert _total([c for c, _ in cleaned]) == Counter()
    sole = next(c for c, counts in cleaned if counts)
    assert sole["company"] == {"title": "ИП", "accredited": False}


def test_habr_card_fixture() -> None:
    card = parse_habr_card((FIXTURES / "habr/vacancy_card_1000150058.html").read_text(encoding="utf-8"))
    vacancy, _ = clean_habr_vacancy(card.vacancy.model_dump(by_alias=True))
    company, _ = clean_habr_card_company(card.company)
    description, _ = clean_text(card.job_posting.description)
    assert _total([vacancy, company, {"description": description}]) == Counter()


def test_telegram_fixture_texts() -> None:
    posts = _telegram_posts()
    assert len(posts) == 14
    planted = _total([{"text": p} for p in posts])
    assert planted["nick"] > 0
    assert _total([{"text": clean_text(p)[0]} for p in posts]) == Counter()


@pytest.mark.parametrize("loader", [_trudvsem_api, _trudvsem_snapshot, _habr_list])
def test_cleaning_is_idempotent(loader: Any) -> None:
    for record in loader():
        cleaner = {
            _trudvsem_api: clean_trudvsem_api,
            _trudvsem_snapshot: clean_trudvsem_snapshot,
            _habr_list: clean_habr_vacancy,
        }[loader]
        once, _ = cleaner(record)
        twice, counts = cleaner(once)
        assert twice == once
        assert counts == Counter()
