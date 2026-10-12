"""Personal data in structured records, per source, before anything reaches the raw layer.

Contacts are dropped whole (no hashes: a hash of a phone number is still personal data). Employers
who are people lose their name, INN, OGRNIP, every code and link built from it and the street-level
address. Every remaining string goes through `clean_text`. The returned counter is `pii_removed`.
"""

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

from radar.pii.employer import clean_employer_name, coarse_address, digits, is_ip_name, is_person_employer
from radar.pii.text import DEFAULT_KEEP_HANDLES, clean_text_counts, scan_text

Record = dict[str, Any]

_CONTACT_KIND = {
    "телефон": "phone",
    "мобильный телефон": "phone",
    "факс": "phone",
    "эл. почта": "email",
    "электронная почта": "email",
    "скайп": "nick",
}

# Values that are identifiers, dates or numbers: no free text, and long digit runs there are not phones.
_NOT_TEXT = frozenset(
    {
        "id",
        "creationDate",
        "dateModify",
        "datePublished",
        "changeTime",
        "hireDate",
        "salaryMin",
        "salaryMax",
        "codeProfession",
        "stateRegionCode",
        "federalDistrictCode",
        "oksoCode",
        "latitude",
        "longitude",
        "lat",
        "lng",
        "addressCode",
        "inn",
        "kpp",
        "ogrn",
        "companyCode",
        "companycode",
        "url",
        "vacancyUrl",
        "vac_url",
        "creation-date",
        "date_modify",
        "region_code",
        "code_profession",
        "href",
        "src",
        "date",
        "source_id",
        "content_hash",
        "fetched_at",
        "published_at",
        "modified_at",
        "snapshot_version",
        "link",
        "client_logo",
    }
)
# UUIDs and hex digests (`*_hash`) can hold any digit run; they are not checked for codes either.
_OPAQUE = frozenset({"id", "source_id"})

_OGRNIP_RUN = re.compile(r"(?<!\d)3\d{14}(?!\d)")
_CONTACT_KEYS = frozenset(
    {"contactPerson", "contact_person", "contactList", "contact_list", "contact", "phone", "phones", "fax", "faxes"}
)
_EMPLOYER_NAME_KEYS = frozenset({"fullCompanyName", "name", "title"})


def _count_contacts(items: Any, counts: Counter[str], type_key: str, value_key: str) -> None:
    # Many entries carry only a type ("Скайп", "Другое") and no value: nothing personal was removed.
    for item in items if isinstance(items, list) else [items]:
        if not isinstance(item, dict) or not str(item.get(value_key) or "").strip():
            continue
        kind = _CONTACT_KIND.get(str(item.get(type_key) or "").strip().lower(), "contact_other")
        counts[kind] += 1


def _clean_strings(node: Any, counts: Counter[str], key: str = "") -> Any:
    if isinstance(node, str):
        if key in _NOT_TEXT or not node:
            return node
        cleaned, found = clean_text_counts(node)
        for kind, n in found:
            counts[kind] += n
        return cleaned
    if isinstance(node, dict):
        return {k: _clean_strings(v, counts, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_clean_strings(v, counts, key) for v in node]
    return node


def _drop(mapping: Record, keys: Iterable[str]) -> bool:
    removed = False
    for key in keys:
        if mapping.pop(key, None) not in (None, "", [], {}):
            removed = True
    return removed


def _set_company(out: Record, company: Record) -> None:
    # An employer can lose every company field; the original object must not survive in that case.
    if company:
        out["company"] = company
    else:
        out.pop("company", None)


def clean_trudvsem_snapshot(record: Record) -> tuple[Record, Counter[str]]:
    """One `<vacancy>` of the open-data XML snapshot, already turned into a dict."""
    out = dict(record)
    counts: Counter[str] = Counter()
    if out.pop("contactPerson", None):
        counts["contact_person"] += 1
    _count_contacts(out.pop("contactList", None), counts, "contactType", "contactValue")
    company = dict(out.get("company") or {})
    if company.pop("email", None):
        counts["email"] += 1
    external = out.get("codeExternalSystem")
    if isinstance(external, str) and "@" in external:
        # Integration account of another job board; the board's domain is what carries meaning.
        out["codeExternalSystem"] = external.rsplit("@", 1)[1]
        counts["email"] += 1
    if is_person_employer(out.get("fullCompanyName"), company.get("inn"), company.get("ogrn")):
        out["fullCompanyName"], removed = clean_employer_name(out.get("fullCompanyName"))
        counts["employer_name"] += removed
        counts["inn"] += _drop(company, ["inn"])
        counts["ogrn"] += _drop(company, ["ogrn"])
        counts["company_code"] += _drop(company, ["companyCode"])
        _drop(company, ["url", "site", "kpp"])
        out.pop("vacancyUrl", None)
        address = out.get("vacancyAddress")
        coarse = coarse_address(address)
        dropped = _drop(
            out, ["vacancyAddressHouse", "vacancyAddressAdditionalInfo", "addressOffice", "geo", "addressCode"]
        )
        if coarse is None:
            out.pop("vacancyAddress", None)
        else:
            out["vacancyAddress"] = coarse
        counts["address"] += dropped or coarse != address
    _set_company(out, company)
    out = _clean_strings(out, counts)
    return out, +counts


def clean_trudvsem_api(vacancy: Record) -> tuple[Record, Counter[str]]:
    """`results.vacancies[].vacancy` of the trudvsem API."""
    out = dict(vacancy)
    counts: Counter[str] = Counter()
    if out.pop("contact_person", None):
        counts["contact_person"] += 1
    _count_contacts(out.pop("contact_list", None), counts, "contact_type", "contact_value")
    company = dict(out.get("company") or {})
    if company.pop("email", None):
        counts["email"] += 1
    if is_person_employer(company.get("name"), company.get("inn"), company.get("ogrn")):
        company["name"], removed = clean_employer_name(company.get("name"))
        counts["employer_name"] += removed
        counts["inn"] += _drop(company, ["inn"])
        counts["ogrn"] += _drop(company, ["ogrn"])
        counts["company_code"] += _drop(company, ["companycode"])
        _drop(company, ["url", "site", "kpp"])
        out.pop("vac_url", None)
        addresses = (out.get("addresses") or {}).get("address") or []
        coarse = [{"location": coarse_address(a.get("location"))} for a in addresses]
        if coarse != addresses:
            out["addresses"] = {"address": coarse}
        counts["address"] += _drop(out, ["addressOffice"]) or coarse != addresses
    _set_company(out, company)
    out = _clean_strings(out, counts)
    return out, +counts


def _clean_habr_company(company: Record, counts: Counter[str], name_key: str) -> Record:
    company = dict(company)
    if is_ip_name(company.get(name_key)):
        counts["employer_name"] += company[name_key] != "ИП"
        company[name_key] = "ИП"
        # The alias, page link, id and logo all lead back to the person.
        if _drop(company, ["alias_name", "href", "id", "logo", "rating", "url", "vacancies"]):
            counts["company_code"] += 1
    return company


def clean_habr_vacancy(vacancy: Record) -> tuple[Record, Counter[str]]:
    """A Habr Career vacancy: a list item or the `vacancy` object of a card."""
    out = dict(vacancy)
    counts: Counter[str] = Counter()
    if isinstance(out.get("company"), dict):
        out["company"] = _clean_habr_company(out["company"], counts, "title")
    out = _clean_strings(out, counts)
    return out, +counts


def clean_habr_card_company(company: Record) -> tuple[Record, Counter[str]]:
    """The top-level `company` object of a card's SSR state."""
    counts: Counter[str] = Counter()
    out = _clean_strings(_clean_habr_company(company, counts, "name"), counts)
    return out, +counts


_SUPERJOB_PLACE = ("latitude", "longitude", "metro")
# What leads from a SuperJob employer who is a person back to them: the client page (its URL slug is
# the transliterated name), the site, the logo, the description and the addresses.
_SUPERJOB_CLIENT_TRACE = ("id", "link", "url", "description", "client_logo", "address", "addresses")
_IP_TOKEN = re.compile(r"(?<![\w-])ИП(?![\w-])")
_WORD = re.compile(r"[^\W\d_]+")


def _is_whole_person(name: Any) -> bool:
    """ "ИП Фамилия Имя Отчество", or nothing but a full name."""
    if not isinstance(name, str) or not name.strip():
        return False
    return is_ip_name(name) or clean_text_counts(name.strip(" «»\"'"))[0] == "[name]"


def _is_person_name(name: Any) -> bool:
    """A SuperJob employer name that is a person: a whole name, or a brand with "(ИП Фамилия Имя)"."""
    if _is_whole_person(name):
        return True
    return isinstance(name, str) and "[name]" in clean_text_counts(name)[0] and bool(_IP_TOKEN.search(name))


def _clean_person_employer(name: Any, person_words: set[str]) -> tuple[Any, bool]:
    """The other name of the same employer can be a bare surname ("Фамилия" next to "Фамилия Имя
    Отчество"); a name made only of the person's words goes whole, a brand stays."""
    if not isinstance(name, str) or not name.strip():
        return name, False
    words = {w.lower() for w in _WORD.findall(name)}
    if not is_ip_name(name) and words and words <= person_words:
        return "[name]", True
    cleaned, removed = clean_employer_name(name)
    return cleaned, bool(removed)


def _count_numbers(single: Any, many: Any) -> int:
    numbers = [p for p in many if isinstance(p, dict) and str(p.get("number") or "").strip()] if many else []
    return len(numbers) or int(bool(str(single or "").strip()))


def clean_superjob_vacancy(vacancy: Record) -> tuple[Record, Counter[str]]:
    """An object of SuperJob `GET /2.0/vacancies/`.

    Contacts go whole; the place keeps the town and the address loses the street, the house, the
    coordinates and the metro. An employer who is a person keeps "ИП", its brand or a placeholder.
    """
    out = dict(vacancy)
    counts: Counter[str] = Counter()
    if str(out.pop("contact", None) or "").strip():
        counts["contact_person"] += 1
    if str(out.pop("email", None) or "").strip():
        counts["email"] += 1
    counts["phone"] += _count_numbers(out.pop("phone", None), out.pop("phones", None))
    counts["phone"] += _count_numbers(out.pop("fax", None), out.pop("faxes", None))
    address = out.get("address")
    town = coarse_address(address) if isinstance(address, str) and address else None
    place = _drop(out, _SUPERJOB_PLACE)
    if town is None:
        place = _drop(out, ["address"]) or place
    else:
        out["address"] = town
    counts["address"] += place or town != address
    client = dict(out.get("client") or {})
    for key in ("address", "addresses"):
        value = client.get(key)
        if isinstance(value, str) and value:
            client[key] = coarse_address(value)
            counts["address"] += client[key] != value
        elif value:
            client[key] = []
            counts["address"] += 1
    names = (out.get("firm_name"), client.get("title"))
    if any(_is_person_name(n) for n in names):
        person_words = {w.lower() for n in names if _is_whole_person(n) for w in _WORD.findall(str(n))}
        out["firm_name"], removed = _clean_person_employer(out.get("firm_name"), person_words)
        if client.get("title"):
            client["title"], client_removed = _clean_person_employer(client["title"], person_words)
            removed = removed or client_removed
        counts["employer_name"] += removed
        traced = _drop(client, _SUPERJOB_CLIENT_TRACE)
        counts["company_code"] += _drop(out, ["id_client", "client_logo", "firm_activity"]) or traced
    if "client" in out:
        out["client"] = client
    out = _clean_strings(out, counts)
    return out, +counts


def _walk(node: Any, key: str = "", parent: str = "") -> Iterator[tuple[str, str, Any]]:
    yield key, parent, node
    if isinstance(node, dict):
        for k, v in node.items():
            if k != "pii_removed":
                yield from _walk(v, k, key)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key, parent)


def scan_record(record: Any, keep: Iterable[str] = DEFAULT_KEEP_HANDLES) -> Counter[str]:
    """Personal data still present in a record: structured fields plus every free-text string.

    Used on cleaned records (must find nothing) and on raw fixtures (must find what was planted).
    `keep` lists the handles that are not people; for Telegram, the channels of its config.
    """
    keep = frozenset(h.lower() for h in keep)
    found: Counter[str] = Counter()
    for key, parent, value in _walk(record):
        if key in _CONTACT_KEYS and value:
            found["contact"] += 1
        elif key == "email" and parent in ("company", "payload") and value:
            found["email"] += 1
        elif key == "inn" and len(digits(value)) == 12:
            found["inn"] += 1
        elif key == "ogrn" and len(digits(value)) == 15:
            found["ogrn"] += 1
        elif (
            key in _EMPLOYER_NAME_KEYS
            and isinstance(value, str)
            and is_ip_name(value)
            and value not in ("ИП", "Глава КФХ")
        ):
            found["employer_name"] += 1
        if isinstance(value, str) and value:
            opaque = key in _OPAQUE or key.endswith("_hash")
            if not opaque and _OGRNIP_RUN.search(value):
                found["company_code"] += 1
            if not opaque and key not in _NOT_TEXT:
                for finding in scan_text(value, keep):
                    found[finding.kind] += 1
    return found
