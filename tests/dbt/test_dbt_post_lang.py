"""Macro post_lang (dbt/macros/telegram_lang.sql) is the rule of radar.extract.telegram_corpus.post_lang
in RE2: these tests keep its constants and patterns equal to the Python ones. Agreement on real posts
needs ClickHouse, so it is not checked here."""

import re
from pathlib import Path
from typing import Any

from radar.extract import baseline
from radar.extract.telegram_corpus import _HANDLE, _HASHTAG, _URL
from radar.pii.text import PLACEHOLDER

ROOT = Path(__file__).resolve().parents[2]
MACRO = (ROOT / "dbt" / "macros" / "telegram_lang.sql").read_text(encoding="utf-8")


def test_letter_shares_are_those_of_the_baseline() -> None:
    assert f"> {baseline._EN_LATIN_SHARE}, 'en'" in MACRO
    assert f") / {{{{ cyr }}}} > {baseline._OTHER_CYRILLIC_SHARE}, 'other'" in MACRO
    assert f"'{baseline._CYRILLIC.pattern}'" in MACRO
    assert f"'{baseline._LATIN.pattern}'" in MACRO
    assert f"'{baseline._NON_RUSSIAN_CYRILLIC.pattern}'" in MACRO


def test_own_text_cuts_what_the_python_cuts() -> None:
    placeholders = re.search(r"return\(\[([^\]]*)\]\)", MACRO)
    assert placeholders
    assert {f"[{p}]" for p in re.findall(r"'(\w+)'", placeholders.group(1))} == set(PLACEHOLDER.values())
    # A change to these patterns has to be carried into the macro by hand.
    assert _URL.pattern == r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+[a-z]{2,}/\S*"
    assert _HASHTAG.pattern == r"#[^\s#]+"
    assert _HANDLE.pattern == r"@\w+"


def test_stg_telegram_filters_by_the_seed_language(manifest: dict[str, Any]) -> None:
    node = next(n for n in manifest["nodes"].values() if n["name"] == "stg_telegram")
    assert "macro.radar.post_lang" in node["depends_on"]["macros"]
    assert "WHERE channel_lang = '' OR {{ post_lang('body') }} = channel_lang" in node["raw_code"]
