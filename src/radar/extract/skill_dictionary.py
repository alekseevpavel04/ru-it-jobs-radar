"""Skills by the marts' skill dictionary v0 (dbt/seeds/skill_dictionary.csv), for the regex
baseline until the tuning builds the dictionary of the extraction schema.

A synonym matches between non-word characters, where + and # count as word characters so that c,
c++ and c# stay apart: the rule of the dbt macro `skill_dictionary_constant`, with one addition, a
version number may follow a spelling that ends in a symbol ("C++11" is C++, as the guide asks). The
CSV is read, not copied, so the marts and the baseline count the same spellings.

Entries the annotation guide does not count as skills are left out. The guide counts a named tool,
language, library, framework, database, platform, protocol, standard or method; it does not count
human languages, personal qualities, general activities ("сбор требований", "интеграция",
"тестирование") or fields of knowledge without a proper name. The marts' dictionary has entries of
every one of those kinds: they match words like "алгоритм", "нейросеть" or "прогнозирование" in
almost any text.
"""

import csv
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from radar.config import ROOT

DICTIONARY = ROOT / "dbt" / "seeds" / "skill_dictionary.csv"

NOT_SKILL_CATEGORIES = frozenset({"language_human", "soft"})
NOT_SKILLS = frozenset(
    # general activities
    {"Requirements", "API integration", "Manual testing", "Test automation", "Prompt engineering", "Fine-tuning"}
    # fields of knowledge
    | {"Statistics", "Algorithms", "Multithreading", "Networking", "Virtualization", "Cryptography", "Embedded"}
    | {"System design", "Microservices", "Design patterns", "Clean architecture", "Product analytics", "Big Data"}
    | {"Machine learning", "Deep learning", "Gradient boosting", "NLP", "Computer vision", "Recommender systems"}
    | {"Time series", "Speech recognition"}
)

# Spellings that are ordinary English words. In an English text they count only when not written in
# lowercase: "Go" and "REST" are skills, "go" and "the rest" are not.
ENGLISH_WORDS = frozenset(
    {"go", "rest", "less", "solid", "chef", "node", "argo", "gin", "rabbit", "elastic", "arm", "spring"}
    | {"swift", "rust", "dart", "julia", "ci", "lan", "wan", "yarn", "hive", "triton", "astra"}
)

_WORD = "a-zа-я0-9_+#"


@dataclass(frozen=True)
class Entry:
    skill: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Hit:
    skill: str
    start: int
    end: int


def _pattern(synonyms: str) -> re.Pattern[str]:
    # Longer spellings first, so that "rest api" wins over "rest" at the same position.
    alternatives = sorted(synonyms.split("|"), key=len, reverse=True)
    return re.compile(rf"(?<![{_WORD}])(?:{'|'.join(alternatives)})(?![a-zа-я_+#])(?!(?<=[a-zа-я0-9])\d)")


@cache
def load(path: Path = DICTIONARY) -> tuple[Entry, ...]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Entry(row["skill"], _pattern(row["synonyms"]))
        for row in rows
        if row["category"] not in NOT_SKILL_CATEGORIES and row["skill"] not in NOT_SKILLS
    )


def find_skills(
    text: str, normalized: str, english: bool = False, entries: tuple[Entry, ...] | None = None
) -> list[str]:
    """Canonical skills in order of first appearance.

    `normalized` is `text` lowercased with ё as е, character for character, so spans line up. A match
    inside the span of another skill's match is dropped: "MS SQL" is MS SQL Server, not also SQL, and
    "1С-Битрикс" is Bitrix, not also 1С.
    """
    hits: list[Hit] = []
    for entry in entries if entries is not None else load():
        for m in entry.pattern.finditer(normalized):
            if english and m.group(0) in ENGLISH_WORDS and text[m.start() : m.end()].islower():
                continue
            hits.append(Hit(entry.skill, m.start(), m.end()))
    kept = [
        h
        for h in hits
        if not any(
            o.skill != h.skill and o.start <= h.start and h.end <= o.end and (o.end - o.start) > (h.end - h.start)
            for o in hits
        )
    ]
    return list(dict.fromkeys(h.skill for h in sorted(kept, key=lambda h: h.start)))
