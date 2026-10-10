"""Skills found in a text by their spellings: the skill search of the regex baseline.

The default matcher reads the dictionary v1 (`configs/skills.toml`, `radar.extract.skills`). It
searches the canonical name, the aliases and the inflection regexes of every entry, except
- names listed in [search] names_not_searched (a letter or an ordinary word: "C", "Смета"), whose
  aliases are still searched;
- entries whose only evidence is the gold labels ([provenance] gold_only): searching them would tune
  the baseline's vocabulary to the very labels it is scored on.

A spelling matches between non-word characters, where + and # count as word characters so that c,
c++ and c# stay apart, and a version number may follow a spelling that ends in a symbol ("C++11").
Separators inside a fixed spelling are optional in the text: "node.js" also finds "nodejs" and
"node js". A match inside a longer match of another skill is dropped: "MS SQL" is MS SQL Server, not
also SQL.

`load_v0` reads the marts' dictionary v0 (`dbt/seeds/skill_dictionary.csv`) with the
exclusions the baseline used before v1. It stays to measure v1 against v0 (results/skills/).
"""

import csv
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from radar.config import ROOT
from radar.extract import skills as dictionary_v1

V0_SEED = ROOT / "dbt" / "seeds" / "skill_dictionary.csv"

_WORD = "a-zа-я0-9_+#"
_SEP = r"[\s\-_.]*"
_META = set("[](){}.*+?|\\^$")


@dataclass(frozen=True)
class Entry:
    skill: str
    pattern: re.Pattern[str]
    # Substrings one of which a text must contain for the pattern to have a chance; empty: always run.
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Matcher:
    entries: tuple[Entry, ...]
    english_words: frozenset[str]
    complete: bool


@dataclass(frozen=True)
class Hit:
    skill: str
    start: int
    end: int


def _compile(alternatives: list[str]) -> re.Pattern[str]:
    # Longer alternatives first, so that "rest api" wins over "rest" at the same position.
    ordered = sorted(alternatives, key=len, reverse=True)
    return re.compile(rf"(?<![{_WORD}])(?:{'|'.join(ordered)})(?![a-zа-я_+#])(?!(?<=[a-zа-я0-9])\d)")


def literal(spelling: str) -> tuple[str, str]:
    """Regex for a fixed spelling with optional inner separators, and its longest chunk as anchor."""
    s = dictionary_v1.clean(spelling)
    lead = re.match(r"[\s\-_.]*", s)
    head = lead.group(0) if lead else ""
    chunks = [c for c in re.split(r"[\s\-_.]+", s[len(head) :]) if c]
    return re.escape(head) + _SEP.join(re.escape(c) for c in chunks), max(chunks, key=len, default="")


def regex_anchor(pattern: str) -> str:
    """The literal prefix of a regex: text before its first metacharacter, minus a quantified letter."""
    out: list[str] = []
    for i, ch in enumerate(pattern):
        if ch in _META:
            if ch in "?*{" and out:
                out.pop()
            break
        if i + 1 < len(pattern) and pattern[i + 1] in "?*{":
            break
        out.append(ch)
    return "".join(out)


def build(d: dictionary_v1.Dictionary, search_gold_only: bool = False) -> Matcher:
    entries = []
    for skill in d.skills.values():
        if skill.name in d.gold_only and not search_gold_only:
            continue
        spellings = [*([] if skill.name in d.names_not_searched else [skill.name]), *skill.aliases]
        alternatives, anchors = [], []
        for spelling in dict.fromkeys(spellings):
            regex, anchor = literal(spelling)
            alternatives.append(regex)
            anchors.append(anchor)
        for pattern in skill.patterns:
            alternatives.append(pattern)
            anchors.append(regex_anchor(pattern))
        if alternatives:
            entries.append(Entry(skill.name, _compile(alternatives), () if "" in anchors else tuple(anchors)))
    return Matcher(tuple(entries), d.english_words, complete=True)


@cache
def load() -> Matcher:
    return build(dictionary_v1.load())


V0_NOT_SKILL_CATEGORIES = frozenset({"language_human", "soft"})
V0_NOT_SKILLS = frozenset(
    {"Requirements", "API integration", "Manual testing", "Test automation", "Prompt engineering", "Fine-tuning"}
    | {"Statistics", "Algorithms", "Multithreading", "Networking", "Virtualization", "Cryptography", "Embedded"}
    | {"System design", "Microservices", "Design patterns", "Clean architecture", "Product analytics", "Big Data"}
    | {"Machine learning", "Deep learning", "Gradient boosting", "NLP", "Computer vision", "Recommender systems"}
    | {"Time series", "Speech recognition"}
)
V0_ENGLISH_WORDS = frozenset(
    {"go", "rest", "less", "solid", "chef", "node", "argo", "gin", "rabbit", "elastic", "arm", "spring"}
    | {"swift", "rust", "dart", "julia", "ci", "lan", "wan", "yarn", "hive", "triton", "astra"}
)


def v0_rows(path: Path = V0_SEED) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["category"] not in V0_NOT_SKILL_CATEGORIES and r["skill"] not in V0_NOT_SKILLS]


@cache
def load_v0(path: Path = V0_SEED) -> Matcher:
    entries = tuple(Entry(r["skill"], _compile(r["synonyms"].split("|"))) for r in v0_rows(path))
    return Matcher(entries, V0_ENGLISH_WORDS, complete=False)


def find_skills(text: str, normalized: str, english: bool = False, matcher: Matcher | None = None) -> list[str]:
    """Canonical skills in order of first appearance.

    `normalized` is `text` lowercased with ё as е, character for character, so spans line up.
    """
    m_ = matcher or load()
    hits: list[Hit] = []
    for entry in m_.entries:
        if entry.anchors and not any(a in normalized for a in entry.anchors):
            continue
        for m in entry.pattern.finditer(normalized):
            if english and m.group(0) in m_.english_words and text[m.start() : m.end()].islower():
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
    found = list(dict.fromkeys(h.skill for h in sorted(kept, key=lambda h: h.start)))
    return dictionary_v1.complete(found) if m_.complete else found
