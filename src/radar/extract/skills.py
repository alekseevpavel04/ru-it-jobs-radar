"""Skill normalization by the dictionary v1 (`configs/skills.toml`).

`normalize(raw)` maps one spelling to its canonical name, or to None when the annotation guide does
not count it as a skill. Steps, first hit wins:

1. the whole spelling against names and aliases, ignoring case, ё, spaces, dots, hyphens and
   underscores ("Node JS", "nodejs" and "node.js" are one key);
2. the non-skills of the guide (exact names, prefixes such as "ГОСТ" or "ISO ", regexes);
3. the inflection regexes of the dictionary ("питона", "постгресом");
4. rules, each followed by step 1 again: a version or level suffix dropped ("Python 3.10",
   "C++11/14", "Java 8+", "PHPStan Level 9"), a vendor prefix dropped ("Apache Kafka", "Microsoft
   SQL Server", case 61), a generic word dropped ("СУБД PostgreSQL", "фреймворк Django"), a plural
   "s" dropped or added, Cyrillic look-alike letters read as Latin ("Рython" typed with a Cyrillic Р).

A spelling none of this resolves is unknown: `normalize` returns it as written (whitespace collapsed),
so an extractor never loses a skill the dictionary has not met yet. Unknown spellings are never
matched to a neighbour at run time; `radar.extract.skill_review` proposes neighbours offline and
only reviewed entries enter the dictionary.
"""

import re
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from radar.config import ROOT

DICTIONARY = ROOT / "configs" / "skills.toml"

Status = Literal["known", "rule", "non_skill", "unknown"]

_NON_SKILL = "\x00non-skill"
_SEPARATORS = re.compile(r"[\s\-_.]+")
_EDGE_PUNCT = " \t\n\"'«»“”„`,;:()[]{}*•·"
# A version word needs a separator before it, so that "Server 2019" is not read as "Ser" + "ver 2019".
_VERSION = re.compile(
    r"^(?P<base>.+?)(?:[\s\-]+(?:ver|version|версия|версии|level|уровень|lvl)\.?\s*|[\s\-]*v?\.?)"
    r"\d+(?:[.,]\d+)*(?:\.x)?(?:\s*/\s*\d+(?:[.,]\d+)*)*\s*\+?$"
)
_VENDORS = (
    "apache", "microsoft", "ms", "adobe", "google", "yandex", "яндекс", "amazon", "aws", "vmware",
    "oracle", "ibm", "jetbrains", "atlassian", "hashicorp", "hugging face", "red hat", "nvidia",
)  # fmt: skip
_GENERIC_PREFIX = (
    "язык программирования", "язык", "субд", "фреймворк", "фреймворка", "библиотека", "библиотеки",
    "платформа", "платформы", "система", "инструмент", "сервис", "framework", "library",
)  # fmt: skip
_GENERIC_SUFFIX = (
    "framework", "library", "фреймворк", "библиотека", "язык", "субд", "platform", "платформа",
    "stack", "стек", "api", "sdk",
)  # fmt: skip
_LOOKALIKE = str.maketrans("асеорхкмтвнуАСЕОРХКМТВНУ", "aceopxkmtbhyACEOPXKMTBHY")
# "+" splits only between spaces, so "C++/C#" splits into C++ and C#.
_SPLIT = re.compile(r"\s+\+\s+|\s*(?:/|\\|,|;|&|\(|\))\s*|\s+(?:and|и)\s+")


def key(spelling: str) -> str:
    """Lookup key: lowercase, ё as е, separators removed."""
    text = " ".join(spelling.split()).strip(_EDGE_PUNCT).casefold().replace("ё", "е")
    return _SEPARATORS.sub("", text)


def clean(spelling: str) -> str:
    """Lowercase, ё as е, whitespace collapsed: the form inflection regexes and rules work on."""
    return " ".join(spelling.split()).strip(_EDGE_PUNCT).casefold().replace("ё", "е")


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...]
    patterns: tuple[str, ...]
    parent: str | None


@dataclass(frozen=True)
class Dictionary:
    skills: dict[str, Skill]
    by_key: dict[str, str]
    patterns: tuple[tuple[re.Pattern[str], str], ...]
    non_skill_keys: frozenset[str]
    non_skill_prefixes: tuple[str, ...]
    non_skill_pattern: re.Pattern[str] | None
    names_not_searched: frozenset[str]
    english_words: frozenset[str]
    gold_only: dict[str, str]
    rejected: frozenset[str]

    def lookup(self, spelling: str) -> str | None:
        """Canonical name, `_NON_SKILL`, or None for an unknown spelling."""
        k = key(spelling)
        if not k:
            return None
        if k in self.by_key:
            return self.by_key[k]
        c = clean(spelling)
        if k in self.non_skill_keys or c.startswith(self.non_skill_prefixes):
            return _NON_SKILL
        if self.non_skill_pattern is not None and self.non_skill_pattern.fullmatch(c):
            return _NON_SKILL
        return None

    def without(self, names: Iterable[str]) -> "Dictionary":
        """The same dictionary with some entries removed: for the coverage of a dictionary that
        did not see a split's labels."""
        drop = set(names)
        skills = {n: s for n, s in self.skills.items() if n not in drop}
        return Dictionary(
            skills=skills,
            by_key={k: n for k, n in self.by_key.items() if n not in drop},
            patterns=tuple((p, n) for p, n in self.patterns if n not in drop),
            non_skill_keys=self.non_skill_keys,
            non_skill_prefixes=self.non_skill_prefixes,
            non_skill_pattern=self.non_skill_pattern,
            names_not_searched=self.names_not_searched - drop,
            english_words=self.english_words,
            gold_only={n: v for n, v in self.gold_only.items() if n not in drop},
            rejected=self.rejected,
        )


def _add_key(by_key: dict[str, str], spelling: str, name: str) -> None:
    k = key(spelling)
    if not k:
        raise ValueError(f"{name}: empty spelling")
    if by_key.get(k, name) != name:
        raise ValueError(f"spelling {spelling!r} of {name!r} is already a spelling of {by_key[k]!r}")
    by_key[k] = name


@cache
def load(path: Path = DICTIONARY) -> Dictionary:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    patterns_raw: dict[str, list[str]] = raw.get("patterns", {})
    parents: dict[str, str] = raw.get("parents", {})
    skills: dict[str, Skill] = {}
    by_key: dict[str, str] = {}
    for category, entries in raw["skills"].items():
        for name, aliases in entries.items():
            if name in skills:
                raise ValueError(f"{name!r} is listed twice")
            skills[name] = Skill(name, category, tuple(aliases), tuple(patterns_raw.get(name, ())), parents.get(name))
            for spelling in (name, *aliases):
                _add_key(by_key, spelling, name)

    for section, names in (("patterns", patterns_raw), ("parents", parents)):
        missing = sorted(set(names) - set(skills))
        if missing:
            raise ValueError(f"[{section}] names unknown skills: {missing}")
    unknown_parents = sorted({p for p in parents.values() if p not in skills})
    if unknown_parents:
        raise ValueError(f"[parents] points to unknown skills: {unknown_parents}")

    ns = raw.get("non_skills", {})
    non_skill_keys = frozenset(key(n) for n in ns.get("names", []))
    clash = sorted(n for n in ns.get("names", []) if key(n) in by_key)
    if clash:
        raise ValueError(f"non-skills that are also skill spellings: {clash}")
    ns_patterns = ns.get("patterns", [])
    search = raw.get("search", {})
    not_searched = frozenset(search.get("names_not_searched", []))
    if not not_searched <= skills.keys():
        raise ValueError(f"[search] names unknown skills: {sorted(not_searched - skills.keys())}")
    gold_only: dict[str, str] = raw.get("provenance", {}).get("gold_only", {})
    if not gold_only.keys() <= skills.keys():
        raise ValueError(f"[provenance] names unknown skills: {sorted(gold_only.keys() - skills.keys())}")

    return Dictionary(
        skills=skills,
        by_key=by_key,
        patterns=tuple((re.compile(p), n) for n, ps in patterns_raw.items() for p in ps),
        non_skill_keys=non_skill_keys,
        non_skill_prefixes=tuple(p.casefold().replace("ё", "е") for p in ns.get("prefixes", [])),
        non_skill_pattern=re.compile("|".join(f"(?:{p})" for p in ns_patterns)) if ns_patterns else None,
        names_not_searched=not_searched,
        english_words=frozenset(search.get("english_words", [])),
        gold_only=gold_only,
        rejected=frozenset(key(r) for r in raw.get("review", {}).get("rejected", [])),
    )


@dataclass(frozen=True)
class Resolution:
    raw: str
    canonical: str | None
    status: Status
    rule: str | None = None


def _drop_version(c: str) -> str | None:
    m = _VERSION.match(c)
    return m.group("base").strip() if m else None


def _drop_vendor(c: str) -> str | None:
    for vendor in _VENDORS:
        if c.startswith(vendor + " ") and len(c) > len(vendor) + 1:
            return c[len(vendor) + 1 :]
    return None


def _drop_generic(c: str) -> str | None:
    for word in _GENERIC_PREFIX:
        if c.startswith(word + " ") and len(c) > len(word) + 1:
            return c[len(word) + 1 :]
    for word in _GENERIC_SUFFIX:
        for sep in (" ", "-"):
            if c.endswith(sep + word) and len(c) > len(word) + 1:
                return c[: -len(word) - 1]
    return None


def _plural(c: str) -> str | None:
    if not re.fullmatch(r"[a-z][a-z0-9 \-]{3,}", c):
        return None
    return c[:-1] if c.endswith("s") else c + "s"


def _lookalike(c: str) -> str | None:
    latin = c.translate(_LOOKALIKE)
    if latin == c or re.search(r"[а-я]", latin):
        return None
    return latin


RULES: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("version", _drop_version),
    ("vendor_prefix", _drop_vendor),
    ("generic_word", _drop_generic),
    ("plural", _plural),
    ("lookalike", _lookalike),
)


def _from_lookup(raw: str, hit: str, status: Status, rule: str | None = None) -> Resolution:
    if hit == _NON_SKILL:
        return Resolution(raw, None, "non_skill", rule)
    return Resolution(raw, hit, status, rule)


def resolve(raw: str, dictionary: Dictionary | None = None) -> Resolution:
    d = dictionary or load()
    hit = d.lookup(raw)
    if hit is not None:
        return _from_lookup(raw, hit, "known")
    c = clean(raw)
    for pattern, name in d.patterns:
        if pattern.fullmatch(c):
            return Resolution(raw, name, "known", "pattern")
    # Rules compose: "Microsoft SQL Server 2019" needs the version and then the vendor dropped.
    current = c
    applied: list[str] = []
    for name, rule in RULES:
        changed = rule(current)
        if changed is None or changed == current:
            continue
        hit = d.lookup(changed)
        if hit is not None:
            return _from_lookup(raw, hit, "rule", "+".join([*applied, name]))
        if name != "plural":
            current, applied = changed, [*applied, name]
    return Resolution(raw, None, "unknown")


def normalize(raw: str, dictionary: Dictionary | None = None) -> str | None:
    """Canonical name; None for a non-skill; an unknown spelling as written."""
    r = resolve(raw, dictionary)
    if r.status == "non_skill":
        return None
    return r.canonical or " ".join(raw.split())


def complete(skills: list[str]) -> list[str]:
    """List rules of the guide that need no text: a 1С configuration implies 1С (cases 21, 62)."""
    if "1С" not in skills and any(s.startswith("1С:") for s in skills):
        return ["1С", *skills]
    return skills


def normalize_list(raws: Iterable[str], dictionary: Dictionary | None = None) -> list[str]:
    """An extractor's skill list in canonical names: non-skills dropped, glued spellings split
    ("C#/.NET", "sklearn/catboost", "SQL (PostgreSQL, MySQL)") when every part is known,
    duplicates removed in order of first appearance."""
    d = dictionary or load()
    out: list[str] = []
    for raw in raws:
        r = resolve(raw, d)
        if r.status == "unknown":
            parts = [p for p in _SPLIT.split(raw) if p.strip()]
            resolved = [resolve(p, d) for p in parts]
            if len(parts) > 1 and all(p.status != "unknown" for p in resolved):
                out.extend(p.canonical for p in resolved if p.canonical)
                continue
        name = normalize(raw, d)
        if name is not None:
            out.append(name)
    return complete(list(dict.fromkeys(out)))


def eval_key(raw: str) -> str:
    """Comparison key for `radar.eval` with `[skills] normalization = "dictionary"`: the canonical
    name of a known spelling, otherwise the spelling itself; case and whitespace ignored either way.
    A non-skill keeps its own key: scoring does not drop what a system output."""
    r = resolve(raw)
    name = r.canonical if r.canonical is not None else raw
    return " ".join(name.split()).casefold().replace("ё", "е")
