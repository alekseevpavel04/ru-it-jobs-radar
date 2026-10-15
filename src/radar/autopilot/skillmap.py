"""Extracted skill spellings to dictionary names, as int_extracted_fields maps them in dbt.

A spelling is folded to its lookup key (radar.extract.skills.key, the macro skill_key) and looked up by
the key's hash in the seed dbt/seeds/skill_dictionary_v1.csv. A spelling the seed has not seen is
resolved by radar.extract.skills.normalize_list, which is what dbt/skill_seed.py would write for it on
the next seed run; dbt itself leaves such a spelling out until then. The skills of [text_check] in
configs/skills.toml (HTTP, HTTPS) stay only when the text the extractor read names them outside a link.
"""

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from radar.config import ROOT
from radar.extract import skills

SEED = ROOT / "dbt" / "seeds" / "skill_dictionary_v1.csv"


def key_hash(spelling: str) -> str:
    return hashlib.sha256(skills.key(spelling).encode("utf-8")).hexdigest()[:16]


class SkillMap:
    def __init__(self, seed: Path = SEED, dictionary: skills.Dictionary | None = None) -> None:
        self.dictionary = dictionary or skills.load()
        names: dict[str, set[str]] = defaultdict(set)
        with seed.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                names[row["key_hash"]].update([row["skill"]] if row["skill"] else [])
        self.by_hash = {h: tuple(sorted(n)) for h, n in names.items()}
        self.unseen = 0

    def names(self, spelling: str) -> tuple[str, ...]:
        if not skills.key(spelling):
            return ()
        found = self.by_hash.get(key_hash(spelling))
        if found is not None:
            return found
        self.unseen += 1
        d = self.dictionary
        return tuple(sorted(s for s in skills.normalize_list([spelling], d) if s in d.skills))

    def skills(self, written: Iterable[str], text: str) -> list[str]:
        """Sorted dictionary names of an answer's skills; `text` is what the extractor read."""
        mapped = {name for spelling in written for name in self.names(spelling)}
        d = self.dictionary
        return sorted(n for n in mapped if n not in d.outside_links or skills.named_outside_links(n, text, d))
