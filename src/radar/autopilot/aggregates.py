"""Weekly Telegram counts kept in git, the collector cursors and the text hashes of the open weeks.

    aggregates/telegram/posts/<year>.csv    week,channel,role,grade,posts
    aggregates/telegram/skills/<year>.csv   week,channel,role,grade,skill,posts
    aggregates/telegram/state/<channel>.json   the collector's daily cursor (post ids and hashes)
    aggregates/telegram/week_hashes.json    text hashes of the current and the previous week

Only IT units are counted (staging.int_extracted_weekly_units with is_it: a single-vacancy post after the
weekly cross-post dedup whose extracted role is not non_it), a week is the Moscow Monday of the post and
a year file holds the weeks of that year, so a daily run rewrites only the current year. A grade the
extractor left empty is an empty cell. No post text and no per-post row goes here (DECISIONS 10): counts,
post ids of the cursors and 64-bit text hashes only. Rows are sorted so that a day's diff is the rows of
its week.
"""

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from radar.config import ROOT

AGGREGATES = ROOT / "aggregates" / "telegram"
MSK = timezone(timedelta(hours=3))
POST_COLUMNS = ("week", "channel", "role", "grade", "posts")
SKILL_COLUMNS = ("week", "channel", "role", "grade", "skill", "posts")
HASH_NAME = "sha256(normalized text)[:16]"

PostKey = tuple[str, str, str, str]
SkillKey = tuple[str, str, str, str, str]

# The normalization of the dbt macro telegram_text_hash: links, @names and hashtags are channel
# furniture (each channel signs its copy), punctuation, emoji and spacing do not count either.
_FURNITURE = re.compile(r"https?://[^ \n]+|t[.]me/[^ \n]+|@[a-z0-9_]+|#[^ \n#]+")
_NOT_WORD = re.compile(r"[^a-zа-я0-9+]+")


def normalized_text(text: str | None) -> str:
    lowered = (text or "").lower().replace("ё", "е")
    return _NOT_WORD.sub(" ", _FURNITURE.sub(" ", lowered)).strip()


def text_hash(text: str | None) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()[:16]


def moscow_week(at: datetime) -> date:
    day = at.astimezone(MSK).date()
    return day - timedelta(days=day.weekday())


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


@dataclass
class Counts:
    """IT units by (week, channel, role, grade) and their skill mentions by the same key and skill."""

    posts: Counter[PostKey] = field(default_factory=Counter)
    skills: Counter[SkillKey] = field(default_factory=Counter)

    def add(self, week: date, channel: str, role: str | None, grade: str | None, skills: Iterable[str]) -> None:
        key = (week.isoformat(), channel, role or "", grade or "")
        self.posts[key] += 1
        for skill in set(skills):
            self.skills[(*key, skill)] += 1

    def merge(self, other: "Counts") -> None:
        self.posts.update(other.posts)
        self.skills.update(other.skills)

    def years(self) -> set[int]:
        return {int(k[0][:4]) for k in self.posts} | {int(k[0][:4]) for k in self.skills}


def _render(columns: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return out.getvalue()


def save(counts: Counts, root: Path = AGGREGATES) -> list[Path]:
    """Write a file per year of the counts; years without counts keep their files."""
    written = []
    for year in sorted(counts.years()):
        posts = sorted((*k, str(n)) for k, n in counts.posts.items() if k[0].startswith(f"{year}-") and n)
        skills = sorted((*k, str(n)) for k, n in counts.skills.items() if k[0].startswith(f"{year}-") and n)
        for sub, columns, rows in (("posts", POST_COLUMNS, posts), ("skills", SKILL_COLUMNS, skills)):
            path = root / sub / f"{year}.csv"
            _write_atomic(path, _render(columns, rows))
            written.append(path)
    return written


def _read(path: Path, columns: tuple[str, ...]) -> Iterable[list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if tuple(next(reader)) != columns:
            raise ValueError(f"{path}: expected the columns {','.join(columns)}")
        yield from reader


def load(root: Path = AGGREGATES, years: Iterable[int] | None = None) -> Counts:
    wanted = None if years is None else {str(y) for y in years}

    def files(sub: str) -> list[Path]:
        return [p for p in sorted((root / sub).glob("*.csv")) if wanted is None or p.stem in wanted]

    counts = Counts()
    for path in files("posts"):
        for week, channel, role, grade, n in _read(path, POST_COLUMNS):
            counts.posts[(week, channel, role, grade)] += int(n)
    for path in files("skills"):
        for week, channel, role, grade, skill, n in _read(path, SKILL_COLUMNS):
            counts.skills[(week, channel, role, grade, skill)] += int(n)
    return counts


def trend_rows(counts: Counts) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows of radar.site.skill_trends SIZE_SQL, SKILL_SQL and DIRECTION_SQL, by year of the week."""
    sizes: Counter[tuple[int, str, str, str]] = Counter()
    directions: Counter[tuple[int, str, str]] = Counter()
    mentions: Counter[tuple[int, str, str, str, str]] = Counter()
    for (week, channel, role, grade), n in counts.posts.items():
        sizes[(int(week[:4]), channel, role, grade)] += n
        directions[(int(week[:4]), channel, role)] += n
    for (week, channel, role, grade, skill), n in counts.skills.items():
        mentions[(int(week[:4]), channel, role, grade, skill)] += n

    def null(value: str) -> str | None:
        return value or None

    return (
        [{"year": y, "channel": c, "role": null(r), "grade": null(g), "posts": n} for (y, c, r, g), n in sizes.items()],
        [
            {"year": y, "channel": c, "role": null(r), "grade": null(g), "skill": s, "posts": n}
            for (y, c, r, g, s), n in mentions.items()
        ],
        [{"year": y, "channel": c, "role": null(r), "posts": n} for (y, c, r), n in directions.items()],
    )


# --- cursors and week hashes ---------------------------------------------------------------------


def read_cursors(root: Path = AGGREGATES) -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "state").glob("*.json"))}


def write_cursor(channel: str, state: Mapping[str, Any], root: Path = AGGREGATES) -> None:
    _write_atomic(root / "state" / f"{channel}.json", json.dumps(state, ensure_ascii=False, indent=1) + "\n")


def read_week_hashes(root: Path = AGGREGATES) -> dict[str, set[str]]:
    path = root / "week_hashes.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc["hash"] != HASH_NAME:
        raise ValueError(f"{path}: hashes are {doc['hash']}, this code makes {HASH_NAME}")
    return {week: set(hashes) for week, hashes in doc["weeks"].items()}


def kept_weeks(today: date) -> tuple[str, str]:
    """The previous and the current Moscow week of a day: a post a run sees was published in one of them."""
    monday = today - timedelta(days=today.weekday())
    return ((monday - timedelta(days=7)).isoformat(), monday.isoformat())


def write_week_hashes(hashes: Mapping[str, set[str]], today: date, root: Path = AGGREGATES) -> None:
    weeks = kept_weeks(today)
    doc = {"hash": HASH_NAME, "weeks": {w: sorted(hashes.get(w, set())) for w in weeks}}
    _write_atomic(root / "week_hashes.json", json.dumps(doc, indent=0) + "\n")
