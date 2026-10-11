"""Units of deduplication: the latest staged record of each (source, source_id) that may be IT.

trudvsem rows, API and snapshot alike, count where is_it_candidate holds, the same filter that
feeds the extraction corpus; Habr is an IT site and is taken whole. The API slice is not all IT:
on 2026-10-08 its non-IT rows were mass postings (loaders, bank clerks) whose clusters of hundreds
of identical texts would swamp the pair counts. Telegram posts are not dedup units here:
radar.extract.telegram_corpus groups them itself.

The API and the snapshots are the same portal and share vacancy ids. Records with one id in both
form one group before any text comparison: the group is represented by its latest record, and the
other members ride along. Candidate search then runs over groups, so these certain links neither
inflate the candidate counts nor end up in the labeling sample.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from radar.dedup.normalize import normalize_employer

# The embedding step imports this module inside the WSL training venv, which has no httpx.
if TYPE_CHECKING:
    from radar.stream.clickhouse import ClickHouse

SOURCES = ("trudvsem", "trudvsem_snapshot", "habr")

_SELECT = {
    "trudvsem": """
        SELECT source, source_id, toString(dt) AS dt, title, employer_name, employer_inn, region, address,
               CAST([] AS Array(String)) AS cities, employment = 'Дистанционная (удаленная) работа' AS remote,
               salary_text, salary_from, salary_to, text,
               toString(published_at) AS published_at, toString(modified_at) AS modified_at
        FROM staging.stg_trudvsem_api
        WHERE is_it_candidate
        ORDER BY source_id, dt DESC
        LIMIT 1 BY source_id
    """,
    "trudvsem_snapshot": """
        SELECT source, source_id, toString(dt) AS dt, title, employer_name, employer_inn, region, address,
               CAST([] AS Array(String)) AS cities, employment = 'Дистанционная (удаленная) работа' AS remote,
               salary_text, salary_from, salary_to, text,
               toString(published_at) AS published_at, toString(modified_at) AS modified_at
        FROM staging.stg_trudvsem_snapshot
        WHERE is_it_candidate
        ORDER BY source_id, dt DESC
        LIMIT 1 BY source_id
    """,
    "habr": """
        SELECT source, source_id, toString(dt) AS dt, title, employer_name,
               CAST(NULL AS Nullable(String)) AS employer_inn, CAST(NULL AS Nullable(String)) AS region,
               CAST(NULL AS Nullable(String)) AS address, cities, remote_work AS remote,
               salary_text, salary_from, salary_to, text,
               toString(published_at) AS published_at, CAST(NULL AS Nullable(String)) AS modified_at
        FROM staging.stg_habr
        ORDER BY source_id, dt DESC
        LIMIT 1 BY source_id
    """,
}

# Sources of one portal whose ids mean the same vacancy, latest first within a group.
ID_SHARED = ("trudvsem", "trudvsem_snapshot")


@dataclass(frozen=True)
class Unit:
    uid: str
    source: str
    source_id: str
    dt: str
    title: str
    employer_name: str | None
    employer_inn: str | None
    location: str | None
    address: str | None
    remote: bool | None
    salary_text: str | None
    text: str
    published_at: str | None
    group: str

    @property
    def employer_key(self) -> str | None:
        """INN where the portal gives it, else the normalized name."""
        if self.employer_inn:
            return f"inn:{self.employer_inn}"
        name = normalize_employer(self.employer_name)
        return f"name:{name}" if name else None

    @property
    def employer_name_key(self) -> str | None:
        """Normalized name only: the key that can meet across Habr (no INN) and trudvsem."""
        return normalize_employer(self.employer_name)

    def document(self) -> str:
        """Title, employer, place and text as one string, the input of both methods."""
        head = ". ".join(p for p in (self.title, self.employer_name, self.location) if p)
        return f"{head}.\n{self.text}" if self.text else head


def fetch(ch: "ClickHouse") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        body = ch.query(_SELECT[source] + " FORMAT JSONEachRow")
        rows.extend(json.loads(line) for line in body.splitlines() if line)
    return rows


def group_key(source: str, source_id: str) -> str:
    return f"trudvsem:{source_id}" if source in ID_SHARED else f"{source}:{source_id}"


def to_units(rows: list[dict[str, object]]) -> list[Unit]:
    units = []
    for r in rows:
        source, source_id = str(r["source"]), str(r["source_id"])
        cities = r.get("cities") or []
        location = r.get("region") or (", ".join(str(c) for c in cities) if isinstance(cities, list) else None)
        units.append(
            Unit(
                uid=f"{source}:{source_id}",
                source=source,
                source_id=source_id,
                dt=str(r["dt"]),
                title=str(r.get("title") or ""),
                employer_name=_str_or_none(r.get("employer_name")),
                employer_inn=_str_or_none(r.get("employer_inn")),
                location=_str_or_none(location),
                address=_str_or_none(r.get("address")),
                remote=None if r.get("remote") is None else bool(r["remote"]),
                salary_text=_str_or_none(r.get("salary_text")),
                text=str(r.get("text") or ""),
                published_at=_str_or_none(r.get("published_at")),
                group=group_key(source, source_id),
            )
        )
    return units


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def representatives(units: list[Unit]) -> list[Unit]:
    """One unit per group: the latest day, and the API over the snapshot on the same day."""
    rank = {s: i for i, s in enumerate(SOURCES)}
    best: dict[str, Unit] = {}
    for u in units:
        cur = best.get(u.group)
        if cur is None or (u.dt, -rank[u.source]) > (cur.dt, -rank[cur.source]):
            best[u.group] = u
    return sorted(best.values(), key=lambda u: u.uid)


_STRING_COLUMNS = (
    "uid",
    "source",
    "source_id",
    "dt",
    "title",
    "employer_name",
    "employer_inn",
    "location",
    "address",
    "salary_text",
    "text",
    "published_at",
    "group",
)
SCHEMA = pa.schema([pa.field(name, pa.string()) for name in _STRING_COLUMNS] + [pa.field("remote", pa.bool_())])


def write_units(units: list[Unit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [getattr(u, name) for u in units] for name in SCHEMA.names}
    pq.write_table(pa.table(columns, schema=SCHEMA), path)


def inputs_digest(units: list[Unit]) -> str:
    """Ties stored vectors to the exact units and documents, in order, they were computed from."""
    h = hashlib.sha256()
    for u in units:
        h.update(u.uid.encode("utf-8") + b"\0" + u.document().encode("utf-8") + b"\0")
    return h.hexdigest()


def read_units(path: Path) -> list[Unit]:
    return [Unit(**row) for row in pq.read_table(path).to_pylist()]
