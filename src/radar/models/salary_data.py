"""Training frame of the salary model, v0 features, and a split without near-duplicate leakage.

Population: IT dedup entities (staging.int_vacancy_entities, role bucket not `not_it`) of trudvsem
(API and snapshots, one portal) and Habr Career with a stated RUB salary, one row per entity, fields
of its latest record. Part-time vacancies are left out (trudvsem schedule «Неполный рабочий день» or
employment «Частичная занятость», Habr part_time): the target is a full-time monthly amount.

Target: the midpoint of the stated range in RUB per month; a single bound stands for itself. Both
sources publish monthly amounts without a period or gross/net field, so amounts stay as stated, as
in mart_salaries: nothing is converted and gross or net is not known per vacancy. The source is a
feature, so a systematic difference between the sources is absorbed rather than corrected.

Groups: an entity, a text cluster of the dedup (one template posted in 70 regions), a group of the
extraction corpus split and an employer with one normalized title are joined into one connected
component, and a component lies in one split. The corpus test and dev groups are pinned to test and
dev, so no model of the project trains on a gold vacancy; components holding a `withheld` corpus
group (near-duplicates of gold records) are left out. The rest is split by a salted hash of the
component's smallest entity id.
"""

import csv
import hashlib
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from radar.dedup.normalize import normalize_title
from radar.models.salary_features import ROLE_GROUP, UNKNOWN

log = logging.getLogger("radar.models.salary_data")

TRAIN, CALIB, DEV, TEST, WITHHELD = "train", "calib", "dev", "test", "withheld"
PART_TIME_SCHEDULE = "Неполный рабочий день"
PART_TIME_EMPLOYMENT = ("Частичная занятость", "part_time")
INTERNSHIP_EMPLOYMENT = "Стажировка"
NUMBERS = ("salary_from", "salary_to", "experience_min_years")

_SALARIED = """
    SELECT entity_id FROM staging.int_vacancy_entities
    WHERE role_bucket != 'not_it' AND salary_currency_code = 'RUB'
        AND (salary_from IS NOT NULL OR salary_to IS NOT NULL)
"""

ENTITIES_SQL = f"""
WITH latest AS (
    SELECT entity_id, family, latest_source, latest_source_id, latest_dt, first_seen, employer_key,
        salary_from, salary_to, role_bucket, grade, work_format, region_group, experience_min_years,
        no_experience, skills
    FROM staging.int_vacancy_entities
    WHERE entity_id IN ({_SALARIED})
),
terms AS (
    SELECT 'trudvsem_snapshot' AS source, source_id, dt, employment, schedule
    FROM staging.stg_trudvsem_snapshot
    WHERE (source_id, dt) IN (SELECT latest_source_id, latest_dt FROM latest WHERE latest_source = 'trudvsem_snapshot')
    LIMIT 1 BY source_id, dt
    UNION ALL
    SELECT 'trudvsem' AS source, source_id, dt, employment, schedule
    FROM staging.stg_trudvsem_api
    WHERE (source_id, dt) IN (SELECT latest_source_id, latest_dt FROM latest WHERE latest_source = 'trudvsem')
    LIMIT 1 BY source_id, dt
    UNION ALL
    SELECT 'habr' AS source, source_id, dt, coalesce(employment, lower(card_employment_type)) AS employment,
        CAST(NULL AS Nullable(String)) AS schedule
    FROM staging.stg_habr
    WHERE (source_id, dt) IN (SELECT latest_source_id, latest_dt FROM latest WHERE latest_source = 'habr')
    LIMIT 1 BY source_id, dt
),
titles AS (
    SELECT source, source_id, title FROM staging.int_vacancy_records WHERE entity_id IN ({_SALARIED})
)
SELECT l.entity_id AS entity_id, l.family AS family, l.first_seen AS first_seen,
    l.employer_key AS employer_key, ti.title AS title,
    l.salary_from AS salary_from, l.salary_to AS salary_to,
    l.role_bucket AS role_bucket, l.grade AS grade, l.work_format AS work_format,
    l.region_group AS region_group, l.experience_min_years AS experience_min_years,
    l.no_experience AS no_experience, arrayStringConcat(l.skills, '|') AS skills,
    t.employment AS employment, t.schedule AS schedule
FROM latest AS l
LEFT JOIN terms AS t ON t.source = l.latest_source AND t.source_id = l.latest_source_id AND t.dt = l.latest_dt
LEFT JOIN titles AS ti ON ti.source = l.latest_source AND ti.source_id = l.latest_source_id
SETTINGS join_use_nulls = 1
FORMAT TabSeparatedWithNames
"""

RECORDS_SQL = f"""
SELECT r.entity_id AS entity_id, concat(r.source, ':', r.source_id) AS record_id, v.text_cluster_id AS text_cluster_id
FROM staging.int_vacancy_records AS r
LEFT JOIN (
    SELECT source, source_id, any(text_cluster_id) AS text_cluster_id
    FROM staging.vacancy_entities
    GROUP BY source, source_id
) AS v ON r.source = v.source AND r.source_id = v.source_id
WHERE r.entity_id IN ({_SALARIED})
SETTINGS join_use_nulls = 1
FORMAT TabSeparatedWithNames
"""


def _tsv(text: str) -> pd.DataFrame:
    return pd.read_csv(
        StringIO(text), sep="\t", quoting=csv.QUOTE_NONE, na_values=["\\N"], keep_default_na=False, dtype=str
    )


def load_clickhouse() -> tuple[pd.DataFrame, pd.DataFrame]:
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        entities = _tsv(ch.query(ENTITIES_SQL))
        records = _tsv(ch.query(RECORDS_SQL))
    for column in entities.columns:
        if column not in NUMBERS:
            entities[column] = [v if isinstance(v, str) else None for v in entities[column]]
    for column in NUMBERS:
        entities[column] = pd.to_numeric(entities[column])
    entities["no_experience"] = entities["no_experience"].map({"1": True, "0": False})
    entities["first_seen"] = pd.to_datetime(entities["first_seen"], errors="coerce").dt.date
    return entities, records


def load_corpus_groups(path: Path) -> dict[str, tuple[str, str]]:
    """Record id (`source:source_id`) -> (corpus group, its split) of the extraction corpus split."""
    table = pq.read_table(path, columns=["group_id", "split", "member_ids"]).to_pydict()
    return {
        m: (g, s)
        for g, s, members in zip(table["group_id"], table["split"], table["member_ids"], strict=True)
        for m in members
    }


def records_of(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def midpoint(salary_from: float | None, salary_to: float | None) -> float:
    bounds = [b for b in (salary_from, salary_to) if b is not None and np.isfinite(b) and b > 0]
    return float(np.mean(bounds)) if bounds else np.nan


def is_part_time(employment: str | None, schedule: str | None) -> bool:
    return employment in PART_TIME_EMPLOYMENT or bool(schedule and schedule.startswith(PART_TIME_SCHEDULE))


def v0_feature_row(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Feature row (salary_features.feature_row keys) from the rule fields of the marts."""
    experience = rec["experience_min_years"]
    if experience is None or (isinstance(experience, float) and np.isnan(experience)):
        experience = 0.0 if rec["no_experience"] is True else None
    skills = rec["skills"]
    return {
        "role_bucket": rec["role_bucket"],
        "role_group": ROLE_GROUP.get(rec["role_bucket"], UNKNOWN),
        "grade": rec["grade"] if isinstance(rec["grade"], str) else UNKNOWN,
        "region_group": rec["region_group"],
        "work_format": rec["work_format"],
        "employment_type": "internship" if rec["employment"] == INTERNSHIP_EMPLOYMENT else UNKNOWN,
        "source": rec["family"],
        "experience_min": experience,
        "skills": sorted(skills.split("|")) if isinstance(skills, str) and skills else [],
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def components(links: Iterable[tuple[str, str]], entities: Iterable[str]) -> dict[str, str]:
    """Entity id -> id of its component (the smallest entity id in it)."""
    uf = UnionFind()
    ids = sorted(set(entities))
    for e in ids:
        uf.find("e:" + e)
    for a, b in links:
        uf.union(a, b)
    members: dict[str, list[str]] = {}
    for e in ids:
        members.setdefault(uf.find("e:" + e), []).append(e)
    return {e: min(group) for group in members.values() for e in group}


def unit(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class SplitShares:
    salt: str
    dev: float
    test: float
    calib: float


def assign(component: str, pinned: set[str], shares: SplitShares) -> str:
    """Split of one component: pinned corpus splits first (test, dev, withheld), then the hash."""
    for split in (TEST, DEV, WITHHELD):
        if split in pinned:
            return split
    u = unit(component, shares.salt)
    if u < shares.test:
        return TEST
    if u < shares.test + shares.dev:
        return DEV
    return CALIB if unit(component, shares.salt + ":calib") < shares.calib else TRAIN


def group_links(
    entities: pd.DataFrame, records: pd.DataFrame, corpus: Mapping[str, tuple[str, str]]
) -> list[tuple[str, str]]:
    links = []
    for e, cluster, rid in zip(records["entity_id"], records["text_cluster_id"], records["record_id"], strict=True):
        if isinstance(cluster, str) and cluster:
            links.append(("e:" + e, "t:" + cluster))
        if rid in corpus:
            links.append(("e:" + e, "c:" + corpus[rid][0]))
    for e, employer, title in zip(entities["entity_id"], entities["employer_key"], entities["title"], strict=True):
        key = normalize_title(title) if isinstance(title, str) else None
        if isinstance(employer, str) and employer and key:
            links.append(("e:" + e, f"j:{employer}|{key}"))
    return links


def build_frame(
    entities: pd.DataFrame,
    records: pd.DataFrame,
    corpus: Mapping[str, tuple[str, str]],
    shares: SplitShares,
    min_monthly: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rows of the model with features, target, group and split; counts of every filter step."""
    counts = {"entities_it_rub_salary": len(entities)}
    comp = components(group_links(entities, records, corpus), entities["entity_id"])
    pinned: dict[str, set[str]] = {}
    for e, rid in zip(records["entity_id"], records["record_id"], strict=True):
        if rid in corpus:
            pinned.setdefault(comp[e], set()).add(corpus[rid][1])
    split_of = {c: assign(c, pinned.get(c, set()), shares) for c in set(comp.values())}

    rows = []
    for rec in records_of(entities):
        mid = midpoint(rec["salary_from"], rec["salary_to"])
        group = comp[rec["entity_id"]]
        rows.append(
            {
                "entity_id": rec["entity_id"],
                "group_id": group,
                "split": split_of[group],
                "first_seen": rec["first_seen"],
                "salary_from": rec["salary_from"],
                "salary_to": rec["salary_to"],
                "salary_mid": mid,
                "part_time": is_part_time(rec["employment"], rec["schedule"]),
                **v0_feature_row(rec),
            }
        )
    frame = pd.DataFrame(rows)
    counts["part_time"] = int(frame["part_time"].sum())
    frame = frame[~frame["part_time"]]
    # A bound stored as NaN rather than NULL passes the SQL filter and carries no amount.
    no_amount = frame["salary_mid"].isna()
    counts["no_amount"] = int(no_amount.sum())
    frame = frame[~no_amount]
    low = frame["salary_mid"] < min_monthly
    counts["below_min_monthly"] = int(low.sum())
    frame = frame[~low]
    counts["withheld_near_gold"] = int((frame["split"] == WITHHELD).sum())
    frame = frame[frame["split"] != WITHHELD].drop(columns=["part_time"]).reset_index(drop=True)
    counts["rows"] = len(frame)
    counts["groups"] = int(frame["group_id"].nunique())
    for split in (TRAIN, CALIB, DEV, TEST):
        part = frame[frame["split"] == split]
        counts[f"rows_{split}"] = len(part)
        counts[f"groups_{split}"] = int(part["group_id"].nunique())
    counts["largest_group"] = int(frame["group_id"].value_counts().max()) if len(frame) else 0
    return frame, counts


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
