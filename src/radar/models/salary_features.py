"""Features of the salary model, defined on the extraction schema plus the source.

Every feature is computable from `VacancyFields` alone, so the same encoder serves three inputs:
the v0 training frame (rule fields of the marts, `salary_data`), the student's fields for the whole
corpus once the corpus run loads them, and `POST /salary {fields}`. Swapping v0 for the corpus run means building
the frame from the extracted fields with `feature_row`; the encoder and the models do not change.

Region follows the marts macro `region_group`: remote work first, then Moscow, Saint Petersburg,
any other stated place, else unknown.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

UNKNOWN = "unknown"

# Schema role (5.2) -> role bucket of the marts (macros/marts.sql, seed role_groups.csv).
ROLE_BUCKET = {
    "ml_engineer": "ml_ds",
    "data_scientist": "ml_ds",
    "mlops": "ml_ds",
    "data_engineer": "data_engineering",
    "data_analyst": "data_analytics",
    "business_analyst": "business_system_analysis",
    "system_analyst": "business_system_analysis",
    "backend": "backend",
    "frontend": "frontend",
    "fullstack": "fullstack",
    "mobile": "mobile",
    "software_engineer": "software_engineer",
    "developer_1c": "developer_1c",
    "embedded": "embedded",
    "devops": "devops",
    "qa": "qa",
    "sysadmin": "sysadmin_support",
    "support": "sysadmin_support",
    "security": "security",
    "designer": "design",
    "manager": "management",
    "other_it": "other_it",
}

ROLE_GROUP = {
    "ml_ds": "ml_ds_data",
    "data_engineering": "ml_ds_data",
    "data_analytics": "ml_ds_data",
    "software_engineer": "dev",
    "backend": "dev",
    "frontend": "dev",
    "fullstack": "dev",
    "mobile": "dev",
    "embedded": "dev",
    "developer_1c": "dev",
    "business_system_analysis": "analyst",
    "qa": "qa_devops_sys",
    "devops": "qa_devops_sys",
    "sysadmin_support": "qa_devops_sys",
    "security": "qa_devops_sys",
    "design": "other_it",
    "management": "other_it",
    "other_it": "other_it",
}

MOSCOW = "Москва"
SPB = "Санкт-Петербург"

CATEGORICAL = ("role_bucket", "role_group", "grade", "region_group", "work_format", "employment_type", "source")
NUMERIC = ("experience_min",)
# Columns permuted together when importance is measured: a block is one input of the schema.
BLOCKS = {
    "role": ("role_bucket", "role_group"),
    "grade": ("grade",),
    "region": ("region_group",),
    "work_format": ("work_format",),
    "experience": ("experience_min",),
    "employment_type": ("employment_type",),
    "source": ("source",),
    "skills": ("skills",),
}


class OutOfScope(ValueError):
    """The model covers IT vacancies only."""


def region_group(cities: Sequence[str], work_format: str | None) -> str:
    if work_format == "remote":
        return "remote"
    if MOSCOW in cities:
        return "moscow"
    if SPB in cities:
        return "spb"
    return "other" if cities else UNKNOWN


def feature_row(fields: Mapping[str, Any], source: str) -> dict[str, Any]:
    """One feature row from extraction fields (5.2). A missing field reads as unknown, never guessed."""
    role = fields.get("role")
    if role == "non_it":
        raise OutOfScope("the salary model covers IT vacancies only")
    bucket = ROLE_BUCKET.get(role, UNKNOWN) if role else UNKNOWN
    experience = fields.get("experience_years") or {}
    work_format = fields.get("work_format")
    return {
        "role_bucket": bucket,
        "role_group": ROLE_GROUP.get(bucket, UNKNOWN),
        "grade": fields.get("grade") or UNKNOWN,
        "region_group": region_group(list(fields.get("cities") or []), work_format),
        "work_format": work_format or UNKNOWN,
        "employment_type": fields.get("employment_type") or UNKNOWN,
        "source": source,
        "experience_min": experience.get("min"),
        "skills": sorted(set(fields.get("skills") or [])),
    }


@dataclass(frozen=True)
class Encoded:
    categorical: NDArray[np.int64]  # (n, len(CATEGORICAL)), -1 for a level unseen in training
    numeric: Array  # (n, len(NUMERIC)), NaN when unknown
    skills: Array  # (n, len(skill_vocab)) of 0 and 1


@dataclass(frozen=True)
class Encoder:
    levels: dict[str, tuple[str, ...]]
    skill_vocab: tuple[str, ...]

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]], min_skill_count: int) -> "Encoder":
        levels = {c: tuple(sorted({str(r[c]) for r in rows})) for c in CATEGORICAL}
        counts = Counter(s for r in rows for s in r["skills"])
        vocab = tuple(sorted(s for s, n in counts.items() if n >= min_skill_count))
        return cls(levels, vocab)

    def encode(self, rows: Sequence[Mapping[str, Any]]) -> Encoded:
        index = {c: {v: i for i, v in enumerate(self.levels[c])} for c in CATEGORICAL}
        skill_index = {s: i for i, s in enumerate(self.skill_vocab)}
        cat = np.full((len(rows), len(CATEGORICAL)), -1, dtype=np.int64)
        num = np.full((len(rows), len(NUMERIC)), np.nan)
        skills = np.zeros((len(rows), len(self.skill_vocab)))
        for i, r in enumerate(rows):
            for j, c in enumerate(CATEGORICAL):
                cat[i, j] = index[c].get(str(r[c]), -1)
            for j, c in enumerate(NUMERIC):
                if r[c] is not None and not (isinstance(r[c], float) and np.isnan(r[c])):
                    num[i, j] = float(r[c])
            for s in r["skills"]:
                k = skill_index.get(s)
                if k is not None:
                    skills[i, k] = 1.0
        return Encoded(cat, num, skills)

    def tree_matrix(self, e: Encoded) -> Array:
        """Category codes (NaN for unseen), numbers, skill indicators; the first columns are categorical."""
        cat = e.categorical.astype(np.float64)
        cat[e.categorical < 0] = np.nan
        return np.hstack([cat, e.numeric, e.skills])

    def linear_matrix(self, e: Encoded, fill: Array) -> Array:
        """One-hot categories (an unseen level is all zeros), numbers imputed with `fill` plus a
        missing flag each, skill indicators."""
        blocks = []
        for j, c in enumerate(CATEGORICAL):
            onehot = np.zeros((len(e.categorical), len(self.levels[c])))
            seen = e.categorical[:, j] >= 0
            onehot[np.flatnonzero(seen), e.categorical[seen, j]] = 1.0
            blocks.append(onehot)
        missing = np.isnan(e.numeric)
        blocks += [np.where(missing, fill, e.numeric), missing.astype(np.float64), e.skills]
        return np.hstack(blocks)

    def tree_columns(self) -> list[str]:
        return [*CATEGORICAL, *NUMERIC, *(f"skill:{s}" for s in self.skill_vocab)]

    def block_columns(self, block: str) -> list[int]:
        """Positions in tree_matrix of the columns of one importance block."""
        names = self.tree_columns()
        wanted = BLOCKS[block]
        return [i for i, n in enumerate(names) if n in wanted or ("skills" in wanted and n.startswith("skill:"))]
