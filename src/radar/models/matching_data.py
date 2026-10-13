"""Pool, synthetic resumes and graded relevance for resume-to-vacancy matching.

Pool: every IT dedup entity of the portals (trudvsem API and snapshots, Habr Career, SuperJob) whose
latest record carries valid fields of the extraction model with a role other than `non_it`; one
document per entity, the extractor input of that record (title, facts header, text) as its text.

Queries: no resume data exists in the project, so resumes are written from the extracted fields of a
seed vacancy by templates: a role title from a fixed list for its schema role, the grade, the
experience, part of the seed's skills plus a few skills common for the role, the city and remote
readiness. Nothing of the vacancy text is copied, so a retriever cannot find the seed by its wording.
Seeds need `min_skills` extracted skills and are split into dev and test by a salted hash of the
entity id.

Relevance of a pool vacancy for a resume is a function of the extracted fields of both, written
before any retriever was run: 0 when the role bucket differs, the grades are more than `grade_gap`
steps apart or the place does not fit; 2 when at least `coverage` of the vacancy's extracted skills
are on the resume; 1 otherwise. The retrievers read text only, never these fields, so none of them is
fitted to the labels; a field-based reranker would be, and is not evaluated here.
"""

import hashlib
import json
import logging
import math
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from radar.config import ROOT
from radar.models.salary_features import ROLE_BUCKET, UNKNOWN

log = logging.getLogger("radar.models.matching_data")

CONFIG = ROOT / "configs" / "matching.toml"
DEV, TEST = "dev", "test"
PORTALS = ("trudvsem", "trudvsem_snapshot", "habr", "superjob")
GRADE_RANK = {"intern": 0, "junior": 1, "middle": 2, "senior": 3, "lead": 4}

ROLE_TITLES = {
    "ml_engineer": ("ML-инженер", "Machine Learning Engineer", "Инженер машинного обучения"),
    "data_scientist": ("Data Scientist", "Специалист по анализу данных"),
    "mlops": ("MLOps-инженер", "Инженер MLOps"),
    "data_engineer": ("Data Engineer", "Инженер данных"),
    "data_analyst": ("Аналитик данных", "Data Analyst"),
    "business_analyst": ("Бизнес-аналитик", "Business Analyst"),
    "system_analyst": ("Системный аналитик", "System Analyst"),
    "backend": ("Backend-разработчик", "Серверный разработчик"),
    "frontend": ("Frontend-разработчик", "Фронтенд-разработчик"),
    "fullstack": ("Fullstack-разработчик", "Фулстек-разработчик"),
    "mobile": ("Мобильный разработчик", "Mobile-разработчик"),
    "software_engineer": ("Программист", "Разработчик программного обеспечения"),
    "developer_1c": ("Программист 1С", "Разработчик 1С"),
    "embedded": ("Разработчик встраиваемых систем", "Программист микроконтроллеров"),
    "devops": ("DevOps-инженер", "Инженер DevOps"),
    "qa": ("Тестировщик", "QA-инженер"),
    "sysadmin": ("Системный администратор", "Сисадмин"),
    "support": ("Специалист технической поддержки", "Инженер техподдержки"),
    "security": ("Специалист по информационной безопасности", "Инженер ИБ"),
    "designer": ("UI/UX-дизайнер", "Дизайнер интерфейсов"),
    "manager": ("Руководитель IT-проектов", "Project manager"),
    "other_it": ("IT-специалист", "Специалист в IT"),
}
GRADE_WORDS = {
    "intern": ("Стажёр", "Intern"),
    "junior": ("Junior", "Младший"),
    "middle": ("Middle", "Middle"),
    "senior": ("Senior", "Старший"),
    "lead": ("Lead", "Ведущий"),
}
TEMPLATES = (
    "{title}\n{experience}Ключевые навыки: {skills}.\n{place}",
    "Желаемая должность: {title}.\n{place}{experience}Стек: {skills}.",
    "Резюме. {title}.\nНавыки: {skills}.\n{experience}{place}",
)


@dataclass(frozen=True)
class QueryConfig:
    salt: str
    dev: int
    test: int
    min_skills: int
    keep_min: float
    keep_max: float
    extra_max: int
    extra_pool: int
    grade_gap: int
    coverage: float


def load_query_config(path: Path = CONFIG) -> QueryConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    q, r = raw["queries"], raw["relevance"]
    return QueryConfig(
        salt=q["salt"],
        dev=int(q["dev"]),
        test=int(q["test"]),
        min_skills=int(q["min_skills"]),
        keep_min=float(q["keep_min"]),
        keep_max=float(q["keep_max"]),
        extra_max=int(q["extra_max"]),
        extra_pool=int(q["extra_pool"]),
        grade_gap=int(r["grade_gap"]),
        coverage=float(r["coverage"]),
    )


# --- pool ---------------------------------------------------------------------------------------

POOL_SQL = """
SELECT e.entity_id AS entity_id, e.family AS family, e.latest_source AS source,
    e.latest_source_id AS source_id, e.is_open_now AS is_open_now, toString(f.dt) AS day,
    f.content_hash AS content_hash, f.role AS role, f.grade AS grade, f.skills AS skills, f.cities AS cities,
    f.work_format AS work_format, f.experience_min AS experience_min
FROM staging.int_vacancy_entities AS e
INNER JOIN (
    SELECT * FROM staging.stg_vacancy_fields
    WHERE model_id = {model:String} AND valid AND source IN ('trudvsem', 'trudvsem_snapshot', 'habr', 'superjob')
    ORDER BY dt DESC, created_at DESC
    LIMIT 1 BY source, source_id
) AS f ON f.source = e.latest_source AND f.source_id = e.latest_source_id
WHERE f.role != 'non_it'
ORDER BY e.entity_id
FORMAT JSONEachRow
"""

RAW_SQL = """
SELECT source_id, toString(dt) AS day, content_hash, title, text, payload, url
FROM raw.vacancies
WHERE source = {{source:String}} AND (source_id, dt, content_hash) IN ({keys})
ORDER BY fetched_at DESC
LIMIT 1 BY source_id, dt, content_hash
FORMAT JSONEachRow
"""
RAW_BATCH = 1000


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _json_lines(body: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in body.splitlines() if line]


def load_pool(extraction_model: str) -> pd.DataFrame:
    """Pool documents with their fields and the extractor input as text (needs ClickHouse)."""
    from radar.extract.inputs import input_for_record
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        rows = _json_lines(ch.query(POOL_SQL, model=extraction_model))
        texts: dict[tuple[str, str], tuple[str, str | None, str | None]] = {}
        for source in PORTALS:
            keys = [(r["source_id"], r["day"], r["content_hash"]) for r in rows if r["source"] == source]
            for start in range(0, len(keys), RAW_BATCH):
                literal = ", ".join(
                    f"({_quote(i)}, toDate({_quote(d)}), {_quote(h)})" for i, d, h in keys[start : start + RAW_BATCH]
                )
                for raw in _json_lines(ch.query(RAW_SQL.format(keys=literal), source=source)):
                    text = input_for_record({"source": source, **raw})
                    texts[(source, raw["source_id"])] = (text, raw.get("title"), raw.get("url"))
    missing = [r for r in rows if (r["source"], r["source_id"]) not in texts]
    if missing:
        raise RuntimeError(f"{len(missing)} pool records have no raw row, e.g. {missing[0]['source_id']}")
    frame = pd.DataFrame(rows)
    frame["text"] = [texts[(s, i)][0] for s, i in zip(frame["source"], frame["source_id"], strict=True)]
    frame["title"] = [texts[(s, i)][1] for s, i in zip(frame["source"], frame["source_id"], strict=True)]
    frame["url"] = [texts[(s, i)][2] for s, i in zip(frame["source"], frame["source_id"], strict=True)]
    frame["is_open_now"] = frame["is_open_now"].fillna(0).astype(int).astype(bool)
    return frame.drop(columns=["day", "content_hash"])


# --- queries ------------------------------------------------------------------------------------


def unit(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def role_skill_counts(pool: pd.DataFrame) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for role, skills in zip(pool["role"], pool["skills"], strict=True):
        out.setdefault(role, Counter()).update(set(skills))
    return out


def _years(n: float) -> str:
    k = int(n)
    if k == 0:
        return "без опыта"
    word = "год" if k % 10 == 1 and k % 100 != 11 else "года" if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14 else "лет"
    return f"{k} {word}"


def render_resume(profile: Mapping[str, Any], rng: np.random.Generator) -> str:
    """Resume text for a profile (role, grade, skills, experience, city, remote)."""
    title = str(rng.choice(ROLE_TITLES.get(profile["role"], ROLE_TITLES["other_it"])))
    if profile["grade"] in GRADE_WORDS:
        title = f"{rng.choice(GRADE_WORDS[profile['grade']])} {title}"
    experience = profile["experience_years"]
    experience_line = "" if experience is None else f"Опыт работы: {_years(experience)}.\n"
    place = []
    if profile["city"]:
        place.append(f"Город: {profile['city']}.")
    if profile["remote"]:
        place.append("Рассматриваю удалённую работу.")
    place_line = " ".join(place) + "\n" if place else ""
    template = str(rng.choice(TEMPLATES))
    return template.format(
        title=title, experience=experience_line, skills=", ".join(profile["skills"]), place=place_line
    ).strip()


def make_profile(
    seed: Mapping[str, Any], role_skills: Mapping[str, Counter[str]], cfg: QueryConfig, rng: np.random.Generator
) -> dict[str, Any]:
    skills = sorted(set(seed["skills"]))
    keep = max(1, math.ceil(len(skills) * rng.uniform(cfg.keep_min, cfg.keep_max)))
    kept = [str(s) for s in rng.choice(skills, size=keep, replace=False)]
    common = [s for s, _ in role_skills.get(seed["role"], Counter()).most_common(cfg.extra_pool) if s not in skills]
    n_extra = min(int(rng.integers(0, cfg.extra_max + 1)), len(common))
    extra = [str(s) for s in rng.choice(common, size=n_extra, replace=False)] if n_extra else []
    resume_skills = kept + extra
    rng.shuffle(resume_skills)
    cities = list(seed["cities"])
    experience = seed["experience_min"]
    return {
        "role": seed["role"],
        "grade": seed["grade"],
        "skills": resume_skills,
        "experience_years": None if experience is None or not np.isfinite(experience) else float(experience),
        "city": cities[0] if cities else None,
        "remote": seed["work_format"] == "remote",
    }


def make_queries(pool: pd.DataFrame, cfg: QueryConfig) -> pd.DataFrame:
    """Dev and test resumes, one per seed vacancy."""
    role_skills = role_skill_counts(pool)
    eligible = pool[[len(set(s)) >= cfg.min_skills for s in pool["skills"]]]
    split = np.where([unit(e, cfg.salt) < 0.5 for e in eligible["entity_id"]], DEV, TEST)
    order = np.argsort([unit(e, cfg.salt + ":order") for e in eligible["entity_id"]], kind="stable")
    rows: list[dict[str, Any]] = []
    for name, size in ((DEV, cfg.dev), (TEST, cfg.test)):
        picked = [i for i in order if split[i] == name][:size]
        for i in picked:
            seed = eligible.iloc[i]
            rng = np.random.default_rng(int(unit(seed["entity_id"], cfg.salt + ":resume") * 2**32))
            profile = make_profile(seed, role_skills, cfg, rng)
            rows.append(
                {
                    "query_id": f"{name}-{len(rows):04d}",
                    "split": name,
                    "seed_entity_id": seed["entity_id"],
                    "resume": render_resume(profile, rng),
                    **{f"profile_{k}": v for k, v in profile.items()},
                }
            )
    return pd.DataFrame(rows)


# --- relevance ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolFields:
    bucket: NDArray[np.str_]
    grade: NDArray[np.int64]  # -1 unknown
    remote: NDArray[np.bool_]
    cities: list[frozenset[str]]
    skills: list[frozenset[str]]

    @classmethod
    def of(cls, pool: pd.DataFrame) -> "PoolFields":
        return cls(
            bucket=np.array([ROLE_BUCKET.get(r, UNKNOWN) for r in pool["role"]], dtype=str),
            grade=np.array([GRADE_RANK.get(g, -1) if isinstance(g, str) else -1 for g in pool["grade"]]),
            remote=(pool["work_format"] == "remote").to_numpy(dtype=bool),
            cities=[frozenset(c) for c in pool["cities"]],
            skills=[frozenset(s) for s in pool["skills"]],
        )


def relevance(profile: Mapping[str, Any], pool: PoolFields, cfg: QueryConfig) -> NDArray[np.int8]:
    """Grade 0, 1 or 2 of every pool document for one resume profile (module docstring)."""
    bucket = ROLE_BUCKET.get(profile["role"], UNKNOWN)
    rank = GRADE_RANK.get(profile["grade"], -1) if isinstance(profile["grade"], str) else -1
    ok = pool.bucket == bucket
    if rank >= 0:
        ok &= (pool.grade < 0) | (np.abs(pool.grade - rank) <= cfg.grade_gap)
    city = profile["city"]
    if city:
        place = np.array([not c or city in c for c in pool.cities])
        ok &= pool.remote | place
    out = ok.astype(np.int8)
    resume = frozenset(profile["skills"])
    for i in np.flatnonzero(ok):
        skills = pool.skills[i]
        if skills and len(skills & resume) / len(skills) >= cfg.coverage:
            out[i] = 2
    return np.asarray(out, dtype=np.int8)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in r.items()} for r in frame.to_dict("records")]


def profile_of(query: Mapping[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("profile_"): v for k, v in query.items() if k.startswith("profile_")}


def qrels(queries: pd.DataFrame, pool: pd.DataFrame, cfg: QueryConfig) -> list[NDArray[np.int8]]:
    fields = PoolFields.of(pool)
    return [relevance(profile_of(q), fields, cfg) for q in records(queries)]


def label_counts(labels: Iterable[NDArray[np.int8]]) -> dict[str, list[int]]:
    rows = list(labels)
    return {"grade1": [int((r == 1).sum()) for r in rows], "grade2": [int((r == 2).sum()) for r in rows]}


def texts_digest(texts: Sequence[str]) -> str:
    return hashlib.sha256("\x1e".join(texts).encode()).hexdigest()
