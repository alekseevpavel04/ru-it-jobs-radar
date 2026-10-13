from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from radar.models.matching import (
    BM25,
    MatchIndex,
    match,
    per_query,
    ranks,
    reranked,
    rrf,
    served_choice,
    standardize,
)
from radar.models.matching_data import (
    PoolFields,
    QueryConfig,
    make_queries,
    relevance,
    render_resume,
)

QCFG = QueryConfig(
    salt="t",
    dev=5,
    test=5,
    min_skills=2,
    keep_min=0.6,
    keep_max=1.0,
    extra_max=2,
    extra_pool=5,
    grade_gap=1,
    coverage=0.5,
)


def pool_frame() -> pd.DataFrame:
    rows = [
        ("e0", "backend", "middle", ["Python", "Django", "PostgreSQL"], ["Москва"], "office"),
        ("e1", "backend", "senior", ["Go", "Kafka"], ["Москва"], "office"),
        ("e2", "backend", "lead", ["Python", "Django"], ["Москва"], "office"),
        ("e3", "qa", "middle", ["Python", "Selenium"], ["Москва"], "office"),
        ("e4", "backend", None, ["Python"], ["Казань"], "remote"),
        ("e5", "backend", "junior", ["Python", "Django", "Redis"], ["Казань"], "office"),
        ("e6", "backend", "middle", [], [], None),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": e,
                "role": r,
                "grade": g,
                "skills": s,
                "cities": c,
                "work_format": w,
                "experience_min": 2.0,
                "text": f"{r} {' '.join(s)} {' '.join(c)}",
                "title": f"Вакансия {e}",
                "url": None,
                "source": "habr",
                "source_id": e,
                "is_open_now": e != "e5",
            }
            for e, r, g, s, c, w in rows
        ]
    )


def test_relevance_rules() -> None:
    fields = PoolFields.of(pool_frame())
    profile = {"role": "backend", "grade": "middle", "skills": ["Python", "Django"], "city": "Москва", "remote": False}
    rel = relevance(profile, fields, QCFG)
    # e0 covers 2 of 3 skills; e1 shares none; e2 is two grades up; e3 another role;
    # e4 remote with unknown grade; e5 another city in office; e6 has no skills to cover.
    assert rel.tolist() == [2, 1, 0, 0, 2, 0, 1]


def test_resume_is_written_from_fields_only() -> None:
    pool = pool_frame()
    queries = make_queries(pool, QCFG)
    assert set(queries["split"]) <= {"dev", "test"}
    assert len(queries) == len(make_queries(pool, QCFG))
    assert queries.equals(make_queries(pool, QCFG))
    for q in queries.to_dict("records"):
        seed = pool.set_index("entity_id").loc[q["seed_entity_id"]]
        assert seed["title"] not in q["resume"]
        kept = set(q["profile_skills"]) & set(seed["skills"])
        assert len(kept) >= 0.6 * len(seed["skills"])


def test_render_resume_mentions_profile() -> None:
    profile = {
        "role": "developer_1c",
        "grade": None,
        "skills": ["1С:Предприятие"],
        "experience_years": 0.0,
        "city": None,
        "remote": True,
    }
    text = render_resume(profile, np.random.default_rng(0))
    assert "1С:Предприятие" in text
    assert "без опыта" in text
    assert "удалённую" in text


def test_bm25_prefers_rare_matching_terms() -> None:
    bm25 = BM25(1.2, 0.75).fit(["python django", "python flask", "java spring", "c# .net"])
    s = bm25.scores(["django python", "c#"])
    assert s[0].argmax() == 0
    assert s[1].argmax() == 3


def test_ranks_and_rrf() -> None:
    a = np.array([[0.1, 0.9, 0.5]])
    assert ranks(a).tolist() == [[2, 0, 1]]
    fused = rrf(a, np.array([[0.9, 0.1, 0.5]]), 60)
    assert fused[0, 2] < fused[0, 0] == pytest.approx(fused[0, 1])
    z = standardize(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert z[0].mean() == pytest.approx(0) and (z[1] == 0).all()


def test_metrics_on_known_rankings() -> None:
    labels = [np.array([2, 0, 1, 0], dtype=np.int8)]
    perfect = per_query(np.array([[4.0, 1.0, 3.0, 0.0]]), labels, ndcg_k=3, recall_k=1)
    assert perfect["ndcg"][0] == pytest.approx(1.0)
    assert perfect["mrr"][0] == 1.0 and perfect["recall"][0] == 1.0
    worst = per_query(np.array([[0.0, 4.0, 1.0, 3.0]]), labels, ndcg_k=3, recall_k=1)
    assert worst["mrr"][0] == 0.25 and worst["recall"][0] == 0.0
    assert 0 < worst["ndcg"][0] < 1


def test_rerank_moves_candidates_to_the_top() -> None:
    first = np.array([[0.9, 0.8, 0.1, 0.0]])
    out = reranked(first, {0: ([1, 2], [5.0, 7.0])}, depth=2)
    assert np.argsort(-out[0]).tolist()[:2] == [2, 1]


def index_for(system: str) -> MatchIndex:
    pool = pool_frame()
    vectors = np.eye(len(pool), dtype=np.float32)
    index = MatchIndex(
        system=system,
        model="intfloat/multilingual-e5-small" if system != "bm25" else None,
        fusion_weight=0.5 if system.startswith("fusion") else None,
        bm25=BM25(1.2, 0.75).fit(list(pool["text"])),
        vectors=None if system == "bm25" else vectors,
        max_length=32,
        items=[{"vacancy_entity_id": e} for e in pool["entity_id"]],
        open_now=pool["is_open_now"].to_numpy(dtype=bool),
        version="test",
        built_at="now",
    )
    # A fake encoder pointing at e5, so the dense part is known without a model.
    index._encode = lambda texts: (np.eye(len(pool), dtype=np.float32)[[5]], 0)
    return index


def test_match_bm25_and_open_filter() -> None:
    out = match("Python Django Redis Казань", k=3, index=index_for("bm25"))
    ids = [i["vacancy_entity_id"] for i in out["items"]]
    assert "e5" not in ids  # closed
    assert len(ids) == 3 and out["system"] == "bm25"
    assert [i["rank"] for i in out["items"]] == [1, 2, 3]
    everything = match("Python Django Redis Казань", k=3, open_only=False, index=index_for("bm25"))
    assert everything["items"][0]["vacancy_entity_id"] == "e5"
    with pytest.raises(ValueError):
        match("  ", index=index_for("bm25"))


def test_match_dense_uses_the_query_encoder() -> None:
    out = match("anything", k=1, open_only=False, index=index_for("dense_e5-small"))
    assert out["items"][0]["vacancy_entity_id"] == "e5"
    fused = match("Go Kafka", k=7, open_only=False, index=index_for("fusion_e5-small@50"))
    assert {i["vacancy_entity_id"] for i in fused["items"][:2]} == {"e1", "e5"}


def test_served_choice_respects_the_serving_list() -> None:
    per = {
        "bm25": {"ndcg": np.array([0.3])},
        "dense_bge-m3": {"ndcg": np.array([0.6])},
        "fusion_e5-small@70": {"ndcg": np.array([0.5])},
    }

    cfg = SimpleNamespace(served_models=["intfloat/multilingual-e5-small"])
    assert served_choice({"per_query": per}, cfg) == "fusion_e5-small@70"  # type: ignore[arg-type]
