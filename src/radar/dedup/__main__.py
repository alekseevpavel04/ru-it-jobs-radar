import argparse
import json
import logging
import random
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from radar.config import ROOT, data_dir, load_dotenv
from radar.dedup import minhash, neighbours, pairs, units
from radar.dedup.normalize import jaccard, normalize_title, shingles
from radar.results import write_result

log = logging.getLogger("radar.dedup")

TEXT_CHARS = 1500
EVAL_CONFIG = ROOT / "configs" / "eval.toml"


def work_dir() -> Path:
    return data_dir() / "dedup"


def cmd_units(_: argparse.Namespace) -> int:
    from radar.stream.clickhouse import ClickHouse

    with ClickHouse() as ch:
        rows = units.fetch(ch)
    found = units.to_units(rows)
    path = work_dir() / "units.parquet"
    units.write_units(found, path)
    by_source = Counter(u.source for u in found)
    log.info("%d units to %s: %s", len(found), path, dict(sorted(by_source.items())))
    return 0


def _share(n: int, d: int) -> dict[str, Any]:
    return {"value": round(n / d, 4) if d else 0.0, "n": n, "denominator": d}


def unit_view(u: units.Unit) -> dict[str, Any]:
    return {
        "source": u.source,
        "source_id": u.source_id,
        "title": u.title,
        "employer": u.employer_name,
        "city": u.location,
        "address": u.address,
        "remote": u.remote,
        "salary_text": u.salary_text,
        "published_at": u.published_at,
        "seen_on": u.dt,
        "has_text": bool(u.text),
        "text": u.text[:TEXT_CHARS],
    }


def cmd_candidates(args: argparse.Namespace) -> int:
    p = pairs.Params()
    seed = int(tomllib.loads(EVAL_CONFIG.read_text(encoding="utf-8"))["bootstrap"]["seed"])
    wd = work_dir()
    all_units = units.read_units(wd / "units.parquet")
    emb_meta = json.loads((wd / "embeddings.json").read_text(encoding="utf-8"))
    if emb_meta["inputs_sha"] != units.inputs_digest(all_units):
        raise SystemExit("embeddings.npy was computed for another units.parquet; rerun radar.dedup.embed")
    vectors_all = np.load(wd / "embeddings.npy")
    row_of = {u.uid: n for n, u in enumerate(all_units)}

    reps = units.representatives(all_units)
    vectors = np.ascontiguousarray(vectors_all[[row_of[u.uid] for u in reps]], dtype=np.float32)
    members: dict[str, list[str]] = {}
    for u in all_units:
        members.setdefault(u.group, []).append(u.uid)

    started = time.perf_counter()
    docs = [shingles(u.document(), p.shingle_k) if u.text else set() for u in reps]
    lsh = set(minhash.lsh_pairs(docs, p.lsh_threshold, p.num_perm, seed))
    minhash_seconds = time.perf_counter() - started

    started = time.perf_counter()
    idx, sim = neighbours.top_k(vectors, p.knn_k)
    knn = neighbours.knn_pairs(idx, sim, p.knn_min_cosine)
    knn_seconds = time.perf_counter() - started

    candidates = pairs.merge_candidates(
        lsh,
        knn,
        lambda i, j: jaccard(docs[i], docs[j]),
        lambda i, j: neighbours.cosine(vectors, i, j),
    )
    keys = pairs.Keys(
        source=[u.source for u in reps],
        employer=[u.employer_key for u in reps],
        employer_name=[u.employer_name_key for u in reps],
        title=[normalize_title(u.title) for u in reps],
    )
    rng = random.Random(seed)
    strata, take = pairs.build_strata(keys, candidates, p, rng)

    def scores(i: int, j: int) -> dict[str, Any]:
        return {
            "jaccard": round(jaccard(docs[i], docs[j]), 4),
            "cosine": round(neighbours.cosine(vectors, i, j), 4),
            "by_minhash": (i, j) in lsh,
            "by_embedding": (i, j) in knn,
        }

    drawn = [(s.name, d) for s in strata for d in s.pairs]
    split = pairs.split_tuning({s.name: [d.pair for d in s.pairs] for s in strata}, p.tuning_share, rng)
    rng.shuffle(drawn)
    labeling, meta_pairs = [], []
    for n, (stratum, d) in enumerate(drawn, start=1):
        i, j = d.pair
        a, b = (reps[i], reps[j]) if rng.random() < 0.5 else (reps[j], reps[i])
        pair_id = f"d{n:04d}"
        labeling.append({"pair_id": pair_id, "a": unit_view(a), "b": unit_view(b)})
        meta_pairs.append(
            {
                "pair_id": pair_id,
                "stratum": stratum,
                "split": split[(i, j)],
                "p_draw": d.p_draw,
                "a_uid": a.uid,
                "b_uid": b.uid,
                "a_group_members": members[a.group],
                "b_group_members": members[b.group],
                **scores(i, j),
            }
        )

    with (wd / "pairs_for_labeling.jsonl").open("w", encoding="utf-8") as f:
        for row in labeling:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "seed": seed,
        "seed_source": "configs/eval.toml [bootstrap] seed",
        "params": p.as_dict(),
        "strata": [
            {
                "name": s.name,
                "population": s.population,
                "anchors": s.anchors,
                "census": s.census,
                "draws": s.draws,
                "drawn": len(s.pairs),
            }
            for s in strata
        ],
        "pairs": meta_pairs,
    }
    (wd / "pairs_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_candidates(wd / "candidates.parquet", reps, candidates, p)

    by_source = Counter(u.source for u in all_units)
    with_text = Counter(u.source for u in all_units if u.text)
    linked = [g for g, m in members.items() if len(m) == 2]
    unit_of = {u.uid: u for u in all_units}
    same_title = sum(1 for g in linked if len({normalize_title(unit_of[uid].title) for uid in members[g]}) == 1)
    bands = Counter(pairs.band(c, p) or "below_border" for c in candidates)
    src = {i: u.source for i, u in enumerate(reps)}
    split_counts = Counter(m["split"] for m in meta_pairs)
    metrics: dict[str, dict[str, Any]] = {
        **{f"units.{s}": {"value": by_source[s]} for s in units.SOURCES},
        "units.total": {"value": len(all_units)},
        **{f"units_with_text.{s}": _share(with_text[s], by_source[s]) for s in units.SOURCES},
        "id_links.snapshot_api": {"value": len(linked)},
        "id_links.share_of_api_units": _share(len(linked), by_source["trudvsem"]),
        "id_links.same_title": _share(same_title, len(linked)),
        "groups.total": {"value": len(reps)},
        **{f"groups.{s}": {"value": sum(1 for u in reps if u.source == s)} for s in units.SOURCES},
        "pairs.minhash": {"value": len(lsh)},
        "pairs.embedding": {"value": len(knn)},
        "pairs.both": {"value": len(lsh & knn.keys())},
        "pairs.union": {"value": len(candidates)},
        "pairs.cross_source": _share(sum(1 for c in candidates if src[c.i] != src[c.j]), len(candidates)),
        **{f"pairs.band.{b}": {"value": bands[b]} for b in [*pairs.BAND_STRATA, "below_border"]},
        **{f"sample.population.{s.name}": {"value": s.population} for s in strata},
        **{f"sample.drawn.{s.name}": {"value": len(s.pairs)} for s in strata},
        "sample.total": {"value": len(meta_pairs)},
        "sample.tuning": {"value": split_counts["tuning"]},
        "sample.heldout": {"value": split_counts["heldout"]},
        "seconds.embedding": {"value": emb_meta["seconds"]},
        "seconds.minhash": {"value": round(minhash_seconds, 1)},
        "seconds.knn": {"value": round(knn_seconds, 1)},
        "embedding.peak_vram_mib": {"value": emb_meta["peak_vram_mib"]},
        "embedding.truncated_texts": _share(emb_meta["truncated_texts"], emb_meta["texts"]),
    }
    path = write_result(
        "dedup/candidates.json",
        metrics,
        source_script="src/radar/dedup/__main__.py",
        embedding={k: emb_meta[k] for k in ("model", "device", "dtype", "batch_size", "max_length", "prefix")},
        params=p.as_dict(),
        seed=seed,
        allocation_used=take,
        staging_days=staging_days(all_units),
        units_rule=units.__doc__,
        blocking=pairs.__doc__,
    )
    log.info("%d candidates, %d pairs sampled; %s", len(candidates), len(meta_pairs), path)
    return 0


def staging_days(all_units: list[units.Unit]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in units.SOURCES:
        days = sorted({u.dt for u in all_units if u.source == s})
        out[s] = [days[0], days[-1]] if days else []
    return out


def write_candidates(path: Path, reps: list[units.Unit], candidates: list[pairs.Candidate], p: pairs.Params) -> None:
    table = pa.table(
        {
            "a_uid": [reps[c.i].uid for c in candidates],
            "b_uid": [reps[c.j].uid for c in candidates],
            "jaccard": [c.jaccard for c in candidates],
            "cosine": [c.cosine for c in candidates],
            "by_minhash": [c.by_minhash for c in candidates],
            "by_embedding": [c.by_embedding for c in candidates],
            "band": [pairs.band(c, p) for c in candidates],
        }
    )
    pq.write_table(table, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m radar.dedup")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("units", help="units.parquet from staging").set_defaults(fn=cmd_units)
    sub.add_parser("candidates", help="candidate pairs, labeling sample, results").set_defaults(fn=cmd_candidates)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
