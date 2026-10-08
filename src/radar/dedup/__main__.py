import argparse
import json
import logging
import random
import sys
import time
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from radar.config import ROOT, data_dir, load_dotenv
from radar.dedup import entities, evaluate, minhash, neighbours, pairs, settings, units
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
    p = settings.load().params
    seed = int(tomllib.loads(EVAL_CONFIG.read_text(encoding="utf-8"))["bootstrap"]["seed"])
    wd = work_dir()
    all_units = units.read_units(wd / "units.parquet")
    emb_meta = json.loads((wd / "embeddings.json").read_text(encoding="utf-8"))
    if emb_meta["inputs_sha"] != units.inputs_digest(all_units):
        raise SystemExit("embeddings.npy was computed for another units.parquet; rerun radar.dedup.embed")
    if emb_meta["model"] != p.embedding_model:
        raise SystemExit(f"embeddings.npy is from {emb_meta['model']}, configs/dedup.toml asks {p.embedding_model}")
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
    meta_path = wd / "pairs_meta.json"
    if meta_path.exists() and not args.resample:
        # The sample is labeled: redrawing it from changed data would detach the labels.
        log.info("labeled sample kept as is: %s", meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        if args.resample and (wd / "labels").exists():
            raise SystemExit("data/dedup/labels exists: a new sample would not match those labels")
        meta = draw_sample(wd, reps, members, keys, candidates, docs, vectors, lsh, knn, p, seed)
    write_candidates(wd / "candidates.parquet", reps, candidates, p)

    by_source = Counter(u.source for u in all_units)
    with_text = Counter(u.source for u in all_units if u.text)
    linked = [g for g, m in members.items() if len(m) == 2]
    unit_of = {u.uid: u for u in all_units}
    same_title = sum(1 for g in linked if len({normalize_title(unit_of[uid].title) for uid in members[g]}) == 1)
    bands = Counter(pairs.band(c, p) or "below_border" for c in candidates)
    src = {i: u.source for i, u in enumerate(reps)}
    split_counts = Counter(m["split"] for m in meta["pairs"])
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
        **{f"sample.population.{s['name']}": {"value": s["population"]} for s in meta["strata"]},
        **{f"sample.drawn.{s['name']}": {"value": s["drawn"]} for s in meta["strata"]},
        "sample.total": {"value": len(meta["pairs"])},
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
        sample_drawn_on=meta.get("drawn_on", "stage A, 2026-10-08"),
        staging_days=staging_days(all_units),
        units_rule=units.__doc__,
        blocking=pairs.__doc__,
    )
    log.info("%d candidates; %s", len(candidates), path)
    return 0


def draw_sample(
    wd: Path,
    reps: list[units.Unit],
    members: dict[str, list[str]],
    keys: pairs.Keys,
    candidates: list[pairs.Candidate],
    docs: list[set[str]],
    vectors: NDArray[np.float32],
    lsh: set[tuple[int, int]],
    knn: dict[tuple[int, int], float],
    p: settings.Params,
    seed: int,
) -> dict[str, Any]:
    """Draw the stratified labeling sample and write it with its metadata."""
    rng = random.Random(seed)
    strata, _ = pairs.build_strata(keys, candidates, p, rng)

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
    meta: dict[str, Any] = {
        "drawn_on": datetime.now(UTC).date().isoformat(),
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
    return meta


def staging_days(all_units: list[units.Unit]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in units.SOURCES:
        days = sorted({u.dt for u in all_units if u.source == s})
        out[s] = [days[0], days[-1]] if days else []
    return out


def write_candidates(path: Path, reps: list[units.Unit], candidates: list[pairs.Candidate], p: settings.Params) -> None:
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


def read_edges(path: Path) -> list[entities.Edge]:
    cols = pq.read_table(path, columns=["a_uid", "b_uid", "jaccard", "cosine"]).to_pydict()
    return [
        entities.Edge(a, b, j, c)
        for a, b, j, c in zip(cols["a_uid"], cols["b_uid"], cols["jaccard"], cols["cosine"], strict=True)
    ]


def cmd_link(args: argparse.Namespace) -> int:
    s = settings.load()
    wd = work_dir()
    all_units = units.read_units(wd / "units.parquet")
    reps = units.representatives(all_units)
    by_uid = {u.uid: u for u in reps}
    candidates = read_edges(wd / "candidates.parquet")
    nodes = [u.uid for u in reps]

    edges = entities.rule_edges(by_uid, candidates, s.rule)
    unconstrained, _ = entities.components(nodes, edges)
    traits = {u.uid: entities.traits_of(u) for u in reps}
    start = s.rule.jaccard if s.rule.jaccard is not None else 0.0
    entity_groups, refused, recut = entities.split_large(nodes, edges, start, s.max_component, traits)
    text_edges = [e for e in candidates if e.jaccard >= s.text_cluster_jaccard]
    text_groups, _ = entities.components(nodes, text_edges)

    rows = entities.mapping(all_units, entity_groups, text_groups)
    entities.write_mapping(rows, str(wd / "entities.parquet"))
    info = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": s.rule.as_dict(),
        "rule_edges": len(edges),
        "text_edges": len(text_edges),
        "largest_component_unconstrained": max(len(g) for g in unconstrained),
        "links_refused_by_traits": refused,
        "largest_component_constrained": max(len(g) for g in entity_groups),
        "max_component": s.max_component,
        "components_recut": recut,
        "sizes_in": "groups (an API record and a snapshot record with one id are one group)",
        "units_digest": units.inputs_digest(all_units),
    }
    (wd / "entities.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    log.info("%d rows, %s", len(rows), json.dumps(info))
    if not args.no_clickhouse:
        from radar.stream.clickhouse import ClickHouse

        with ClickHouse() as ch:
            log.info("%s: %d rows", entities.TABLE, entities.load_clickhouse(ch, rows))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _interval_metric(point: float, samples: NDArray[np.float64], confidence: float) -> dict[str, Any]:
    low, high, n = evaluate.interval(samples, confidence)
    return {"value": round(point, 4), "low": round(low, 4), "high": round(high, 4), "resamples_defined": n}


def cmd_evaluate(_: argparse.Namespace) -> int:
    s = settings.load()
    boot = tomllib.loads(EVAL_CONFIG.read_text(encoding="utf-8"))["bootstrap"]
    wd = work_dir()
    views = {r["pair_id"]: r for r in read_jsonl(wd / "pairs_for_labeling.jsonl")}
    sample = json.loads((wd / "pairs_meta.json").read_text(encoding="utf-8"))
    meta = {m["pair_id"]: m for m in sample["pairs"]}
    populations = {st["name"]: st["population"] for st in sample["strata"]}
    final = {r["pair_id"]: r["label"] for r in read_jsonl(wd / "labels" / "final.jsonl")}
    passes = [
        {r["pair_id"]: r["label"] for f in sorted((wd / "labels" / name).glob("*.jsonl")) for r in read_jsonl(f)}
        for name in ("pass_a", "pass_b")
    ]

    labeled = evaluate.labeled_pairs(views, meta, final)
    tuning = [x for x in labeled if x.split == "tuning"]
    heldout = [x for x in labeled if x.split == "heldout"]
    best, tuned = evaluate.select(evaluate.grid(s.grid_jaccard, s.grid_cosine), tuning, populations)
    if best != s.rule:
        raise SystemExit(f"configs/dedup.toml [rule] is {s.rule}, the tuning selection picks {best}")

    mapping = pq.read_table(wd / "entities.parquet").to_pydict()
    entity_of = {
        f"{src}:{sid}": e
        for src, sid, e in zip(mapping["source"], mapping["source_id"], mapping["vacancy_entity_id"], strict=True)
    }
    same_entity: dict[str, bool] = {pid: entity_of[m["a_uid"]] == entity_of[m["b_uid"]] for pid, m in meta.items()}

    def by_rule(x: evaluate.Labeled) -> bool:
        return evaluate.decide(s.rule, x)

    def by_entity(x: evaluate.Labeled) -> bool:
        return same_entity[x.pair_id]

    metrics: dict[str, dict[str, Any]] = {}
    n_resamples, seed, confidence = boot["n_resamples"], boot["seed"], boot["confidence"]
    for name, predicted in (("rule", by_rule), ("entity", by_entity)):
        for estimator, pops in (("hajek", populations), ("hansen_hurwitz", None)):
            w = evaluate.Weighted(heldout, pops)
            point = w.estimate(predicted)
            precision, recall = w.bootstrap(predicted, n_resamples, seed)
            metrics[f"heldout.{name}.{estimator}.precision"] = _interval_metric(point.precision, precision, confidence)
            metrics[f"heldout.{name}.{estimator}.recall"] = _interval_metric(point.recall, recall, confidence)
        for split_name, xs in (("heldout", heldout), ("tuning", tuning)):
            tp = sum(predicted(x) and x.same for x in xs)
            fp = sum(predicted(x) and not x.same for x in xs)
            fn = sum(not predicted(x) and x.same for x in xs)
            metrics[f"{split_name}.{name}.raw.tp"] = {"value": tp}
            metrics[f"{split_name}.{name}.raw.fp"] = {"value": fp}
            metrics[f"{split_name}.{name}.raw.fn"] = {"value": fn}
            metrics[f"{split_name}.{name}.raw.precision"] = _share(tp, tp + fp)
            metrics[f"{split_name}.{name}.raw.recall"] = _share(tp, tp + fn)

    hajek = evaluate.Weighted(heldout, populations).estimate(by_rule)
    missed = evaluate.missed_upper(heldout, populations)
    found = hajek.recall * hajek.positive_pairs
    metrics["heldout.rule.recall_worst_case"] = {
        "value": round(found / (hajek.positive_pairs + sum(missed.values())), 4),
        "strata_without_positives": sorted(missed),
    }
    metrics["tuning.rule.hajek.f1"] = {"value": round(tuned.f1, 4)}

    pairs_ids = sorted(final)
    agree = sum(passes[0][p] == passes[1][p] for p in pairs_ids)
    binary = [p for p in pairs_ids if passes[0][p] != "unsure" and passes[1][p] != "unsure"]
    labels_count = Counter(final.values())
    metrics |= {
        "labels.same": {"value": labels_count["same"]},
        "labels.different": {"value": labels_count["different"]},
        "labels.unsure_excluded": _share(labels_count["unsure"], len(final)),
        "labels.heldout_evaluated": {"value": len(heldout)},
        "labels.tuning_evaluated": {"value": len(tuning)},
        "agreement.pairs": _share(agree, len(pairs_ids)),
        "agreement.kappa_same_different": {
            "value": round(evaluate.cohen_kappa([passes[0][p] for p in binary], [passes[1][p] for p in binary]), 4),
            "pairs": len(binary),
        },
        "agreement.kappa_three_labels": {
            "value": round(
                evaluate.cohen_kappa([passes[0][p] for p in pairs_ids], [passes[1][p] for p in pairs_ids]), 4
            ),
            "pairs": len(pairs_ids),
        },
    }
    metrics |= entity_stats(wd, mapping)

    path = write_result(
        "dedup/eval.json",
        metrics,
        source_script="src/radar/dedup/__main__.py",
        rule=s.rule.as_dict(),
        selection="highest Hajek-weighted F1 on the tuning half over the grid of configs/dedup.toml [selection]",
        entities_build=json.loads((wd / "entities.json").read_text(encoding="utf-8")),
        bootstrap={"n_resamples": n_resamples, "seed": seed, "confidence": confidence},
        estimators=evaluate.__doc__,
        notes=[
            "unsure pairs are excluded from every estimate",
            "id links between trudvsem API and snapshots are taken as the same vacancy and are not in the sample",
            "rule.* scores the pairwise decision; entity.* scores membership in one connected component",
            "rule.* held-out scores are clean: the rule was fixed on the tuning half before the held-out half"
            " was scored",
            "entity.* held-out scores are optimistic: the component constraints (one INN, one settlement, one"
            " street; place kept for remote sole proprietors) were added after the held-out pairwise errors had"
            " been seen, and the sole-proprietor case removes held-out pair d0004 directly",
            "recall over all pairs is not estimable from this sample: four strata hold no positive pair, and"
            " recall_worst_case is the rule-of-three bound if they held 3/n of their pairs",
        ],
    )
    log.info("%s", path)
    return 0


def entity_stats(wd: Path, mapping: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    sources_of: dict[str, set[str]] = {}
    size: Counter[str] = Counter()
    for src, e in zip(mapping["source"], mapping["vacancy_entity_id"], strict=True):
        sources_of.setdefault(e, set()).add(src)
        size[e] += 1
    text_size = Counter(mapping["text_cluster_id"])
    out: dict[str, dict[str, Any]] = {
        "entities.records": {"value": len(mapping["source"])},
        "entities.total": {"value": len(size)},
        "entities.multi_record": _share(sum(1 for n in size.values() if n > 1), len(size)),
        "entities.multi_source": _share(sum(1 for v in sources_of.values() if len(v) > 1), len(size)),
        "entities.habr_and_trudvsem": _share(
            sum(1 for v in sources_of.values() if "habr" in v and len(v) > 1), len(size)
        ),
        "entities.largest": {"value": max(size.values())},
        "text_clusters.total": {"value": len(text_size)},
        "text_clusters.largest": {"value": max(text_size.values())},
        "text_clusters.multi_record": _share(sum(1 for n in text_size.values() if n > 1), len(text_size)),
    }
    for source in units.SOURCES:
        out[f"entities.with_{source}"] = {"value": sum(1 for v in sources_of.values() if source in v)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m radar.dedup")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("units", help="units.parquet from staging").set_defaults(fn=cmd_units)
    cand = sub.add_parser("candidates", help="candidate pairs; the labeling sample if there is none yet")
    cand.add_argument("--resample", action="store_true", help="draw a new sample (refused once labels exist)")
    cand.set_defaults(fn=cmd_candidates)
    link = sub.add_parser("link", help="entities.parquet and staging.vacancy_entities under configs/dedup.toml")
    link.add_argument("--no-clickhouse", action="store_true", help="write the parquet file only")
    link.set_defaults(fn=cmd_link)
    sub.add_parser("evaluate", help="rule selection on tuning, held-out scores, results/dedup/eval.json").set_defaults(
        fn=cmd_evaluate
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
