"""Deduplication of vacancies across trudvsem API, trudvsem snapshots and Habr.

Steps, each writing under data/dedup/, settings in configs/dedup.toml:
  python -m radar.dedup units        units.parquet from staging (ClickHouse)
  radar.dedup.embed inside WSL       embeddings.npy and embeddings.json (GPU)
  python -m radar.dedup candidates   candidates.parquet, results/dedup/candidates.json; the labeling
                                     sample (pairs_for_labeling.jsonl, pairs_meta.json) if none exists
  python -m radar.dedup link         entities.parquet and the table staging.vacancy_entities
  python -m radar.dedup evaluate     rule selection on the tuning half, results/dedup/eval.json
"""
