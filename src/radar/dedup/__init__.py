"""Deduplication of vacancies across trudvsem API, trudvsem snapshots and Habr.

Steps, each writing under data/dedup/:
  python -m radar.dedup units        units.parquet from staging (ClickHouse)
  radar.dedup.embed inside WSL       embeddings.npy and embeddings.json (GPU)
  python -m radar.dedup candidates   candidates.parquet, pairs_for_labeling.jsonl,
                                     results/dedup/candidates.json
"""
