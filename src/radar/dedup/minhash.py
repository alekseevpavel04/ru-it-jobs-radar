"""MinHash LSH over word shingles: which units share enough text to be compared at all.

LSH only proposes pairs; every proposed pair then gets its exact Jaccard from the shingle sets, so
the score in the candidates is not the MinHash estimate.
"""

from collections.abc import Iterator, Sequence

from datasketch import MinHash, MinHashLSH  # type: ignore[import-untyped]

from radar.dedup.normalize import jaccard


def signatures(docs: Sequence[set[str]], num_perm: int, seed: int) -> list[MinHash | None]:
    out: list[MinHash | None] = []
    for doc in docs:
        if not doc:
            out.append(None)
            continue
        m = MinHash(num_perm=num_perm, seed=seed)
        m.update_batch([s.encode("utf-8") for s in doc])
        out.append(m)
    return out


def lsh_pairs(docs: Sequence[set[str]], threshold: float, num_perm: int, seed: int) -> Iterator[tuple[int, int]]:
    """Index pairs (i < j) that land in a common LSH band; empty documents take no part."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs = signatures(docs, num_perm, seed)
    with lsh.insertion_session() as session:
        for i, sig in enumerate(sigs):
            if sig is not None:
                session.insert(i, sig, check_duplication=False)
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for j in lsh.query(sig):
            if j > i:
                yield i, j


def exact_jaccard(docs: Sequence[set[str]], pairs: Sequence[tuple[int, int]]) -> list[float]:
    return [jaccard(docs[i], docs[j]) for i, j in pairs]
