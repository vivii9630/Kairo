"""Pairwise-similarity helper for graph builders.

Given a list of (id, embedding) pairs, returns the top-k neighbors per
id above a cosine threshold. Kept deliberately O(n^2) for now — builders
run at ingest time, not on the query hot path, and n is bounded by the
corpus size in a single build call. When corpora exceed ~5k docs we swap
this for an approximate index (HNSW) in Phase 11e.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def top_k_similar_pairs(
    ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    k: int = 5,
    threshold: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """Return (source_id, target_id, cosine) triples for each id's top-k
    similar neighbors above ``threshold``.

    Embeddings are L2-normalized internally so the dot product equals
    cosine similarity. Self-pairs are excluded. Results are deduplicated:
    (a, b) appears at most once — we keep the (lower_id, higher_id)
    orientation so the caller can pick whichever edge direction they want.
    """
    n = len(ids)
    if n < 2 or embeddings.shape[0] != n:
        return []

    vecs = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)

    seen: set[Tuple[str, str]] = set()
    triples: List[Tuple[str, str, float]] = []
    actual_k = min(k, n - 1)

    for i in range(n):
        neighbor_idx = np.argpartition(-sims[i], actual_k - 1)[:actual_k]
        for j in neighbor_idx:
            score = float(sims[i, j])
            if score < threshold:
                continue
            a, b = ids[i], ids[j]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            triples.append((key[0], key[1], score))

    triples.sort(key=lambda t: t[2], reverse=True)
    return triples
