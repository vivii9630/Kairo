"""Tests for Phase 11c weighted similarity edges on DocumentGraphBuilder."""

from __future__ import annotations

import numpy as np

from kairo_core import Document
from kairo_embeddings import get_provider
from kairo_graph import DocumentGraphBuilder
from kairo_graph.builders._similarity import top_k_similar_pairs


def _docs() -> list[Document]:
    return [
        Document(id="d1", text="cats purr when content"),
        Document(id="d2", text="dogs bark at strangers"),
        Document(id="d3", text="python programming language"),
        Document(id="d4", text="python used for data analysis"),
    ]


def test_no_provider_means_zero_edges() -> None:
    builder = DocumentGraphBuilder()
    graph = builder.build(_docs())
    assert graph.num_nodes() == 4
    assert graph.num_edges() == 0


def test_provider_emits_similar_to_edges() -> None:
    builder = DocumentGraphBuilder(
        embedding_provider=get_provider("hash-stub", dim=64),
        similarity_k=3,
        similarity_threshold=0.0,
    )
    graph = builder.build(_docs())
    assert graph.num_nodes() == 4

    edges = list(graph.nx_graph.edges(data=True))
    assert len(edges) > 0
    for _u, _v, data in edges:
        assert data["kind"] == "similar-to"
        assert 0.0 <= data["weight"] <= 1.0 + 1e-6
        # Phase 13b1: every similarity edge is tagged inferred with the
        # cosine score as its confidence.
        assert data["provenance"] == "inferred"
        assert abs(data["confidence"] - data["weight"]) < 1e-6


def test_similarity_threshold_filters_weak_pairs() -> None:
    """A high threshold should leave most pairs out."""
    builder = DocumentGraphBuilder(
        embedding_provider=get_provider("hash-stub", dim=64),
        similarity_k=5,
        similarity_threshold=0.99,
    )
    graph = builder.build(_docs())
    assert graph.num_edges() <= 1  # hash-stub rarely produces > 0.99 cosine


def test_top_k_similar_pairs_is_dedup_and_sorted() -> None:
    ids = ["a", "b", "c"]
    vecs = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pairs = top_k_similar_pairs(ids, vecs, k=2, threshold=0.0)
    # Each unordered pair appears once, sorted by descending score.
    seen = {frozenset((a, b)) for a, b, _ in pairs}
    assert len(seen) == len(pairs)
    scores = [s for _, _, s in pairs]
    assert scores == sorted(scores, reverse=True)
    # a and b are nearly colinear → highest pair.
    assert frozenset(pairs[0][:2]) == {"a", "b"}


def test_single_doc_skips_similarity_step() -> None:
    """Builder must not attempt similarity on a 1-doc corpus."""
    builder = DocumentGraphBuilder(
        embedding_provider=get_provider("hash-stub", dim=32),
    )
    graph = builder.build([Document(id="solo", text="only one")])
    assert graph.num_nodes() == 1
    assert graph.num_edges() == 0
