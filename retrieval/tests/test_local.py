"""Tests for LocalPipeline — BM25-only and BM25+cosine+RRF paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from kairo_core import Document, QueryRequest
from kairo_embeddings import get_provider
from kairo_retrieval import LocalPipeline


def _corpus() -> list[Document]:
    return [
        Document(id="d1", text="Cats purr when they are happy and content."),
        Document(id="d2", text="Dogs bark to warn their owners of intruders."),
        Document(id="d3", text="Felines often sleep in sunny spots on the floor."),
        Document(id="d4", text="Python is a programming language used for data analysis."),
        Document(
            id="d5",
            text="Machine learning models need labeled data to train effectively.",
            metadata={"topic": "ml"},
        ),
        Document(
            id="d6",
            text="Data pipelines move records from source systems into warehouses.",
            metadata={"topic": "data"},
        ),
    ]


def _make_pipeline(
    tmp_path: Path, *, with_embeddings: bool
) -> LocalPipeline:
    store = tmp_path / "store.json"
    provider = get_provider("hash-stub", dim=64) if with_embeddings else None
    pipe = LocalPipeline(store_path=store, embedding_provider=provider)
    pipe.ingest_documents(_corpus())
    return pipe


def test_bm25_only_retrieves_lexical_match(tmp_path: Path) -> None:
    pipe = _make_pipeline(tmp_path, with_embeddings=False)
    result = pipe._retrieve(QueryRequest(query="cats purr", top_k=3))
    ids = [ev.document_id for ev in result.evidences]
    assert ids[0] == "d1"
    assert result.plan and "BM25" in result.plan
    assert "RRF" not in result.plan


def test_hybrid_rrf_runs_two_retrievers(tmp_path: Path) -> None:
    pipe = _make_pipeline(tmp_path, with_embeddings=True)
    result = pipe._retrieve(QueryRequest(query="cats purr", top_k=3))
    ids = [ev.document_id for ev in result.evidences]
    assert "d1" in ids
    assert result.plan and "RRF" in result.plan
    # RRF scores are small (1/(60+rank+1)), strictly positive, non-increasing.
    scores = [ev.score for ev in result.evidences]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_filters_restrict_candidate_pool(tmp_path: Path) -> None:
    pipe = _make_pipeline(tmp_path, with_embeddings=False)
    result = pipe._retrieve(
        QueryRequest(query="pipelines warehouses", top_k=5, filters={"topic": "data"})
    )
    ids = [ev.document_id for ev in result.evidences]
    assert ids == ["d6"]

    # A query that only matches d5 (topic=ml) should NOT leak d5 through
    # a topic=data filter — we'd either get d6 (degraded match) or empty.
    result = pipe._retrieve(
        QueryRequest(query="labeled models", top_k=5, filters={"topic": "data"})
    )
    ids = [ev.document_id for ev in result.evidences]
    assert "d5" not in ids


def test_empty_corpus_returns_no_evidences(tmp_path: Path) -> None:
    pipe = LocalPipeline(store_path=tmp_path / "store.json")
    result = pipe._retrieve(QueryRequest(query="anything", top_k=3))
    assert result.evidences == []


def test_rrf_fusion_weights_top_ranks(tmp_path: Path) -> None:
    """A doc ranked #1 by both retrievers should score higher than one
    ranked #1 by one and absent from the other."""
    rankings = [
        ["a", "b", "c"],
        ["a", "d", "b"],
    ]
    fused = LocalPipeline._rrf_fuse(rankings)
    scores = dict(fused)
    # 'a' is rank-1 in both → 2 * 1/(60+1)
    assert scores["a"] == pytest.approx(2.0 / 61)
    # 'b' is rank-2 in first, rank-3 in second → 1/62 + 1/63
    assert scores["b"] == pytest.approx(1.0 / 62 + 1.0 / 63)
    # 'a' wins overall
    assert fused[0][0] == "a"


def test_graph_walk_surfaces_neighbor_not_in_lexical_match(tmp_path: Path) -> None:
    """Phase 13a: a doc that's lexically irrelevant but graph-adjacent to a
    top hit should get pulled into the result set via the 1-hop walk."""
    docs = [
        Document(id="d1", text="alpha alpha alpha"),
        Document(id="d2", text="totally unrelated zebra giraffe"),
        Document(id="d3", text="another disconnected text about apples"),
        Document(id="d4", text="oranges grapes melons"),
        Document(id="d5", text="pineapple mango papaya"),
    ]
    # d2 is a graph-adjacent 'similar-to' neighbor of d1. Without the walk,
    # querying "alpha" returns only d1 (the only BM25 match). With the walk,
    # d2 should also appear since it's reachable from d1.
    neighbors = {"d1": [("d2", 0.7)]}
    pipe = LocalPipeline(
        store_path=tmp_path / "store.json",
        graph_neighbors=lambda doc_id: neighbors.get(doc_id, []),
        hop_decay=0.5,
    )
    pipe.ingest_documents(docs)

    no_walk_pipe = LocalPipeline(store_path=tmp_path / "store_nowalk.json")
    no_walk_pipe.ingest_documents(docs)

    walked = pipe._retrieve(QueryRequest(query="alpha", top_k=3))
    walked_ids = [ev.document_id for ev in walked.evidences]
    assert walked_ids[0] == "d1"
    assert "d2" in walked_ids
    assert "1-hop graph walk" in (walked.plan or "")

    base = no_walk_pipe._retrieve(QueryRequest(query="alpha", top_k=3))
    base_ids = [ev.document_id for ev in base.evidences]
    assert base_ids == ["d1"]


def test_graph_walk_boost_skips_unknown_docs(tmp_path: Path) -> None:
    """Neighbors pointing to doc ids not in the corpus must be silently
    dropped — the adjacency callable could be stale relative to the store."""
    docs = [Document(id="d1", text="alpha"), Document(id="d2", text="beta")]
    neighbors = {"d1": [("ghost_doc", 0.99), ("d2", 0.5)]}
    pipe = LocalPipeline(
        store_path=tmp_path / "store.json",
        graph_neighbors=lambda doc_id: neighbors.get(doc_id, []),
    )
    pipe.ingest_documents(docs)

    result = pipe._retrieve(QueryRequest(query="alpha", top_k=5))
    ids = {ev.document_id for ev in result.evidences}
    assert "ghost_doc" not in ids
    assert ids <= {"d1", "d2"}


def test_graph_walk_callable_exception_does_not_break_query(tmp_path: Path) -> None:
    """A buggy adjacency callable shouldn't take down retrieval."""

    def explode(_doc_id: str):
        raise RuntimeError("graph backend offline")

    docs = [Document(id="d1", text="alpha")]
    pipe = LocalPipeline(
        store_path=tmp_path / "store.json",
        graph_neighbors=explode,
    )
    pipe.ingest_documents(docs)

    result = pipe._retrieve(QueryRequest(query="alpha", top_k=3))
    assert [ev.document_id for ev in result.evidences] == ["d1"]
