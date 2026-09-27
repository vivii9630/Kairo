"""Phase 13b1 — provenance + confidence tags on every edge builder."""

from __future__ import annotations

import pytest

from kairo_core import Document
from kairo_graph import (
    DocumentGraphBuilder,
    DocumentLayeredBuilder,
    PROVENANCE_INFERRED,
    PROVENANCE_STRUCTURAL,
    PROVENANCE_VALUES,
    edge_attrs,
)
from kairo_graph.builders.code import CodeGraphBuilder
from kairo_graph.builders.tabular import TabularGraphBuilder


# ---------------------------------------------------------------------------
# edge_attrs helper
# ---------------------------------------------------------------------------

def test_edge_attrs_defaults_to_structural_full_confidence() -> None:
    attrs = edge_attrs()
    assert attrs == {"provenance": "structural", "confidence": 1.0}


def test_edge_attrs_rejects_bad_provenance() -> None:
    with pytest.raises(ValueError):
        edge_attrs(provenance="made-up")


def test_edge_attrs_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        edge_attrs(confidence=1.5)
    with pytest.raises(ValueError):
        edge_attrs(confidence=-0.01)


def test_edge_attrs_preserves_extra_kwargs() -> None:
    attrs = edge_attrs(provenance="inferred", confidence=0.7, score=0.7, why="cosine")
    assert attrs["score"] == 0.7
    assert attrs["why"] == "cosine"


# ---------------------------------------------------------------------------
# Per-builder provenance coverage
# ---------------------------------------------------------------------------

def _assert_every_edge_tagged(graph) -> None:
    for _u, _v, data in graph.nx_graph.edges(data=True):
        assert "provenance" in data, f"edge missing provenance: {data}"
        assert "confidence" in data, f"edge missing confidence: {data}"
        assert data["provenance"] in PROVENANCE_VALUES
        assert 0.0 <= float(data["confidence"]) <= 1.0


def test_code_builder_edges_are_structural() -> None:
    src = (
        "import json\n"
        "\n"
        "def foo():\n"
        "    return bar()\n"
        "\n"
        "def bar():\n"
        "    return 1\n"
    )
    docs = [Document(id="m", text=src, metadata={"path": "m.py"})]
    graph = CodeGraphBuilder().build(docs)
    _assert_every_edge_tagged(graph)
    for _u, _v, data in graph.nx_graph.edges(data=True):
        assert data["provenance"] == PROVENANCE_STRUCTURAL
        assert data["confidence"] == 1.0


def test_tabular_builder_edges_are_structural() -> None:
    docs = [
        Document(
            id=f"r{i}",
            text='{"a": 1, "b": 2}',
            metadata={"path": "t.csv"},
        )
        for i in range(3)
    ]
    graph = TabularGraphBuilder().build(docs)
    _assert_every_edge_tagged(graph)
    for _u, _v, data in graph.nx_graph.edges(data=True):
        assert data["provenance"] == PROVENANCE_STRUCTURAL


def test_document_builder_similar_edges_are_inferred() -> None:
    from kairo_embeddings import get_provider

    docs = [
        Document(id="d1", text="cats purr when content"),
        Document(id="d2", text="dogs bark at strangers"),
        Document(id="d3", text="python programming language"),
    ]
    graph = DocumentGraphBuilder(
        embedding_provider=get_provider("hash-stub", dim=64),
        similarity_k=2,
        similarity_threshold=0.0,
    ).build(docs)
    _assert_every_edge_tagged(graph)
    for _u, _v, data in graph.nx_graph.edges(data=True):
        assert data["kind"] == "similar-to"
        assert data["provenance"] == PROVENANCE_INFERRED
        # confidence carries the cosine score
        assert data["confidence"] == pytest.approx(data["weight"])


def test_layered_builder_mixes_structural_and_inferred() -> None:
    docs = [
        Document(id="a", text="Cats purr. Felines sleep often."),
        Document(id="b", text="Dogs bark. Canines howl."),
    ]
    lg = DocumentLayeredBuilder().build(docs)

    seen = {"structural": 0, "inferred": 0}
    for layer in ("document", "semantic", "detail"):
        for _u, _v, data in lg.layer(layer).nx_graph.edges(data=True):
            assert data["provenance"] in PROVENANCE_VALUES
            seen[data["provenance"]] = seen.get(data["provenance"], 0) + 1

    # Inter-layer edges too
    for edge in lg.inter_layer_edges:
        assert edge.attrs.get("provenance") in PROVENANCE_VALUES

    # We expect both kinds to appear: document/semantic sequence & grounds
    # are structural, co_occurs / near are inferred.
    assert seen["structural"] > 0
    assert seen["inferred"] > 0
