"""Phase 13b2 — concept nodes built from Leiden communities."""

from __future__ import annotations

from kairo_core import Document
from kairo_graph import (
    DocumentGraphBuilder,
    KairoGraph,
    add_concept_nodes,
    detect_communities,
    summarize_community,
)


# ---------------------------------------------------------------------------
# summarize_community
# ---------------------------------------------------------------------------

def test_summarize_uses_text_preview_attr() -> None:
    g = KairoGraph()
    g.add_node("a", text_preview="cats purr cats sleep cats hunt")
    g.add_node("b", text_preview="cats meow loudly")
    out = summarize_community(g, ["a", "b"], top_k_terms=3)
    # "cats" appears 4 times, should lead.
    assert out.split()[0] == "cats"


def test_summarize_skips_stopwords_and_short_tokens() -> None:
    g = KairoGraph()
    g.add_node("a", text_preview="the cat is on the mat and it is fine ok")
    out = summarize_community(g, ["a"])
    tokens = out.split()
    # Stopwords "the" / "and" and sub-3-char tokens "is" / "on" / "ok" / "it"
    # must be dropped.
    for banned in ("the", "and", "is", "on", "ok", "it"):
        assert banned not in tokens


def test_summarize_returns_empty_when_no_text() -> None:
    g = KairoGraph()
    g.add_node("a")  # no text_preview / full_text / label content
    assert summarize_community(g, ["a"]) == ""


# ---------------------------------------------------------------------------
# add_concept_nodes
# ---------------------------------------------------------------------------

def test_add_concept_nodes_creates_belongs_to_edges() -> None:
    g = KairoGraph()
    for nid in ("d1", "d2", "d3"):
        g.add_node(nid, kind="document", text_preview=f"shared term {nid}")
    communities = [["d1", "d2", "d3"]]
    concepts = add_concept_nodes(g, communities)

    assert concepts == ["concept:0"]
    assert "concept:0" in g.nx_graph.nodes
    concept_data = g.nx_graph.nodes["concept:0"]
    assert concept_data["kind"] == "concept"
    assert concept_data["size"] == 3

    belongs = [
        (u, v, d) for u, v, d in g.nx_graph.edges(data=True)
        if d.get("kind") == "belongs_to"
    ]
    assert len(belongs) == 3
    for u, v, d in belongs:
        assert v == "concept:0"
        assert u in {"d1", "d2", "d3"}
        # belongs_to is a derived edge, tagged inferred
        assert d["provenance"] == "inferred"
        assert d["confidence"] == 1.0


def test_add_concept_nodes_handles_multiple_communities() -> None:
    g = KairoGraph()
    for nid in ("a1", "a2", "b1", "b2"):
        g.add_node(nid, kind="document", text_preview=nid)
    concepts = add_concept_nodes(g, [["a1", "a2"], ["b1", "b2"]])
    assert concepts == ["concept:0", "concept:1"]
    assert g.nx_graph.nodes["concept:0"]["size"] == 2
    assert g.nx_graph.nodes["concept:1"]["size"] == 2


def test_add_concept_nodes_accepts_custom_summary_fn() -> None:
    g = KairoGraph()
    g.add_node("x", text_preview="irrelevant")

    def constant_fn(_graph, _members) -> str:
        return "pluggable-summary"

    add_concept_nodes(g, [["x"]], summary_fn=constant_fn)
    assert g.nx_graph.nodes["concept:0"]["summary"] == "pluggable-summary"


# ---------------------------------------------------------------------------
# Builder integration
# ---------------------------------------------------------------------------

def test_document_builder_opt_in_produces_concept_nodes() -> None:
    """When ``detect_communities=True`` the builder runs detection + concept
    insertion after similarity edges. Default remains no concepts."""
    from kairo_embeddings import get_provider

    # Two clear clusters: animal docs vs. programming docs.
    docs = [
        Document(id="a1", text="cats purr felines hunt small prey"),
        Document(id="a2", text="cats sleep felines bathe often"),
        Document(id="a3", text="felines groom cats chase laser dots"),
        Document(id="p1", text="python decorator function signature"),
        Document(id="p2", text="python generator yield expression"),
        Document(id="p3", text="python dataclass field attribute"),
    ]
    provider = get_provider("hash-stub", dim=128)

    default_graph = DocumentGraphBuilder(
        embedding_provider=provider,
        similarity_k=3,
        similarity_threshold=0.0,
    ).build(docs)
    assert not any(
        d.get("kind") == "concept"
        for _, d in default_graph.nx_graph.nodes(data=True)
    )

    concept_graph = DocumentGraphBuilder(
        embedding_provider=provider,
        similarity_k=3,
        similarity_threshold=0.0,
        detect_communities=True,
    ).build(docs)

    concepts = [
        (nid, d) for nid, d in concept_graph.nx_graph.nodes(data=True)
        if d.get("kind") == "concept"
    ]
    assert len(concepts) >= 1
    for _nid, cdata in concepts:
        assert cdata["size"] >= 2
        assert isinstance(cdata["summary"], str)


def test_detect_communities_then_add_concept_nodes_composes() -> None:
    """The two helpers chain cleanly — the output of detect_communities is
    the input shape add_concept_nodes expects."""
    g = KairoGraph()
    for nid in ("A", "B", "C", "X", "Y", "Z"):
        g.add_node(nid, kind="document", text_preview=nid)
    for u, v in [("A", "B"), ("B", "C"), ("C", "A")]:
        g.add_edge(u, v, kind="similar-to", weight=0.9,
                   provenance="inferred", confidence=0.9)
    for u, v in [("X", "Y"), ("Y", "Z"), ("Z", "X")]:
        g.add_edge(u, v, kind="similar-to", weight=0.9,
                   provenance="inferred", confidence=0.9)
    g.add_edge("A", "X", kind="similar-to", weight=0.05,
               provenance="inferred", confidence=0.05)

    communities = detect_communities(g, min_community_size=2)
    concepts = add_concept_nodes(g, communities)
    assert len(concepts) == len(communities) == 2
    # Every original doc now has exactly one belongs_to edge.
    for doc in ("A", "B", "C", "X", "Y", "Z"):
        out_kinds = [
            data.get("kind")
            for _, _, data in g.nx_graph.out_edges(doc, data=True)
        ]
        assert out_kinds.count("belongs_to") == 1
