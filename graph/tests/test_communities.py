"""Phase 13b1 — community detection (Leiden preferred, Louvain fallback)."""

from __future__ import annotations

from kairo_graph import KairoGraph, detect_communities, leiden_available


def _two_cluster_graph() -> KairoGraph:
    """A → B → C → A densely connected, X → Y → Z likewise, and one
    weak bridge A → X. Leiden/Louvain should split into two communities."""
    g = KairoGraph()
    for nid in ("A", "B", "C", "X", "Y", "Z"):
        g.add_node(nid, kind="document")
    for u, v in [("A", "B"), ("B", "C"), ("C", "A")]:
        g.add_edge(u, v, kind="similar-to", weight=0.9,
                   provenance="inferred", confidence=0.9)
    for u, v in [("X", "Y"), ("Y", "Z"), ("Z", "X")]:
        g.add_edge(u, v, kind="similar-to", weight=0.9,
                   provenance="inferred", confidence=0.9)
    g.add_edge("A", "X", kind="similar-to", weight=0.05,
               provenance="inferred", confidence=0.05)
    return g


def test_detect_communities_returns_two_groups() -> None:
    g = _two_cluster_graph()
    communities = detect_communities(g, min_community_size=2)
    assert len(communities) == 2
    # Each community is one of the triangles.
    groups = [set(c) for c in communities]
    assert {"A", "B", "C"} in groups
    assert {"X", "Y", "Z"} in groups


def test_detect_communities_ignores_other_edge_kinds() -> None:
    g = KairoGraph()
    for nid in ("a", "b", "c"):
        g.add_node(nid)
    g.add_edge("a", "b", kind="calls", weight=1.0,
               provenance="structural", confidence=1.0)
    g.add_edge("b", "c", kind="calls", weight=1.0,
               provenance="structural", confidence=1.0)
    assert detect_communities(g, edge_kind="similar-to") == []


def test_detect_communities_skips_singletons() -> None:
    g = KairoGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b", kind="similar-to", weight=0.8,
               provenance="inferred", confidence=0.8)
    # With min_community_size=3 our 2-node component drops out.
    assert detect_communities(g, min_community_size=3) == []


def test_detect_communities_ignores_zero_or_negative_weights() -> None:
    g = KairoGraph()
    for nid in ("a", "b", "c"):
        g.add_node(nid)
    g.add_edge("a", "b", kind="similar-to", weight=0.0,
               provenance="inferred", confidence=0.0)
    g.add_edge("b", "c", kind="similar-to", weight=-0.1,
               provenance="inferred", confidence=0.0)
    assert detect_communities(g) == []


def test_leiden_available_flag_is_consistent_with_backend() -> None:
    """The helper should agree with what happens if we try to import."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    assert leiden_available() is installed
