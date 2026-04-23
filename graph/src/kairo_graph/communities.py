"""Phase 13b1 — community detection on weighted similarity graphs.

Preferred backend is Leiden via ``leidenalg`` + ``python-igraph``
(stricter quality guarantees than Louvain — monotonic modularity
improvement, no bad local optima). When those optional deps aren't
installed we fall back to NetworkX's ``louvain_communities`` so
callers always get a result.

Only one public entry point — :func:`detect_communities` — because
that's the stable contract downstream (13b2 concept-node builder,
retrieval filtering) will depend on. Swapping the backend later
shouldn't require touching callers.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence, Set

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kairo-graph requires networkx. Install with: pip install networkx"
    ) from exc

from .graph import KairoGraph
from .provenance import PROVENANCE_INFERRED, edge_attrs


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")
_DEFAULT_STOPWORDS: Set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "was", "one", "our", "out", "has", "have", "had", "its", "from",
    "this", "that", "with", "been", "their", "they", "them", "were",
    "into", "than", "then", "what", "when", "your", "about", "would",
    "there", "could", "other", "after", "first", "where", "these",
    "those", "such", "some", "more", "most", "very", "just", "also",
    "only", "over", "under", "which", "while", "who", "whom", "why",
}


def _build_subgraph(graph: KairoGraph, edge_kind: str) -> "nx.Graph":
    """Undirected projection over edges matching *edge_kind*, preserving
    the max weight whenever both directions are present."""
    sub: nx.Graph = nx.Graph()
    for u, v, data in graph.nx_graph.edges(data=True):
        if data.get("kind") != edge_kind:
            continue
        weight = float(data.get("weight", 1.0))
        if weight <= 0:
            continue
        if sub.has_edge(u, v):
            sub[u][v]["weight"] = max(sub[u][v]["weight"], weight)
        else:
            sub.add_edge(u, v, weight=weight)
    return sub


def _leiden(sub: "nx.Graph", *, resolution: float, seed: int) -> List[List[str]]:
    import igraph as ig
    import leidenalg

    node_list = list(sub.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}
    edges = [(node_idx[u], node_idx[v]) for u, v in sub.edges()]
    weights = [float(sub[u][v].get("weight", 1.0)) for u, v in sub.edges()]

    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    return [[node_list[i] for i in community] for community in partition]


def _louvain(sub: "nx.Graph", *, resolution: float, seed: int) -> List[List[str]]:
    from networkx.algorithms.community import louvain_communities

    raw = louvain_communities(
        sub, weight="weight", resolution=resolution, seed=seed
    )
    return [list(c) for c in raw]


def leiden_available() -> bool:
    """True when Leiden can run — Louvain is the fallback otherwise."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        return False
    return True


def detect_communities(
    graph: KairoGraph,
    *,
    edge_kind: str = "similar-to",
    resolution: float = 1.0,
    seed: int = 0,
    min_community_size: int = 2,
) -> List[List[str]]:
    """Group nodes by community over the weighted *edge_kind* subgraph.

    Uses Leiden when ``leidenalg`` + ``python-igraph`` are importable,
    Louvain otherwise. Singletons (size < *min_community_size*) are
    dropped — a concept cluster over a single doc is never informative.

    Returns a list of communities, each a sorted list of node ids,
    ordered by descending size (ties broken by lowest id for stable
    snapshot hashing).
    """
    sub = _build_subgraph(graph, edge_kind)
    if sub.number_of_edges() == 0:
        return []

    if leiden_available():
        raw = _leiden(sub, resolution=resolution, seed=seed)
    else:
        raw = _louvain(sub, resolution=resolution, seed=seed)

    sized = [sorted(c) for c in raw if len(c) >= min_community_size]
    sized.sort(key=lambda c: (-len(c), c[0]))
    return sized


# ---------------------------------------------------------------------------
# Concept summarization (default = lightweight, no LLM)
# ---------------------------------------------------------------------------

def summarize_community(
    graph: KairoGraph,
    member_ids: Sequence[str],
    *,
    top_k_terms: int = 8,
    text_attr_candidates: Sequence[str] = ("text_preview", "full_text", "label"),
    stopwords: Set[str] = _DEFAULT_STOPWORDS,
) -> str:
    """Build a lightweight summary as the top-k most frequent tokens.

    Pulls text from the first matching attribute on each member node
    (defaults align with DocumentGraphBuilder's ``text_preview`` and
    DocumentLayeredBuilder's ``full_text``). Stopwords and short tokens
    are skipped; result is space-joined.

    Designed to be swapped: :func:`add_concept_nodes` accepts a
    ``summary_fn`` with the same signature, so a caller can plug in an
    Ollama-backed summarizer later without touching this module.
    """
    nx_g = graph.nx_graph
    counts: Dict[str, int] = {}
    for nid in member_ids:
        if nid not in nx_g:
            continue
        data = nx_g.nodes[nid]
        text = ""
        for attr in text_attr_candidates:
            val = data.get(attr)
            if val:
                text = str(val)
                break
        for tok in _WORD_RE.findall(text.lower()):
            if tok in stopwords or len(tok) <= 2:
                continue
            counts[tok] = counts.get(tok, 0) + 1

    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k_terms]
    return " ".join(t for t, _ in top)


SummaryFn = Callable[[KairoGraph, Sequence[str]], str]


# ---------------------------------------------------------------------------
# Concept-node insertion
# ---------------------------------------------------------------------------

def add_concept_nodes(
    graph: KairoGraph,
    communities: List[List[str]],
    *,
    summary_fn: SummaryFn = summarize_community,
    concept_prefix: str = "concept:",
) -> List[str]:
    """Mutate *graph* to add one concept node per community.

    For each community, creates a node ``{concept_prefix}{i}`` with
    ``kind="concept"``, ``label`` = first 80 chars of the summary, and
    attrs ``summary`` (full summary string) + ``size`` (member count).
    Adds a ``belongs_to`` edge from every member → the concept, tagged
    inferred since the cluster itself was derived by community
    detection (not present in the source material).

    Returns the created concept node ids in community order.
    """
    belongs_to = edge_attrs(provenance=PROVENANCE_INFERRED, confidence=1.0)
    concept_ids: List[str] = []
    for idx, members in enumerate(communities):
        cid = f"{concept_prefix}{idx}"
        summary = summary_fn(graph, members)
        graph.add_node(
            cid,
            kind="concept",
            label=(summary[:80] if summary else cid),
            summary=summary,
            size=len(members),
        )
        for member in members:
            graph.add_edge(member, cid, kind="belongs_to", **belongs_to)
        concept_ids.append(cid)
    return concept_ids
