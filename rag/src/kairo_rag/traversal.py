"""Graph traversal and KNN search over a 3-layer Kairo graph.

Two retrieval modes work together:

1. **KNN (vector)** — cosine similarity over per-node embeddings to find
   the starting neighborhoods in each layer.
2. **Graph walk** — BFS/weighted expansion from the KNN hits, following
   intra-layer edges and optionally crossing inter-layer edges to pull
   in structurally connected evidence.

The traversal is deterministic: given the same graph, embeddings, and
query vector, the output is identical every run.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from kairo_core import InterLayerEdge, TraversalStep, TraversalTrace
from kairo_graph import LayeredKairoGraph
from kairo_embeddings.store import EmbeddingStore

from .layered_store import LayerEmbeddingStore

# Type alias for layer names.
LayerName = str


# ---------------------------------------------------------------------------
# KNN search across layers
# ---------------------------------------------------------------------------

def knn_search(
    query_vec: np.ndarray,
    layer_embeddings: LayerEmbeddingStore,
    *,
    top_k: int = 5,
    layers: Optional[List[LayerName]] = None,
) -> Dict[LayerName, List[Tuple[str, float]]]:
    """Run cosine-similarity KNN search independently in each layer.

    Returns ``{layer_name: [(node_id, score), ...]}`` for the requested
    layers (defaults to all three).
    """
    layers = layers or ["document", "semantic", "detail"]
    results: Dict[LayerName, List[Tuple[str, float]]] = {}
    for layer_name in layers:
        store = layer_embeddings.get(layer_name)
        if store is None or len(store) == 0:
            results[layer_name] = []
            continue
        results[layer_name] = store.similar(query_vec, k=top_k)
    return results


# ---------------------------------------------------------------------------
# Graph-walk expansion
# ---------------------------------------------------------------------------

def graph_walk(
    graph: LayeredKairoGraph,
    seeds: Dict[LayerName, List[str]],
    *,
    max_depth: int = 2,
    max_nodes_per_layer: int = 30,
    cross_layer: bool = True,
    agent_id: str = "traversal",
) -> TraversalTrace:
    """BFS expansion from *seeds* across the 3-layer graph.

    Parameters
    ----------
    graph : LayeredKairoGraph
        The composite graph to traverse.
    seeds : dict
        ``{layer_name: [node_id, ...]}`` — starting nodes per layer
        (typically from :func:`knn_search`).
    max_depth : int
        Maximum BFS hops within a single layer.
    max_nodes_per_layer : int
        Stop expanding a layer after visiting this many nodes.
    cross_layer : bool
        Whether to follow inter-layer edges.
    agent_id : str
        Identifier recorded in :class:`TraversalStep` entries.

    Returns
    -------
    TraversalTrace
        Full record of visited nodes, steps, and crossed edges.
    """
    visited: Dict[str, Set[str]] = {
        "document": set(),
        "semantic": set(),
        "detail": set(),
    }
    steps: List[TraversalStep] = []
    crossed: List[InterLayerEdge] = []

    # BFS per layer
    for layer_name in ("document", "semantic", "detail"):
        layer_graph = graph.layer(layer_name)
        queue: deque[Tuple[str, int]] = deque()
        for nid in seeds.get(layer_name, []):
            if nid in layer_graph.nx_graph:
                queue.append((nid, 0))

        while queue and len(visited[layer_name]) < max_nodes_per_layer:
            nid, depth = queue.popleft()
            if nid in visited[layer_name]:
                continue
            visited[layer_name].add(nid)
            steps.append(TraversalStep(
                agent_id=agent_id,
                node_id=nid,
                layer=layer_name,  # type: ignore[arg-type]
                action="visit",
            ))

            if depth < max_depth:
                for neighbor in layer_graph.nx_graph.successors(nid):
                    if neighbor not in visited[layer_name]:
                        queue.append((neighbor, depth + 1))
                        steps.append(TraversalStep(
                            agent_id=agent_id,
                            node_id=nid,
                            layer=layer_name,  # type: ignore[arg-type]
                            action="expand",
                        ))

    # Cross-layer expansion
    if cross_layer:
        all_visited = set()
        for s in visited.values():
            all_visited |= s

        for edge in graph.inter_layer_edges:
            src_visited = edge.source in visited.get(edge.source_layer, set())
            tgt_visited = edge.target in visited.get(edge.target_layer, set())
            if src_visited and not tgt_visited:
                visited.setdefault(edge.target_layer, set()).add(edge.target)
                crossed.append(edge)
                steps.append(TraversalStep(
                    agent_id=agent_id,
                    node_id=edge.target,
                    layer=edge.target_layer,  # type: ignore[arg-type]
                    action="cross_layer",
                ))
            elif tgt_visited and not src_visited:
                visited.setdefault(edge.source_layer, set()).add(edge.source)
                crossed.append(edge)
                steps.append(TraversalStep(
                    agent_id=agent_id,
                    node_id=edge.source,
                    layer=edge.source_layer,  # type: ignore[arg-type]
                    action="cross_layer",
                ))

    return TraversalTrace(
        query="",  # caller fills this in
        steps=steps,
        visited_nodes={k: sorted(v) for k, v in visited.items()},
        crossed_edges=crossed,
    )


# ---------------------------------------------------------------------------
# Cluster labelling (for viz + supervisor routing)
# ---------------------------------------------------------------------------

def cluster_visited_nodes(
    graph: LayeredKairoGraph,
    visited: Dict[str, List[str]],
    *,
    max_clusters: int = 5,
) -> Dict[str, List[str]]:
    """Group visited nodes into topical clusters based on graph connectivity.

    Uses a simple connected-components approach on the subgraph induced by
    visited nodes.  Returns ``{cluster_label: [node_id, ...]}``.
    """
    try:
        import networkx as nx
    except ImportError:
        return {"default": [nid for ids in visited.values() for nid in ids]}

    # Build a subgraph of all visited nodes across layers
    sub = nx.Graph()
    for layer_name, node_ids in visited.items():
        layer_g = graph.layer(layer_name)
        for nid in node_ids:
            if nid in layer_g.nx_graph:
                sub.add_node(nid, layer=layer_name)
                for neighbor in layer_g.nx_graph.successors(nid):
                    if neighbor in node_ids:
                        sub.add_edge(nid, neighbor)

    # Also add inter-layer edges between visited nodes
    all_visited = {nid for ids in visited.values() for nid in ids}
    for edge in graph.inter_layer_edges:
        if edge.source in all_visited and edge.target in all_visited:
            sub.add_edge(edge.source, edge.target)

    components = list(nx.connected_components(sub))
    components.sort(key=len, reverse=True)

    clusters: Dict[str, List[str]] = {}
    for i, comp in enumerate(components[:max_clusters]):
        clusters[f"cluster_{i}"] = sorted(comp)
    # Overflow into last cluster
    if len(components) > max_clusters:
        overflow = clusters.get(f"cluster_{max_clusters - 1}", [])
        for comp in components[max_clusters:]:
            overflow.extend(sorted(comp))
        clusters[f"cluster_{max_clusters - 1}"] = overflow

    return clusters
