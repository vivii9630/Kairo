from __future__ import annotations

from typing import Dict, List, Tuple

from kairo_core.models import GraphData, GraphDiff, GraphEdge, GraphNode


def diff_graphs(old: GraphData, new: GraphData) -> GraphDiff:
    """Structural diff between two graphs.

    A node is "changed" if its id is in both graphs but its (kind, label, attrs)
    differ. Edges are compared as (source, target, kind) triples.
    """
    old_nodes = _nodes_by_id(old.nodes)
    new_nodes = _nodes_by_id(new.nodes)

    added_nodes = sorted(set(new_nodes) - set(old_nodes))
    removed_nodes = sorted(set(old_nodes) - set(new_nodes))
    changed_nodes = sorted(
        nid
        for nid in set(old_nodes) & set(new_nodes)
        if _node_fingerprint(old_nodes[nid]) != _node_fingerprint(new_nodes[nid])
    )

    old_edges = _edge_keys(old.edges)
    new_edges = _edge_keys(new.edges)
    added_edges = sorted(new_edges - old_edges)
    removed_edges = sorted(old_edges - new_edges)

    return GraphDiff(
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        changed_nodes=changed_nodes,
        added_edges=[list(e) for e in added_edges],
        removed_edges=[list(e) for e in removed_edges],
    )


def _nodes_by_id(nodes: List[GraphNode]) -> Dict[str, GraphNode]:
    return {n.id: n for n in nodes}


def _node_fingerprint(node: GraphNode) -> Tuple:
    return (node.kind, node.label, tuple(sorted(node.attrs.items())))


def _edge_keys(edges: List[GraphEdge]):
    return {(e.source, e.target, e.kind) for e in edges}
