from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kairo-graph requires networkx. Install with: pip install networkx"
    ) from exc

from kairo_core import GraphData, GraphEdge, GraphNode


class KairoGraph:
    """Directed graph with deterministic hashing and JSON round-trip.

    Thin wrapper over :class:`networkx.DiGraph` so we can swap the backing
    store (Kuzu, Memgraph, etc.) later without changing call sites.
    """

    def __init__(self, graph: Optional["nx.DiGraph"] = None) -> None:
        self._g: nx.DiGraph = graph if graph is not None else nx.DiGraph()

    @property
    def nx_graph(self) -> "nx.DiGraph":
        return self._g

    def add_node(self, node_id: str, *, kind: str = "node", label: str = "", **attrs: Any) -> None:
        self._g.add_node(node_id, kind=kind, label=label, **attrs)

    def add_edge(self, source: str, target: str, *, kind: str = "edge", **attrs: Any) -> None:
        self._g.add_edge(source, target, kind=kind, **attrs)

    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def to_data(self) -> GraphData:
        reserved_node = {"kind", "label"}
        reserved_edge = {"kind"}
        nodes = [
            GraphNode(
                id=str(nid),
                kind=data.get("kind", "node"),
                label=data.get("label", ""),
                attrs={k: v for k, v in data.items() if k not in reserved_node},
            )
            for nid, data in self._g.nodes(data=True)
        ]
        edges = [
            GraphEdge(
                source=str(u),
                target=str(v),
                kind=data.get("kind", "edge"),
                attrs={k: v for k, v in data.items() if k not in reserved_edge},
            )
            for u, v, data in self._g.edges(data=True)
        ]
        return GraphData(nodes=nodes, edges=edges)

    @classmethod
    def from_data(cls, data: GraphData) -> "KairoGraph":
        g = cls()
        for node in data.nodes:
            g._g.add_node(node.id, kind=node.kind, label=node.label, **node.attrs)
        for edge in data.edges:
            g._g.add_edge(edge.source, edge.target, kind=edge.kind, **edge.attrs)
        return g

    def hash(self) -> str:
        """SHA-256 over a canonical (sorted) serialization of nodes and edges.

        Used by ``kairo-temporal`` to detect unchanged snapshots and to skip
        writing a new history node when the graph is byte-identical to its
        parent.
        """
        data = self.to_data()
        payload = {
            "nodes": sorted(
                (n.model_dump() for n in data.nodes),
                key=lambda n: n["id"],
            ),
            "edges": sorted(
                (e.model_dump() for e in data.edges),
                key=lambda e: (e["source"], e["target"], e["kind"]),
            ),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
