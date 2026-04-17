"""TemporalRAG engine — the core orchestrator for 3-layer retrieval.

Flow
----
1. Receive query + documents (or a pre-built layered graph).
2. Build the 3-layer graph via :class:`LayeredGraphBuilder`.
3. Embed all nodes via :class:`EmbeddingProvider`.
4. KNN search in each layer to find seed neighborhoods.
5. BFS graph-walk from seeds, expanding intra- and cross-layer.
6. Cluster visited nodes for supervisor routing.
7. Collect evidence from visited detail-layer nodes.
8. Snapshot the graph+embeddings at time *t*.
9. On follow-up queries in the same session, check session store first —
   reconstruct from saved state if a prior snapshot exists.

The engine is **deterministic**: given the same graph state, embeddings,
and query, the traversal and evidence selection are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from kairo_core import (
    Document,
    Evidence,
    InterLayerEdge,
    LayeredGraphData,
    TraversalTrace,
)
from kairo_embeddings.provider import EmbeddingProvider
from kairo_graph import LayeredKairoGraph
from kairo_graph.builders.layered_document import DocumentLayeredBuilder

from .layered_store import LayerEmbeddingStore
from .session import GraphSnapshot, SessionStore
from .traversal import cluster_visited_nodes, graph_walk, knn_search


@dataclass
class RAGResult:
    """Output of a single TemporalRAG query."""

    answer_context: str
    evidences: List[Evidence]
    trace: TraversalTrace
    snapshot_id: str
    from_cache: bool = False
    clusters: Dict[str, List[str]] = field(default_factory=dict)
    graph_data: Optional[LayeredGraphData] = None


class TemporalRAGEngine:
    """Orchestrates 3-layer graph construction, retrieval, and snapshotting.

    Parameters
    ----------
    provider : EmbeddingProvider
        The embedding backend (hash-stub for tests, sentence-transformers
        or Ollama for production).
    top_k : int
        Number of nearest neighbors per layer in KNN search.
    walk_depth : int
        BFS expansion depth from seed nodes.
    max_nodes_per_layer : int
        Stop expanding a layer after this many visited nodes.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        top_k: int = 5,
        walk_depth: int = 2,
        max_nodes_per_layer: int = 30,
    ) -> None:
        self.provider = provider
        self.top_k = top_k
        self.walk_depth = walk_depth
        self.max_nodes_per_layer = max_nodes_per_layer
        self._sessions: Dict[str, SessionStore] = {}

    # -- session management -------------------------------------------------

    def get_session(self, session_id: str) -> SessionStore:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionStore(session_id)
        return self._sessions[session_id]

    # -- main query entrypoint ----------------------------------------------

    def query(
        self,
        query: str,
        documents: List[Document],
        *,
        session_id: str = "default",
        reuse_snapshot: bool = True,
    ) -> RAGResult:
        """Run a full 3-layer retrieval for *query*.

        If *reuse_snapshot* is True and the session has a prior snapshot
        for the same query, the graph is reconstructed from cache instead
        of rebuilding.
        """
        session = self.get_session(session_id)

        # -- Check session cache --------------------------------------------
        if reuse_snapshot:
            cached = session.find_by_query(query)
            if cached is not None:
                return self._query_from_snapshot(query, cached)

        # -- Build fresh graph + embeddings ---------------------------------
        builder = DocumentLayeredBuilder()
        graph = builder.build(documents)

        layer_emb = LayerEmbeddingStore(self.provider.dim)
        layer_emb.embed_graph(graph, self.provider)

        return self._search_and_snapshot(query, graph, layer_emb, session)

    def query_from_graph(
        self,
        query: str,
        graph: LayeredKairoGraph,
        *,
        session_id: str = "default",
        layer_emb: Optional[LayerEmbeddingStore] = None,
    ) -> RAGResult:
        """Run retrieval on a pre-built layered graph."""
        session = self.get_session(session_id)

        if layer_emb is None:
            layer_emb = LayerEmbeddingStore(self.provider.dim)
            layer_emb.embed_graph(graph, self.provider)

        return self._search_and_snapshot(query, graph, layer_emb, session)

    # -- internals ----------------------------------------------------------

    def _search_and_snapshot(
        self,
        query: str,
        graph: LayeredKairoGraph,
        layer_emb: LayerEmbeddingStore,
        session: SessionStore,
    ) -> RAGResult:
        """KNN → walk → cluster → evidence → snapshot."""
        # 1. Embed query
        query_vec = self.provider.embed([query])[0]

        # 2. KNN per layer
        knn_hits = knn_search(
            query_vec, layer_emb, top_k=self.top_k
        )

        # 3. Seed nodes for graph walk
        seeds: Dict[str, List[str]] = {
            layer: [nid for nid, _score in hits]
            for layer, hits in knn_hits.items()
        }

        # 4. Graph walk (BFS expansion)
        trace = graph_walk(
            graph,
            seeds,
            max_depth=self.walk_depth,
            max_nodes_per_layer=self.max_nodes_per_layer,
        )
        trace.query = query

        # 5. Cluster visited nodes
        clusters = cluster_visited_nodes(
            graph, trace.visited_nodes
        )
        trace.clusters = clusters

        # 6. Collect evidence from detail layer
        evidences = self._collect_evidence(
            graph, trace, knn_hits, query_vec, layer_emb
        )

        # 7. Build answer context from evidences
        answer_context = "\n\n".join(
            f"[{ev.document_id}] (score={ev.score:.3f}): {ev.text}"
            for ev in evidences
        )

        # 8. Snapshot
        snap = session.save_snapshot(graph, layer_emb, query)

        return RAGResult(
            answer_context=answer_context,
            evidences=evidences,
            trace=trace,
            snapshot_id=snap.id,
            from_cache=False,
            clusters=clusters,
            graph_data=graph.to_data(),
        )

    def _query_from_snapshot(
        self, query: str, snapshot: GraphSnapshot
    ) -> RAGResult:
        """Reconstruct graph from snapshot and re-run search."""
        graph = snapshot.reconstruct_graph()
        layer_emb = snapshot.reconstruct_embeddings()

        query_vec = self.provider.embed([query])[0]

        knn_hits = knn_search(
            query_vec, layer_emb, top_k=self.top_k
        )

        seeds = {
            layer: [nid for nid, _ in hits]
            for layer, hits in knn_hits.items()
        }

        trace = graph_walk(
            graph, seeds,
            max_depth=self.walk_depth,
            max_nodes_per_layer=self.max_nodes_per_layer,
        )
        trace.query = query

        clusters = cluster_visited_nodes(graph, trace.visited_nodes)
        trace.clusters = clusters

        evidences = self._collect_evidence(
            graph, trace, knn_hits, query_vec, layer_emb
        )

        answer_context = "\n\n".join(
            f"[{ev.document_id}] (score={ev.score:.3f}): {ev.text}"
            for ev in evidences
        )

        return RAGResult(
            answer_context=answer_context,
            evidences=evidences,
            trace=trace,
            snapshot_id=snapshot.id,
            from_cache=True,
            clusters=clusters,
            graph_data=graph.to_data(),
        )

    def _collect_evidence(
        self,
        graph: LayeredKairoGraph,
        trace: TraversalTrace,
        knn_hits: Dict[str, list],
        query_vec: np.ndarray,
        layer_emb: LayerEmbeddingStore,
    ) -> List[Evidence]:
        """Gather evidence from visited detail-layer nodes, scored by KNN."""
        # Build a score map from KNN hits across all layers
        score_map: Dict[str, float] = {}
        for layer_hits in knn_hits.values():
            for nid, score in layer_hits:
                score_map[nid] = max(score_map.get(nid, 0.0), score)

        evidences: List[Evidence] = []
        detail_graph = graph.detail

        # Collect from visited detail nodes
        detail_visited = trace.visited_nodes.get("detail", [])
        for nid in detail_visited:
            if nid not in detail_graph.nx_graph:
                continue
            data = detail_graph.nx_graph.nodes[nid]
            text = data.get("full_text", data.get("label", ""))
            if not text:
                continue

            # Score: use KNN score if available, else compute on the fly
            score = score_map.get(nid)
            if score is None:
                store = layer_emb.get("detail")
                if store and nid in store:
                    idx = store._ids.index(nid)
                    vec = store._matrix[idx]
                    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
                    score = float(vec @ q)
                else:
                    score = 0.0

            evidences.append(Evidence(
                document_id=nid,
                text=text,
                score=score,
                metadata={"layer": "detail", "kind": data.get("kind", "")},
            ))

        # Also include high-scoring semantic nodes as supporting evidence
        sem_visited = trace.visited_nodes.get("semantic", [])
        for nid in sem_visited:
            score = score_map.get(nid, 0.0)
            if score < 0.3:
                continue
            data = graph.semantic.nx_graph.nodes.get(nid, {})
            label = data.get("label", "")
            if label:
                evidences.append(Evidence(
                    document_id=nid,
                    text=f"[concept] {label}",
                    score=score,
                    metadata={"layer": "semantic", "kind": "concept"},
                ))

        # Sort by score descending
        evidences.sort(key=lambda e: e.score, reverse=True)
        return evidences
