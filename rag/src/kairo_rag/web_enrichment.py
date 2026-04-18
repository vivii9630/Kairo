"""Web enrichment — inject live Tavily hits into a :class:`RAGResult`.

Purpose
-------
When a user query asks for comparisons, suggestions, or latest trends,
local documents alone rarely answer well.  :class:`WebEnricher`:

1. Decides whether a query *needs* web context (cheap keyword heuristic).
2. Picks a query string biased toward the top-scoring semantic concepts
   from the traversal so web results match what the graph actually surfaced.
3. Calls :class:`kairo_agents.providers.tavily.TavilyClient`.
4. Appends each hit as a **detail-layer node** (``kind="web"``) to
   ``rag_result.graph_data`` with new :class:`InterLayerEdge` objects
   linking the top semantic concepts → web nodes.
5. Updates ``trace.visited_nodes["detail"]``, ``trace.crossed_edges``,
   and ``trace.steps`` so the supervisor notices web nodes and the 3D
   viz lights them up.
6. Appends a low-score :class:`Evidence` per hit so agents can cite them.

The enricher never raises on Tavily failures — it logs into the trace
step metadata and returns the untouched result so engine output stays
deterministic-on-success, graceful-on-failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kairo_core import (
    Evidence,
    GraphNode,
    InterLayerEdge,
    TraversalStep,
)

if TYPE_CHECKING:  # avoid import cycle at runtime
    from kairo_agents.providers.tavily import TavilyClient, TavilyResult

    from .temporal_rag import RAGResult


# Keywords that flag a query as web-enrichable.  Kept tight: the detail
# layer should stay local-first; web context only kicks in when the user
# explicitly asks for external perspective.
_ENRICH_KEYWORDS = frozenset(
    {
        "suggest",
        "suggestion",
        "suggestions",
        "improve",
        "improvement",
        "improvements",
        "better",
        "best",
        "compare",
        "comparison",
        "alternative",
        "alternatives",
        "latest",
        "recent",
        "trend",
        "trends",
        "recommend",
        "recommendation",
        "benchmark",
        "state-of-the-art",
        "sota",
        "modern",
        "industry",
    }
)


@dataclass
class WebEnricherConfig:
    """Knobs for the enricher — safe to tune per deployment.

    Parameters
    ----------
    max_hits : int
        Hard cap on injected web nodes per query.
    concepts_to_link : int
        How many top-scoring semantic concepts get linked to each web
        node via an inter-layer edge.
    web_agent_id : str
        Identifier used in the :class:`TraversalStep` entries.
    always_on : bool
        If True, skip the keyword heuristic and enrich every query (the
        UI can still scope this via ``TAVILY_API_KEY`` env presence).
    """

    max_hits: int = 4
    concepts_to_link: int = 3
    web_agent_id: str = "web_researcher"
    always_on: bool = False


class WebEnricher:
    """Stateless enricher that mutates a :class:`RAGResult` in place."""

    def __init__(
        self,
        client: "TavilyClient",
        *,
        config: Optional[WebEnricherConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or WebEnricherConfig()

    # -- gating -------------------------------------------------------------

    def should_enrich(self, query: str) -> bool:
        """Cheap heuristic: does this query benefit from fresh web context?"""
        if self.config.always_on:
            return True
        q = (query or "").lower()
        if not q:
            return False
        # Word-boundary-ish scan so "improvement" also matches "improve".
        return any(kw in q for kw in _ENRICH_KEYWORDS)

    # -- main entrypoint ----------------------------------------------------

    def enrich(
        self,
        query: str,
        rag_result: "RAGResult",
    ) -> "RAGResult":
        """Inject web hits into *rag_result* and return the same object.

        On any failure, the result is left untouched and a ``note`` step
        is appended to ``rag_result.trace.steps`` describing the error —
        the calling :class:`EngineRunner` stays oblivious either way.
        """
        if not self.should_enrich(query):
            return rag_result

        try:
            hits = self.client.search(query)
        except Exception as exc:  # noqa: BLE001 — fail soft, record to trace
            self._record_failure(rag_result, f"{type(exc).__name__}: {exc}")
            return rag_result

        if not hits:
            self._record_failure(rag_result, "no Tavily results")
            return rag_result

        hits = hits[: self.config.max_hits]
        top_concepts = self._top_semantic_concepts(rag_result)

        self._inject(rag_result, query, hits, top_concepts)
        return rag_result

    # -- internals ----------------------------------------------------------

    def _top_semantic_concepts(
        self, rag_result: "RAGResult"
    ) -> List[str]:
        """Pick semantic node ids to link to new web nodes.

        Preference order:
        1. Top-score evidences with ``layer == "semantic"``.
        2. Otherwise the first N visited semantic nodes from the trace.
        """
        evidences = getattr(rag_result, "evidences", None) or []
        ranked = [
            ev.document_id
            for ev in evidences
            if ev.metadata.get("layer") == "semantic"
        ]
        if ranked:
            return ranked[: self.config.concepts_to_link]

        visited_sem = (
            rag_result.trace.visited_nodes.get("semantic", []) if rag_result.trace else []
        )
        return list(visited_sem[: self.config.concepts_to_link])

    def _inject(
        self,
        rag_result: "RAGResult",
        query: str,
        hits: List["TavilyResult"],
        concept_ids: List[str],
    ) -> None:
        graph_data = rag_result.graph_data
        trace = rag_result.trace

        new_node_ids: List[str] = []
        new_inter_edges: List[InterLayerEdge] = []

        for i, hit in enumerate(hits):
            nid = f"web_{i:02d}"
            title = hit.title or hit.url or f"Web result {i}"
            snippet = (hit.content or "")[:500]

            node = GraphNode(
                id=nid,
                kind="web",
                label=title[:80],
                attrs={
                    "url": hit.url,
                    "full_text": snippet,
                    "score": hit.score,
                    "source": "tavily",
                },
            )
            if graph_data is not None:
                graph_data.detail.nodes.append(node)
            new_node_ids.append(nid)

            # Link this hit to each top semantic concept — the edge makes
            # the 3D viz show a diagonal line from concept down to the web
            # node, which is the visual signal we want on screen.
            for concept_id in concept_ids:
                edge = InterLayerEdge(
                    source=concept_id,
                    target=nid,
                    source_layer="semantic",
                    target_layer="detail",
                    kind="enriched_by",
                    attrs={"web_url": hit.url},
                )
                if graph_data is not None:
                    graph_data.inter_layer_edges.append(edge)
                new_inter_edges.append(edge)

            # Append as evidence so analyzer/synthesizer agents can cite.
            rag_result.evidences.append(
                Evidence(
                    document_id=nid,
                    text=f"[web:{title}] {snippet}",
                    score=max(0.0, min(1.0, hit.score or 0.0)),
                    metadata={
                        "layer": "detail",
                        "kind": "web",
                        "url": hit.url,
                        "title": title,
                    },
                )
            )

        # Update trace so the supervisor sees web nodes in the detail layer
        # and the 3D viz highlights them.
        visited = trace.visited_nodes.setdefault("detail", [])
        for nid in new_node_ids:
            if nid not in visited:
                visited.append(nid)

        trace.crossed_edges.extend(new_inter_edges)

        # Add traversal steps so the UI can render a web-research phase
        # between KNN seeding and synthesis.
        base_step = len(trace.steps)
        for offset, (nid, hit) in enumerate(zip(new_node_ids, hits)):
            trace.steps.append(
                TraversalStep(
                    agent_id=self.config.web_agent_id,
                    node_id=nid,
                    layer="detail",
                    action="cross_layer" if offset == 0 else "visit",
                    score=float(hit.score or 0.0),
                    metadata={
                        "url": hit.url,
                        "title": hit.title,
                        "step_index": base_step + offset,
                    },
                )
            )

        # Single cluster so the supervisor can route a web_researcher agent.
        if new_node_ids:
            trace.clusters["web_hits"] = list(new_node_ids)

    def _record_failure(self, rag_result: "RAGResult", reason: str) -> None:
        """Leave a breadcrumb in the trace when enrichment is skipped."""
        if rag_result.trace is None:
            return
        # Represented as a zero-score 'score' action on a virtual node so
        # the TraversalTrace schema stays well-typed without new kinds.
        rag_result.trace.steps.append(
            TraversalStep(
                agent_id=self.config.web_agent_id,
                node_id="__web_enrichment_skipped__",
                layer="detail",
                action="score",
                score=0.0,
                metadata={"reason": reason, "skipped": True},
            )
        )
