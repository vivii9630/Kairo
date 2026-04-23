"""Phase 13b3 — augment a KairoGraph with LLM-extracted edges.

Unlike the other builders which *build* a graph from scratch, this one
*augments* an existing graph (typically the output of
:class:`DocumentGraphBuilder` + community detection) with edges and
concept nodes reported by an :class:`ExtractorProvider`.

Keep-separate rationale: LLM extraction is slow, opt-in, and may fail
per-document. Isolating it from the structural builders means a user
can always fall back to the fast deterministic graph, and a bad
extractor run doesn't corrupt the guaranteed-correct structural edges.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from kairo_core import Document, ExtractionResult

from ..graph import KairoGraph
from ..provenance import edge_attrs
from ..extractors.protocol import ExtractorProvider


class LLMExtractedGraphBuilder:
    """Runs an ExtractorProvider over documents and merges results.

    Parameters
    ----------
    extractor :
        Any object conforming to :class:`ExtractorProvider`.
    create_missing_concepts :
        If True (default), edges referencing node ids that don't exist
        yet cause those ids to be created as ``concept`` nodes.
        Otherwise the edge is dropped. False is safer for code/tabular
        corpora where spurious concept nodes would pollute the schema.
    relation_whitelist :
        If provided, only edges whose ``relation`` is in this set are
        kept. Useful to constrain a model that likes to invent
        free-form relation names.
    collision_policy :
        What to do when the graph already has an edge between the
        subject and object. ``"skip"`` (default) keeps the existing
        edge — important because KairoGraph wraps ``nx.DiGraph`` which
        allows only one edge per directed pair, so naive overwrites
        would silently clobber structural/similarity edges produced by
        earlier builders. ``"overwrite"`` replaces the existing edge
        with the extracted one (use only when you know extraction is
        the higher-authority source).
    """

    name = "llm_extracted"

    def __init__(
        self,
        extractor: ExtractorProvider,
        *,
        create_missing_concepts: bool = True,
        relation_whitelist: Optional[Iterable[str]] = None,
        collision_policy: str = "skip",
    ) -> None:
        if collision_policy not in {"skip", "overwrite"}:
            raise ValueError(
                f"collision_policy must be 'skip' or 'overwrite', "
                f"got {collision_policy!r}"
            )
        self.extractor = extractor
        self.create_missing_concepts = create_missing_concepts
        self.relation_whitelist = (
            set(relation_whitelist) if relation_whitelist is not None else None
        )
        self.collision_policy = collision_policy

    def augment(
        self,
        graph: KairoGraph,
        documents: Sequence[Document],
        *,
        context_docs: Sequence[Document] = (),
    ) -> List[ExtractionResult]:
        """Run the extractor over each document, merge edges into *graph*.

        Returns the per-document results so callers can inspect errors,
        report stats, or persist the extraction audit log.
        """
        results: List[ExtractionResult] = []
        for doc in documents:
            result = self.extractor.extract(doc, context=context_docs)
            results.append(result)
            if result.error:
                continue
            self._merge(graph, result)
        return results

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _merge(self, graph: KairoGraph, result: ExtractionResult) -> None:
        nx_g = graph.nx_graph
        for concept in result.concepts:
            if concept.id in nx_g:
                continue
            graph.add_node(
                concept.id,
                kind="concept",
                label=concept.label or concept.id,
                description=concept.description,
                source_doc_id=concept.source_doc_id or "",
                extractor=result.extractor,
                confidence=float(concept.confidence),
            )

        for edge in result.edges:
            if (
                self.relation_whitelist is not None
                and edge.relation not in self.relation_whitelist
            ):
                continue
            subj = edge.subject
            obj = edge.object
            if subj not in nx_g:
                if not self.create_missing_concepts:
                    continue
                graph.add_node(
                    subj,
                    kind="concept",
                    label=subj,
                    source_doc_id=edge.source_doc_id or "",
                    extractor=result.extractor,
                )
            if obj not in nx_g:
                if not self.create_missing_concepts:
                    continue
                graph.add_node(
                    obj,
                    kind="concept",
                    label=obj,
                    source_doc_id=edge.source_doc_id or "",
                    extractor=result.extractor,
                )
            if nx_g.has_edge(subj, obj) and self.collision_policy == "skip":
                continue
            graph.add_edge(
                subj,
                obj,
                kind=edge.relation,
                **edge_attrs(
                    provenance=edge.provenance,
                    confidence=edge.confidence,
                    rationale=edge.rationale,
                    extractor=result.extractor,
                    source_doc_id=edge.source_doc_id or "",
                ),
            )
