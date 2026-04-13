from __future__ import annotations

from typing import Iterable

from kairo_core import Document

from ..graph import KairoGraph


class DocumentGraphBuilder:
    """Minimal document-to-graph builder.

    Every document becomes one node. Richer edges (chunk links, entity
    mentions, semantic similarity) land in later phases without breaking
    this builder's contract.
    """

    name = "document"

    def build(self, documents: Iterable[Document]) -> KairoGraph:
        graph = KairoGraph()
        for doc in documents:
            attrs = {f"meta_{k}": v for k, v in doc.metadata.items()}
            graph.add_node(
                doc.id,
                kind="document",
                label=doc.id,
                text_preview=doc.text[:200],
                **attrs,
            )
        return graph
