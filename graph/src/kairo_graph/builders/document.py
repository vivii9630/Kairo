from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from kairo_core import Document

from ..communities import add_concept_nodes, detect_communities
from ..graph import KairoGraph
from ..provenance import PROVENANCE_INFERRED, edge_attrs
from ._similarity import top_k_similar_pairs

try:
    from kairo_embeddings import EmbeddingProvider
except ImportError:  # pragma: no cover — embeddings is always installed alongside graph in practice
    EmbeddingProvider = None  # type: ignore[assignment]


class DocumentGraphBuilder:
    """Document-to-graph builder.

    Every document becomes one node. When an ``embedding_provider`` is
    supplied, pairwise cosine similarity is computed and each document
    gets up to ``similarity_k`` ``similar-to`` edges to its nearest
    neighbors above ``similarity_threshold``. Without a provider, the
    builder emits a node-only graph (backward-compatible behavior).
    """

    name = "document"

    def __init__(
        self,
        *,
        embedding_provider: Optional["EmbeddingProvider"] = None,
        similarity_k: int = 5,
        similarity_threshold: float = 0.3,
        detect_communities: bool = False,
        community_resolution: float = 1.0,
        community_min_size: int = 2,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.similarity_k = similarity_k
        self.similarity_threshold = similarity_threshold
        self.detect_communities = detect_communities
        self.community_resolution = community_resolution
        self.community_min_size = community_min_size

    def build(self, documents: Iterable[Document]) -> KairoGraph:
        graph = KairoGraph()
        docs = list(documents)
        for doc in docs:
            attrs = {f"meta_{k}": v for k, v in doc.metadata.items()}
            graph.add_node(
                doc.id,
                kind="document",
                label=doc.id,
                text_preview=doc.text[:200],
                **attrs,
            )

        if self.embedding_provider is not None and len(docs) >= 2:
            texts = [doc.text for doc in docs]
            ids = [doc.id for doc in docs]
            vecs = np.asarray(
                self.embedding_provider.embed(texts), dtype=np.float32
            )
            for source, target, score in top_k_similar_pairs(
                ids,
                vecs,
                k=self.similarity_k,
                threshold=self.similarity_threshold,
            ):
                graph.add_edge(
                    source,
                    target,
                    kind="similar-to",
                    weight=score,
                    **edge_attrs(provenance=PROVENANCE_INFERRED, confidence=score),
                )

            if self.detect_communities:
                communities = detect_communities(
                    graph,
                    resolution=self.community_resolution,
                    min_community_size=self.community_min_size,
                )
                if communities:
                    add_concept_nodes(graph, communities)

        return graph
