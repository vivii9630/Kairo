"""Per-layer embedding store wrapper for 3-layer graphs.

Wraps three :class:`EmbeddingStore` instances (one per layer) and provides
helpers for batch embedding of all nodes in a :class:`LayeredKairoGraph`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from kairo_embeddings.store import EmbeddingStore
from kairo_embeddings.provider import EmbeddingProvider
from kairo_graph import LayeredKairoGraph

LAYER_NAMES = ("document", "semantic", "detail")


class LayerEmbeddingStore:
    """Three ``EmbeddingStore`` instances, one per graph layer."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._stores: Dict[str, EmbeddingStore] = {
            name: EmbeddingStore(dim) for name in LAYER_NAMES
        }

    def get(self, layer: str) -> Optional[EmbeddingStore]:
        return self._stores.get(layer)

    def __getitem__(self, layer: str) -> EmbeddingStore:
        return self._stores[layer]

    def total_vectors(self) -> int:
        return sum(len(s) for s in self._stores.values())

    # -- batch embedding ----------------------------------------------------

    def embed_graph(
        self,
        graph: LayeredKairoGraph,
        provider: EmbeddingProvider,
    ) -> None:
        """Embed every node in *graph* into the corresponding layer store.

        Uses the node's ``label`` (falling back to ``text_preview`` or
        ``full_text`` attrs) as the text to embed.
        """
        for layer_name in LAYER_NAMES:
            layer_graph = graph.layer(layer_name)
            ids: List[str] = []
            texts: List[str] = []
            for nid, data in layer_graph.nx_graph.nodes(data=True):
                text = (
                    data.get("label", "")
                    or data.get("text_preview", "")
                    or data.get("full_text", "")
                    or str(nid)
                )
                ids.append(str(nid))
                texts.append(text)
            if not texts:
                continue
            vectors = provider.embed(texts)
            store = self._stores[layer_name]
            for node_id, vec in zip(ids, vectors):
                store.add(node_id, vec)

    # -- serialization (for snapshot persistence) ---------------------------

    def to_dict(self) -> Dict[str, Dict]:
        """Serialize to a dict of {layer: {ids, matrix_list, dim}}."""
        result = {}
        for name, store in self._stores.items():
            result[name] = {
                "dim": store.dim,
                "ids": list(store._ids),
                "matrix": store._matrix.tolist(),
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "LayerEmbeddingStore":
        """Reconstruct from serialized dict."""
        first = next(iter(data.values()))
        dim = first["dim"]
        obj = cls(dim)
        for name in LAYER_NAMES:
            if name in data:
                entry = data[name]
                store = obj._stores[name]
                store._ids = list(entry["ids"])
                store._matrix = np.array(entry["matrix"], dtype=np.float32)
        return obj
