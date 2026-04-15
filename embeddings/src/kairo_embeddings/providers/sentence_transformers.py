from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..credentials import Credentials

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SentenceTransformersProvider requires the optional "
        "'sentence-transformers' extra. Install with: "
        'pip install -e "./embeddings[sentence-transformers]"'
    ) from exc


class SentenceTransformersProvider:
    """Local real-model embedding provider backed by sentence-transformers.

    Default model is ``all-MiniLM-L6-v2`` (384-dim, small, fast, CPU-friendly).
    Credentials are accepted for interface uniformity but unused — the model
    runs entirely locally.
    """

    name = "sentence-transformers"

    def __init__(
        self,
        *,
        model: str = "all-MiniLM-L6-v2",
        credentials: Optional[Credentials] = None,
    ):
        del credentials  # unused — local model
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)
