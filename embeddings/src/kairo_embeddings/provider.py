from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns a list of texts into a (len(texts), dim) float matrix.

    Implementations live under ``kairo_embeddings.providers`` — one file
    per backend. Register new ones via :func:`register_provider` or by
    adding to the builtin registry in ``registry.py``.
    """

    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray: ...
