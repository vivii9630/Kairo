from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from ..credentials import Credentials


class HashStubProvider:
    """Deterministic, zero-dependency embedding stub.

    Tokenizes input text and hashes each token into a fixed-dim float
    vector bucket with a signed increment. Not semantically meaningful,
    but byte-stable across runs — perfect for CI, plumbing tests, and
    letting new users try Kairo without downloading a model.
    """

    name = "hash-stub"

    def __init__(self, *, dim: int = 64, credentials: Optional[Credentials] = None):
        del credentials  # accepted for interface uniformity; unused
        self.dim = int(dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                out[i, bucket] += sign
        return out


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]
