from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np


class EmbeddingStore:
    """In-memory embedding store with cosine top-k and numpy + JSON persistence.

    Keys are node_ids; values are L2-normalized float32 vectors stacked into
    a single matrix for fast similarity. The interface is intentionally tiny
    so a FAISS / LanceDB / pgvector backend can swap in later without
    changing callers.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._ids: List[str] = []
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._ids

    def add(self, node_id: str, vector: np.ndarray) -> None:
        v = _ensure_vector(vector, self.dim)
        v = _normalize(v)
        if node_id in self._ids:
            idx = self._ids.index(node_id)
            self._matrix[idx] = v
        else:
            self._ids.append(node_id)
            self._matrix = np.vstack([self._matrix, v[None, :]])

    def add_many(self, items: Iterable[Tuple[str, np.ndarray]]) -> None:
        for node_id, vec in items:
            self.add(node_id, vec)

    def similar(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        q = _normalize(_ensure_vector(query, self.dim))
        scores = self._matrix @ q
        k = min(k, len(self._ids))
        top = np.argsort(-scores)[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in top]

    def save(
        self,
        matrix_path: Union[Path, str],
        index_path: Union[Path, str],
    ) -> None:
        m_path = Path(matrix_path)
        i_path = Path(index_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        i_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(m_path, self._matrix)
        i_path.write_text(
            json.dumps({"dim": self.dim, "ids": self._ids}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        matrix_path: Union[Path, str],
        index_path: Union[Path, str],
    ) -> "EmbeddingStore":
        m_path = Path(matrix_path)
        i_path = Path(index_path)
        meta = json.loads(i_path.read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]))
        store._ids = list(meta["ids"])
        matrix = np.load(m_path).astype(np.float32)
        if matrix.shape != (len(store._ids), store.dim):
            raise ValueError(
                f"matrix shape {matrix.shape} does not match index "
                f"({len(store._ids)}, {store.dim})"
            )
        store._matrix = matrix
        return store


def _ensure_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    if v.shape[0] != dim:
        raise ValueError(f"vector dim {v.shape[0]} != store dim {dim}")
    return v


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm
