from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from kairo_core import (
    AgentStep,
    Document,
    Evidence,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kairo-retrieval requires rank_bm25. Install with: pip install rank_bm25"
    ) from exc

from kairo_embeddings import EmbeddingProvider


RRF_K = 60
CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATE_POOL = 50
DEFAULT_HOP_DECAY = 0.5

# Duck-typed adjacency: given a doc id, return iterable of (neighbor_id, weight).
# Weight is typically cosine similarity in [0, 1] from the similarity edges
# built in DocumentGraphBuilder. Callers construct this from any backing store
# (KairoGraph, Kuzu, plain dict) so retrieval doesn't depend on kairo-graph.
GraphNeighbors = Callable[[str], Iterable[Tuple[str, float]]]


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


class LocalPipeline:
    """Local retrieval + simple RAG scaffold with persistence.

    Retrieval is BM25 over tokens, optionally fused with semantic cosine
    via Reciprocal Rank Fusion (RRF, k=60). When no ``embedding_provider``
    is supplied the pipeline stays BM25-only — the hash-stub embedding is
    intentionally NOT used as a similarity proxy (it carries no semantic
    signal).
    """

    def __init__(
        self,
        store_path: Path | str = ".kairo_store.json",
        *,
        embedding_provider: Optional[EmbeddingProvider] = None,
        graph_neighbors: Optional[GraphNeighbors] = None,
        hop_decay: float = DEFAULT_HOP_DECAY,
    ):
        self.store_path = Path(store_path)
        self._docs: Dict[str, Document] = {}
        self._embedding_provider = embedding_provider
        self._graph_neighbors = graph_neighbors
        self._hop_decay = hop_decay
        self._doc_ids: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._doc_vecs: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for raw in data.get("documents", []):
                doc = Document.model_validate(raw)
                self._docs[doc.id] = doc
        self._rebuild_indexes()

    def _save(self) -> None:
        payload = {"documents": [doc.model_dump() for doc in self._docs.values()]}
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _rebuild_indexes(self) -> None:
        self._doc_ids = list(self._docs.keys())
        if not self._doc_ids:
            self._doc_tokens = []
            self._bm25 = None
            self._doc_vecs = None
            return
        self._doc_tokens = [_tokenize(self._docs[doc_id].text) for doc_id in self._doc_ids]
        self._bm25 = BM25Okapi(self._doc_tokens)
        if self._embedding_provider is not None:
            texts = [self._docs[doc_id].text for doc_id in self._doc_ids]
            vecs = np.asarray(self._embedding_provider.embed(texts), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._doc_vecs = vecs / norms
        else:
            self._doc_vecs = None

    def ingest_documents(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            self._docs[doc.id] = doc
        self._save()
        self._rebuild_indexes()

    def ingest_paths(self, paths: Iterable[Path]) -> None:
        from kairo_ingest import ingest_any

        for path in paths:
            docs = ingest_any(Path(path))
            self.ingest_documents(docs)

    def _filtered_indexes(self, filters: Dict[str, object]) -> List[int]:
        if not filters:
            return list(range(len(self._doc_ids)))
        keep: List[int] = []
        for idx, doc_id in enumerate(self._doc_ids):
            meta = self._docs[doc_id].metadata
            if all(meta.get(k) == v for k, v in filters.items()):
                keep.append(idx)
        return keep

    def _bm25_ranking(
        self,
        query_tokens: List[str],
        candidate_idx: List[int],
        pool_size: int,
    ) -> List[str]:
        if self._bm25 is None or not candidate_idx or not query_tokens:
            return []
        all_scores = self._bm25.get_scores(query_tokens)
        subset = [(i, float(all_scores[i])) for i in candidate_idx]
        # Drop zero/negative-score docs IF at least one positive exists.
        # Zero-score docs in the ranking would otherwise eat RRF rank slots
        # and let the graph walk boost arbitrary unmatched neighbors. The
        # "no positives at all" branch keeps the single-doc / all-stopword
        # case from returning empty for any query.
        positives = [(i, s) for i, s in subset if s > 0]
        chosen = positives if positives else subset
        chosen.sort(key=lambda x: x[1], reverse=True)
        chosen = chosen[:pool_size]
        return [self._doc_ids[i] for i, _ in chosen]

    def _cosine_ranking(
        self,
        query_text: str,
        candidate_idx: List[int],
        pool_size: int,
    ) -> List[str]:
        if (
            self._embedding_provider is None
            or self._doc_vecs is None
            or not candidate_idx
        ):
            return []
        q_vec = np.asarray(
            self._embedding_provider.embed([query_text])[0], dtype=np.float32
        )
        norm = float(np.linalg.norm(q_vec))
        if norm == 0.0:
            return []
        q_vec = q_vec / norm
        sims = self._doc_vecs @ q_vec
        subset = [(i, float(sims[i])) for i in candidate_idx]
        subset.sort(key=lambda x: x[1], reverse=True)
        subset = subset[:pool_size]
        return [self._doc_ids[i] for i, _ in subset]

    @staticmethod
    def _rrf_fuse(rankings: List[List[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank_idx, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_idx + 1)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _graph_walk_boost(
        self,
        seeds: List[Tuple[str, float]],
        candidate_filter: Optional[set],
    ) -> List[Tuple[str, float]]:
        """Apply 1-hop additive boost from graph neighbors.

        For every seed (doc_id, fused_score), pull weighted neighbors and
        increment their score by ``seed_score * edge_weight * hop_decay``.
        Neighbors not in the filtered candidate set are dropped.
        """
        if self._graph_neighbors is None or not seeds:
            return seeds
        scores: Dict[str, float] = {doc_id: s for doc_id, s in seeds}
        for seed_id, seed_score in seeds:
            try:
                neighbors = self._graph_neighbors(seed_id)
            except Exception:  # noqa: BLE001 - adjacency callable is user-supplied
                continue
            for nb_id, weight in neighbors:
                if nb_id == seed_id or nb_id not in self._docs:
                    continue
                if candidate_filter is not None and nb_id not in candidate_filter:
                    continue
                boost = seed_score * float(weight) * self._hop_decay
                if boost <= 0:
                    continue
                scores[nb_id] = scores.get(nb_id, 0.0) + boost
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _retrieve(self, request: QueryRequest) -> RetrievalResult:
        query_tokens = _tokenize(request.query)
        candidate_idx = self._filtered_indexes(request.filters or {})
        pool = max(request.top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATE_POOL)

        rankings: List[List[str]] = []
        bm25_rank = self._bm25_ranking(query_tokens, candidate_idx, pool)
        if bm25_rank:
            rankings.append(bm25_rank)
        cosine_rank = self._cosine_ranking(request.query, candidate_idx, pool)
        if cosine_rank:
            rankings.append(cosine_rank)

        if not rankings:
            return RetrievalResult(evidences=[], plan="no retrievers produced results")

        fused = self._rrf_fuse(rankings)
        if self._graph_neighbors is not None:
            seed_count = max(request.top_k, 10)
            candidate_set = (
                {self._doc_ids[i] for i in candidate_idx}
                if request.filters
                else None
            )
            fused = self._graph_walk_boost(fused[:seed_count], candidate_set)
        fused = fused[: request.top_k]

        evidences: List[Evidence] = [
            Evidence(
                document_id=doc_id,
                text=self._docs[doc_id].text,
                score=score,
                metadata=self._docs[doc_id].metadata,
            )
            for doc_id, score in fused
        ]

        if bm25_rank and cosine_rank:
            plan = f"Hybrid BM25 + cosine via RRF (k={RRF_K})"
        elif cosine_rank:
            plan = "Cosine semantic retrieval (BM25 had no matches)"
        else:
            plan = "BM25 lexical retrieval (no embedding provider configured)"
        if self._graph_neighbors is not None:
            plan += f" + 1-hop graph walk (decay={self._hop_decay})"
        return RetrievalResult(evidences=evidences, plan=plan)

    def query(self, request: QueryRequest) -> QueryResponse:
        retrieval = self._retrieve(request)
        context = "\n\n".join(ev.text for ev in retrieval.evidences)
        answer = context[:1024] if context else "No relevant evidence found."
        steps = [
            AgentStep(name="planner", input=request.query, output=retrieval.plan or "planned"),
            AgentStep(
                name="retriever",
                input=request.query,
                output=f"{len(retrieval.evidences)} evidences",
            ),
            AgentStep(name="responder", input=context, output=answer),
        ]
        return QueryResponse(answer=answer, evidences=retrieval.evidences, steps=steps)
