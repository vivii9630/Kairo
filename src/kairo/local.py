from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import Document, Evidence, QueryRequest, RetrievalResult, QueryResponse, AgentStep
from .ingest import ingest_any


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def _tf(term: str, tokens: List[str]) -> float:
    return tokens.count(term) / max(len(tokens), 1)


def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    # Simple term-frequency similarity (placeholder for vector/graph retrieval)
    return sum(_tf(t, doc_tokens) for t in query_tokens)


class LocalPipeline:
    """Minimal local retrieval + simple RAG scaffold with persistence."""

    def __init__(self, store_path: Path | str = ".kairo_store.json"):
        self.store_path = Path(store_path)
        self._docs: Dict[str, Document] = {}
        self._load()

    # Persistence helpers
    def _load(self) -> None:
        if self.store_path.exists():
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for raw in data.get("documents", []):
                doc = Document.model_validate(raw)
                self._docs[doc.id] = doc

    def _save(self) -> None:
        payload = {"documents": [doc.model_dump() for doc in self._docs.values()]}
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Public API
    def ingest_documents(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            self._docs[doc.id] = doc
        self._save()

    def ingest_paths(self, paths: Iterable[Path]) -> None:
        for path in paths:
            docs = ingest_any(Path(path))
            self.ingest_documents(docs)

    def _retrieve(self, request: QueryRequest) -> RetrievalResult:
        query_tokens = _tokenize(request.query)
        scored: List[Tuple[float, Document]] = []

        for doc in self._docs.values():
            if request.filters:
                # naive filter check
                mismatch = False
                for k, v in request.filters.items():
                    if doc.metadata.get(k) != v:
                        mismatch = True
                        break
                if mismatch:
                    continue

            doc_tokens = _tokenize(doc.text)
            score = _score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        evidences: List[Evidence] = []
        for score, doc in scored[: request.top_k]:
            evidences.append(
                Evidence(document_id=doc.id, text=doc.text, score=score, metadata=doc.metadata)
            )
        plan = "Lexical term-frequency retrieval (placeholder for hybrid/vector/graph)."
        return RetrievalResult(evidences=evidences, plan=plan)

    def query(self, request: QueryRequest) -> QueryResponse:
        retrieval = self._retrieve(request)
        # Simple concatenation as a placeholder for summarization
        context = "\n\n".join(ev.text for ev in retrieval.evidences)
        answer = context[:1024] if context else "No relevant evidence found."
        steps = [
            AgentStep(name="planner", input=request.query, output=retrieval.plan or "planned"),
            AgentStep(name="retriever", input=request.query, output=f"{len(retrieval.evidences)} evidences"),
            AgentStep(name="responder", input=context, output=answer),
        ]
        return QueryResponse(answer=answer, evidences=retrieval.evidences, steps=steps)

