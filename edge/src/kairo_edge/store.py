from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from kairo_core import Document, Evidence, QueryRequest, QueryResponse, RetrievalResult


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not doc_tokens:
        return 0.0
    n = len(doc_tokens)
    return sum(doc_tokens.count(t) / n for t in query_tokens)


class EdgeStore:
    """Tiny pure-Python retrieval store for mobile/ARM targets.

    Keeps documents in a JSON file and performs lexical term-frequency scoring.
    No numpy, no pandas, no binary wheels. Good enough for on-device search
    over a few thousand short documents.
    """

    def __init__(self, store_path: Path | str = ".kairo_edge.json"):
        self.store_path = Path(store_path)
        self._docs: Dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for raw in data.get("documents", []):
                doc = Document.model_validate(raw)
                self._docs[doc.id] = doc

    def _save(self) -> None:
        payload = {"documents": [doc.model_dump() for doc in self._docs.values()]}
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            self._docs[doc.id] = doc
        self._save()

    def query(self, request: QueryRequest) -> QueryResponse:
        query_tokens = _tokenize(request.query)
        scored: List[Tuple[float, Document]] = []
        for doc in self._docs.values():
            doc_tokens = _tokenize(doc.text)
            s = _score(query_tokens, doc_tokens)
            if s > 0:
                scored.append((s, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        evidences = [
            Evidence(document_id=d.id, text=d.text, score=s, metadata=d.metadata)
            for s, d in scored[: request.top_k]
        ]
        context = "\n\n".join(ev.text for ev in evidences)
        answer = context[:512] if context else "No relevant evidence found."
        return QueryResponse(answer=answer, evidences=evidences, steps=[])
