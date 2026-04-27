from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kairo_core import Document, Evidence, QueryRequest, QueryResponse

try:  # Rust-backed fast path. Pure-Python fallback stays importable.
    from kairo_edge_py import Document as _RustDocument
    from kairo_edge_py import EdgeStore as _RustEdgeStore
except Exception:  # pragma: no cover - depends on optional native wheel
    _RustDocument = None
    _RustEdgeStore = None


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not doc_tokens:
        return 0.0
    n = len(doc_tokens)
    return sum(doc_tokens.count(t) / n for t in query_tokens)


class _PythonEdgeStore:
    """Tiny pure-Python retrieval store for mobile/ARM targets.

    Keeps documents in a JSON file and performs lexical term-frequency scoring.
    No numpy, no pandas, no binary wheels. Good enough for on-device search
    over a few thousand short documents.
    """

    def __init__(self, store_path: Path | str | None = ".kairo_edge.json"):
        self.store_path = None if store_path is None else Path(store_path)
        self._docs: Dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path is not None and self.store_path.exists():
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for raw in data.get("documents", []):
                doc = Document.model_validate(raw)
                self._docs[doc.id] = doc

    def _save(self) -> None:
        if self.store_path is None:
            return
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


def _metadata_from_rust(value: Any) -> Dict[str, Any]:
    """Best-effort conversion from PyO3 metadata dict values.

    The Rust bridge currently returns JSON values as strings for simple
    Python interoperability. Parse values back to JSON when possible so
    numeric/bool metadata remains useful to callers.
    """
    out: Dict[str, Any] = {}
    for key, raw in dict(value).items():
        if isinstance(raw, str):
            try:
                out[key] = json.loads(raw)
            except json.JSONDecodeError:
                out[key] = raw
        else:
            out[key] = raw
    return out


def _rust_document(doc: Document):
    if _RustDocument is None:
        raise RuntimeError("kairo_edge_py is not installed")
    return _RustDocument(id=doc.id, text=doc.text, metadata=doc.metadata)


def _rust_evidence(ev: Any) -> Evidence:
    return Evidence(
        document_id=ev.document_id,
        text=ev.text,
        score=ev.score,
        metadata=_metadata_from_rust(ev.metadata),
    )


class EdgeStore:
    """Public edge retrieval store.

    Uses the Rust/PyO3 backend when ``kairo_edge_py`` is installed and
    falls back to the legacy pure-Python implementation otherwise. The
    public API remains the Python ``kairo_core`` shape:
    ``add(Iterable[Document])`` and ``query(QueryRequest)``.
    """

    def __init__(
        self,
        store_path: Path | str | None = ".kairo_edge.json",
        *,
        prefer_rust: bool = True,
    ):
        self.store_path = None if store_path is None else Path(store_path)
        self._uses_rust = bool(prefer_rust and _RustEdgeStore is not None)
        if self._uses_rust:
            native_path = None if self.store_path is None else str(self.store_path)
            self._inner = _RustEdgeStore(native_path)
        else:
            self._inner = _PythonEdgeStore(self.store_path)

    @property
    def backend(self) -> str:
        return "rust" if self._uses_rust else "python"

    def add(self, docs: Iterable[Document]) -> None:
        docs_list = list(docs)
        if self._uses_rust:
            self._inner.add([_rust_document(doc) for doc in docs_list])
            return
        self._inner.add(docs_list)

    def query(self, request: QueryRequest) -> QueryResponse:
        if self._uses_rust:
            response = self._inner.query(request.query, top_k=request.top_k)
            return QueryResponse(
                answer=response.answer,
                evidences=[_rust_evidence(ev) for ev in response.evidences],
                steps=[],
            )
        return self._inner.query(request)

    def __len__(self) -> int:
        if self._uses_rust:
            return len(self._inner)
        return len(self._inner._docs)
