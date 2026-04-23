"""Phase 13d — streaming synthesis for the /ask/stream SSE route.

Provides :func:`stream_rag_answer`, a generator that:

1. Retrieves top-k evidences from a LocalPipeline over the caller's
   documents.
2. Builds a synthesis prompt from those evidences.
3. Calls ``ollama_stream_chat`` and yields Server-Sent Events as
   Ollama emits tokens.
4. After the model signals completion, runs citation verification
   against the collected answer and emits a terminal ``done`` event
   carrying the report + evidence ids.

This path is deliberately lighter than the multi-agent EngineRunner
used by /ask: no planner, no layered graph, no web enrichment. The
trade-off is intentional — true token streaming through the full
multi-agent pipeline requires an engine refactor that isolates the
synthesizer as the only streamable step. That lands in a later phase.
Until then /ask keeps the full quality path; /ask/stream gives the
progressive-render UX.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional, Sequence

from kairo_agents.providers.ollama import OllamaConfig, ollama_stream_chat
from kairo_core import Document, Evidence, QueryRequest
from kairo_rag import CitationVerifier
from kairo_retrieval import LocalPipeline


_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's question using ONLY "
    "the evidence provided below. If the evidence does not contain the "
    "answer, say so plainly — do not invent facts. Keep the response "
    "concise (3-8 sentences)."
)


def sse(event_type: str, **payload: object) -> str:
    """Format a single Server-Sent Event frame (``data: {json}\\n\\n``)."""
    body = {"type": event_type, **payload}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def parse_sse_payload(frame: str) -> dict:
    """Reverse of :func:`sse` — extract the JSON payload from an
    ``data: ...\\n\\n`` frame. Raises ``ValueError`` on malformed input.
    """
    prefix = "data: "
    if not frame.startswith(prefix):
        raise ValueError("not an SSE data frame")
    body = frame[len(prefix):].rstrip("\n")
    return json.loads(body)


def _build_synthesis_messages(
    query: str,
    evidences: Sequence[Evidence],
) -> List[dict]:
    context_blocks: List[str] = []
    for idx, ev in enumerate(evidences, start=1):
        context_blocks.append(f"[{idx}] id={ev.document_id}\n{ev.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no evidence)"
    user = f"Question: {query}\n\nEvidence:\n{context}\n\nAnswer:"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def stream_rag_answer(
    *,
    query: str,
    documents: Sequence[Document],
    thread_id: str,
    ollama_config: Optional[OllamaConfig] = None,
    top_k: int = 5,
    pipeline: Optional[LocalPipeline] = None,
) -> Iterator[str]:
    """Yield SSE frames: retrieve → stream synthesis → done."""
    # 1. Retrieval pass.
    if pipeline is None:
        pipeline = LocalPipeline()
    pipeline.ingest_documents(list(documents))
    retrieval = pipeline._retrieve(QueryRequest(query=query, top_k=top_k))
    evidences = retrieval.evidences

    yield sse(
        "retrieve",
        plan=retrieval.plan or "",
        evidence_count=len(evidences),
        cited_node_ids=[ev.document_id for ev in evidences],
    )

    # 2. Stream synthesis.
    messages = _build_synthesis_messages(query, evidences)
    accumulated: List[str] = []
    stream_error: Optional[str] = None
    for kind, content in ollama_stream_chat(messages, config=ollama_config):
        if kind == "token":
            accumulated.append(content)
            yield sse("token", content=content)
        elif kind == "error":
            stream_error = content
            yield sse("error", message=content)
            break
        elif kind == "done":
            # content here is the fully reassembled answer — prefer it
            # over our accumulated copy if they disagree (they shouldn't).
            if content and not accumulated:
                accumulated.append(content)
            break

    if stream_error is not None:
        yield sse("done", thread_id=thread_id, answer="", aborted=True)
        return

    answer = "".join(accumulated)

    # 3. Citation verification over the streamed answer.
    report = CitationVerifier().verify(answer, evidences)

    yield sse(
        "done",
        thread_id=thread_id,
        answer=answer,
        cited_node_ids=[ev.document_id for ev in evidences],
        citation_report=report.model_dump(),
        aborted=False,
    )
