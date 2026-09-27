"""Phase 13d — SSE helpers + stream_rag_answer integration."""

from __future__ import annotations

import io
import json
from typing import Iterable, Iterator

import pytest

from kairo_ai.streaming import parse_sse_payload, sse, stream_rag_answer
from kairo_core import Document


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def test_sse_roundtrips_through_parse() -> None:
    frame = sse("token", content="hi", extra=42)
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    parsed = parse_sse_payload(frame)
    assert parsed == {"type": "token", "content": "hi", "extra": 42}


def test_parse_sse_payload_rejects_non_data_frame() -> None:
    with pytest.raises(ValueError):
        parse_sse_payload("event: foo\n\n")


def test_parse_sse_payload_handles_unicode() -> None:
    frame = sse("token", content="héllo 🌍")
    assert parse_sse_payload(frame)["content"] == "héllo 🌍"


# ---------------------------------------------------------------------------
# stream_rag_answer integration (Ollama mocked)
# ---------------------------------------------------------------------------

class _FakeOllamaResponse:
    def __init__(self, chunks: Iterable[dict]) -> None:
        body = "".join(json.dumps(c) + "\n" for c in chunks)
        self._stream = io.BytesIO(body.encode("utf-8"))

    def __enter__(self): return self
    def __exit__(self, *exc): return None
    def __iter__(self):
        for raw in self._stream:
            yield raw


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch, chunks) -> None:
    from kairo_agents.providers import ollama as ollama_mod
    monkeypatch.setattr(
        ollama_mod.urllib.request, "urlopen",
        lambda _req, timeout=0: _FakeOllamaResponse(chunks),
    )


def _frames_to_events(frames: Iterator[str]) -> list:
    events = []
    for frame in frames:
        events.append(parse_sse_payload(frame))
    return events


def test_stream_rag_answer_emits_retrieve_token_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_ollama(monkeypatch, [
        {"message": {"content": "Cats "}, "done": False},
        {"message": {"content": "purr."}, "done": True},
    ])

    docs = [
        Document(id="d1", text="Cats purr when they are content."),
        Document(id="d2", text="Dogs bark at strangers."),
    ]
    frames = list(stream_rag_answer(
        query="why do cats purr",
        documents=docs,
        thread_id="t1",
    ))
    events = _frames_to_events(iter(frames))

    assert events[0]["type"] == "retrieve"
    assert events[0]["evidence_count"] >= 1
    assert "d1" in events[0]["cited_node_ids"]

    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["Cats ", "purr."]

    done = events[-1]
    assert done["type"] == "done"
    assert done["thread_id"] == "t1"
    assert done["answer"] == "Cats purr."
    assert done["aborted"] is False
    assert "citation_report" in done
    # The citation report shape comes from CitationReport.model_dump().
    assert "overall_support" in done["citation_report"]


def test_stream_rag_answer_surfaces_ollama_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairo_agents.providers import ollama as ollama_mod

    def boom(_req, timeout=0):
        raise ollama_mod.urllib.error.URLError("conn refused")
    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", boom)

    frames = list(stream_rag_answer(
        query="anything",
        documents=[Document(id="d1", text="text")],
        thread_id="t1",
    ))
    events = _frames_to_events(iter(frames))
    kinds = [e["type"] for e in events]
    assert "error" in kinds
    # Final frame should still be a done event so the client's
    # state-machine can cleanly close.
    assert events[-1]["type"] == "done"
    assert events[-1]["aborted"] is True
    assert events[-1]["answer"] == ""
