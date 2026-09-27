"""Phase 13d — ollama_stream_chat generator.

Mocks urllib.request.urlopen so tests don't require a live Ollama.
"""

from __future__ import annotations

import io
import json
from typing import Iterable

import pytest

from kairo_agents.providers import ollama as ollama_mod
from kairo_agents.providers.ollama import ollama_stream_chat


class _FakeResponse:
    """Stand-in for the urlopen context manager response.

    Ollama sends newline-delimited JSON when ``stream=true``; this
    response replays a scripted list of chunks so we can assert the
    generator's emitted events deterministically.
    """

    def __init__(self, chunks: Iterable[dict]) -> None:
        body = "".join(json.dumps(c) + "\n" for c in chunks)
        self._stream = io.BytesIO(body.encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def __iter__(self):
        for raw in self._stream:
            yield raw


def _install_fake(monkeypatch: pytest.MonkeyPatch, chunks):
    def fake_urlopen(_req, timeout=0):  # noqa: ARG001
        return _FakeResponse(chunks)
    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", fake_urlopen)


def test_stream_emits_tokens_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": "!"}, "done": True},
    ])
    events = list(ollama_stream_chat([{"role": "user", "content": "hi"}]))
    assert events[0] == ("token", "Hello")
    assert events[1] == ("token", " world")
    assert events[2] == ("token", "!")
    assert events[-1] == ("done", "Hello world!")


def test_stream_skips_malformed_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama sometimes interleaves non-JSON keepalives; those must
    not abort the stream or emit spurious events."""
    class _Mixed:
        def __init__(self) -> None:
            body = (
                json.dumps({"message": {"content": "a"}}) + "\n"
                + "<garbage>\n"
                + json.dumps({"message": {"content": "b"}, "done": True}) + "\n"
            )
            self._stream = io.BytesIO(body.encode("utf-8"))

        def __enter__(self): return self
        def __exit__(self, *exc): return None
        def __iter__(self):
            for raw in self._stream:
                yield raw

    monkeypatch.setattr(
        ollama_mod.urllib.request, "urlopen",
        lambda _req, timeout=0: _Mixed(),
    )
    events = list(ollama_stream_chat([{"role": "user", "content": "hi"}]))
    tokens = [c for k, c in events if k == "token"]
    assert tokens == ["a", "b"]
    assert events[-1] == ("done", "ab")


def test_stream_surfaces_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_req, timeout=0):
        raise ollama_mod.urllib.error.URLError("connection refused")
    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", boom)

    events = list(ollama_stream_chat([{"role": "user", "content": "hi"}]))
    assert len(events) == 1
    kind, content = events[0]
    assert kind == "error"
    assert "connection refused" in content


def test_stream_skips_empty_token_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [
        {"message": {"content": ""}, "done": False},
        {"message": {"content": "real"}, "done": False},
        {"done": True},  # final chunk with no content
    ])
    events = list(ollama_stream_chat([{"role": "user", "content": "hi"}]))
    tokens = [c for k, c in events if k == "token"]
    assert tokens == ["real"]
    assert events[-1] == ("done", "real")
