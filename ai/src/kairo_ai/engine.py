"""Engine wiring — lazy ``EngineRunner`` with Ollama think-fn.

The `/ask` route calls :func:`get_engine` once on first query, which
builds an :class:`EngineRunner` backed by:

* ``kairo_embeddings.hash-stub`` for fast, zero-dep node embeddings.
* ``kairo_agents.providers.ollama.ollama_think_fn`` for every agent.
* A small default seed corpus so queries have *something* to graph
  before Phase 7f ingestion (GitHub URL, CSV attachment).

If Ollama is not reachable at call time, :func:`get_engine` returns
``None`` and the route falls back to the stub responder in
:mod:`kairo_ai.app`.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

from kairo_core import Document
from kairo_embeddings.registry import get_provider
from kairo_agents.providers.ollama import (
    OllamaConfig,
    check_ollama,
    list_ollama_models,
    ollama_think_fn,
)
from kairo_agents.providers.tavily import tavily_client_from_env
from kairo_rag.engine_runner import EngineRunner
from kairo_rag.web_enrichment import WebEnricher, WebEnricherConfig


# Kairo's own self-description — ensures an empty-attachment demo query
# still gets a coherent, grounded answer. Replaced by user-supplied docs
# once Phase 7f ingestion lands.
_SEED_DOCS: List[Document] = [
    Document(
        id="kairo_overview",
        text=(
            "Kairo is a retrieval-native temporal RAG engine. It represents "
            "knowledge as a 3-layer graph: a document layer for high-level "
            "structure, a semantic layer for concepts and meaning, and a "
            "detail layer for fine-grained facts and evidence. Queries are "
            "answered by KNN search plus a bounded graph walk that may cross "
            "layers, with every result snapshotted so earlier states can be "
            "revisited."
        ),
        metadata={"source": "seed", "topic": "overview"},
    ),
    Document(
        id="kairo_layers",
        text=(
            "The document layer captures files, sections, and headings. The "
            "semantic layer extracts concepts and connects related ideas "
            "through co-occurrence and shared-concept edges. The detail layer "
            "holds sentences, numbers, and attributes. Inter-layer edges "
            "(contains, grounds, refines) link a high-level node to its "
            "supporting detail so traversal can zoom from overview to evidence."
        ),
        metadata={"source": "seed", "topic": "layers"},
    ),
    Document(
        id="kairo_agents",
        text=(
            "Kairo's supervisor is deterministic: it analyzes graph topology "
            "and produces a reproducible plan assigning roles to agents. "
            "Roles include explorer (surveys the graph), analyzer (examines "
            "clusters), synthesizer (writes the final answer), and validator "
            "(checks grounding). Creativity lives only inside each agent's "
            "LLM call, never in the routing itself."
        ),
        metadata={"source": "seed", "topic": "agents"},
    ),
    Document(
        id="kairo_snapshots",
        text=(
            "Every query produces a snapshot: the 3-layer graph hash plus the "
            "embedding store. Snapshots form a DAG so branching and rollback "
            "are schema-native. A session cache lets repeated queries reuse "
            "a prior snapshot without rebuilding, keeping follow-up "
            "interactions fast and deterministic."
        ),
        metadata={"source": "seed", "topic": "snapshots"},
    ),
    Document(
        id="kairo_traces",
        text=(
            "Traversal traces record each agent's movement through the graph: "
            "which nodes were visited, which inter-layer edges were crossed, "
            "and which clusters were formed. The 3D visualization replays "
            "these steps so the user sees exactly how the answer was reached "
            "— evidence with provenance, not a black box."
        ),
        metadata={"source": "seed", "topic": "traces"},
    ),
]


_lock = threading.Lock()
_runner: Optional[EngineRunner] = None
_model_used: Optional[str] = None
_web_enabled: bool = False


def _pick_ollama_model() -> Optional[str]:
    """Choose an available local Ollama model, preferring small llamas.

    Respects ``KAIRO_OLLAMA_MODEL`` if set and present locally.
    """
    env = os.environ.get("KAIRO_OLLAMA_MODEL")
    models = list_ollama_models()
    if not models:
        return None
    if env and env in models:
        return env
    # Preference order for demo: small, fast, llama-like.
    for pref in ("llama3.2:latest", "llama3.2:3b", "llama3.2:1b",
                 "qwen2.5:1.5b", "qwen2.5:3b"):
        if pref in models:
            return pref
    return models[0]


def get_engine() -> Optional[EngineRunner]:
    """Return a process-wide EngineRunner or None if Ollama is down.

    Thread-safe and lazy: the runner is built on first successful call
    and reused afterwards.  If ``TAVILY_API_KEY`` is set in the
    environment, a :class:`WebEnricher` is attached so queries asking
    for suggestions / alternatives / recent trends get web-sourced
    detail nodes injected into the 3-layer graph.
    """
    global _runner, _model_used, _web_enabled

    if _runner is not None:
        return _runner

    with _lock:
        if _runner is not None:
            return _runner

        if not check_ollama():
            return None

        model = _pick_ollama_model()
        if model is None:
            return None

        provider = get_provider("hash-stub", dim=128)
        think_fn = ollama_think_fn(OllamaConfig(model=model))

        # Optional web enrichment — only built when TAVILY_API_KEY is set.
        enricher: Optional[WebEnricher] = None
        tavily = tavily_client_from_env()
        if tavily is not None:
            enricher = WebEnricher(
                tavily,
                config=WebEnricherConfig(max_hits=4, concepts_to_link=3),
            )
            _web_enabled = True

        _runner = EngineRunner(
            embedding_provider=provider,
            think_fn=think_fn,
            available_agents=3,
            top_k=5,
            walk_depth=2,
            verbose=False,
            web_enricher=enricher,
        )
        _model_used = model
        return _runner


def seed_documents() -> List[Document]:
    """Documents used when the user has not attached anything yet."""
    return list(_SEED_DOCS)


def current_model() -> Optional[str]:
    """Tag describing the Ollama model in use; used for UI display."""
    return _model_used


def web_enabled() -> bool:
    """True when Tavily web enrichment is wired into the runner."""
    return _web_enabled
