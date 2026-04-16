"""FastAPI app factory for Kairo AI.

Three route groups today:

* ``GET  /plugins`` — flattens :func:`kairo_connectors.list_manifests`
  into the shape the plugin picker consumes. The registry already
  gates optional deps via ``ImportError``, so whichever connectors are
  actually importable in the current environment show up here.
* ``POST /ask``     — stubbed answer path. Returns realistic shapes
  (thread id, trace steps, citations) so the frontend can be built
  against stable contracts before the Phase 7 orchestrator lands.
* ``*    /threads`` — in-memory thread storage via :class:`ThreadStore`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from kairo_connectors import list_manifests
from kairo_connectors.plugin import PluginManifest

from .models import (
    AskRequest,
    AskResponse,
    Citation,
    Message,
    PluginSummary,
    Thread,
    ThreadCreateRequest,
    TraceStep,
)
from .state import ThreadStore


DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def create_app(*, cors_origins: List[str] | None = None) -> FastAPI:
    app = FastAPI(
        title="Kairo AI",
        version="0.1.0",
        description="Chat-native backend orchestrating connectors, engine, and agents.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or DEFAULT_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = ThreadStore()
    app.state.thread_store = store

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/plugins", response_model=List[PluginSummary])
    def get_plugins() -> List[PluginSummary]:
        return [_flatten_manifest(m) for m in list_manifests()]

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")

        thread = (
            store.get(req.thread_id)
            if req.thread_id
            else store.create(plugin=req.plugin)
        )
        if thread is None:
            raise HTTPException(status_code=404, detail=f"unknown thread: {req.thread_id}")

        now = datetime.now(timezone.utc)
        store.append_message(
            thread.id,
            Message(role="user", content=req.query, created_at=now),
        )

        answer_text = _stub_answer(req.query, req.plugin)
        trace = _stub_trace(req.query, req.plugin)
        store.append_message(
            thread.id,
            Message(role="assistant", content=answer_text, created_at=datetime.now(timezone.utc)),
        )

        return AskResponse(
            thread_id=thread.id,
            answer=answer_text,
            citations=[],
            trace=trace,
            stubbed=True,
        )

    @app.get("/threads", response_model=List[Thread])
    def list_threads() -> List[Thread]:
        return store.list()

    @app.post("/threads", response_model=Thread)
    def create_thread(req: ThreadCreateRequest) -> Thread:
        return store.create(title=req.title, plugin=req.plugin)

    @app.get("/threads/{thread_id}", response_model=Thread)
    def get_thread(thread_id: str) -> Thread:
        thread = store.get(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
        return thread

    return app


def _flatten_manifest(m: PluginManifest) -> PluginSummary:
    auth = m.auth
    return PluginSummary(
        name=m.name,
        label=m.label,
        description=m.description,
        uri_example=m.uri_example,
        icon=m.icon,
        tags=list(m.tags),
        auth_required=auth is not None,
        auth_methods=[cls.__name__ for cls in (auth.methods if auth else [])],
        auth_scopes=list(auth.scopes) if auth else [],
        auth_instructions=auth.instructions if auth else "",
    )


def _stub_answer(query: str, plugin: str | None) -> str:
    target = f" against the {plugin!r} connector" if plugin else ""
    return (
        f"[stubbed] Kairo AI will route this query{target} through the "
        "TemporalRAG orchestrator once Phase 7 lands. You asked: "
        f"{query.strip()!r}."
    )


def _stub_trace(query: str, plugin: str | None) -> List[TraceStep]:
    return [
        TraceStep(step=1, kind="plan", summary=f"Parse query: {query.strip()[:80]}"),
        TraceStep(
            step=2,
            kind="retrieve",
            summary=(
                f"Would fetch via '{plugin}' connector"
                if plugin
                else "No plugin selected — would route to default retriever"
            ),
        ),
        TraceStep(step=3, kind="synthesize", summary="Compose answer from evidence (stubbed)"),
    ]


app = create_app()
