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
from kairo_core.models import (
    GraphData,
    GraphEdge,
    GraphNode,
    InterLayerEdge,
    LayeredGraphData,
    TraversalStep,
    TraversalTrace,
)

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
        graph_data, traversal_trace = _stub_layered_graph(req.query)
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
            graph_data=graph_data,
            traversal_trace=traversal_trace,
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


def _stub_layered_graph(query: str) -> tuple[LayeredGraphData, TraversalTrace]:
    """Synthetic 3-layer graph so Phase 7d 3D viz renders pre-engine-wiring.

    Replaced in Phase 7e by the real :class:`TemporalRAGEngine` output.
    """
    doc_nodes = [
        GraphNode(id=f"D{i}", kind="document", label=f"Doc {i}") for i in range(4)
    ]
    doc_edges = [
        GraphEdge(source=f"D{i}", target=f"D{i + 1}", kind="sequence")
        for i in range(3)
    ]

    sem_nodes = [
        GraphNode(id=f"S{i}", kind="concept", label=lbl)
        for i, lbl in enumerate(
            ["data", "model", "metrics", "findings", "decision", "risk"]
        )
    ]
    sem_edges = [
        GraphEdge(source="S0", target="S1", kind="co_occurs"),
        GraphEdge(source="S1", target="S2", kind="co_occurs"),
        GraphEdge(source="S2", target="S3", kind="co_occurs"),
        GraphEdge(source="S3", target="S4", kind="co_occurs"),
        GraphEdge(source="S4", target="S5", kind="shared_concept"),
        GraphEdge(source="S0", target="S3", kind="shared_concept"),
    ]

    det_labels = [
        "n=1024", "lr=1e-4", "acc=0.92", "f1=0.88",
        "infra:gpu", "latency<100ms", "owner:team-a", "bug-#42",
        "rollback plan", "stakeholder:legal",
    ]
    det_nodes = [
        GraphNode(id=f"X{i}", kind="fact", label=lbl)
        for i, lbl in enumerate(det_labels)
    ]
    det_edges = [
        GraphEdge(source=f"X{i}", target=f"X{i + 1}", kind="follows")
        for i in range(len(det_nodes) - 1)
    ] + [GraphEdge(source="X0", target="X3", kind="near")]

    inter_edges = [
        InterLayerEdge(
            source="D0", target="S0",
            source_layer="document", target_layer="semantic", kind="contains",
        ),
        InterLayerEdge(
            source="D1", target="S1",
            source_layer="document", target_layer="semantic", kind="contains",
        ),
        InterLayerEdge(
            source="D2", target="S3",
            source_layer="document", target_layer="semantic", kind="contains",
        ),
        InterLayerEdge(
            source="D3", target="S4",
            source_layer="document", target_layer="semantic", kind="contains",
        ),
        InterLayerEdge(
            source="S0", target="X0",
            source_layer="semantic", target_layer="detail", kind="grounds",
        ),
        InterLayerEdge(
            source="S1", target="X1",
            source_layer="semantic", target_layer="detail", kind="grounds",
        ),
        InterLayerEdge(
            source="S2", target="X2",
            source_layer="semantic", target_layer="detail", kind="grounds",
        ),
        InterLayerEdge(
            source="S2", target="X3",
            source_layer="semantic", target_layer="detail", kind="grounds",
        ),
        InterLayerEdge(
            source="S3", target="X8",
            source_layer="semantic", target_layer="detail", kind="grounds",
        ),
    ]

    graph = LayeredGraphData(
        document=GraphData(nodes=doc_nodes, edges=doc_edges),
        semantic=GraphData(nodes=sem_nodes, edges=sem_edges),
        detail=GraphData(nodes=det_nodes, edges=det_edges),
        inter_layer_edges=inter_edges,
    )

    visited = {
        "document": ["D0", "D1", "D2"],
        "semantic": ["S0", "S1", "S2", "S3"],
        "detail": ["X2", "X3", "X8"],
    }
    steps = [
        TraversalStep(agent_id="explorer", node_id="D0", layer="document", action="visit", score=0.9),
        TraversalStep(agent_id="explorer", node_id="D1", layer="document", action="expand", score=0.8),
        TraversalStep(agent_id="explorer", node_id="S1", layer="semantic", action="cross_layer", score=0.85),
        TraversalStep(agent_id="analyzer", node_id="S2", layer="semantic", action="visit", score=0.78),
        TraversalStep(agent_id="analyzer", node_id="S3", layer="semantic", action="expand", score=0.72),
        TraversalStep(agent_id="analyzer", node_id="X2", layer="detail", action="cross_layer", score=0.7),
        TraversalStep(agent_id="synthesizer", node_id="X3", layer="detail", action="visit", score=0.65),
        TraversalStep(agent_id="synthesizer", node_id="X8", layer="detail", action="cluster", score=0.6),
    ]
    crossed = [
        e for e in inter_edges
        if e.source in visited[e.source_layer] and e.target in visited[e.target_layer]
    ]
    traversal = TraversalTrace(
        query=query,
        steps=steps,
        visited_nodes=visited,
        crossed_edges=crossed,
        clusters={"c0": ["S1", "S2", "X2", "X3"]},
    )
    return graph, traversal


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
