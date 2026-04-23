"""FastAPI app factory for Kairo AI.

Three route groups today:

* ``GET  /plugins`` — flattens :func:`kairo_connectors.list_manifests`
  into the shape the plugin picker consumes. The registry already
  gates optional deps via ``ImportError``, so whichever connectors are
  actually importable in the current environment show up here.
* ``POST /ask``     — routes the query through the real TemporalRAG
  engine when Ollama is reachable; otherwise falls back to a synthetic
  3-layer graph stub so the UI remains functional.
* ``*    /threads`` — in-memory thread storage via :class:`ThreadStore`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from kairo_analytics import (
    AnalyticsRunner,
    build_plans,
    has_chart_intent,
    infer_columns,
)
from kairo_connectors import list_manifests
from kairo_connectors.plugin import PluginManifest
from kairo_core import AnalyticsPlan, ChartSpec, ForecastSpec
from kairo_core.models import (
    GraphData,
    GraphEdge,
    GraphNode,
    InterLayerEdge,
    LayeredGraphData,
    TraversalStep,
    TraversalTrace,
)

from .engine import current_model, get_engine, seed_documents
from .streaming import parse_sse_payload, stream_rag_answer
from .ingest import (
    GitHubFetchError,
    IngestStore,
    TabularFetchError,
    ingest_csv,
    ingest_github,
    ingest_xlsx,
)
from .models import (
    AskRequest,
    AskResponse,
    Citation,
    IngestResponse,
    IngestUrlRequest,
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

    ingest_store = IngestStore()
    app.state.ingest_store = ingest_store

    analytics_runner = AnalyticsRunner()
    app.state.analytics_runner = analytics_runner

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/plugins", response_model=List[PluginSummary])
    def get_plugins() -> List[PluginSummary]:
        return [_flatten_manifest(m) for m in list_manifests()]

    # Citation verifier reused across requests — stateless, cheap.
    from kairo_rag import CitationVerifier
    citation_verifier = CitationVerifier()

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

        # Pick the document set the engine will graph:
        #   ingest_id → the user-uploaded docs (GitHub / CSV / XLSX)
        #   otherwise → the Kairo self-description seed corpus.
        documents = seed_documents()
        ingest_label: Optional[str] = None
        ingest_obj = None
        if req.ingest_id:
            ingest_obj = ingest_store.get(req.ingest_id)
            if ingest_obj is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown ingest_id: {req.ingest_id}",
                )
            documents = ingest_obj.documents
            ingest_label = ingest_obj.label

        # Session id per (thread, ingest) so the snapshot cache doesn't
        # alias a GitHub-graph with a seed-graph in the same thread.
        session_id = thread.id
        if req.ingest_id:
            session_id = f"{thread.id}:{req.ingest_id}"

        runner = get_engine()
        cited_node_ids: List[str] = []
        citation_report = None
        if runner is None:
            # Ollama unreachable — return the stub so the UI still works.
            answer_text, trace, graph_data, traversal_trace, stubbed = (
                _stub_response(req.query, req.plugin)
            )
        else:
            try:
                result = runner.run(
                    req.query, documents, session_id=session_id
                )
                answer_text = result.answer
                trace = _trace_from_result(result)
                graph_data = result.graph_data
                traversal_trace = result.rag_result.trace
                cited_node_ids = _cited_ids(result)
                citation_report = citation_verifier.verify(
                    answer_text, result.rag_result.evidences
                )
                stubbed = False
            except Exception as exc:  # noqa: BLE001 — surface engine errors in-UI
                answer_text, trace, graph_data, traversal_trace, stubbed = (
                    _stub_response(
                        req.query,
                        req.plugin,
                        note=f"Engine error: {type(exc).__name__}: {exc}",
                    )
                )

        charts, forecasts, analytics_steps = _run_analytics(
            req.query, ingest_obj, analytics_runner, starting_step=len(trace) + 1
        )
        if analytics_steps:
            trace.extend(analytics_steps)

        store.append_message(
            thread.id,
            Message(role="assistant", content=answer_text, created_at=datetime.now(timezone.utc)),
        )

        return AskResponse(
            thread_id=thread.id,
            answer=answer_text,
            citations=[],
            trace=trace,
            stubbed=stubbed,
            graph_data=graph_data,
            traversal_trace=traversal_trace,
            cited_node_ids=cited_node_ids,
            charts=charts,
            forecasts=forecasts,
            citation_report=citation_report,
        )

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Phase 13d — token-streaming variant of /ask.

        Runs a lightweight retrieve → stream-synthesis path instead of
        the full multi-agent pipeline, so we can pipe Ollama tokens
        straight to the client as they arrive. Trade-off is intentional:
        /ask keeps the full-quality multi-agent answer, /ask/stream
        gives progressive-render UX. True multi-agent streaming waits
        on an engine refactor.
        """
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")

        thread = (
            store.get(req.thread_id)
            if req.thread_id
            else store.create(plugin=req.plugin)
        )
        if thread is None:
            raise HTTPException(
                status_code=404, detail=f"unknown thread: {req.thread_id}"
            )

        now = datetime.now(timezone.utc)
        store.append_message(
            thread.id,
            Message(role="user", content=req.query, created_at=now),
        )

        documents = seed_documents()
        if req.ingest_id:
            ingest_obj = ingest_store.get(req.ingest_id)
            if ingest_obj is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown ingest_id: {req.ingest_id}",
                )
            documents = ingest_obj.documents

        model = current_model()
        cfg = None
        if model:
            from kairo_agents.providers.ollama import OllamaConfig
            cfg = OllamaConfig(model=model)

        captured_answer: List[str] = []

        def event_generator():
            for frame in stream_rag_answer(
                query=req.query,
                documents=documents,
                thread_id=thread.id,
                ollama_config=cfg,
            ):
                # Capture the final answer so we can record the
                # assistant turn once the stream completes. The client
                # has already seen the tokens — this is bookkeeping.
                if frame.startswith("data: "):
                    try:
                        payload = parse_sse_payload(frame)
                        if payload.get("type") == "done":
                            captured_answer.append(payload.get("answer", ""))
                    except Exception:
                        pass
                yield frame
            # After the generator finishes, persist the assistant message.
            final_answer = captured_answer[0] if captured_answer else ""
            if final_answer:
                store.append_message(
                    thread.id,
                    Message(
                        role="assistant",
                        content=final_answer,
                        created_at=datetime.now(timezone.utc),
                    ),
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable proxy buffering
            },
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest_url(req: IngestUrlRequest) -> IngestResponse:
        """Ingest a public GitHub repo URL.

        Returns an ``ingest_id`` to attach to subsequent ``/ask``
        requests so the engine graphs the repo's documents instead of
        the default seed corpus.
        """
        url = req.source_url.strip()
        if not url:
            raise HTTPException(
                status_code=422, detail="source_url must not be empty"
            )
        try:
            docs, label = ingest_github(url)
        except GitHubFetchError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        ingest = ingest_store.save(source=url, label=label, docs=docs, kind="github")
        return IngestResponse(
            ingest_id=ingest.id,
            source=url,
            label=label,
            doc_count=len(docs),
            kind="github",
        )

    @app.post("/ingest/upload", response_model=IngestResponse)
    async def ingest_upload(file: UploadFile = File(...)) -> IngestResponse:
        """Ingest a CSV or XLSX attachment.

        File type is dispatched by extension. The parser caps rows and
        drops empty rows; one schema doc is prepended so the semantic
        layer can see the column names.
        """
        filename = file.filename or "upload"
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="empty file")

        lower = filename.lower()
        try:
            if lower.endswith(".csv"):
                docs, label, dataframe = ingest_csv(data, filename)
                kind = "csv"
            elif lower.endswith(".xlsx") or lower.endswith(".xlsm"):
                docs, label, dataframe = ingest_xlsx(data, filename)
                kind = "xlsx"
            else:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"unsupported file type: {filename!r}. "
                        "Supported: .csv, .xlsx"
                    ),
                )
        except TabularFetchError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        ingest = ingest_store.save(
            source=filename, label=label, docs=docs,
            kind=kind, dataframe=dataframe,
        )
        return IngestResponse(
            ingest_id=ingest.id,
            source=filename,
            label=label,
            doc_count=len(docs),
            kind=kind,
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


def _stub_response(
    query: str,
    plugin: Optional[str],
    *,
    note: Optional[str] = None,
) -> Tuple[str, List[TraceStep], LayeredGraphData, TraversalTrace, bool]:
    """Full stub payload, used when the engine is unavailable."""
    prefix = "[stubbed]"
    if note:
        prefix = f"[stubbed — {note}]"
    target = f" against the {plugin!r} connector" if plugin else ""
    answer = (
        f"{prefix} Kairo's TemporalRAG engine isn't reachable right now — "
        f"start Ollama and retry. Query{target}: {query.strip()!r}."
    )
    trace = _stub_trace(query, plugin)
    graph_data, traversal = _stub_layered_graph(query)
    return answer, trace, graph_data, traversal, True


def _run_analytics(
    query: str,
    ingest,
    runner: AnalyticsRunner,
    *,
    starting_step: int,
) -> Tuple[List[ChartSpec], List[ForecastSpec], List[TraceStep]]:
    """Run the analytics pipeline against a tabular ingest, if applicable.

    Returns (charts, forecasts, trace_steps). Empty lists mean the ingest
    isn't tabular, there's no chart intent, or no plan survived validation.
    """
    if ingest is None or ingest.dataframe is None or ingest.dataframe.empty:
        return [], [], []
    if not has_chart_intent(query):
        return [], [], []

    columns = infer_columns(ingest.dataframe)
    plans: List[AnalyticsPlan] = build_plans(query, columns)
    if not plans:
        return [], [], []

    charts: List[ChartSpec] = []
    forecasts: List[ForecastSpec] = []
    steps: List[TraceStep] = []
    step_no = starting_step
    for plan in plans:
        result = runner.run(ingest.dataframe, plan)
        if result.error:
            steps.append(
                TraceStep(
                    step=step_no,
                    kind="note",
                    summary=f"analyst: {plan.tool} failed — {result.error}",
                    metadata={"tool": plan.tool, "columns": plan.columns},
                )
            )
        else:
            if result.chart is not None:
                charts.append(result.chart)
            if result.forecast is not None:
                forecasts.append(result.forecast)
            summary = f"analyst: {plan.tool}({', '.join(plan.columns)})"
            if plan.rationale:
                summary = f"{summary} — {plan.rationale}"
            steps.append(
                TraceStep(
                    step=step_no,
                    kind="tool",
                    summary=summary,
                    metadata={
                        "tool": plan.tool,
                        "columns": plan.columns,
                        "params": plan.params,
                    },
                )
            )
        step_no += 1
    return charts, forecasts, steps


def _cited_ids(result, *, max_ids: int = 12, min_score: float = 0.15) -> List[str]:
    """Evidence ids that actually grounded the synthesized answer.

    The RAG engine already scores evidences by KNN similarity + graph
    proximity.  We pass the top N (above a soft floor) to the UI so the
    3D viz can halo them.  Web-kind evidence is always included since it
    was injected specifically to enrich this query.
    """
    evidences = list(getattr(result.rag_result, "evidences", []) or [])
    if not evidences:
        return []

    # Web hits are always cited — they're here because the enricher
    # decided this query needed them.
    web_ids = [
        ev.document_id for ev in evidences
        if ev.metadata.get("kind") == "web"
    ]

    # Then top-scoring non-web evidence up to the cap.
    ranked = sorted(
        (ev for ev in evidences if ev.metadata.get("kind") != "web"),
        key=lambda e: e.score,
        reverse=True,
    )
    top = [ev.document_id for ev in ranked if ev.score >= min_score]

    merged: List[str] = []
    seen = set()
    for nid in web_ids + top:
        if nid and nid not in seen:
            merged.append(nid)
            seen.add(nid)
        if len(merged) >= max_ids:
            break
    return merged


def _trace_from_result(result) -> List[TraceStep]:
    """Map an :class:`EngineResult` into the UI's ``TraceStep`` list.

    One step for the plan, one per agent finding, plus a final
    snapshot marker carrying the traversal snapshot id.
    """
    plan = result.plan
    steps: List[TraceStep] = [
        TraceStep(
            step=1,
            kind="plan",
            summary=(
                f"Supervisor planned {len(plan.tasks)} agent(s): "
                + ", ".join(f"{t.role}" for t in plan.tasks)
            ),
            metadata={"reasoning": plan.scaling_reason or ""},
        )
    ]
    for i, finding in enumerate(result.findings, start=2):
        if finding.role == "synthesizer":
            kind = "synthesize"
        elif finding.role == "web_researcher":
            kind = "tool"
        else:
            kind = "retrieve"
        output = (finding.output or "").strip()
        summary = output[:160] + ("…" if len(output) > 160 else "")
        if not summary:
            summary = f"{finding.role} ({finding.agent_name}) produced no output"
        steps.append(
            TraceStep(
                step=i,
                kind=kind,
                summary=f"{finding.role}: {summary}",
                metadata={"agent": finding.agent_name, "role": finding.role},
            )
        )
    steps.append(
        TraceStep(
            step=len(steps) + 1,
            kind="note",
            summary=(
                f"Snapshot {result.rag_result.snapshot_id[:8]} "
                f"({'cache' if result.rag_result.from_cache else 'fresh'}) "
                f"· model={current_model() or 'n/a'}"
            ),
            snapshot_id=result.rag_result.snapshot_id,
        )
    )
    return steps


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
