<div align="center">

# Kairo

**A retrieval-native temporal engine for agents, RAG, and evolving knowledge graphs.**

[Repository](https://github.com/vivii9630/Kairo) · [Documentation](docs/) · [Issues](https://github.com/vivii9630/Kairo/issues) · [License](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/vivii9630/Kairo?style=flat&logo=github)](https://github.com/vivii9630/Kairo/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/vivii9630/Kairo?style=flat&logo=github)](https://github.com/vivii9630/Kairo/network/members)
[![GitHub issues](https://img.shields.io/github/issues/vivii9630/Kairo)](https://github.com/vivii9630/Kairo/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/vivii9630/Kairo)](https://github.com/vivii9630/Kairo/pulls)
[![License: MIT](https://img.shields.io/github/license/vivii9630/Kairo)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/vivii9630/Kairo)

![Kairo temporal graph retrieval engine](Image/Gemini_Generated_Image_5j6rva5j6rva5j6r.png)

</div>

Kairo is a Python framework for building agentic systems that reason over documents, structured databases, and knowledge that changes over time. Unlike retrieval stacks bolted onto an agent loop, Kairo plans retrieval *before* the agent runs — across lexical, vector, graph, and temporal paths — and hands the agent evidence-first task packets with full provenance.

Kairo ships as a set of **independent, composable packages**. Install only what you need; extend by dropping new tools into their own package without touching the rest.

---

## Why Kairo

| | Classic RAG / Agent Stacks | **Kairo** |
|---|---|---|
| Retrieval planning | Agent decides on the fly | **Retrieval-native** — lexical / vector / graph / temporal routes planned before the agent runs |
| Temporal reasoning | Static chunks, "latest" snapshot | **Historical state reconstruction** — answer from what the graph looked like at time *t* |
| Evidence handling | Raw chunks stuffed into context | **Evidence-first task packets** with provenance, emitted before LLM reasoning |
| Agent topology | DAG or single loop | **Directed graphs with cycles** — plan/critique/retry flows are first-class |
| Edge / mobile | Requires full stack | **`kairo-edge`** — pure-Python, zero heavy deps, ARM-friendly |
| Extensibility | Monolith | **Independent packages** — add a new tool or connector as its own package, ship it on its own cadence |

---

## What's novel

- **Retrieval-native planning.** Lexical, vector, graph, and temporal retrieval routes are planned and executed *before* agents run, so agents reason over curated evidence instead of improvising search.
- **Temporal + graph state.** Kairo reconstructs historical states and graph paths, not just static chunks — ask "what did we believe on 2024-03-01?" and get an answer grounded in that epoch.
- **Evidence-first task packets.** Every step receives structured evidence with provenance before an LLM sees it, making runs auditable and replayable.
- **Flow-based agents.** Directed graphs of steps and tools, cycles allowed — express plan/critique/retry loops natively.
- **Modular by construction.** Core, retrieval, ingestion, agents, RAG, flow, CLI, connectors, and edge are separate packages that ship and version independently.

---

## Architecture

Kairo is organized as a **layered stack of independent packages**. Each layer depends only on the ones below it; each package in a layer can be installed and worked on in isolation.

```
┌────────────┬──────────────────────────────────────────────────────────┐
│  apps      │  kairo (meta) · kairo-cli · kairo-ai · kairo-ui          │
├────────────┼──────────────────────────────────────────────────────────┤
│  patterns  │  kairo-rag  ·  kairo-flow  ·  kairo-agents               │
├────────────┼──────────────────────────────────────────────────────────┤
│  temporal  │  kairo-temporal  ·  kairo-embeddings  ·  kairo-graph     │
├────────────┼──────────────────────────────────────────────────────────┤
│  io        │  kairo-retrieval  ·  kairo-ingest  ·  kairo-connectors   │
├────────────┼──────────────────────────────────────────────────────────┤
│  foundation│  kairo-core                             kairo-edge (alt) │
└────────────┴──────────────────────────────────────────────────────────┘
```

| Package | Purpose | Depends on | Status |
|---|---|---|---|
| [`kairo-core`](core/) | Shared types (`Document`, `Message`, `Evidence`, `QueryRequest`, `Snapshot`, `HistoryNode`, `GraphDiff`, `TemporalQuery`, …) | — | ✅ shipped |
| [`kairo-ingest`](ingest/) | Multi-format loaders (txt, jsonl, csv, xlsx, pdf, docx, audio) | core | ✅ shipped |
| [`kairo-retrieval`](retrieval/) | Local lexical pipeline + hosted HTTP client. **BM25 + cosine** fused via Reciprocal Rank Fusion (Phase 11a), **1-hop graph walk** with weight-decayed neighbor boost (Phase 13a), eval harness with recall@k / MRR / nDCG@k | core, embeddings | ✅ shipped |
| [`kairo-connectors`](connectors/) | Adapters for external sources via the `ConnectorPlugin` protocol. Ships GitHub, Slack, Google Drive, Gmail; pluggable auth layer (Bearer / OAuth2 / API key) + env + file credential stores | core | ✅ shipped (Phases 4 + 6a/6b/6c) |
| [`kairo-graph`](graph/) | `GraphBuilder` protocol, NetworkX-backed `KairoGraph` with deterministic SHA-256 hashing and JSON round-trip, per-content-type builders. Every edge carries **`provenance`** (structural / extracted / inferred / ambiguous) + **`confidence`** (Phase 13b1). **Leiden community detection** with Louvain fallback, concept-node builder (13b2). Pluggable **`ExtractorProvider`** protocol + `OllamaExtractor` + `LLMExtractedGraphBuilder` for LLM-extracted relationships (13b3). `CodeGraphBuilder` emits **docstring** and **rationale-comment** nodes (13b4) | core, embeddings | ✅ shipped |
| [`kairo-embeddings`](embeddings/) | Pluggable `EmbeddingProvider` registry (hash-stub, sentence-transformers, OpenAI, Cohere, Voyage, Bedrock), `EmbeddingStore` with cosine top-k | core | ✅ shipped (Phase 2) |
| [`kairo-temporal`](temporal/) | `HistoryGraph` (branching DAG of snapshots), rollback, walk, diff — owns the `.kairo/` filesystem layout | core, graph, embeddings | ✅ shipped (Phase 3) |
| [`kairo-rag`](rag/) | RAG patterns: plain, hybrid, **temporal** `TemporalRAG` orchestrator (Phase 7). Post-generation **`CitationVerifier`** checks every claim against cited evidences and flags `answer_downgraded` when support drops below threshold (Phase 13c) | core, agents, retrieval | ✅ shipped |
| [`kairo-flow`](flow/) | Directed-graph runner for agent/tool steps (cycles supported) | core, retrieval | ✅ shipped |
| [`kairo-agents`](agents/) | Agents, inter-agent messaging, subagent delegation, verbose reasoning; supervisor multi-agent + Kairo-native tool-use lands in Phase 8 | core | ✅ shipped |
| [`kairo-cli`](cli/) | `kairo` command: ingest, query, and upcoming temporal commands (`init`, `snapshot`, `ask`, `log`, `rollback`, `branch`) | core, ingest, retrieval | ✅ shipped |
| [`kairo-edge`](edge/) | ARM / mobile build — pure-Python, zero heavy deps | core | ✅ shipped |
| [`kairo-ai`](ai/) | Backend orchestration for the **Kairo AI** product: FastAPI shell exposing `/plugins`, `/ask`, `/ingest`, `/threads`. Full multi-agent `EngineRunner` behind `/ask` (Phase 7e). **`/ask/stream`** SSE route pipes Ollama tokens through as they arrive (Phase 13d) | core, connectors, rag, retrieval, agents | ✅ shipped |
| [`kairo-ui`](ui/) | Chat-native frontend: logo-first landing, plugin picker wired to the live manifest registry, trace panel, thread list. Next.js (App Router) + Tailwind + TypeScript, talks to `kairo-ai` over HTTP | kairo-ai | ✅ shipped (Phase 10, thin slice) |
| `kairo-auth` | Credential broker for embedding / LLM / connector providers; other packages pull keys from here | core | 🔒 reserved (future) |

Each package lives in its own directory with its own `pyproject.toml` and `README.md`, so you can work on, commit, and publish them independently.

---

## Temporal RAG engine *(in build)*

Kairo is being extended with a **temporal RAG engine** that turns any project directory — docs, CSVs, source code, or a GitHub URL — into a durable, time-aware knowledge base. Point Kairo at a folder, and every time the content changes it writes a new versioned snapshot of the knowledge graph into a local `.kairo/` directory. Agents can then query the current state, roll back to any prior point in time, branch off alternate histories, or ask "what changed between these two versions?"

### Design at a glance

- **Snapshots, not deltas.** Each version of the knowledge graph is persisted as a full, addressable snapshot. Embeddings index those snapshots for semantic lookup — they are never used to reconstruct a graph (embeddings are lossy; snapshots are the source of truth).
- **Branching history DAG.** `Snapshot.parents: List[str]` from day one, so merges and forks are first-class. Git-shaped: `HEAD`, `refs/branches/`, an append-only history index.
- **Pluggable embeddings.** Users pick their provider (hash-stub, sentence-transformers, OpenAI, Cohere, Voyage, Bedrock, …) via a single `get_provider(name, …)` registry. A `Credentials` seam is designed in on day one so the planned `kairo-auth` package can broker API keys across all tools without breaking callers.
- **`.kairo/` is the source of truth.** No browser cache, no session-scoped storage — temporal guarantees require durable state on disk.

### `.kairo/` directory layout *(git-shaped)*

```
project-root/
├── src/, docs/, data.csv, README.md, ...
└── .kairo/
    ├── config.toml          # history_graph_id, embedding provider, branch policy
    ├── HEAD                 # current branch ref, e.g. "refs/branches/main"
    ├── refs/
    │   └── branches/
    │       ├── main         # -> snapshot_id
    │       └── experiment   # -> snapshot_id
    ├── history.json         # DAG: snapshot_id -> { parents: [...], timestamp, message }
    ├── snapshots/
    │   └── <snapshot_id>.json   # serialized KairoGraph + metadata
    └── embeddings/
        └── <snapshot_id>.npy    # embedding matrix aligned to graph nodes
```

### Phased build plan

| Phase | Scope | Package(s) | Status |
|---|---|---|---|
| 1 | Core temporal types + `KairoGraph` + `DocumentGraphBuilder` + round-trip | `kairo-core`, `kairo-graph` | ✅ shipped |
| 2 | Pluggable `EmbeddingProvider` + registry + `EmbeddingStore` | `kairo-embeddings` | ✅ shipped |
| 3 | `HistoryGraph`, snapshots, rollback, `.kairo/` filesystem store | `kairo-temporal` | ✅ shipped |
| 4 | `ConnectorPlugin` protocol + GitHub URL connector (shallow clone → hand off to ingest) | `kairo-connectors` | ✅ shipped |
| 5 | Per-type graph builders (code AST, tabular rows/cols/FKs) | `kairo-graph` | ✅ shipped |
| 6 | Protocol evolution + auth layer + Slack / Google Drive / Gmail (Jira deferred) | `kairo-connectors` | ✅ shipped (6a/6b/6c) |
| 7 | `TemporalRAG` orchestrator, multi-agent supervisor, Ollama engine, GitHub/CSV ingest, web enrichment, grounding halos | `kairo-rag`, `kairo-agents`, `kairo-ai`, `kairo-ui` | ✅ shipped (7a–7h) |
| 8 | In-chat analytics: charts (bar/line/histogram/scatter/pairwise), forecasting (exp-smoothing), kmeans + linear regression | `kairo-analytics` | ✅ shipped (8a–8e) |
| 9 | FastAPI shell: `/plugins`, `/ask`, `/ingest`, `/threads` | `kairo-ai` | ✅ shipped |
| 10 | Chat-native UI: plugin picker, ask box, 3D graph viz, trace panel, thread list | `kairo-ui` | ✅ shipped |
| 11 | **Retrieval & graph quality** — BM25 + cosine via RRF, `sentence-transformers` as real-semantic default, weighted `similar-to` edges from `DocumentGraphBuilder`, recall@k / MRR / nDCG@k eval harness | `kairo-retrieval`, `kairo-embeddings`, `kairo-graph` | ✅ shipped (11a–11d) |
| 12 | **STDP plasticity** — event pipeline + reward-modulated edge updates + goal conditioning + staleness decay. Implicit behavioral signals (regenerate / copy / follow-up), never thumbs-up/down alone | `kairo-plasticity` (planned), `kairo-ai` | ⏳ planned |
| 13 | **Graph intelligence & answer trust** — 1-hop graph walk in retrieval, Leiden communities, edge `provenance` + `confidence` tags on every builder, concept nodes, pluggable `ExtractorProvider` + `OllamaExtractor` + `LLMExtractedGraphBuilder`, docstring + rationale-comment nodes in `CodeGraphBuilder`, `CitationVerifier` over generated answers, SSE `/ask/stream` | `kairo-retrieval`, `kairo-graph`, `kairo-rag`, `kairo-ai`, `kairo-agents` | ✅ shipped (13a, 13b1–4, 13c, 13d) |
| 14e | **Edge runtime, Rust-first** — Rust workspace (`kairo-edge-core`, `kairo-edge-cli`, `kairo-edge-py` PyO3 wheel, `kairo-edge-llama-cpp`), Kairo Protocol v0.1 wire-format spec, `DeviceProfile` / `ProviderCapability` selection layer, `EdgePipeline` with extractive baseline, `LlamaCppProvider` scaffold + transparency metadata, real `llama-cpp-2` backend gated behind a `real-backend` Cargo feature, six-profile emulator (pi-zero/pi-4/pi-5/phone-mid/browser-lite/browser-gpu) | `kairo-edge`, `kairo-edge-core`, `kairo-edge-py`, `kairo-edge-llama-cpp` | 🚧 in progress (14e.0–14e.4.1 shipped, 14e.4.2 next) |

Each phase lands as a package-scoped commit so phases can be released, revisited, or contributed to independently.

---

## Graph intelligence & answer trust *(Phase 13)*

Phase 13 turned the graph from decorative into operational and added first-class answer-trust signals.

### Edge provenance & confidence

Every edge produced by any builder now carries two attributes beyond `kind`:

- `provenance ∈ {structural, extracted, inferred, ambiguous}` — how the edge was produced.
- `confidence ∈ [0.0, 1.0]` — how certain the builder is the edge is real.

| Edge source | Provenance | Confidence |
|---|---|---|
| AST (imports, calls, contains) | `structural` | `1.0` |
| Docstring / rationale-comment nodes | `structural` | `1.0` |
| Tabular schema (has_column, has_row) | `structural` | `1.0` |
| Cosine `similar-to` between docs | `inferred` | cosine score |
| Leiden `belongs_to` (doc → concept) | `inferred` | `1.0` (cluster-derived) |
| LLM-extracted relationships | `extracted` / `ambiguous` | extractor-reported |

Retrieval, the graph walk, and future plasticity updates can filter or weight by these attrs. The `edge_attrs()` helper in `kairo_graph.provenance` is the single choke point — invalid provenance or out-of-range confidence raises, so bad values can't leak past a builder.

### Leiden community detection + concept nodes

`detect_communities()` groups nodes over the weighted `similar-to` subgraph using **Leiden** via `leidenalg` + `python-igraph` (Louvain fallback when the extra isn't installed). `add_concept_nodes()` inserts one `concept` node per community with a lightweight token-frequency summary; each member doc gains a `belongs_to` edge. The summarizer is pluggable — pass a `summary_fn=(graph, member_ids) -> str` to swap in an Ollama-backed or Claude-backed summary without touching the builder.

Install the extra:

```bash
pip install -e "./graph[leiden]"
```

### Pluggable relationship extraction

The `ExtractorProvider` protocol lets any backend plug in:

```python
from kairo_graph import LLMExtractedGraphBuilder, OllamaExtractor

extractor = OllamaExtractor(model="llama3.2")
builder = LLMExtractedGraphBuilder(extractor, collision_policy="skip")
results = builder.augment(graph, documents)
```

`OllamaExtractor` is the local-first default. Cloud backends (Claude via Anthropic SDK, OpenAI, Cohere) slot in as `ExtractorProvider` subclasses without builder changes. Extracted edges are tagged `provenance="extracted"` with the model's self-reported confidence; `collision_policy="skip"` (default) guards against clobbering existing structural edges in the DiGraph.

### Citation verification

`CitationVerifier` in `kairo-rag` checks every sentence of a generated answer against the cited evidences via asymmetric token overlap. Returns a `CitationReport`:

```python
from kairo_rag import CitationVerifier

report = CitationVerifier(claim_threshold=0.35).verify(answer, evidences)
print(report.overall_support)          # 0.0–1.0
print(report.answer_downgraded)        # True when support < 0.6
for claim in report.claims:
    print(claim.supported, claim.sentence)
```

Wired into the `/ask` response as `citation_report`. Trivial / connective sentences ("Here is the summary:") auto-pass so they don't skew the support score. Embedding-similarity and LLM-judge strategies can slot in behind the same report contract later.

### SSE token streaming

`POST /ask/stream` returns a `text/event-stream`. Four event types:

- `retrieve` — one frame with the plan + evidence count + `cited_node_ids`.
- `token` — one frame per Ollama chunk (usually a few chars each).
- `error` — transport / parse failure; followed by a terminal `done`.
- `done` — reassembled answer + `CitationReport` over what was streamed.

The route uses a lightweight retrieve → stream-synthesize path; the full multi-agent pipeline stays on `/ask`. True multi-agent streaming lands when the synthesizer is isolated as the only streamable step.

### Retrieval eval harness

```bash
python scripts/eval_retrieval.py
```

Reports recall@k / MRR / nDCG@k across `bm25_only`, `hybrid_hash`, `hybrid_hash+walk`, `hybrid_default`, and `hybrid_default+walk` configs against a small labeled query set at `scripts/eval_data.json`.

---

## Installation

### Full install (recommended for getting started)

```bash
# optional: create and activate a venv
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# install the meta package with dev extras
pip install -e .[dev]
```

This installs the `kairo` meta package and the `kairo` CLI.

### Install individual packages

Pick only what you need. All packages are pip-installable from the repo root in editable mode:

```bash
# minimal: shared types only
pip install -e ./core

# retrieval + ingestion stack
pip install -e ./core ./ingest ./retrieval

# agents + RAG
pip install -e ./core ./agents ./retrieval ./rag

# directed-graph flows
pip install -e ./core ./retrieval ./flow

# CLI
pip install -e ./core ./ingest ./retrieval ./cli

# edge / ARM / mobile — tiny footprint
pip install -e ./core ./edge
```

### Optional extras

```bash
# connectors — each source opts into its own deps
pip install -e "./connectors[s3]"
pip install -e "./connectors[notion]"
pip install -e "./connectors[web]"
pip install -e "./connectors[slack]"

# retrieval with bundled ingestion
pip install -e "./retrieval[ingest]"
```

### Requirements

- Python **3.9+**
- Works on Linux, macOS, and Windows

---

## Quickstart

### CLI

```bash
# Ingest a supported file (txt / jsonl / csv / xls / xlsx / pdf / docx / audio)
kairo ingest examples/sample.jsonl

# Query locally against the ingested store
kairo query "How does Kairo handle temporal reconstruction?" --mode local --top-k 3

# Query a hosted endpoint
kairo query "Explain temporal reconstruction" \
  --mode hosted \
  --base-url https://your.api \
  --api-key YOUR_KEY \
  --top-k 3
```

### Local ingestion + query

```python
from pathlib import Path
from kairo import LocalPipeline, QueryRequest
from kairo.ingest import ingest_any

pipe = LocalPipeline(store_path=".kairo_store.json")
docs = ingest_any(Path("examples/sample.jsonl"))
pipe.ingest_documents(docs)

resp = pipe.query(QueryRequest(query="Explain temporal reconstruction", top_k=3))
print(resp.answer)
```

### Hosted client

```python
from kairo import HostedClient, QueryRequest

client = HostedClient(base_url="https://your.api", api_key="YOUR_API_KEY")
resp = client.query(QueryRequest(query="Explain temporal reconstruction", top_k=3))
print(resp.answer)
client.close()
```

### Agents with inter-agent messaging

```python
from kairo_agents import Agent, MessageBus

bus = MessageBus(verbose=True)
planner = Agent("planner", role="task planner",
                instruction="Break goals into steps.", verbose=True, bus=bus)
writer = Agent("writer", role="writer",
               instruction="Draft prose.", verbose=True)
planner.add_subagent(writer)

planner.think("Plan a short article.")
planner.delegate("writer", "Draft an opening line.")
```

Plug in any LLM by passing `think_fn=(role, instruction, input) -> str`.

### Run the Kairo AI app locally

Two processes: the FastAPI backend (`ai/`) and the Next.js frontend (`ui/`).

```bash
# 1. Backend — serves /plugins, /ask, /threads on :8000
pip install -e ./core ./connectors ./ai
uvicorn kairo_ai.app:app --reload

# 2. Frontend — in a second terminal
cd ui
npm install
npm run dev   # http://localhost:3000
```

The plugin picker is wired to the live connector registry — whichever connectors are importable in your environment (GitHub always; Slack, Drive, Gmail when their extras are installed) appear in the picker. The `/ask` endpoint returns realistic stubbed responses today; the Phase 7 `TemporalRAG` orchestrator slots in behind it without the UI changing.

> Requires **Node ≥16.14** (pinned to Next.js 13.5 for wider Node compatibility). Set `NEXT_PUBLIC_KAIRO_API_URL` in `ui/.env.local` if the backend is on a non-default host.

### Flow (directed graph, cycles allowed)

```python
from kairo import FlowGraph, FlowNode, retrieval_node, QueryRequest

flow = FlowGraph()
flow.add_node(retrieval_node("retriever"))
flow.add_node(FlowNode(
    name="postprocess",
    run=lambda req, pipeline: pipeline.query(req),
    next_nodes=[],
))
flow.nodes["retriever"].next_nodes = ["postprocess"]

resp = flow.run(start="retriever", request=QueryRequest(query="Hybrid retrieval", top_k=3))
print(resp.answer)
```

---

## Repository layout

```
Kairo/
├── core/          # kairo-core         — shared types (incl. temporal: Snapshot, HistoryNode, ...)
├── ingest/        # kairo-ingest       — document loaders
├── retrieval/     # kairo-retrieval    — local + hosted retrieval
├── connectors/    # kairo-connectors   — external source adapters (ConnectorPlugin protocol)
├── graph/         # kairo-graph        — GraphBuilder + NetworkX KairoGraph + hashing
├── embeddings/    # kairo-embeddings   — pluggable providers + store
├── temporal/      # kairo-temporal     — history DAG, snapshots, rollback, .kairo/
├── rag/           # kairo-rag          — RAG patterns (plain, hybrid, temporal)
├── flow/          # kairo-flow         — directed-graph runner
├── agents/        # kairo-agents       — agents & messaging
├── cli/           # kairo-cli          — command-line interface
├── edge/          # kairo-edge         — ARM / mobile minimal build
├── ai/            # kairo-ai           — FastAPI backend for the Kairo AI product       (Phase 9)
├── ui/            # kairo-ui           — chat-native Next.js frontend                   (Phase 10)
├── src/kairo/     # kairo              — meta package / SDK facade
├── examples/
└── pyproject.toml # meta package
```

Every subpackage is self-contained: its own `src/`, `pyproject.toml`, and `README.md`. You can branch, commit, and release a single package without pulling the rest of the repo into the change.

### Adding a new tool or package

1. Create a new top-level directory, e.g. `mytool/`.
2. Add `mytool/pyproject.toml` declaring the package name (`kairo-mytool`) and its dependencies on existing `kairo-*` packages.
3. Put source under `mytool/src/kairo_mytool/`.
4. Add a `mytool/README.md` describing the package.
5. Install it in editable mode: `pip install -e ./mytool`.

No changes required in the meta package unless you want to re-export symbols from `kairo`.

---

## Security & privacy

- **Local mode** makes no network calls; data stays in `.kairo_store.json`.
- **Hosted mode** only calls the `base_url` you provide, with your `api_key`.
- No telemetry, no background uploads.

---

## License

MIT — see [LICENSE](LICENSE).
