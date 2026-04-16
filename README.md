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
| [`kairo-retrieval`](retrieval/) | Local lexical pipeline + hosted HTTP client; vector/graph/hybrid land here | core | ✅ shipped |
| [`kairo-connectors`](connectors/) | Adapters for external sources via the `ConnectorPlugin` protocol. Ships GitHub, Slack, Google Drive, Gmail; pluggable auth layer (Bearer / OAuth2 / API key) + env + file credential stores | core | ✅ shipped (Phases 4 + 6a/6b/6c) |
| [`kairo-graph`](graph/) | `GraphBuilder` protocol, NetworkX-backed `KairoGraph` with deterministic SHA-256 hashing and JSON round-trip, per-content-type builders | core | ✅ shipped (Phase 1) |
| [`kairo-embeddings`](embeddings/) | Pluggable `EmbeddingProvider` registry (hash-stub, sentence-transformers, OpenAI, Cohere, Voyage, Bedrock), `EmbeddingStore` with cosine top-k | core | ✅ shipped (Phase 2) |
| [`kairo-temporal`](temporal/) | `HistoryGraph` (branching DAG of snapshots), rollback, walk, diff — owns the `.kairo/` filesystem layout | core, graph, embeddings | ✅ shipped (Phase 3) |
| [`kairo-rag`](rag/) | RAG patterns: plain, hybrid, **temporal** (`TemporalRAG` orchestrator lands here in Phase 7) | core, agents, retrieval | ✅ shipped |
| [`kairo-flow`](flow/) | Directed-graph runner for agent/tool steps (cycles supported) | core, retrieval | ✅ shipped |
| [`kairo-agents`](agents/) | Agents, inter-agent messaging, subagent delegation, verbose reasoning; supervisor multi-agent + Kairo-native tool-use lands in Phase 8 | core | ✅ shipped |
| [`kairo-cli`](cli/) | `kairo` command: ingest, query, and upcoming temporal commands (`init`, `snapshot`, `ask`, `log`, `rollback`, `branch`) | core, ingest, retrieval | ✅ shipped |
| [`kairo-edge`](edge/) | ARM / mobile build — pure-Python, zero heavy deps | core | ✅ shipped |
| [`kairo-ai`](ai/) | Backend orchestration for the **Kairo AI** product: FastAPI shell exposing `/plugins`, `/ask`, `/threads`. Stubbed answer path today; gets the Phase 7 `TemporalRAG` orchestrator behind it next | core, connectors | ✅ shipped (Phase 9, stub) |
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
| 7 | `TemporalRAG` orchestrator (composes temporal + embeddings + graph) | `kairo-rag` | ⏳ next |
| 8 | Supervisor multi-agent + **Kairo-native dict-based tool-use protocol** | `kairo-agents`, `kairo-flow` | ⏳ planned |
| 9 | FastAPI shell: `/plugins`, `/ask` (stubbed), `/threads` | `kairo-ai` | ✅ shipped (stub) |
| 10 | Chat-native UI: plugin picker, ask box, trace panel, thread list | `kairo-ui` | ✅ shipped (thin slice) |

Each phase lands as a package-scoped commit so phases can be released, revisited, or contributed to independently.

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
