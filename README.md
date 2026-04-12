# Kairo

![Kairo temporal graph retrieval engine](Image/Gemini_Generated_Image_5j6rva5j6rva5j6r.png)

**A retrieval-native temporal engine for agents, RAG, and evolving knowledge graphs.**

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

```
┌────────────────────────────────────────────────────────────────┐
│                          kairo (meta)                          │
└────────────────────────────────────────────────────────────────┘
            │
   ┌────────┼────────────────────────────────────────┐
   ▼        ▼                                        ▼
┌──────┐ ┌──────────┐ ┌──────────┐ ┌─────┐ ┌──────┐ ┌────────────┐
│ core │ │ retrieval│ │  ingest  │ │ rag │ │ flow │ │ connectors │
└──────┘ └──────────┘ └──────────┘ └─────┘ └──────┘ └────────────┘
            │              │         │       │
            └──────────────┴─────────┴───────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
                ┌────────┐ ┌──────┐
                │ agents │ │ edge │
                └────────┘ └──────┘
                    │
                    ▼
                ┌──────┐
                │ cli  │
                └──────┘
```

| Package | Purpose | Depends on |
|---|---|---|
| [`kairo-core`](core/) | Shared types: `Document`, `Message`, `Evidence`, `Query` | — |
| [`kairo-ingest`](ingest/) | Multi-format loaders (txt, jsonl, csv, xlsx, pdf, docx, audio) | core |
| [`kairo-retrieval`](retrieval/) | Local lexical pipeline + hosted HTTP client; vector/graph/hybrid land here | core |
| [`kairo-agents`](agents/) | Agents, inter-agent messaging, subagent delegation, verbose reasoning | core |
| [`kairo-rag`](rag/) | RAG patterns: plain, hybrid, temporal | core, agents, retrieval |
| [`kairo-flow`](flow/) | Directed-graph runner for agent/tool steps (cycles supported) | core, retrieval |
| [`kairo-connectors`](connectors/) | Adapters for external sources (S3, Notion, web, Slack, ...) | core |
| [`kairo-cli`](cli/) | `kairo` command for ingest + query | core, ingest, retrieval |
| [`kairo-edge`](edge/) | ARM / mobile build — pure-Python, zero heavy deps | core |

Each package lives in its own directory with its own `pyproject.toml` and `README.md`, so you can work on, commit, and publish them independently.

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
├── core/          # kairo-core         — shared types
├── ingest/        # kairo-ingest       — document loaders
├── retrieval/     # kairo-retrieval    — local + hosted retrieval
├── agents/        # kairo-agents       — agents & messaging
├── rag/           # kairo-rag          — RAG patterns
├── flow/          # kairo-flow         — directed-graph runner
├── connectors/    # kairo-connectors   — external source adapters
├── cli/           # kairo-cli          — command-line interface
├── edge/          # kairo-edge         — ARM / mobile minimal build
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
