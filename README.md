# Kairo

# StateGraphRAG

StateGraphRAG is a retrieval native engine for building agentic systems over documents, structured databases, and evolving knowledge graphs.

Unlike standard retrieval systems that fetch static chunks and pass them directly to an LLM, StateGraphRAG treats retrieval as the core intelligence of the system. It first understands the task, selects the right retrieval path, reconstructs historical graph or document state when needed, distills the evidence, and only then lets the agent reason and answer.

The project is designed for problems where truth depends on context, structure, and time.

## Why this project exists

Most current agent frameworks are strong at orchestration, tool calling, and workflow design. Most current RAG systems are strong at static retrieval over text. However, real enterprise and research systems often require more than that.

In many settings:

- documents change over time
- graph relationships evolve
- users ask what was true at a particular point in time
- answers depend on graph structure, not only text similarity
- raw retrieval returns too much noisy context
- systems need evidence, provenance, and historical reconstruction

StateGraphRAG is designed to solve this gap.

## Core idea

The system follows this flow:

`Query -> Task Understanding -> Retrieval Planning -> Evidence Retrieval -> Temporal or Graph Reconstruction -> Context Distillation -> Agent Reasoning -> Grounded Answer`

This makes the engine retrieval native rather than agent first.

## Main capabilities

### Task oriented retrieval
The engine understands the task before retrieving. It decides whether the request is best answered through lexical search, vector search, SQL retrieval, graph traversal, hybrid retrieval, or temporal reconstruction.

### Context distillation
The engine does not pass large unfiltered retrieval results to the agent. It compresses evidence into a task packet containing the most relevant facts, graph paths, rows, contradictions, gaps, and supporting context.

### Graph reasoning
The system supports graph aware retrieval and reasoning over entities, relations, paths, neighborhoods, and multi hop structure.

### Temporal state retrieval
The system stores meaningful historical states of documents or subgraphs with version aware identifiers and embeddings, so retrieval can target what was semantically true at a specific time.

### Graph reconstruction
The engine can traverse historical state links, load snapshots or replay deltas, rebuild the relevant graph or document state, and answer from that recovered historical truth.

### Evidence grounded answering
Every answer should be tied to supporting evidence such as retrieved chunks, SQL rows, graph paths, state identifiers, or reconstructed snapshots.

## What makes this different

StateGraphRAG is not just another multi agent framework.

StateGraphRAG is not just another GraphRAG wrapper.

StateGraphRAG is a retrieval native temporal graph reasoning engine.

The core novelty is that it reasons over state evolution, not only static entities and relations.

## The seven core components

### 1. Agent Orchestration Layer
Coordinates agents, workflows, handoffs, retries, approvals, and final response assembly.

### 2. Connector and Tool Gateway
Provides unified access to relational databases, graph stores, vector stores, files, APIs, cloud storage, and external tools.

### 3. Task Oriented Retrieval and Context Distillation Engine
Classifies the task, selects the retrieval path, ranks and filters evidence, and produces a compact task packet.

### 4. Knowledge Index and Versioned State Store
Stores chunked documents, embeddings, metadata indexes, graph entities, relations, state versions, snapshots, and delta chains.

### 5. Graph Reasoning and Temporal State Reconstruction Engine
Supports graph traversal, state history navigation, snapshot loading, delta replay, and time aware reasoning.

### 6. Memory and Evidence Layer
Stores session context, retrieved evidence, intermediate results, graph paths, citations, and provenance.

### 7. Governance, Observability, and Evaluation Layer
Supports tracing, permissions, auditing, cost tracking, latency tracking, retrieval evaluation, and regression testing.

## Main novelty

### Temporal State Retrieval with Graph Reconstruction

Each meaningful document or graph state is stored with:

- a `history_id`
- a semantic embedding for that state
- a snapshot reference or delta chain
- temporal metadata
- graph links to previous and next states

At retrieval time, the system can:

- find the right historical state semantically
- traverse the history graph
- reconstruct the graph or document at that time
- distill the relevant evidence
- answer based on that recovered historical truth

This is stronger than plain GraphRAG because it reasons over state evolution, not only static entities and relations.

## Example questions the engine should answer

- What was the graph state before the facility update?
- Which policy version was active in March 2025?
- How did this entity relationship change between two versions?
- Which upstream and downstream assets were connected at that historical state?
- What evidence supports the answer at that time rather than now?

## Proposed repository structure

```text
stategraphrag/
├── README.md
├── pyproject.toml
├── .env.example
├── LICENSE
├── CONTRIBUTING.md
├── docs/
│   ├── temporal_state_graph_reconstruction.md
│   ├── planning.md
│   ├── roadmap.md
│   └── milestones.md
├── stategraphrag/
│   ├── agents/
│   ├── connectors/
│   ├── retrieval/
│   ├── state_store/
│   ├── graph/
│   ├── memory/
│   ├── observability/
│   └── api/
└── tests/
