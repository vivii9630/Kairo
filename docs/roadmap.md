# Roadmap

## Overview

This roadmap describes the staged evolution of Kairo from planning to a usable retrieval native engine with temporal graph reasoning and historical reconstruction.

The roadmap is divided into phases so that the project grows in a realistic and technically grounded way.

### Comparison: temporal graph retrieval vs common baselines

| Aspect | Naive LLM | Standard RAG | GraphRAG | Kairo temporal graph retrieval |
| --- | --- | --- | --- | --- |
| Retrieval target | Entire prompt context; no retrieval | Static text chunks | Graph nodes/relations | Documents, graph paths, and historical states |
| Temporal awareness | None | None | Limited (current graph) | Explicit time travel with snapshots/deltas |
| Graph reasoning | None | None | Yes, current graph | Yes, across graph and time |
| Evidence distillation | None | Minimal ranking | Graph-structured context | Task packet with distilled evidence and gaps |
| State reconstruction | None | None | Rare/partial | Reconstructs historical state before answering |
| Provenance | Not available | Chunk citations | Path-level evidence | State IDs, paths, snapshots, and provenance |

---

## Phase 0: Architecture and Design

### Goal

Define the conceptual, data, and module foundations of the system.

### Deliverables

- project identity and repository structure
- README and documentation set
- seven component architecture
- temporal state model
- retrieval and reconstruction design
- milestone definitions
- naming and module boundaries

### Success criteria

- core project concept is documented clearly
- responsibilities of each component are stable
- temporal state novelty is framed precisely

---

## Phase 1: Core Data and Connector Foundation

### Goal

Build the basic ingestion and access layer that allows the engine to work across sources.

### Deliverables

- PostgreSQL connector
- Neo4j connector
- filesystem and document ingestion
- vector store abstraction
- metadata normalization
- document chunking pipeline
- graph entity and relation ingestion

### Success criteria

- data can be ingested from multiple source types
- connectors expose a common interface
- the system can store documents, entities, and relations consistently

---

## Phase 2: Retrieval Native Core

### Goal

Build a task aware retrieval engine instead of direct tool only execution.

### Deliverables

- task classifier
- query rewriting
- retrieval router
- lexical retrieval
- vector retrieval
- hybrid retrieval
- evidence ranking
- context distillation
- task packet generation

### Success criteria

- the engine can choose retrieval mode by task
- the output to the agent is a compact evidence packet
- retrieval quality improves over naive chunk passing

---

## Phase 3: Versioned State Storage

### Goal

Introduce historical state as a first class knowledge object.

### Deliverables

- `history_id` model
- state level embeddings
- temporal metadata
- snapshot store
- delta store
- state lineage links
- state indexing strategy

### Success criteria

- meaningful states can be stored and retrieved
- the system can distinguish current and historical states
- embeddings exist at the state level where needed

---

## Phase 4: Graph Reasoning and Temporal Reconstruction

### Goal

Enable graph based and history aware answering.

### Deliverables

- entity linking
- graph traversal
- path extraction
- temporal graph traversal
- historical state lookup
- snapshot loading
- delta replay
- reconstructed graph evidence extraction
- state comparison support

### Success criteria

- the system can reconstruct a historical state
- the system can answer from reconstructed graph context
- graph paths and state provenance can be shown as evidence

---

## Phase 5: Memory and Evidence Grounding

### Goal

Store and expose supporting evidence throughout execution.

### Deliverables

- session memory
- evidence packet store
- provenance model
- citation generation
- graph path evidence output
- version aware memory references

### Success criteria

- answers can be audited
- retrieved evidence is preserved across execution
- historical answers record which state was used

---

## Phase 6: Orchestration and Workflow Layer

### Goal

Add controlled execution and flexible agent workflows.

### Deliverables

- orchestrator runtime
- task router
- workflow graph
- retry and timeout support
- human approval hooks
- response assembly

### Success criteria

- the system supports repeatable multi step execution
- workflows can invoke retrieval and reconstruction cleanly
- answers remain evidence grounded

---

## Phase 7: Observability and Evaluation

### Goal

Make the system measurable and testable.

### Deliverables

- trace collection
- step logs
- latency and cost metrics
- retrieval evaluation
- graph reasoning evaluation
- temporal reconstruction evaluation
- regression suite

### Success criteria

- retrieval behavior is inspectable
- historical reconstruction can be validated
- regressions can be detected automatically

---

## Phase 8: Developer Experience and Examples

### Goal

Make the project approachable for contributors and users.

### Deliverables

- example scripts
- minimal API service
- local development setup
- architecture diagrams
- sample data and demos
- contribution guide

### Success criteria

- a new user can run a simple example
- the architecture is easy to understand
- contributors can extend modules without deep rewrites

---

## Phase 9: Advanced Research and Scaling

### Goal

Extend the core engine into a more mature retrieval platform.

### Potential directions

- learned retrieval routing
- graph aware reranking
- state summarization layers
- temporal query language
- multi tenant connector governance
- streaming updates into temporal state store
- benchmark datasets for historical graph answering
- uncertainty aware retrieval and answering

### Success criteria

- advanced features remain aligned with the retrieval native design
- system quality improves without weakening transparency
- performance scales to larger graphs and histories

---

## Guiding principles across all phases

1. Retrieval quality before agent complexity  
2. Historical correctness before broad feature expansion  
3. Evidence and provenance from the start  
4. Modular design over tightly coupled code  
5. Clear interfaces between storage, retrieval, reconstruction, and orchestration  

## Summary

The roadmap is intentionally staged so that Kairo develops from a clear architecture into a usable engine, then into a robust and extensible platform. The system should first become good at retrieval, state, and reconstruction before it becomes ambitious about complex agent behaviors.
