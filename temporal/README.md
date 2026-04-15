# kairo-temporal

Temporal engine for Kairo. Owns the `.kairo/` directory layout and provides
the branching history DAG of graph snapshots.

## What it does

- Creates durable **snapshots** of a `KairoGraph` + its embedding index, keyed
  by a deterministic `graph_hash` and a user-supplied `history_graph_id`.
- Maintains a **branching history DAG** (`HistoryGraph`) — snapshots can have
  multiple parents / children, just like git commits.
- Detects graph changes between snapshots via `GraphDiff`
  (added/removed/changed nodes + edges).
- Supports **rollback** (check out a prior snapshot) and **walk** (traverse
  ancestors/descendants of a snapshot).
- Persists everything under `.kairo/` on disk as the source of truth.

## `.kairo/` layout

```
.kairo/
├── HEAD                    # current branch or snapshot id
├── refs/
│   └── branches/
│       └── main            # file containing a snapshot id
├── history.json            # branching DAG (nodes + parent links)
└── snapshots/
    └── <snapshot_id>/
        ├── snapshot.json   # Snapshot metadata
        ├── graph.json      # GraphData (nodes + edges)
        └── embeddings/
            ├── matrix.npy
            └── index.json
```

## Scope boundaries

This package knows nothing about agents, connectors, UIs, or specific data
sources. It is a pure temporal engine over `(graph, embeddings)` pairs.
