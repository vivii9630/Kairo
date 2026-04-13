# kairo-graph

Graph construction primitives for Kairo. Provides a `GraphBuilder` protocol,
a NetworkX-backed `KairoGraph` with deterministic hashing and JSON round-trip,
and a growing set of per-content-type builders.

Used by `kairo-temporal` to snapshot a graph at a point in time and by
`kairo-rag` to feed retrieval over graph structure.

## Quickstart

```python
from kairo_core import Document
from kairo_graph import KairoGraph, DocumentGraphBuilder, save_graph, load_graph

docs = [
    Document(id="a", text="Temporal graphs capture evolving states."),
    Document(id="b", text="Retrieval planning precedes generation."),
]

builder = DocumentGraphBuilder()
graph = builder.build(docs)

print(graph.num_nodes(), graph.num_edges())
print(graph.hash())

save_graph(graph, ".kairo/snapshots/demo.json")
reloaded = load_graph(".kairo/snapshots/demo.json")
assert reloaded.hash() == graph.hash()
```

## Builders

| Builder | Input | Status |
|---|---|---|
| `DocumentGraphBuilder` | `Iterable[Document]` | Phase 1 — one node per document |
| `CodeGraphBuilder` | source tree | Phase 6 |
| `TabularGraphBuilder` | CSV / dataframe | Phase 6 |

Add a new builder by dropping a file under `kairo_graph/builders/` that
implements the `GraphBuilder` protocol.

## Install

```bash
pip install -e ./core ./graph
```
