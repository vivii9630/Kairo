# kairo-flow

Directed graph of agent/tool steps; supports cycles.

```python
from kairo_flow import FlowGraph, retrieval_node
from kairo_core import QueryRequest

graph = FlowGraph()
graph.add_node(retrieval_node("retriever"))
resp = graph.run("retriever", QueryRequest(query="temporal graphs", top_k=3))
```

Install:

```bash
pip install -e ./core ./retrieval ./flow
```
