# kairo-retrieval

Local lexical retrieval pipeline + hosted HTTP client. Vector, graph, and
hybrid strategies will land here.

```python
from kairo_retrieval import LocalPipeline, HostedClient
from kairo_core import QueryRequest

pipe = LocalPipeline()
resp = pipe.query(QueryRequest(query="temporal graphs", top_k=5))
```

Install:

```bash
pip install -e ./core ./retrieval
# With direct-file ingestion support:
pip install -e ./core ./ingest ./retrieval[ingest]
```
