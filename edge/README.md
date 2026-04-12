# kairo-edge

ARM/mobile-minimal build. Pure-Python only — no pandas, pypdf, httpx, or
soundfile. Deploy this on a Raspberry Pi, inside Termux, or bundled into a
BeeWare iOS/Android app.

```python
from kairo_edge import EdgeStore
from kairo_core import Document, QueryRequest

store = EdgeStore()
store.add([Document(id="1", text="temporal graphs are great")])
resp = store.query(QueryRequest(query="temporal graphs", top_k=3))
```

Install:

```bash
pip install -e ./core ./edge
```
