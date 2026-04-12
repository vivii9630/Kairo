# kairo-ingest

Multi-format document loaders for Kairo: txt, jsonl, csv, xlsx, pdf, docx,
audio. Produces `kairo_core.Document` instances.

```python
from kairo_ingest import ingest_any
docs = ingest_any(Path("examples/sample.jsonl"))
```

Install:

```bash
pip install -e ./core ./ingest
```
