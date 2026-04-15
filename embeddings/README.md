# kairo-embeddings

Pluggable embedding providers and a numpy-backed embedding store for the
Kairo temporal RAG engine. Used by `kairo-temporal` to index snapshots of
the knowledge graph and by `kairo-rag` to run semantic lookup.

## Design

- **One `EmbeddingProvider` protocol, many backends.** Add a provider by
  dropping a file under `providers/` that exposes `name`, `dim`, and
  `embed(texts) -> np.ndarray`.
- **`get_provider(name, ...)` registry.** Users pick their backend by name;
  nothing else in Kairo needs to know which one they chose.
- **`Credentials` seam for future `kairo-auth`.** Every provider takes an
  optional `credentials` kwarg. For now, build one by hand or from env
  vars; later, `kairo-auth` will hand one back with the same shape and no
  provider code will need to change.
- **`EmbeddingStore`** is a tiny in-memory numpy matrix with cosine top-k
  and JSON + `.npy` persistence. Swap for FAISS / LanceDB later by
  re-implementing the same interface.

## Providers shipped in Phase 2

| Name | Purpose | Install |
|---|---|---|
| `hash-stub` | Zero-dependency deterministic stub. Not semantically meaningful, but stable. Default, CI-friendly, lets you try Kairo without downloading a model. | included |
| `sentence-transformers` | Real local embedding model (default `all-MiniLM-L6-v2`, 384-dim). | `pip install -e "./embeddings[sentence-transformers]"` |

## Providers with reserved extras (not implemented yet)

`openai`, `cohere`, `voyage`, `bedrock` — each is one file under
`providers/` when we add it, and its optional dependency is already
declared in `pyproject.toml` so the install story is ready.

## Quickstart

```python
from kairo_embeddings import get_provider, EmbeddingStore

# Pick any backend — this is the one line users change.
provider = get_provider("hash-stub", dim=64)
# provider = get_provider("sentence-transformers", model="all-MiniLM-L6-v2")

store = EmbeddingStore(dim=provider.dim)

texts = {
    "a": "Temporal graphs capture evolving states.",
    "b": "Retrieval planning precedes generation.",
    "c": "Branching history enables rollback.",
}
vectors = provider.embed(list(texts.values()))
for (node_id, _), vec in zip(texts.items(), vectors):
    store.add(node_id, vec)

query = provider.embed(["how does history traversal work?"])[0]
for node_id, score in store.similar(query, k=2):
    print(node_id, round(score, 3))

# Round-trip
store.save(".kairo_tmp/emb.npy", ".kairo_tmp/index.json")
reloaded = EmbeddingStore.load(".kairo_tmp/emb.npy", ".kairo_tmp/index.json")
```

## Install

```bash
# base (hash-stub only; zero heavy deps)
pip install -e ./core ./embeddings

# with real local model
pip install -e ./core -e "./embeddings[sentence-transformers]"
```

## The `Credentials` seam

Providers that need API keys take a `credentials` kwarg:

```python
from kairo_embeddings import Credentials, get_provider
import os

creds = Credentials(api_key=os.environ["OPENAI_API_KEY"])
provider = get_provider("openai", model="text-embedding-3-small", credentials=creds)
```

When `kairo-auth` ships, you'll get the same `Credentials` object back
from a call like `kairo_auth.get("openai")`. No provider code changes.
