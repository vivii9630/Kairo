"""Phase 11d / 13a — retrieval eval harness.

Runs a small labeled query set against each available LocalPipeline
config and reports recall@k, MRR, and nDCG@k in a comparison table.

Configs it will try:

  - ``bm25_only``         — LocalPipeline with no embedding provider
  - ``hybrid_hash``       — LocalPipeline + hash-stub (expected poor; included
                            so you can see the fusion pipeline works even
                            when the semantic branch is garbage)
  - ``hybrid_hash+walk``  — same but with a 1-hop graph walk over similarity
                            edges built from hash-stub embeddings
  - ``hybrid_default``    — LocalPipeline + get_provider("default"). Skipped
                            automatically when sentence-transformers isn't
                            installed.
  - ``hybrid_default+walk`` — same plus 1-hop graph walk over real-embedding
                              similarity edges.

Usage::

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 3
    python scripts/eval_retrieval.py --data scripts/eval_data.json

The eval data JSON has two keys: ``corpus`` (list of ``{id, text}``) and
``queries`` (list of ``{query, relevant: [doc_id, ...]}``).
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from kairo_core import Document, QueryRequest
from kairo_embeddings import EmbeddingProvider, available_providers, get_provider
from kairo_graph import DocumentGraphBuilder
from kairo_retrieval import LocalPipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "scripts" / "eval_data.json"


@dataclass
class EvalQuery:
    query: str
    relevant: List[str]


@dataclass
class EvalResult:
    config: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    n_queries: int


def _load_data(path: Path) -> tuple[List[Document], List[EvalQuery]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus = [Document(id=row["id"], text=row["text"]) for row in data["corpus"]]
    queries = [EvalQuery(query=q["query"], relevant=list(q["relevant"])) for q in data["queries"]]
    return corpus, queries


def _recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def _mrr(retrieved: List[str], relevant: List[str]) -> float:
    rel_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel_set:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in rel_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _run_config(
    name: str,
    factory: Callable[[Path], LocalPipeline],
    corpus: List[Document],
    queries: List[EvalQuery],
    k: int,
) -> EvalResult:
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / f"{name}.json"
        pipe = factory(store_path)
        pipe.ingest_documents(corpus)

        recalls: List[float] = []
        mrrs: List[float] = []
        ndcgs: List[float] = []
        for q in queries:
            res = pipe._retrieve(QueryRequest(query=q.query, top_k=k))
            retrieved = [ev.document_id for ev in res.evidences]
            recalls.append(_recall_at_k(retrieved, q.relevant, k))
            mrrs.append(_mrr(retrieved, q.relevant))
            ndcgs.append(_ndcg_at_k(retrieved, q.relevant, k))

        return EvalResult(
            config=name,
            recall_at_k=sum(recalls) / len(recalls),
            mrr=sum(mrrs) / len(mrrs),
            ndcg_at_k=sum(ndcgs) / len(ndcgs),
            n_queries=len(queries),
        )


def _build_graph_neighbors(
    corpus: List[Document],
    provider: EmbeddingProvider,
    *,
    k: int = 5,
    threshold: float = 0.2,
) -> Callable[[str], Iterable[Tuple[str, float]]]:
    """Build a similarity graph over *corpus* and return a neighbor lookup.

    The graph is built once with DocumentGraphBuilder using *provider*'s
    embeddings; the returned callable then services per-doc neighbor queries
    from an in-memory adjacency dict (handles both edge directions since
    similarity edges are stored with lower-id-first orientation).
    """
    builder = DocumentGraphBuilder(
        embedding_provider=provider,
        similarity_k=k,
        similarity_threshold=threshold,
    )
    graph = builder.build(corpus)
    nx_g = graph.nx_graph

    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for u, v, data in nx_g.edges(data=True):
        if data.get("kind") != "similar-to":
            continue
        weight = float(data.get("weight", 0.0))
        adjacency.setdefault(u, []).append((v, weight))
        adjacency.setdefault(v, []).append((u, weight))

    def neighbors(doc_id: str) -> List[Tuple[str, float]]:
        return adjacency.get(doc_id, [])

    return neighbors


def _configs(corpus: List[Document]) -> Dict[str, Callable[[Path], LocalPipeline]]:
    hash_provider = get_provider("hash-stub", dim=64)
    hash_neighbors = _build_graph_neighbors(corpus, hash_provider)

    configs: Dict[str, Callable[[Path], LocalPipeline]] = {
        "bm25_only": lambda p: LocalPipeline(store_path=p),
        "hybrid_hash": lambda p: LocalPipeline(
            store_path=p,
            embedding_provider=hash_provider,
        ),
        "hybrid_hash+walk": lambda p: LocalPipeline(
            store_path=p,
            embedding_provider=hash_provider,
            graph_neighbors=hash_neighbors,
        ),
    }
    if "sentence-transformers" in available_providers():
        default_provider = get_provider("default")
        default_neighbors = _build_graph_neighbors(corpus, default_provider)
        configs["hybrid_default"] = lambda p: LocalPipeline(
            store_path=p,
            embedding_provider=default_provider,
        )
        configs["hybrid_default+walk"] = lambda p: LocalPipeline(
            store_path=p,
            embedding_provider=default_provider,
            graph_neighbors=default_neighbors,
        )
    return configs


def _print_table(results: List[EvalResult], k: int) -> None:
    header = f"{'config':<24} {'recall@' + str(k):<12} {'MRR':<8} {'nDCG@' + str(k):<10} n"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.config:<24} "
            f"{r.recall_at_k:<12.3f} "
            f"{r.mrr:<8.3f} "
            f"{r.ndcg_at_k:<10.3f} "
            f"{r.n_queries}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=5, help="top-k cutoff for metrics")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)

    corpus, queries = _load_data(args.data)
    results: List[EvalResult] = []
    for name, factory in _configs(corpus).items():
        print(f"running {name} ...")
        results.append(_run_config(name, factory, corpus, queries, args.k))
    print()
    _print_table(results, args.k)

    if "sentence-transformers" not in available_providers():
        print()
        print(
            "note: sentence-transformers provider unavailable — `hybrid_default` "
            "and `hybrid_default+walk` were skipped. either it isn't installed "
            "(`pip install -e \"./embeddings[sentence-transformers]\"`) or its "
            "transitive deps failed to import in this env (the registry "
            "swallows the error and degrades to hash-stub)."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
