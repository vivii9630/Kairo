"""Phase 11d — retrieval eval harness.

Runs a small labeled query set against each available LocalPipeline
config and reports recall@k, MRR, and nDCG@k in a comparison table.

Configs it will try:

  - ``bm25_only``     — LocalPipeline with no embedding provider
  - ``hybrid_hash``   — LocalPipeline + hash-stub (expected poor; included
                        so you can see that the fusion pipeline itself
                        works even when the semantic branch is garbage)
  - ``hybrid_default``— LocalPipeline + get_provider("default"). Skipped
                        automatically when sentence-transformers isn't
                        installed.

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
from typing import Callable, Dict, List, Optional

from kairo_core import Document, QueryRequest
from kairo_embeddings import available_providers, get_provider
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


def _configs() -> Dict[str, Callable[[Path], LocalPipeline]]:
    configs: Dict[str, Callable[[Path], LocalPipeline]] = {
        "bm25_only": lambda p: LocalPipeline(store_path=p),
        "hybrid_hash": lambda p: LocalPipeline(
            store_path=p,
            embedding_provider=get_provider("hash-stub", dim=64),
        ),
    }
    if "sentence-transformers" in available_providers():
        configs["hybrid_default"] = lambda p: LocalPipeline(
            store_path=p,
            embedding_provider=get_provider("default"),
        )
    return configs


def _print_table(results: List[EvalResult], k: int) -> None:
    header = f"{'config':<20} {'recall@' + str(k):<12} {'MRR':<8} {'nDCG@' + str(k):<10} n"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.config:<20} "
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
    for name, factory in _configs().items():
        print(f"running {name} ...")
        results.append(_run_config(name, factory, corpus, queries, args.k))
    print()
    _print_table(results, args.k)

    if "sentence-transformers" not in available_providers():
        print()
        print(
            "note: sentence-transformers not installed — `hybrid_default` "
            "was skipped. install with:  pip install -e \"./embeddings[sentence-transformers]\""
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
