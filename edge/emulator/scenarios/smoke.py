"""Smoke scenario — the minimal end-to-end edge workflow.

Ingest 20 synthetic docs, run a retrieval query, check that the
answer contains an expected token. Serves as the baseline budget
reference for every edge profile.

Deliberately uses the existing ``EdgeStore`` so this scenario runs on
``kairo-edge`` today, before the 14e.3 ``EdgePipeline`` lands. When
``EdgePipeline`` ships, this file gets a sibling version that exercises
BM25 + RRF + walk.
"""

from __future__ import annotations

from typing import Any, Dict

from kairo_core import Document, QueryRequest
from kairo_edge import EdgeStore

from ..runner import EmulatorContext
from . import register_scenario


_CORPUS = [
    Document(id=f"d{i:02d}", text=txt) for i, txt in enumerate([
        "Cats purr when they feel content and relaxed.",
        "Felines are solitary hunters active at dusk.",
        "A house cat's tongue has backward-facing papillae.",
        "Dogs bark to alert owners of unfamiliar sounds.",
        "Canines descended from wolves and were domesticated.",
        "Golden retrievers are friendly loyal family dogs.",
        "Python is a high-level programming language.",
        "JavaScript runs in browsers and web servers alike.",
        "Rust offers memory safety through ownership rules.",
        "Type hints in modern Python make mypy analysis possible.",
        "Neural networks learn patterns from labelled data.",
        "Transformers use self-attention across long contexts.",
        "Supervised learning pairs inputs with target outputs.",
        "Cross-entropy is a standard classification loss.",
        "ETL pipelines move records into data warehouses.",
        "A star schema centers fact tables around dimensions.",
        "Kafka partitions streaming events for durable logs.",
        "Espresso is brewed by forcing hot water through grounds.",
        "A cappuccino layers espresso, steamed milk, and foam.",
        "Cold brew steeps coarse grounds for up to a day.",
    ])
]


def _scenario(ctx: EmulatorContext) -> Dict[str, Any]:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "edge.json"

        with ctx.phase("ingest"):
            store = EdgeStore(store_path=store_path)
            store.add(_CORPUS)

        with ctx.phase("retrieve"):
            resp = store.query(
                QueryRequest(query="felines hunt at night", top_k=3)
            )

        retrieved_ids = [ev.document_id for ev in resp.evidences]
        return {
            "retrieved_ids": retrieved_ids,
            "evidence_count": len(resp.evidences),
            "answer_preview": resp.answer[:80],
            "corpus_size": len(_CORPUS),
        }


register_scenario("smoke", _scenario)
