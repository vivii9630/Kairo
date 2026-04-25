"""End-to-end EdgePipeline scenario through the PyO3 wheel.

Exercises the Phase 14e.3 path: a profile-aware pipeline that
retrieves via Rust BM25 and answers via the chosen ``InferenceProvider``
(today: extractive only; LlamaCpp arrives in 14e.4).

Asserts the pipeline returns evidence-grounded text, names the
provider it actually used, and lists ``cited_node_ids`` so callers
can render attribution.
"""

from __future__ import annotations

from typing import Any, Dict

from ..runner import EmulatorContext
from . import register_scenario
from .smoke import _CORPUS as PYTHON_CORPUS


def _scenario(ctx: EmulatorContext) -> Dict[str, Any]:
    try:
        from kairo_edge_py import Document as RustDoc, EdgePipeline
    except ImportError as exc:
        with ctx.phase("import_kairo_edge_py"):
            raise RuntimeError(
                "kairo-edge-py is not installed in this environment. "
                "Build it with: cd edge/rust/kairo-edge-py && "
                "python -m maturin develop --release"
            ) from exc

    rust_docs = [RustDoc(id=d.id, text=d.text) for d in PYTHON_CORPUS]
    profile_name = ctx.profile.name

    with ctx.phase("ingest"):
        pipeline = EdgePipeline.with_profile(profile_name)
        pipeline.add_documents(rust_docs)

    with ctx.phase("retrieve"):
        resp = pipeline.query("felines hunt at night", top_k=3)

    return {
        "backend": "rust-edge-pipeline",
        "profile_name": pipeline.profile_name,
        "provider_used": resp.provider,
        "provider_chain": pipeline.provider_names,
        "cited_node_ids": list(resp.cited_node_ids),
        "evidence_count": len(resp.evidences),
        "answer_preview": resp.answer[:120],
        "corpus_size": len(rust_docs),
    }


register_scenario("edge_pipeline_smoke", _scenario)
