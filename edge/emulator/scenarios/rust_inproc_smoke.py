"""In-process Rust scenario via the kairo-edge-py PyO3 wheel.

Same workload as ``smoke`` and ``rust_smoke``, but Python calls the
Rust ``EdgeStore`` directly via the PyO3 bindings — no subprocess. Use
this scenario alongside ``rust_smoke`` to see the Windows process-spawn
overhead (~20 ms) vs. the actual algorithmic cost.

Skipped silently when the ``kairo_edge_py`` extension isn't installed,
so the broader emulator suite keeps running on machines that haven't
built the wheel yet (e.g. fresh CI before the Rust step).
"""

from __future__ import annotations

from typing import Any, Dict

from ..runner import EmulatorContext
from . import register_scenario
from .smoke import _CORPUS as PYTHON_CORPUS


def _scenario(ctx: EmulatorContext) -> Dict[str, Any]:
    try:
        from kairo_edge_py import Document as RustDoc, EdgeStore as RustStore
    except ImportError as exc:
        # Surface the missing extension as a phase failure so the run
        # is honest about why it didn't run, rather than silently
        # passing zero work.
        with ctx.phase("import_kairo_edge_py"):
            raise RuntimeError(
                "kairo-edge-py is not installed in this environment. "
                "Build it with: cd edge/rust/kairo-edge-py && "
                "python -m maturin develop --release"
            ) from exc

    rust_docs = [RustDoc(id=d.id, text=d.text) for d in PYTHON_CORPUS]

    with ctx.phase("ingest"):
        store = RustStore()
        store.add(rust_docs)

    with ctx.phase("retrieve"):
        resp = store.query("felines hunt at night", top_k=3)

    retrieved_ids = [ev.document_id for ev in resp.evidences]
    return {
        "backend": "rust-via-pyo3",
        "retrieved_ids": retrieved_ids,
        "evidence_count": len(resp.evidences),
        "answer_preview": resp.answer[:80],
        "corpus_size": len(rust_docs),
    }


register_scenario("rust_inproc_smoke", _scenario)
