"""LLM-track scenario: capability selection + soft-fall.

Drives the EdgePipeline with LlamaCppProvider in the candidate chain.
Asserts:

  * On ``pi-zero`` / ``browser-lite``, RAM filtering excludes
    ``llama-cpp`` and the pipeline picks ``extractive`` as the only
    candidate.
  * On ``pi-4`` / ``pi-5`` / ``phone-mid`` / ``browser-gpu``,
    ``select_capability`` picks ``llama-cpp``. The 14e.4 stub
    backend errors at answer-time; the pipeline's soft-fall returns
    a grounded extractive answer instead so the user is never
    left empty-handed. The response reports both ``selected_provider``
    and ``provider`` so the fallback is visible.

Once 14e.4.1 wires the actual ``llama-cpp-2`` backend + a model file
is configured via ``KAIRO_EDGE_LLAMA_MODEL``, this same scenario
will produce a generated answer on the LLM-capable profiles without
any code change here.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..runner import EmulatorContext
from . import register_scenario
from .smoke import _CORPUS as PYTHON_CORPUS


# Profiles where LlamaCppProvider should be selected (when present).
_LLM_PROFILES = {"pi-4", "pi-5", "phone-mid", "browser-gpu"}
_EXTRACTIVE_ONLY_PROFILES = {"pi-zero", "browser-lite"}


def _scenario(ctx: EmulatorContext) -> Dict[str, Any]:
    try:
        from kairo_edge_py import Document as RustDoc, EdgePipeline
    except ImportError as exc:
        with ctx.phase("import_kairo_edge_py"):
            raise RuntimeError(
                "kairo-edge-py is not installed in this environment."
            ) from exc

    rust_docs = [RustDoc(id=d.id, text=d.text) for d in PYTHON_CORPUS]
    profile_name = ctx.profile.name
    model_path = os.environ.get("KAIRO_EDGE_LLAMA_MODEL")

    with ctx.phase("ingest"):
        pipeline = EdgePipeline.with_llama_cpp(profile_name, model_path)
        pipeline.add_documents(rust_docs)

    with ctx.phase("retrieve"):
        resp = pipeline.query("felines hunt at night", top_k=3)

    # Verify the selection contract.
    if profile_name in _EXTRACTIVE_ONLY_PROFILES:
        if resp.provider != "extractive":
            raise AssertionError(
                f"profile={profile_name} expected extractive, "
                f"got {resp.provider!r}"
            )
        if resp.selected_provider != "extractive" or resp.used_fallback:
            raise AssertionError(
                f"profile={profile_name} expected direct extractive, "
                f"got selected_provider={resp.selected_provider!r} "
                f"used_fallback={resp.used_fallback!r}"
            )
    elif profile_name in _LLM_PROFILES:
        # When the stub backend errors, EdgePipeline soft-falls to
        # extractive — that's the *response* provider name. The fact
        # that llama-cpp was selected is reflected in the chain.
        if "llama-cpp" not in pipeline.provider_names:
            raise AssertionError(
                f"profile={profile_name} expected llama-cpp in chain, "
                f"got {pipeline.provider_names!r}"
            )
        if resp.selected_provider != "llama-cpp":
            raise AssertionError(
                f"profile={profile_name} expected selected_provider='llama-cpp', "
                f"got {resp.selected_provider!r}"
            )
        if resp.provider != "extractive" or not resp.used_fallback:
            raise AssertionError(
                f"profile={profile_name} expected extractive fallback, "
                f"got provider={resp.provider!r} "
                f"used_fallback={resp.used_fallback!r}"
            )

    return {
        "backend": "rust-edge-pipeline-with-llama-cpp",
        "profile_name": pipeline.profile_name,
        "provider_chain": pipeline.provider_names,
        "selected_provider": resp.selected_provider,
        "provider_used": resp.provider,
        "used_fallback": resp.used_fallback,
        "provider_error": resp.provider_error,
        "model_path_configured": model_path is not None,
        "cited_node_ids": list(resp.cited_node_ids),
        "answer_preview": resp.answer[:120],
    }


register_scenario("llm_smoke", _scenario)
