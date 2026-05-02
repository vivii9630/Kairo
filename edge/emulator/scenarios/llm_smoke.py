"""LLM-track scenario: capability selection + soft-fall, plus real-backend
detection introduced in Phase 14e.4.2.

Drives the EdgePipeline with LlamaCppProvider in the candidate chain
and adapts assertions based on the build mode of `kairo_edge_py`:

  * **Default wheel build (no `real-llama-cpp` feature):**
    - On ``pi-zero`` / ``browser-lite``, RAM filtering excludes
      ``llama-cpp`` and the pipeline picks ``extractive`` directly.
    - On ``pi-4`` / ``pi-5`` / ``phone-mid`` / ``browser-gpu``,
      ``select_capability`` picks ``llama-cpp``; the scaffold
      backend errors at answer-time and the pipeline soft-falls to
      ``extractive``. ``selected_provider`` and ``provider`` differ;
      ``used_fallback`` is True.

  * **Wheel built with `--features real-llama-cpp` AND
    ``KAIRO_EDGE_LLAMA_MODEL`` set to a readable GGUF:**
    - Pi Zero / browser-lite still pick extractive.
    - LLM-capable profiles select ``llama-cpp`` AND `provider ==
      'llama-cpp'` AND `used_fallback is False`. Answer must be
      non-empty (real generation).

  * **Wheel built with `--features real-llama-cpp` BUT no model
    configured:** identical to the default-wheel soft-fall — the
    provider build sees no model and registers the scaffold so
    selection still works and extractive answers.
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
        from kairo_edge_py import (
            Document as RustDoc,
            EdgePipeline,
            real_llama_cpp_available,
        )
    except ImportError as exc:
        with ctx.phase("import_kairo_edge_py"):
            raise RuntimeError(
                "kairo-edge-py is not installed in this environment."
            ) from exc

    rust_docs = [RustDoc(id=d.id, text=d.text) for d in PYTHON_CORPUS]
    profile_name = ctx.profile.name
    model_path = os.environ.get("KAIRO_EDGE_LLAMA_MODEL")
    has_real = bool(real_llama_cpp_available())
    # The "real path" only holds when the wheel was built with the
    # feature AND a model file is on disk. Either missing → soft-fall
    # contract still applies.
    expect_real = has_real and model_path is not None and os.path.exists(model_path)

    with ctx.phase("ingest"):
        pipeline = EdgePipeline.with_llama_cpp(profile_name, model_path)
        pipeline.add_documents(rust_docs)

    with ctx.phase("retrieve"):
        resp = pipeline.query("felines hunt at night", top_k=3)

    # Verify the selection contract.
    if profile_name in _EXTRACTIVE_ONLY_PROFILES:
        # RAM gate excludes llama-cpp regardless of build mode.
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
        # llama-cpp must be in the candidate chain on these profiles
        # whether the build is real-backend or scaffold.
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

        if expect_real:
            # Real-backend path: provider used is llama-cpp directly,
            # no fallback, answer is non-empty model output.
            if resp.provider != "llama-cpp":
                raise AssertionError(
                    f"profile={profile_name} (real-backend on, model present) "
                    f"expected provider='llama-cpp', got {resp.provider!r}"
                )
            if resp.used_fallback:
                raise AssertionError(
                    f"profile={profile_name} (real-backend on, model present) "
                    f"expected used_fallback=False"
                )
            if not resp.answer.strip():
                raise AssertionError(
                    f"profile={profile_name} (real-backend on, model present) "
                    f"expected non-empty model answer, got empty string"
                )
        else:
            # Scaffold/soft-fall path: provider used is extractive, fallback set.
            if resp.provider != "extractive" or not resp.used_fallback:
                raise AssertionError(
                    f"profile={profile_name} (soft-fall path) expected "
                    f"extractive fallback, got provider={resp.provider!r} "
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
        "real_llama_cpp_available": has_real,
        "model_path_configured": model_path is not None,
        "real_backend_active": expect_real,
        "cited_node_ids": list(resp.cited_node_ids),
        "answer_preview": resp.answer[:120],
    }


register_scenario("llm_smoke", _scenario)
