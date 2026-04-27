# Kairo Edge Implementation Handoff

Last updated: 2026-04-25

## What changed in this work session

- Updated edge documentation to match the current Rust-first plan:
  - `edge/docs/README.md`
  - `edge/docs/EDGE_ARCHITECTURE.md`
  - `edge/docs/KAIRO_PROTOCOL.md`
  - `edge/rust/README.md`
- Added `PipelineResponse` provider transparency in Rust:
  - `provider`: provider that actually produced the answer.
  - `selected_provider`: provider selected by capability before execution.
  - `used_fallback`: true when the selected provider failed and extractive answered.
  - `provider_error`: selected-provider error message when fallback happened.
- Exposed that metadata through PyO3 `PipelineResponse`.
- Updated emulator scenario outputs to surface provider selection/fallback metadata.
- Updated Python `kairo_edge.EdgeStore` so it prefers the Rust/PyO3 backend when `kairo_edge_py` is importable and falls back to the pure-Python store otherwise.

## Current implementation state

- `kairo-edge-core` is still lightweight by default.
- `LlamaCppProvider` is currently a scaffold: it declares capability, validates model-path errors, builds prompts, and intentionally falls back through `EdgePipeline`.
- Real GGUF/llama.cpp inference is still future work for 14e.4.1 and should stay behind an optional boundary.
- The public Python edge API remains `EdgeStore.add(...)` and `EdgeStore.query(...)`; `backend` reports `rust` or `python`.

## Verification run

- `cargo test` in `edge/rust` passed after the metadata changes.
- `python -m compileall edge\src\kairo_edge edge\emulator\scenarios` passed.
- Pure-Python fallback smoke passed with `EdgeStore(None, prefer_rust=False)`.
- `python -m edge.emulator --profile pi-zero --scenario smoke --json` passed.

## Verification — resolved 2026-04-26

The wheel install/import gap from the previous note was worked around without
any global install or permission escalation:

1. `python -m maturin build --release` produced
   `edge/rust/target/wheels/kairo_edge_py-0.1.0-cp39-abi3-win_amd64.whl`.
2. The wheel was unpacked into `edge/rust/target/wheel_unpacked/` (a wheel is a
   regular zip; `python -c "import zipfile; zipfile.ZipFile(...).extractall(...)"`
   is sufficient — no `pip install` needed).
3. Emulator scenarios that need `kairo_edge_py` are run with
   `PYTHONPATH=edge/rust/target/wheel_unpacked`. Example:

```bash
PYTHONPATH="edge/rust/target/wheel_unpacked" \
  python -m edge.emulator --all-profiles --scenario llm_smoke
```

Verified green this session:

- `cargo test --workspace` — 30 unit + 5 integration green.
- `rust_inproc_smoke` — PASS on every profile.
- `edge_pipeline_smoke` — PASS on every profile.
- `llm_smoke` — PASS on every profile, with the selection contract holding:

  | Profile        | Selected   | Used        | Fallback |
  |----------------|------------|-------------|----------|
  | `pi-zero`      | extractive | extractive  | False    |
  | `browser-lite` | extractive | extractive  | False    |
  | `pi-4`         | llama-cpp  | extractive  | True     |
  | `pi-5`         | llama-cpp  | extractive  | True     |
  | `phone-mid`    | llama-cpp  | extractive  | True     |
  | `browser-gpu`  | llama-cpp  | extractive  | True     |

  llama-cpp is the *selected* provider on profiles whose RAM clears 800 MB; the
  current 14e.4 stub `answer()` errors and the pipeline soft-falls to
  extractive, which is exactly the contract the response transparency fields
  document. When 14e.4.1 wires the real llama.cpp backend the same scenario
  will produce a generated answer on the LLM-capable rows without code change.

The wheel install issue itself is an environment problem, not a code bug: a
fresh shell on Windows tends to surface either the system `C:\Python310\python.exe`
or the project `.venv` Python depending on how the prompt was opened, and
`maturin develop` installs into whichever one is currently active. The
PYTHONPATH path above sidesteps that ambiguity by not relying on any
site-packages location at all.

## Recommended next steps

1. Add/update tests for Python `EdgeStore` backend selection (rust vs python).
2. Decide whether to commit current 14e.4 scaffold before starting real
   llama.cpp integration. _(Done at commit time alongside this note.)_
3. Begin 14e.4.1 — the actual `llama-cpp-2` backend, behind an optional
   feature flag or sibling crate so the default core stays lightweight.
