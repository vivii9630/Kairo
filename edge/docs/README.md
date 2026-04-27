# Kairo edge docs

Navigation for everything that lives under `edge/`. The code lives at
[`../src/`](../src/) (Python, deprecation target),
[`../rust/`](../rust/) (Rust, canonical going forward), and
[`../emulator/`](../emulator/) (test harness).

## Documents in this folder

- **[`EDGE_ARCHITECTURE.md`](./EDGE_ARCHITECTURE.md)** — master plan.
  Principles, target device matrix (pi-zero / pi-4 / pi-5 / phone-mid
  / browser-lite / browser-gpu), four-layer architecture, sub-phases
  14e.0 through 14e.8, performance contracts, test strategy, open
  questions. Start here.
- **[`KAIRO_PROTOCOL.md`](./KAIRO_PROTOCOL.md)** — v0.1 wire-format
  spec for `Document` / `Evidence` / `QueryRequest` / `QueryResponse`
  and the `.kairo_edge.json` store file. Versioning policy, encoding
  rules, tolerance contract, conformance criteria. Captures what
  `kairo-edge-core` (Rust) and `kairo_edge` (Python) already emit;
  the cross-runtime validator implementation lands with the fixtures
  in a follow-up commit.

## Related, outside this folder

- **[`../rust/README.md`](../rust/README.md)** — Rust workspace layout,
  cargo commands, wire-format contract, runtime target roadmap.
- **[`../emulator/README.md`](../emulator/README.md)** — how to run
  the edge emulator, write scenarios, add device profiles, gate CI.
- **[`../README.md`](../README.md)** — the edge package README (Python
  side, will be rewritten once Rust reaches parity).

## Current status

| Sub-phase | What | Status |
|---|---|---|
| 14e.0 | Rust workspace + `kairo-edge-core` v0.1 (types, BM25, JSON store) | ✅ shipped (`6cee66a`) |
| 14e.0.1 | `kairo-edge-cli` + `rust_smoke` emulator scenario | ✅ shipped (`f040363`) |
| 14e.0.2 | PyO3 bridge — `kairo-edge-py` wheel + in-process emulator scenario | ✅ shipped (`a895802`) |
| 14e.1 | Kairo Protocol spec v0.1 (doc-only) | ✅ shipped (`bdd21c6`) |
| 14e.1.x | Cross-runtime validator + fixtures | ⏳ next |
| 14e.2 | `DeviceProfile` + `ProviderCapability` + extractive provider | ✅ shipped (`e638ea2`) |
| 14e.3 | `EdgePipeline` with Rust BM25 + extractive baseline + PyO3 wrapper | ✅ shipped (`3e1ba8b`) |
| 14e.4 | `LlamaCppProvider` scaffold + `llm_smoke` + explicit fallback metadata | 🚧 current work |
| 14e.4.1 | Real llama.cpp / GGUF inference behind an optional boundary | ⏳ next |
| 14e.5 | `EdgeGraph` (pure-Rust walk, no networkx) | ⏳ |
| 14e.6 | ONNX embedding provider | ⏳ |
| 14e.6.x | Inference provider expansion (LiteRT-LM / WebLLM adapters) | ⏳ |
| 14e.7 | TypeScript shell (browser-first) | ⏳ |
| 14e.8 | Kotlin / Swift shells (native mobile) | ⏳ deferred |

## Current implementation plan

The active edge milestone is small-LLM answering on Raspberry Pi and
mobile-class devices, while preserving extractive fallback on every
profile. Implementation order:

1. Keep `kairo-edge-core` lightweight and Rust-first; real LLM backends
   must be optional so retrieval-only installs stay small.
2. Make Python `kairo_edge.EdgeStore` prefer the PyO3 Rust backend when
   `kairo_edge_py` is installed, falling back to the pure-Python store.
3. Keep `EdgePipeline` honest about provider behavior by reporting both
   the selected provider and the provider that actually answered.
4. Wire `LlamaCppProvider` as a capability-gated scaffold first; real
   llama.cpp inference follows in 14e.4.1 behind `KAIRO_EDGE_LLAMA_MODEL`.
5. Add protocol fixtures/validator before expanding portable `.kairo/`
   graph and snapshot support.

Everything from 14e.1 onward lands in Rust first (`edge/rust/`) and
only then gets a Python/PyO3 or TS binding. Matches the project memory
at `project_rust_edge.md`: **no more pure-Python `EdgeStore` features**,
only deprecation cleanup.
