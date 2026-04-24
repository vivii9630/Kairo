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
| 14e.1 | Kairo Protocol spec + validator | ⏳ next |
| 14e.2 | `DeviceProfile` + `ProviderCapability` manifests | ⏳ |
| 14e.3 | `EdgePipeline` (BM25 + RRF + optional cosine) | ⏳ |
| 14e.4 | `EdgeGraph` (pure-Rust walk, no networkx) | ⏳ |
| 14e.5 | ONNX embedding provider | ⏳ |
| 14e.6 | Inference providers (extractive, llama.cpp, LiteRT-LM) | ⏳ |
| 14e.7 | TypeScript shell (browser-first) | ⏳ |
| 14e.8 | Kotlin / Swift shells (native mobile) | ⏳ deferred |

Everything from 14e.1 onward lands in Rust first (`edge/rust/`) and
only then gets a Python/PyO3 or TS binding. Matches the project memory
at `project_rust_edge.md`: **no more pure-Python `EdgeStore` features**,
only deprecation cleanup.
