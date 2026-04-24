# Kairo Edge Architecture (KEA)

**Status:** in progress (14e.0 shipped) · **Owner:** kairo-edge package · **Last updated:** 2026-04-23

This document is the production plan for running Kairo on edge devices —
phones, Raspberry Pis, Chromebooks, browsers, and in-process embedded
contexts. It's inspired by how Google's LiteRT-LM structures a portable
on-device runtime, but designed from first principles for Kairo's
specific workload (retrieval + graph + agents, not just LLM inference).

Everything in this plan lands inside the `edge/` package. Heavy
dependencies stay in their original packages; nothing from `edge/` is
allowed to reach up into pandas, torch, statsmodels, or any C-extension
that won't build on ARM without pain.

---

## 1. Why a separate edge architecture

Kairo's main packages optimize for desktop / server deployments —
pydantic, networkx, rank_bm25, and optional torch / statsmodels / sklearn.
That stack is 1–3 GB installed and routinely needs 2+ GB of RAM at
query time. Edge targets can't afford any of that:

| Target | Typical RAM | Storage | Python? | LLM feasible? |
|---|---|---|---|---|
| Raspberry Pi Zero 2 W | 512 MB | 16 GB SD | yes | extractive only |
| Raspberry Pi 4 / 5 | 2–8 GB | 32 GB+ | yes | 1–3B quantized |
| Android (Termux) | 4–12 GB | 64 GB+ | yes | 1–3B quantized |
| iOS | 4–8 GB | 64 GB+ | no (native shell) | 1–3B via LiteRT-LM |
| Browser (Pyodide) | ~500 MB virt | IndexedDB | yes | via WebLLM/LiteRT Web |
| Browser (TS-native) | device RAM | IndexedDB | no | via WebGPU provider |

**The edge package exists because edge is a different product**, not a
smaller version of the same product. Desktop Kairo optimizes for quality;
edge Kairo optimizes for footprint and portability with acceptable
quality degradation.

---

## 2. Design principles

1. **Zero heavy dependencies in the edge core.** Pure Python + optional
   lightweight extras (`onnxruntime`, `llama-cpp-python`). No torch, no
   pandas, no networkx, no C-extensions without pre-built ARM wheels.

2. **Spec first, implementation second.** A wire-format contract
   ([Kairo Protocol](./KAIRO_PROTOCOL.md)) defines the JSON/binary shape
   of every artifact — graphs, snapshots, queries, provider manifests.
   Runtime shells in Python / TypeScript / Kotlin / Swift all read the
   same spec, so a `.kairo/` built on desktop is portable to any edge.

3. **Providers are pluggable; the runtime picks based on device probe.**
   `DeviceProfile` on startup detects RAM / CPU / GPU / NPU / WebGPU
   availability. `ProviderCapability` manifests declare requirements
   per-provider. `select_provider(query, profile)` picks the best fit.
   No provider is hard-coded.

4. **Degrade gracefully, never fail hard.** If no LLM is available,
   answer extractively (return top-k evidence passages). If no semantic
   embeddings, fall back to BM25-only. If no storage, stay in-memory.
   Every scenario has a runnable configuration, even on the smallest
   target.

5. **Measurable from day one.** The
   [emulator](../emulator/README.md) enforces memory / CPU / latency
   budgets per target profile. A scenario that blows the Pi Zero budget
   fails the test, not just shows a warning. Performance is a contract.

6. **One conceptual core, many language shells.** Python is the
   reference implementation. TypeScript ships for browser. Kotlin /
   Swift arrive when there's a real user driving native mobile. Every
   shell is a port of the same algorithms against the same spec — not
   a translation via FFI.

---

## 3. Non-goals

- **Not a rewrite of Kairo.** The desktop package stays. Edge is a
  sibling, not a replacement.
- **Not a single binary.** LiteRT-LM ships a fat C++ binary. Kairo-edge
  ships per-language packages — `kairo-edge` (Python), `@kairo/edge`
  (TS), etc. That matches how Python + JavaScript developers actually
  deploy today.
- **Not for microcontrollers.** ESP32-class devices can't run Python or
  modern JS. If that ever becomes a target, it'll be a separate
  Rust / C track.
- **Not a hardware abstraction layer.** We don't write GPU kernels.
  We delegate to existing runtimes (ONNX, llama.cpp, LiteRT-LM, WebGPU).
- **Not a fork of LiteRT-LM.** We compose with it as an optional
  provider; we don't reimplement LLM inference.

---

## 4. Target device matrix (production profiles)

Profile names are stable contracts — every scenario runs against each
profile in CI. Budgets are wall-clock on realistic hardware (not
emulated; the emulator just enforces the same limits).

| Profile | Peak RAM | Sustained CPU | Storage | Typical scenario |
|---|---|---|---|---|
| `pi-zero`    | 256 MB | 1× 1.0 GHz ARM | 8 GB  | BM25-only, no semantic, no LLM, extractive answer |
| `pi-4`       | 1 GB   | 4× 1.5 GHz ARM | 32 GB | BM25 + ONNX MiniLM semantic, 1B Q4 LLM |
| `pi-5`       | 2 GB   | 4× 2.4 GHz ARM | 32 GB | Full retrieval + 3B Q4 LLM |
| `phone-mid`  | 1 GB   | 4× 2.0 GHz ARM | 32 GB | Full retrieval + LiteRT-LM Gemma 2B int4 |
| `browser-lite` | 400 MB virt | WASM single-thread | 50 MB IndexedDB | TS shell, BM25 only, extractive |
| `browser-gpu`  | 1 GB virt | WebGPU | 500 MB IndexedDB | TS shell + WebLLM/LiteRT Web |

Per-profile budgets are enforced by the emulator and gate every
promotion from dev → release. A retrieval change that adds 50 MB peak
RSS on `pi-4` fails CI until it's optimized or the budget is renegotiated.

---

## 5. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  App shells (per platform)                                       │
│  kairo-cli · kairo-ui (web) · Android app · iOS app · Pi service │
├──────────────────────────────────────────────────────────────────┤
│  Runtime shells — native implementations of the same algorithms  │
│  Python (reference)  · TypeScript  · Kotlin  · Swift             │
│                                                                   │
│  Each shell provides:                                             │
│    • LocalPipeline-edge — BM25 + optional cosine + 1-hop walk    │
│    • KairoGraph-edge   — read / walk / serialize per spec        │
│    • ProviderRegistry  — register + select providers             │
│    • DeviceProfile     — probe + report capability               │
├──────────────────────────────────────────────────────────────────┤
│  Kairo Protocol — spec-only layer (JSON + optional binary)       │
│                                                                   │
│  docs/KAIRO_PROTOCOL.md defines:                                  │
│    • Graph wire format (nodes, edges, provenance, confidence)    │
│    • Snapshot manifest (.kairo/ directory spec)                  │
│    • Provider capability descriptor                              │
│    • Query / Evidence / Response envelopes                       │
│    • Validation rules                                            │
├──────────────────────────────────────────────────────────────────┤
│  Provider plugins — hot-swappable, runtime-selected              │
│                                                                   │
│  InferenceProvider   Ollama · llama.cpp · LiteRT-LM · WebLLM     │
│                      · extractive-only (no LLM)                  │
│  EmbeddingProvider   hash-stub · ONNX-MiniLM · onnxruntime-web   │
│                      · cloud (OpenAI / Voyage)                   │
│  StorageProvider     filesystem · IndexedDB · SQLite · memory    │
│  DeviceProfile       CPU / GPU / NPU / WebGPU / RAM / storage    │
└──────────────────────────────────────────────────────────────────┘
```

### Key architectural decisions

1. **Protocol is the contract.** Every runtime shell validates its
   artifacts against `docs/KAIRO_PROTOCOL.md` schemas. CI fails a shell
   whose output doesn't match the spec. This is what prevents drift
   between Python, TypeScript, and future shells.

2. **Runtime shells own their algorithms.** The TypeScript shell isn't
   Pyodide-wrapped Python — it's native TS implementing BM25 + RRF +
   Louvain + graph walk in ~800 lines. No FFI overhead; no Pyodide
   bloat. Python is canonical / reference.

3. **Provider capability manifests gate selection.** Every provider
   declares `min_ram_mb`, `needs_gpu`, `model_families`, etc. On
   startup the runtime runs `DeviceProfile.probe()` and calls
   `select_provider(query, profile)` to pick a provider that fits. If
   no inference provider fits, the runtime picks `extractive-only`.

4. **Artifacts are portable.** A `.kairo/` directory built by desktop
   Python is readable by `kairo-edge` Python, `@kairo/edge` TypeScript,
   and any future shell. Users can build on desktop, copy to phone,
   query locally.

5. **Ollama stays on desktop.** Edge targets get lighter providers.
   The `InferenceProvider` protocol hides the difference from the
   rest of Kairo — agents and retrieval don't know or care.

---

## 6. Phased roadmap

Each sub-phase is one package-scoped commit, matching the existing
Kairo cadence (11a-d, 13a-d style).

### ✅ Phase 14e.0 — Rust workspace + `kairo-edge-core` v0.1 (shipped 2026-04-23, commit `6cee66a`)

Architectural pivot: **Rust is the canonical edge runtime.** Python's
`kairo_edge` package stays working; it becomes a deprecation target
once the PyO3 bindings crate lands.

- `edge/rust/` Cargo workspace, release profile tuned for size
  (`opt-level=z`, `lto=true`, `strip=symbols`). Release `.rlib` is
  580 KB.
- `kairo-edge-core` ships: `Document` / `Evidence` / `QueryRequest` /
  `QueryResponse` types, pure-Rust Okapi BM25 (k1=1.2, b=0.75,
  zero-filter discipline matches Phase 13a's Python fix), JSON-backed
  `EdgeStore` with in-memory mode for no-FS runtimes.
- Wire-format parity: a Pydantic-shaped JSON from the Python side
  deserializes cleanly into the Rust types (test proves it). Stores
  written by either runtime are readable by the other.
- Tests: 7 green (2 unit + 5 integration). Covered BM25 semantics,
  store round-trip, metadata preservation, Python-JSON interop.

### Phase 14e.1 — Spec + validator (doc-first)
- Write `edge/docs/KAIRO_PROTOCOL.md` — JSON schemas for graph,
  snapshot, query, response, provider manifest.
- Write `edge/src/kairo_edge/protocol.py` — reference validator that
  checks Pydantic round-trip against the schemas.
- Retrofit `kairo-core`'s existing serialization to validate against
  the spec in CI.
- **Deliverable:** spec document + passing validator on every existing
  fixture.

### Phase 14e.2 — DeviceProfile + ProviderCapability manifests
- Add `edge/src/kairo_edge/capability.py` with `DeviceProfile.probe()`
  and `ProviderCapability` dataclass.
- Every existing provider gains a capability manifest file.
- `select_provider(kind, profile)` picks best fit; falls back
  gracefully when nothing fits.
- **Deliverable:** tests that show `pi-zero` profile selects `bm25`
  over `hybrid`, and `extractive-only` inference provider.

### Phase 14e.3 — EdgePipeline (BM25 + RRF + optional cosine)
- Replace the current term-frequency `EdgeStore` with `EdgePipeline`
  — same algorithms as `LocalPipeline` from `kairo-retrieval`, but
  pure-Python (no numpy fallback path) and provider-driven.
- BM25 implemented in pure Python (no `rank_bm25` dependency to
  keep install tiny).
- Cosine path activates only when an embedding provider is installed
  and device profile allows.
- **Deliverable:** `EdgePipeline` passes the same functional tests as
  `LocalPipeline` on the eval harness, minus the numpy-bound paths.

### Phase 14e.4 — EdgeGraph (pure-Python walk, no networkx)
- `kairo_edge.graph.EdgeGraph` — minimal graph implementation (adj
  lists + dicts), reads the Kairo Protocol wire format.
- 1-hop walk + BFS traversal matching `kairo-graph` semantics.
- Optional Louvain implementation (pure Python, ~150 lines) for
  community detection on-device.
- **Deliverable:** round-trips `.kairo/` artifacts built by
  `kairo-graph`; walk results match on the same inputs.

### Phase 14e.5 — ONNX embedding provider
- `kairo_edge.providers.embeddings.OnnxEmbeddings` — quantized
  `all-MiniLM-L6-v2` int8 (~25 MB model + `onnxruntime` ~60 MB
  runtime).
- Capability: `min_ram_mb=150`, `needs_gpu=False`, `arm_ok=True`.
- Replaces `sentence-transformers` as the real-semantic default on
  edge profiles.
- **Deliverable:** `hybrid_default` eval harness config runs on
  `pi-4` profile within memory budget.

### Phase 14e.6 — Inference providers for edge
- `kairo_edge.providers.inference.ExtractiveProvider` — no-LLM
  fallback that returns top-k evidence as "answer".
- `kairo_edge.providers.inference.LlamaCppProvider` —
  `llama-cpp-python` bindings, gated on `needs_gpu=False`,
  `min_ram_mb=800`.
- `kairo_edge.providers.inference.LiteRtLmProvider` (stub for later) —
  Google's runtime when/if Python bindings work on our targets.
- **Deliverable:** `pi-zero` profile runs extractive end-to-end;
  `pi-4` profile runs llama.cpp with a 1B quantized model.

### Phase 14e.7 — TypeScript shell (browser-first)
- New `edge/ts/` directory (mirrors `edge/src/` structure).
- Port `EdgePipeline` + `EdgeGraph` + `ProviderRegistry` to TS in
  ~800 lines.
- Ship as `@kairo/edge` npm package.
- Read the Kairo Protocol wire format identically to Python.
- Browser inference provider: WebLLM adapter.
- **Deliverable:** static HTML page that loads `@kairo/edge` + a
  `.kairo/` artifact + WebLLM, does end-to-end RAG client-side.

### Phase 14e.8 — Kotlin / Swift shells (native mobile)
- Deferred until TS shell validates the Protocol cross-language.
- Likely Kotlin first (JVM ecosystem more like Python), Swift later.
- Each shell is a port, not a wrapper. ~1000 lines per language.
- **Deliverable:** native Android / iOS apps bundling `.kairo/`
  artifacts with LiteRT-LM for inference.

---

## 7. Test and emulation strategy

### Three test tiers

1. **Unit tests** — algorithm correctness, run on developer machine
   in the normal pytest suite. No emulation.
2. **Emulator scenarios** — run inside the `edge/emulator/` harness,
   enforce per-profile budgets, measure time / memory / accuracy.
   Run in CI on every commit touching `edge/`.
3. **Device integration** — real hardware (Raspberry Pi, phone).
   Manual, gated on release candidates. Emulator must pass first.

### What the emulator does

The emulator is a resource-constrained Python harness that runs
Kairo scenarios against a named `DeviceProfile`. It:

- Allocates a memory budget from `DeviceProfile.peak_ram_mb` and
  tracks peak RSS during the scenario.
- Wall-clocks the scenario and compares to the profile's latency
  budget per phase (ingest, retrieve, generate).
- Restricts imports — if a scenario imports pandas on a `pi-zero`
  profile, the emulator fails it instantly.
- Loads only providers whose capability manifest matches the profile.
- Reports a structured JSON result: pass/fail per budget, metrics,
  evidence of what was chosen.

CI uses `pytest` + the emulator scenarios to gate every PR touching
the edge package. Adding a 50 MB static table to retrieval that pushes
`pi-4`'s RSS over budget fails CI immediately.

### What the emulator does NOT do

- It does not emulate ARM instructions on x86. x86 emulation (QEMU)
  is too slow for CI. The CPU budget is a wall-clock cap, not
  simulated cycles.
- It does not emulate GPU absence — if the host machine has CUDA,
  the emulator still disallows providers that declare `needs_gpu=True`
  on a non-GPU profile, via provider capability gating.
- It does not test browser-specific behavior (WebGPU, IndexedDB).
  A separate Playwright-based browser harness covers that, landing
  with Phase 14e.7.

---

## 8. Performance budgets (contracts, not suggestions)

These budgets are what the emulator enforces. Numbers are initial
proposals; will be tightened after Phase 14e.3 establishes baselines.

| Profile | Ingest (1k docs) | Retrieve (top-5) | Generate (100 tok) | Peak RAM |
|---|---|---|---|---|
| `pi-zero`    | ≤ 20s | ≤ 500 ms | N/A (extractive) | 256 MB |
| `pi-4`       | ≤ 10s | ≤ 200 ms | ≤ 15s @ 1B Q4   | 1 GB   |
| `pi-5`       | ≤ 5s  | ≤ 100 ms | ≤ 10s @ 3B Q4   | 2 GB   |
| `phone-mid`  | ≤ 8s  | ≤ 150 ms | ≤ 12s @ 2B int4 | 1 GB   |
| `browser-lite` | ≤ 15s | ≤ 300 ms | N/A              | 400 MB |
| `browser-gpu`  | ≤ 12s | ≤ 200 ms | ≤ 8s @ WebLLM   | 1 GB   |

Going over budget is a PR-blocking regression, same as a failing test.

---

## 9. Migration from the current edge package

`kairo-edge` ships `EdgeStore` today — a term-frequency retrieval
shim. The 14e phases will:

1. Keep `EdgeStore` importable for backwards compatibility but mark
   deprecated.
2. Introduce `EdgePipeline` alongside it (phase 14e.3).
3. Route `EdgeStore.query()` to `EdgePipeline` internally in a later
   minor version.
4. Remove `EdgeStore` in a 0.2 breaking release once downstream users
   migrate.

No user-facing code breaks in the interim.

---

## 10. Open questions

- **Should the Kairo Protocol include a binary format (CBOR /
  FlatBuffers) or JSON-only v1?** JSON is easier to debug and universal;
  binary is smaller and faster to load. Decision deferred to 14e.1 when
  we measure artifact sizes.
- **Does the ONNX embedding model need to be bundled with the package,
  or downloaded on first use?** Bundled is simpler for offline edge;
  download is lighter install. Decision: offer both paths via install
  extras — `kairo-edge[onnx]` downloads, `kairo-edge[onnx-bundled]`
  ships the model.
- **Is Pyodide a target, or just TypeScript?** Pyodide is easier for
  quick demos (reuses Python code) but bulky (~50 MB runtime).
  TypeScript is lighter and ships a better browser experience. Plan
  currently prefers TS; Pyodide stays as an optional fallback for
  "try Kairo in a browser without a TS build."
- **How do we handle `.kairo/` directories larger than a phone's free
  storage?** Likely a streaming read path that mmaps snapshots
  without loading all of them. Out of scope for 14e.1–14e.6; will
  surface naturally during phone profile testing.

---

## Appendix A — Relation to existing Kairo roadmap

Phase 14e runs in parallel with Phase 12 (STDP) — they compose
naturally. STDP on edge is the killer feature: the device learns
from its user's accepted / regenerated signals, data never leaves,
privacy is preserved by construction. That product pitch is why the
edge package is worth building, not just a shrunken desktop.

## Appendix B — Relation to LiteRT-LM

LiteRT-LM solves "how do we run Gemma on a phone." Kairo-edge solves
"how do we run RAG + graphs + agents on a phone." LiteRT-LM is a
potential `InferenceProvider` inside this architecture, not a
replacement for it. We can consume it when Python bindings mature; we
don't depend on it.
