# Kairo edge — Rust workspace

Canonical Rust implementation of Kairo's edge runtime, per the plan in
[`../docs/EDGE_ARCHITECTURE.md`](../docs/EDGE_ARCHITECTURE.md).

Today this ships the Rust core, a small CLI, and PyO3 bindings for
Python. The workspace shape is set up so future crates (WASM target,
optional LLM backends) slot in alongside without restructuring.

| Crate | Status | Purpose |
|---|---|---|
| `kairo-edge-core` | ✅ v0.1 | Core types, BM25 index, JSON-backed `EdgeStore`, capability selection, `EdgePipeline`, extractive fallback, and the `LlamaCppProvider` scaffold. |
| `kairo-edge-py`   | ✅ v0.1 | PyO3 bindings so Python can call Rust `EdgeStore` / `EdgePipeline` in-process. |
| `kairo-edge-wasm` | ⏳ later | `wasm32-unknown-unknown` target for browser runtime. |
| `kairo-edge-cli`  | ✅ v0.1 | Small JSON stdin/stdout binary for emulator and Pi-class subprocess use. |

## Getting started

```bash
# From repo root, enter the Rust workspace
cd edge/rust

# Build everything
cargo build

# Run tests — includes integration tests that round-trip through
# the on-disk JSON store format.
cargo test

# Release build (size-optimized)
cargo build --release
```

## Python / PyO3 dev install

The emulator scenarios `rust_inproc_smoke`, `edge_pipeline_smoke`, and
`llm_smoke` import `kairo_edge_py`. Build and install the extension into
the active Python environment before running those scenarios:

```bash
cd edge/rust/kairo-edge-py
python -m maturin develop --release
```

After that, from the repo root:

```bash
python -m edge.emulator --profile pi-4 --scenario edge_pipeline_smoke --json
python -m edge.emulator --profile pi-4 --scenario llm_smoke --json
```

`llm_smoke` does not require a real model yet; the current
`LlamaCppProvider` scaffold verifies provider selection and fallback.
When 14e.4.1 lands, set `KAIRO_EDGE_LLAMA_MODEL` to a local GGUF file to
exercise real generation.

## Wire format parity

The JSON shape of every persisted artifact matches the Python side so a
store written by either runtime is readable by the other:

```json
{
  "documents": [
    { "id": "d01", "text": "...", "metadata": { "source": "pytest" } }
  ]
}
```

Unknown fields deserialize silently via serde defaults — both runtimes
can evolve their internal representations without breaking cross-runtime
reads, as long as the core fields stay stable.

## Runtime targets

| Target | Status | Notes |
|---|---|---|
| `x86_64-pc-windows-msvc` | ✅ default | dev loop |
| `aarch64-unknown-linux-gnu` | planned | Raspberry Pi / generic ARM |
| `wasm32-unknown-unknown` | planned | Browser runtime |
| `aarch64-apple-darwin` | planned | macOS Apple Silicon / iOS sim |

Release profile in [`Cargo.toml`](./Cargo.toml) is tuned for binary
size (`opt-level = "z"`, `lto = true`, `strip = "symbols"`) so the
resulting artifacts fit comfortably inside the edge profiles defined
in [`../emulator/profile.py`](../emulator/profile.py).
