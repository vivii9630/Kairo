# Kairo edge — Rust workspace

Canonical Rust implementation of Kairo's edge runtime, per the plan in
[`../docs/EDGE_ARCHITECTURE.md`](../docs/EDGE_ARCHITECTURE.md).

Today this ships one crate; the workspace shape is set up so future
crates (PyO3 bindings, WASM target, CLI) slot in alongside without
restructuring.

| Crate | Status | Purpose |
|---|---|---|
| `kairo-edge-core` | ✅ v0.1 | Core types, BM25 index, JSON-backed `EdgeStore`. Wire-format-compatible with the Python `kairo_edge` store. |
| `kairo-edge-py`   | ⏳ next | PyO3 bindings so `import kairo_edge` on the Python side routes through Rust. |
| `kairo-edge-wasm` | ⏳ later | `wasm32-unknown-unknown` target for browser runtime. |
| `kairo-edge-cli`  | ⏳ later | Small `kairo-edge` binary for Pi-class devices. |

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
