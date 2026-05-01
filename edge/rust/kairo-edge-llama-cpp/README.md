# kairo-edge-llama-cpp

Real GGUF inference backend for the Kairo edge runtime. Wraps the
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) crate and
exposes a `LlamaCppRuntime` that implements `InferenceProvider`, so
`EdgePipeline` can produce model-generated answers instead of soft-falling
to extractive on Pi 4 / Pi 5 / phone-mid / browser-gpu profiles.

## Why this is a separate crate

`llama-cpp-2` pulls a C/C++ build of `llama.cpp` via
`llama-cpp-sys-2`. The first compile is slow (5–15 minutes on a clean
system, requires CMake + a working C++ toolchain). Keeping that
complexity out of `kairo-edge-core` means:

- The default `cargo build` and `cargo test` (no flags) — which only hit
  `default-members` in the workspace — stay fast and pure-Rust.
- Edge users who don't need on-device LLM (Pi Zero, browser-lite,
  retrieval-only setups) never see the dependency.
- ARM cross-compile setup is opt-in, not forced on every contributor.

## Build

```bash
# From the workspace root:
cd edge/rust
cargo build -p kairo-edge-llama-cpp                # debug
cargo build -p kairo-edge-llama-cpp --release      # release (recommended)
```

Or pass `--workspace` to opt every member in:

```bash
cargo test --workspace                             # tests every crate
cargo test --workspace --exclude kairo-edge-llama-cpp   # default-members only
```

## Usage

```rust
use kairo_edge_core::{DeviceProfile, EdgePipeline, EdgeStore, Document};
use kairo_edge_llama_cpp::LlamaCppRuntime;
use std::sync::Arc;

let runtime = LlamaCppRuntime::load("/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")?;
let pipeline = EdgePipeline::new(
    EdgeStore::in_memory(),
    DeviceProfile::pi_4(),
    vec![Arc::new(runtime)],
);

pipeline.add_documents([Document::new("d1", "...")])?;
let response = pipeline.query(&request);
println!("{}", response.answer);
println!("provider used: {}", response.provider);
```

## Tests

Unit tests for capability declaration run as part of the regular
`cargo test -p kairo-edge-llama-cpp` and don't require a model file.

The integration test `generates_a_response_when_model_is_configured`
in `tests/real_model.rs` runs **only** when `KAIRO_EDGE_LLAMA_MODEL`
points at a readable GGUF file. Without the env var set, the test
prints a "skipped" line and passes — so `cargo test --workspace` on
CI machines without a model never fails.

```bash
KAIRO_EDGE_LLAMA_MODEL=/path/to/model.gguf \
  cargo test -p kairo-edge-llama-cpp
```
