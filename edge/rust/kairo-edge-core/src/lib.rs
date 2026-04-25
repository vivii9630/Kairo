//! Kairo edge core — canonical Rust implementation.
//!
//! This crate provides the same conceptual surface as the Python
//! `kairo_edge` package today (`Document`, `Evidence`, `QueryRequest`,
//! `QueryResponse`, a lexical store) but in Rust so it can:
//!
//!   * compile to native Linux/ARM for Raspberry Pi class targets
//!     without the CPython interpreter overhead (~30–60 MB saved),
//!   * compile to `wasm32-unknown-unknown` for browser runtimes with
//!     a sub-MB download (vs. Pyodide's tens of MB),
//!   * expose bindings to Python, Kotlin, and Swift later via PyO3 /
//!     UniFFI without re-implementing the algorithms per language.
//!
//! The wire format for `Document` / `Evidence` / `QueryRequest` /
//! `QueryResponse` is deliberately the same JSON shape Python's
//! Pydantic models emit, so a `.kairo_edge.json` store written by
//! either runtime is readable by the other.
//!
//! Algorithms and scope follow
//! [`edge/docs/EDGE_ARCHITECTURE.md`](../../docs/EDGE_ARCHITECTURE.md).
//! v0.1.0 scope: BM25 over a small corpus, JSON-backed store, zero
//! external runtime deps beyond `serde`/`serde_json`.

pub mod bm25;
pub mod capability;
pub mod providers;
pub mod store;
pub mod tokenize;
pub mod types;

pub use bm25::{Bm25Index, BM25_B, BM25_K1};
pub use capability::{
    select_capability, DeviceProfile, ProviderCapability, ProviderKind,
};
pub use providers::{ExtractiveProvider, InferenceProvider, ProviderError};
pub use store::EdgeStore;
pub use types::{Document, Evidence, Metadata, QueryRequest, QueryResponse};
