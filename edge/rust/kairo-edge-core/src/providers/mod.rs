//! Inference / embedding / storage provider traits.
//!
//! v0.1 ships only `InferenceProvider` with one concrete impl
//! (`ExtractiveProvider`). Embedding + storage provider traits land
//! when ONNX (14e.5) and IndexedDB (later) need them.

pub mod extractive;

use crate::types::Evidence;
use crate::capability::ProviderCapability;

/// Errors a provider can surface to the caller.
#[derive(Debug)]
pub enum ProviderError {
    /// The provider was given inputs it can't satisfy (empty
    /// evidence, query past context limit, etc.).
    InvalidInput(String),
    /// Backend transport / library / model load failed.
    Backend(String),
    /// Provider was reached but the underlying model is incompatible
    /// (wrong family, wrong format).
    Incompatible(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::Backend(m) => write!(f, "backend error: {m}"),
            Self::Incompatible(m) => write!(f, "incompatible: {m}"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Synthesizes an answer from a query and a list of evidences. The
/// extractive impl returns concatenated evidence text; future LLM
/// impls (LlamaCppProvider, etc.) call out to a model.
pub trait InferenceProvider: Send + Sync {
    fn capability(&self) -> &ProviderCapability;
    fn answer(&self, query: &str, evidences: &[Evidence]) -> Result<String, ProviderError>;
}

pub use extractive::ExtractiveProvider;
