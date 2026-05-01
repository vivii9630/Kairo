//! Real GGUF inference for the Kairo edge runtime.
//!
//! Wraps [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) and
//! provides a [`LlamaCppRuntime`] that implements
//! `kairo_edge_core::providers::InferenceProvider`, so an
//! `EdgePipeline` can answer directly from the model on capable
//! profiles instead of soft-falling to extractive.
//!
//! ## Build modes
//!
//! - **Default (no features):** the crate compiles as a stub. The
//!   `LlamaCppRuntime` struct still loads (or refuses to load) a model
//!   path, and `answer()` returns a clear "real-backend feature not
//!   enabled" error. The `EdgePipeline` soft-fall to
//!   `ExtractiveProvider` keeps user-facing answers grounded. This
//!   path is what the regular workspace `cargo build`/`cargo test`
//!   exercises — no LLVM, no CMake, no llama.cpp source compile.
//! - **`--features real-backend`:** pulls `llama-cpp-2` +
//!   `llama-cpp-sys-2`. Requires LLVM (`LIBCLANG_PATH` for `bindgen`)
//!   and CMake on the build host. First compile takes 5–15 minutes.
//!   `answer()` runs real generation against a loaded GGUF.
//!
//! See `README.md` and `edge/docs/EDGE_ARCHITECTURE.md` §6 (14e.4 +
//! 14e.4.1) for the full rationale.

use std::path::{Path, PathBuf};

use kairo_edge_core::capability::{ProviderCapability, ProviderKind};
use kairo_edge_core::providers::{InferenceProvider, ProviderError};
use kairo_edge_core::types::Evidence;

#[cfg(feature = "real-backend")]
mod real;

const DEFAULT_N_CTX: u32 = 2048;
const DEFAULT_MAX_TOKENS: u32 = 256;
const DEFAULT_CONTEXT_CHAR_BUDGET: usize = 6_000;

/// `InferenceProvider` backed by a real loaded GGUF model when built
/// with `--features real-backend`, or a clear-erroring stub otherwise.
///
/// Construct with [`LlamaCppRuntime::load`] (path provided directly)
/// or [`LlamaCppRuntime::from_env`] (reads `KAIRO_EDGE_LLAMA_MODEL`).
pub struct LlamaCppRuntime {
    capability: ProviderCapability,
    model_path: PathBuf,
    n_ctx: u32,
    max_tokens: u32,
    context_char_budget: usize,
    /// Real-backend state: loaded model + backend + per-runtime mutex
    /// so concurrent `answer()` calls serialize. Absent in stub mode
    /// so the crate compiles without llama.cpp present.
    #[cfg(feature = "real-backend")]
    inner: real::Inner,
}

impl LlamaCppRuntime {
    /// Validate a GGUF model path and (when `real-backend` is enabled)
    /// load the model into memory. In stub mode the path is recorded
    /// for diagnostics but no model is loaded — `answer()` will return
    /// a clear "feature not enabled" error.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, ProviderError> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(ProviderError::Backend(format!(
                "llama-cpp model file not found: {}",
                model_path.display()
            )));
        }

        let capability = default_capability();
        let n_ctx = DEFAULT_N_CTX;
        let max_tokens = DEFAULT_MAX_TOKENS;
        let context_char_budget = DEFAULT_CONTEXT_CHAR_BUDGET;

        #[cfg(feature = "real-backend")]
        let inner = real::Inner::load(&model_path)?;

        Ok(Self {
            capability,
            model_path,
            n_ctx,
            max_tokens,
            context_char_budget,
            #[cfg(feature = "real-backend")]
            inner,
        })
    }

    /// Read `KAIRO_EDGE_LLAMA_MODEL` and load from there. Returns
    /// `Err(InvalidInput)` when the env var is unset so callers can
    /// fall back to the scaffold provider cleanly.
    pub fn from_env() -> Result<Self, ProviderError> {
        let path = std::env::var_os("KAIRO_EDGE_LLAMA_MODEL").ok_or_else(|| {
            ProviderError::InvalidInput(
                "KAIRO_EDGE_LLAMA_MODEL is not set; \
                 either set it or pass an explicit model_path to LlamaCppRuntime::load"
                    .into(),
            )
        })?;
        Self::load(PathBuf::from(path))
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn with_context_char_budget(mut self, budget: usize) -> Self {
        self.context_char_budget = budget;
        self
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// True when this runtime was built with the real backend feature
    /// and can produce model-generated answers. False when running as
    /// the structural stub.
    pub fn has_real_backend(&self) -> bool {
        cfg!(feature = "real-backend")
    }

    /// Build the evidence-grounded prompt. Public so callers / tests
    /// can inspect it without running inference. Mirrors the scaffold
    /// `LlamaCppProvider::build_prompt` shape so swapping runtimes
    /// doesn't change model behavior.
    pub fn build_prompt(&self, query: &str, evidences: &[Evidence]) -> String {
        let mut out = String::new();
        out.push_str(
            "You are a careful assistant. Answer the user's question \
             using ONLY the evidence provided. If the evidence does not \
             contain the answer, say so plainly.\n\n",
        );
        out.push_str("Evidence:\n");
        let mut budget = self
            .context_char_budget
            .saturating_sub(out.len() + query.len() + 64);
        for (i, ev) in evidences.iter().enumerate() {
            let header = format!("[{}] id={}\n", i + 1, ev.document_id);
            if budget < header.len() {
                break;
            }
            out.push_str(&header);
            budget -= header.len();
            let take: String = ev.text.chars().take(budget).collect();
            let take_bytes = take.len();
            out.push_str(&take);
            out.push('\n');
            budget = budget.saturating_sub(take_bytes + 1);
            if budget == 0 {
                break;
            }
        }
        out.push_str("\nQuestion: ");
        out.push_str(query);
        out.push_str("\nAnswer:");
        out
    }

    fn run_inference(&self, prompt: &str) -> Result<String, ProviderError> {
        #[cfg(feature = "real-backend")]
        {
            return self
                .inner
                .generate(prompt, self.n_ctx, self.max_tokens);
        }

        #[cfg(not(feature = "real-backend"))]
        {
            let _ = prompt;
            Err(ProviderError::Backend(
                "kairo-edge-llama-cpp built without `real-backend` feature; \
                 rebuild with `--features real-backend` (and an LLVM/CMake \
                 toolchain available) to enable real GGUF inference. The \
                 EdgePipeline will soft-fall to ExtractiveProvider."
                    .into(),
            ))
        }
    }
}

impl InferenceProvider for LlamaCppRuntime {
    fn capability(&self) -> &ProviderCapability {
        &self.capability
    }

    fn answer(&self, query: &str, evidences: &[Evidence]) -> Result<String, ProviderError> {
        if query.trim().is_empty() {
            return Err(ProviderError::InvalidInput("query is empty".into()));
        }
        let prompt = self.build_prompt(query, evidences);
        self.run_inference(&prompt)
    }
}

impl std::fmt::Debug for LlamaCppRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppRuntime")
            .field("model_path", &self.model_path)
            .field("n_ctx", &self.n_ctx)
            .field("max_tokens", &self.max_tokens)
            .field("has_real_backend", &cfg!(feature = "real-backend"))
            .finish()
    }
}

fn default_capability() -> ProviderCapability {
    // Mirror the scaffold's declaration so `EdgePipeline`'s selection
    // matrix stays identical whether the scaffold or the real runtime
    // is registered.
    ProviderCapability {
        name: "llama-cpp".into(),
        kind: ProviderKind::Inference,
        min_ram_mb: 800,
        needs_gpu: false,
        needs_webgpu: false,
        model_families: vec![
            "llama".into(),
            "gemma".into(),
            "phi".into(),
            "qwen".into(),
        ],
        version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairo_edge_core::capability::DeviceProfile;
    use std::io::Write;

    fn ev(id: &str, text: &str) -> Evidence {
        Evidence {
            document_id: id.into(),
            text: text.into(),
            score: 1.0,
            metadata: Default::default(),
        }
    }

    #[test]
    fn capability_matches_scaffold() {
        let cap = default_capability();
        assert_eq!(cap.name, "llama-cpp");
        assert_eq!(cap.kind, ProviderKind::Inference);
        assert_eq!(cap.min_ram_mb, 800);
        assert!(!cap.needs_gpu);
        for fam in ["llama", "gemma", "phi", "qwen"] {
            assert!(cap.model_families.iter().any(|f| f == fam));
        }
    }

    #[test]
    fn capability_obeys_pi_zero_ram_gate() {
        let cap = default_capability();
        assert!(!cap.matches(&DeviceProfile::pi_zero()));
        assert!(cap.matches(&DeviceProfile::pi_4()));
    }

    #[test]
    fn load_with_missing_path_returns_clear_error() {
        let r = LlamaCppRuntime::load("/no/such/model.gguf");
        match r {
            Err(ProviderError::Backend(msg)) => {
                assert!(msg.contains("not found"), "msg = {msg}");
            }
            other => panic!("expected Backend error for missing file, got {other:?}"),
        }
    }

    #[test]
    fn from_env_unset_returns_invalid_input() {
        std::env::remove_var("KAIRO_EDGE_LLAMA_MODEL");
        let r = LlamaCppRuntime::from_env();
        match r {
            Err(ProviderError::InvalidInput(msg)) => {
                assert!(msg.contains("KAIRO_EDGE_LLAMA_MODEL"));
            }
            other => panic!("expected InvalidInput when env var unset, got {other:?}"),
        }
    }

    #[test]
    fn build_prompt_includes_evidence_ids_and_query() {
        // Use a fake-but-existing path so load() succeeds in stub mode.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"not a real gguf").unwrap();
        let runtime = match LlamaCppRuntime::load(tmp.path()) {
            Ok(r) => r,
            // With real-backend on, model_load will error on a non-GGUF
            // file. That's fine — this test only runs in stub mode.
            Err(_) if cfg!(feature = "real-backend") => return,
            Err(e) => panic!("stub-mode load failed unexpectedly: {e:?}"),
        };
        let prompt = runtime.build_prompt(
            "felines hunt",
            &[ev("d1", "Felines hunt at dusk."), ev("d2", "Cats purr.")],
        );
        assert!(prompt.contains("id=d1"));
        assert!(prompt.contains("id=d2"));
        assert!(prompt.contains("Question: felines hunt"));
        assert!(prompt.ends_with("Answer:"));
    }

    #[test]
    fn stub_answer_returns_feature_disabled_error() {
        // Skip when the real backend is on — this test asserts the stub
        // contract.
        if cfg!(feature = "real-backend") {
            return;
        }
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"placeholder").unwrap();
        let runtime = LlamaCppRuntime::load(tmp.path()).expect("stub load should succeed");
        let result = runtime.answer("hello", &[]);
        match result {
            Err(ProviderError::Backend(msg)) => {
                assert!(
                    msg.contains("real-backend"),
                    "stub error should reference the feature flag, got: {msg}"
                );
            }
            // InvalidInput on empty query is also acceptable, but we
            // passed a non-empty query so we expect Backend here.
            other => panic!("expected Backend error in stub mode, got {other:?}"),
        }
    }
}
