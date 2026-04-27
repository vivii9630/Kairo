//! `LlamaCppProvider` — capability-gated, model-path-driven LLM
//! inference for Pi 4/5 and mid-range phones.
//!
//! **State at this commit (14e.4):** capability + selection semantics
//! are real and tested. The `answer()` body is intentionally a clear
//! `ProviderError::Backend` so the pipeline's soft-fall to
//! `ExtractiveProvider` exercises end-to-end while the actual
//! `llama-cpp-2` integration (a separate ~50 MB C++ build dep + a
//! quantized model file) is brought in by 14e.4.1.
//!
//! **Why this isn't pure-vapor:** every other piece around the LLM —
//! selection by RAM/profile, model_families filter, error path,
//! Pi-zero filtering, Pi-4 selection, soft-fall to extractive — is
//! exercised by tests in this module. When 14e.4.1 swaps the
//! `answer()` body for a real call into `llama-cpp-2`, none of the
//! surrounding wiring changes.

use std::path::{Path, PathBuf};

use super::{InferenceProvider, ProviderError};
use crate::capability::{ProviderCapability, ProviderKind};
use crate::types::Evidence;

/// Default per-prompt context budget (chars). 1B Q4 models commonly
/// have a 4k token window; 4 chars/token rough rule of thumb says
/// ~16k chars total. We reserve most for the model's output and
/// leave 8k for query + evidences. Tunable per provider instance.
const DEFAULT_CONTEXT_CHAR_BUDGET: usize = 8_192;

pub struct LlamaCppProvider {
    capability: ProviderCapability,
    /// Path to a GGUF model file. `None` means the provider was
    /// constructed but no model was wired — `answer()` will return
    /// a clear configuration error instead of crashing.
    model_path: Option<PathBuf>,
    context_char_budget: usize,
}

impl LlamaCppProvider {
    /// Build a provider with the canonical edge capability — 800 MB
    /// minimum RAM (1B Q4_K_M fits comfortably), supports the
    /// llama / gemma / phi / qwen families, no GPU required.
    pub fn new(model_path: Option<impl Into<PathBuf>>) -> Self {
        Self::with_context_budget(model_path, DEFAULT_CONTEXT_CHAR_BUDGET)
    }

    pub fn with_context_budget(
        model_path: Option<impl Into<PathBuf>>,
        context_char_budget: usize,
    ) -> Self {
        let capability = ProviderCapability {
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
        };
        Self {
            capability,
            model_path: model_path.map(Into::into),
            context_char_budget,
        }
    }

    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    /// Read `KAIRO_EDGE_LLAMA_MODEL` and build a provider configured
    /// from it. Returns a provider with `model_path = None` when the
    /// env var is unset — capability is still declared so selection
    /// works, and `answer()` errors cleanly.
    pub fn from_env() -> Self {
        let path = std::env::var_os("KAIRO_EDGE_LLAMA_MODEL").map(PathBuf::from);
        Self::new(path)
    }

    /// Build the prompt the next LLM call would receive. Exposed so
    /// 14e.4.1's actual `llama-cpp-2` integration can call into this
    /// without re-inventing the format, and so tests assert the
    /// shape today.
    pub fn build_prompt(&self, query: &str, evidences: &[Evidence]) -> String {
        let mut out = String::new();
        out.push_str(
            "You are a careful assistant. Answer the user's question \
             using ONLY the evidence provided. If the evidence does not \
             contain the answer, say so plainly.\n\n",
        );
        out.push_str("Evidence:\n");
        let mut budget = self.context_char_budget.saturating_sub(out.len() + query.len() + 64);
        for (i, ev) in evidences.iter().enumerate() {
            let header = format!("[{}] id={}\n", i + 1, ev.document_id);
            if budget < header.len() {
                break;
            }
            out.push_str(&header);
            budget -= header.len();
            let take = ev.text.chars().take(budget).collect::<String>();
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
}

impl InferenceProvider for LlamaCppProvider {
    fn capability(&self) -> &ProviderCapability {
        &self.capability
    }

    fn answer(&self, _query: &str, _evidences: &[Evidence]) -> Result<String, ProviderError> {
        let path = self.model_path.as_ref().ok_or_else(|| {
            ProviderError::Backend(
                "llama-cpp model_path not configured \
                 (set KAIRO_EDGE_LLAMA_MODEL or pass a path to LlamaCppProvider::new)"
                    .into(),
            )
        })?;
        if !path.exists() {
            return Err(ProviderError::Backend(format!(
                "llama-cpp model file not found: {}",
                path.display()
            )));
        }

        // 14e.4 ships the capability + selection + prompt-building
        // wiring but defers the actual llama-cpp-2 call to 14e.4.1
        // so this commit doesn't pull a heavy C++ build dependency
        // into the workspace before the model + cross-compile story
        // is ready. EdgePipeline soft-falls to ExtractiveProvider on
        // this error, so the user still gets a grounded answer.
        Err(ProviderError::Backend(
            "llama-cpp backend not yet wired (14e.4.1). \
             EdgePipeline will soft-fall to ExtractiveProvider."
                .into(),
        ))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::DeviceProfile;

    fn ev(id: &str, text: &str) -> Evidence {
        Evidence {
            document_id: id.into(),
            text: text.into(),
            score: 1.0,
            metadata: Default::default(),
        }
    }

    #[test]
    fn capability_declares_canonical_edge_llm_shape() {
        let p = LlamaCppProvider::new(None::<PathBuf>);
        let cap = p.capability();
        assert_eq!(cap.name, "llama-cpp");
        assert_eq!(cap.kind, ProviderKind::Inference);
        assert_eq!(cap.min_ram_mb, 800);
        assert!(!cap.needs_gpu);
        assert!(!cap.needs_webgpu);
        for fam in ["llama", "gemma", "phi", "qwen"] {
            assert!(cap.model_families.iter().any(|f| f == fam));
        }
    }

    #[test]
    fn capability_rejects_pi_zero_via_min_ram() {
        let p = LlamaCppProvider::new(None::<PathBuf>);
        assert!(!p.capability().matches(&DeviceProfile::pi_zero()));
    }

    #[test]
    fn capability_accepts_pi_4_phone_mid_pi_5() {
        let p = LlamaCppProvider::new(None::<PathBuf>);
        for profile in [
            DeviceProfile::pi_4(),
            DeviceProfile::phone_mid(),
            DeviceProfile::pi_5(),
        ] {
            assert!(
                p.capability().matches(&profile),
                "llama-cpp should match {} but didn't",
                profile.name
            );
        }
    }

    #[test]
    fn answer_without_model_path_returns_clear_error() {
        let p = LlamaCppProvider::new(None::<PathBuf>);
        let result = p.answer("anything", &[]);
        match result {
            Err(ProviderError::Backend(msg)) => {
                assert!(msg.contains("model_path not configured"));
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
    }

    #[test]
    fn answer_with_missing_model_file_returns_clear_error() {
        let p = LlamaCppProvider::new(Some(PathBuf::from("/definitely/not/here.gguf")));
        let result = p.answer("q", &[]);
        match result {
            Err(ProviderError::Backend(msg)) => {
                assert!(msg.contains("not found"));
            }
            other => panic!("expected Backend error for missing file, got {other:?}"),
        }
    }

    #[test]
    fn build_prompt_includes_evidence_ids_and_query() {
        let p = LlamaCppProvider::new(None::<PathBuf>);
        let prompt = p.build_prompt(
            "felines hunt",
            &[ev("d1", "Felines hunt at dusk."), ev("d2", "Cats purr.")],
        );
        assert!(prompt.contains("id=d1"));
        assert!(prompt.contains("id=d2"));
        assert!(prompt.contains("Question: felines hunt"));
        assert!(prompt.ends_with("Answer:"));
    }

    #[test]
    fn build_prompt_truncates_at_context_budget() {
        let p = LlamaCppProvider::with_context_budget(None::<PathBuf>, 256);
        let huge = "x".repeat(10_000);
        let prompt = p.build_prompt("q", &[ev("d1", &huge)]);
        // Hard upper bound: budget + query + framing ~= < 600 chars.
        assert!(prompt.len() < 600, "prompt should respect budget, got {} chars", prompt.len());
    }

    #[test]
    fn from_env_with_unset_var_yields_no_model_path() {
        // Ensure the var is unset for this test (cargo test inherits env).
        std::env::remove_var("KAIRO_EDGE_LLAMA_MODEL");
        let p = LlamaCppProvider::from_env();
        assert!(p.model_path().is_none());
    }
}
