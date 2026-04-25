//! Zero-LLM extractive answer provider.
//!
//! Returns the evidence text concatenated up to a character budget,
//! prefixed with a short hint of which document each segment came
//! from. Used as the always-available fallback on devices that can't
//! run an LLM (pi-zero) and as the safety net when an LLM provider
//! errors on a more capable device.

use super::{InferenceProvider, ProviderError};
use crate::capability::{ProviderCapability, ProviderKind};
use crate::types::Evidence;

const DEFAULT_CHAR_BUDGET: usize = 1024;
const SEPARATOR: &str = "\n\n";

pub struct ExtractiveProvider {
    capability: ProviderCapability,
    char_budget: usize,
}

impl ExtractiveProvider {
    pub fn new() -> Self {
        Self::with_char_budget(DEFAULT_CHAR_BUDGET)
    }

    pub fn with_char_budget(char_budget: usize) -> Self {
        Self {
            capability: ProviderCapability {
                name: "extractive".into(),
                kind: ProviderKind::Inference,
                min_ram_mb: 0,
                needs_gpu: false,
                needs_webgpu: false,
                model_families: vec![],
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            char_budget,
        }
    }
}

impl Default for ExtractiveProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceProvider for ExtractiveProvider {
    fn capability(&self) -> &ProviderCapability {
        &self.capability
    }

    fn answer(&self, _query: &str, evidences: &[Evidence]) -> Result<String, ProviderError> {
        if evidences.is_empty() {
            return Ok("No relevant evidence found.".to_string());
        }

        let mut budget = self.char_budget;
        let mut out = String::new();
        for (i, ev) in evidences.iter().enumerate() {
            if i > 0 {
                if budget < SEPARATOR.len() {
                    break;
                }
                out.push_str(SEPARATOR);
                budget -= SEPARATOR.len();
            }
            // Reserve a few chars for the doc-id prefix.
            let prefix = format!("[{}] ", ev.document_id);
            if budget < prefix.len() {
                break;
            }
            out.push_str(&prefix);
            budget -= prefix.len();

            // Truncate evidence text at a char (not byte) boundary so
            // we don't slice through a multi-byte UTF-8 char.
            let take = ev.text.chars().take(budget).collect::<String>();
            let take_bytes = take.len();
            out.push_str(&take);
            budget = budget.saturating_sub(take_bytes);

            if budget == 0 {
                break;
            }
        }
        Ok(out)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn ev(id: &str, text: &str, score: f64) -> Evidence {
        Evidence {
            document_id: id.into(),
            text: text.into(),
            score,
            metadata: Default::default(),
        }
    }

    #[test]
    fn empty_evidences_returns_no_results_message() {
        let p = ExtractiveProvider::new();
        let out = p.answer("anything", &[]).unwrap();
        assert!(out.contains("No relevant evidence found"));
    }

    #[test]
    fn single_evidence_is_prefixed_with_id() {
        let p = ExtractiveProvider::new();
        let out = p.answer("q", &[ev("d1", "hello world", 1.0)]).unwrap();
        assert!(out.starts_with("[d1] hello world"));
    }

    #[test]
    fn multiple_evidences_are_separated_by_blank_line() {
        let p = ExtractiveProvider::new();
        let out = p.answer("q", &[ev("d1", "alpha", 1.0), ev("d2", "beta", 0.5)]).unwrap();
        assert!(out.contains("[d1] alpha\n\n[d2] beta"));
    }

    #[test]
    fn char_budget_truncates_long_corpus() {
        let p = ExtractiveProvider::with_char_budget(20);
        let long = "x".repeat(1000);
        let out = p.answer("q", &[ev("d1", &long, 1.0)]).unwrap();
        // 20 chars budget, [d1]_ takes 5, so ~15 x's max
        assert!(out.len() <= 25);
    }

    #[test]
    fn capability_advertises_no_ram_floor_and_no_model() {
        let p = ExtractiveProvider::new();
        let cap = p.capability();
        assert_eq!(cap.name, "extractive");
        assert_eq!(cap.kind, ProviderKind::Inference);
        assert_eq!(cap.min_ram_mb, 0);
        assert!(cap.model_families.is_empty());
        assert!(!cap.needs_gpu);
    }

    #[test]
    fn extractive_matches_every_device_profile() {
        use crate::capability::DeviceProfile;
        let p = ExtractiveProvider::new();
        let cap = p.capability();
        for profile in [
            DeviceProfile::pi_zero(),
            DeviceProfile::pi_4(),
            DeviceProfile::pi_5(),
            DeviceProfile::phone_mid(),
            DeviceProfile::browser_lite(),
            DeviceProfile::browser_gpu(),
        ] {
            assert!(
                cap.matches(&profile),
                "extractive should match {} but didn't",
                profile.name
            );
        }
    }
}
