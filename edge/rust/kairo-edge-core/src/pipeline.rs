//! `EdgePipeline` — composes a retrieval store, a device profile, and
//! a list of inference providers. On every query it retrieves top-k
//! evidences via BM25, picks the best-fit `InferenceProvider` for the
//! profile, and asks it for an answer.
//!
//! Design constraints from the user's plan:
//!   * Extractive must always work — empty provider list still yields
//!     a usable `PipelineResponse` via the bundled fallback.
//!   * Provider selection is profile-driven; no hard-coded model paths.
//!   * Cited node ids accompany every response so callers can render
//!     halos / verify citations.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::capability::{select_capability, DeviceProfile, ProviderKind};
use crate::providers::{ExtractiveProvider, InferenceProvider, ProviderError};
use crate::store::EdgeStore;
use crate::types::{Document, Evidence, QueryRequest};

/// Tagged superset of `QueryResponse`. Carries the chosen provider's
/// name + the list of cited evidence ids so the caller can attribute
/// the answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResponse {
    pub answer: String,
    pub evidences: Vec<Evidence>,
    pub provider: String,
    pub cited_node_ids: Vec<String>,
}

pub struct EdgePipeline {
    store: EdgeStore,
    profile: DeviceProfile,
    providers: Vec<Arc<dyn InferenceProvider>>,
    fallback: Arc<dyn InferenceProvider>,
}

impl EdgePipeline {
    /// Build a pipeline with a profile + provider list. The
    /// `ExtractiveProvider` is added as an always-available fallback;
    /// pass `providers = vec![]` to run pure-extractive on every
    /// query (useful for tests and pi-zero-class targets).
    pub fn new(
        store: EdgeStore,
        profile: DeviceProfile,
        providers: Vec<Arc<dyn InferenceProvider>>,
    ) -> Self {
        Self {
            store,
            profile,
            providers,
            fallback: Arc::new(ExtractiveProvider::new()),
        }
    }

    /// In-memory pipeline with an empty store. Convenience for tests
    /// and short-lived processes.
    pub fn in_memory(profile: DeviceProfile) -> Self {
        Self::new(EdgeStore::in_memory(), profile, vec![])
    }

    pub fn add_documents<I: IntoIterator<Item = Document>>(
        &mut self,
        docs: I,
    ) -> Result<(), crate::store::StoreError> {
        self.store.add(docs)
    }

    pub fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    pub fn provider_names(&self) -> Vec<String> {
        self.providers
            .iter()
            .map(|p| p.capability().name.clone())
            .chain(std::iter::once(self.fallback.capability().name.clone()))
            .collect()
    }

    /// Pick the best-fit provider for this profile. Returns the
    /// fallback when nothing else qualifies — never errors.
    fn pick_provider(&self) -> &dyn InferenceProvider {
        let caps: Vec<&_> = self.providers.iter().map(|p| p.capability()).collect();
        let chosen = select_capability(&caps, &self.profile, ProviderKind::Inference);
        if let Some(c) = chosen {
            for p in &self.providers {
                if p.capability().name == c.name {
                    return p.as_ref();
                }
            }
        }
        self.fallback.as_ref()
    }

    /// End-to-end: retrieve via BM25 + ask the chosen provider.
    /// Falls back to extractive if the chosen provider errors so a
    /// flaky LLM never leaves the user empty-handed.
    pub fn query(&self, request: &QueryRequest) -> PipelineResponse {
        let retrieval = self.store.query(request);
        let evidences = retrieval.evidences;
        let provider = self.pick_provider();

        let answer = match provider.answer(&request.query, &evidences) {
            Ok(a) => a,
            Err(ProviderError::InvalidInput(_))
            | Err(ProviderError::Backend(_))
            | Err(ProviderError::Incompatible(_)) => {
                // Soft-fall to extractive so callers always get an answer.
                self.fallback
                    .answer(&request.query, &evidences)
                    .unwrap_or_else(|_| "No relevant evidence found.".to_string())
            }
        };

        let cited_node_ids: Vec<String> =
            evidences.iter().map(|e| e.document_id.clone()).collect();

        PipelineResponse {
            answer,
            evidences,
            provider: provider.capability().name.clone(),
            cited_node_ids,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::ProviderCapability;

    fn make_docs() -> Vec<Document> {
        vec![
            Document::new("d1", "Cats purr when content."),
            Document::new("d2", "Felines hunt at dusk and prefer solitude."),
            Document::new("d3", "Dogs bark at strangers approaching the door."),
            Document::new("d4", "Python code runs on the JVM via Jython."),
        ]
    }

    #[test]
    fn empty_provider_list_falls_back_to_extractive() {
        let mut p = EdgePipeline::in_memory(DeviceProfile::pi_zero());
        p.add_documents(make_docs()).unwrap();
        let resp = p.query(&QueryRequest::new("felines hunting", 2));
        assert_eq!(resp.provider, "extractive");
        assert!(resp.cited_node_ids.contains(&"d2".to_string()));
        assert!(resp.answer.contains("[d2]"));
    }

    /// A spy provider that always errors so we can prove the
    /// soft-fall to extractive in `EdgePipeline::query`.
    struct AlwaysErrorProvider {
        cap: ProviderCapability,
    }

    impl InferenceProvider for AlwaysErrorProvider {
        fn capability(&self) -> &ProviderCapability {
            &self.cap
        }
        fn answer(&self, _q: &str, _e: &[Evidence]) -> Result<String, ProviderError> {
            Err(ProviderError::Backend("simulated failure".into()))
        }
    }

    #[test]
    fn provider_failure_soft_falls_to_extractive() {
        let cap = ProviderCapability {
            name: "always-error".into(),
            kind: ProviderKind::Inference,
            min_ram_mb: 0,
            needs_gpu: false,
            needs_webgpu: false,
            model_families: vec![],
            version: "0.0.1".into(),
        };
        let bad = Arc::new(AlwaysErrorProvider { cap }) as Arc<dyn InferenceProvider>;
        let mut p = EdgePipeline::new(
            EdgeStore::in_memory(),
            DeviceProfile::pi_4(),
            vec![bad],
        );
        p.add_documents(make_docs()).unwrap();
        let resp = p.query(&QueryRequest::new("felines hunting", 2));
        // Provider that would have been picked (the spy) errored,
        // but EdgePipeline.query falls back to extractive so the
        // user still gets evidence-grounded text.
        assert!(resp.cited_node_ids.contains(&"d2".to_string()));
        assert!(resp.answer.contains("[d2]"));
    }

    #[test]
    fn empty_corpus_yields_empty_evidences_but_still_answers() {
        let p = EdgePipeline::in_memory(DeviceProfile::pi_4());
        let resp = p.query(&QueryRequest::new("anything", 5));
        assert!(resp.evidences.is_empty());
        assert!(resp.cited_node_ids.is_empty());
        assert!(resp.answer.contains("No relevant evidence"));
    }

    #[test]
    fn provider_names_includes_fallback_at_end() {
        let p = EdgePipeline::in_memory(DeviceProfile::pi_4());
        let names = p.provider_names();
        assert_eq!(names, vec!["extractive".to_string()]); // empty providers + fallback
    }

    #[test]
    fn pipeline_response_serializes_through_serde() {
        let mut p = EdgePipeline::in_memory(DeviceProfile::pi_4());
        p.add_documents(make_docs()).unwrap();
        let resp = p.query(&QueryRequest::new("dogs", 1));
        let json = serde_json::to_string(&resp).unwrap();
        let back: PipelineResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.provider, back.provider);
        assert_eq!(resp.cited_node_ids, back.cited_node_ids);
    }
}
