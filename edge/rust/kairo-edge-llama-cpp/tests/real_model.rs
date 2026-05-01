//! Integration test gated on the `real-backend` feature AND the
//! `KAIRO_EDGE_LLAMA_MODEL` env var.
//!
//! - Without the feature: this test is excluded at compile time, so
//!   `cargo test -p kairo-edge-llama-cpp` (no flags) and `cargo test
//!   --workspace` on machines without LLVM never try to build it.
//! - With the feature but without the env var: the test prints a
//!   `skipped` line and passes — useful in CI matrixes that build
//!   real-backend but don't ship a model file.
//! - With both: loads the model and asserts the runtime can produce
//!   non-empty output for a tiny evidence-grounded prompt.

#![cfg(feature = "real-backend")]

use std::path::PathBuf;

use kairo_edge_core::providers::InferenceProvider;
use kairo_edge_core::types::Evidence;
use kairo_edge_llama_cpp::LlamaCppRuntime;

#[test]
fn generates_a_response_when_model_is_configured() {
    let model_env = match std::env::var_os("KAIRO_EDGE_LLAMA_MODEL") {
        Some(v) => PathBuf::from(v),
        None => {
            eprintln!(
                "skipped: KAIRO_EDGE_LLAMA_MODEL not set. \
                 Run with KAIRO_EDGE_LLAMA_MODEL=/path/to/model.gguf \
                 cargo test -p kairo-edge-llama-cpp \
                 to exercise real generation."
            );
            return;
        }
    };

    let runtime = LlamaCppRuntime::load(&model_env)
        .expect("model should load when KAIRO_EDGE_LLAMA_MODEL is valid")
        .with_max_tokens(48);

    let evidences = [Evidence {
        document_id: "doc1".into(),
        text: "The capital of France is Paris.".into(),
        score: 1.0,
        metadata: Default::default(),
    }];

    let answer = runtime
        .answer("What is the capital of France?", &evidences)
        .expect("real model should produce an answer");

    assert!(!answer.trim().is_empty(), "answer should be non-empty");
    eprintln!("model produced ({} chars): {}", answer.len(), answer);
}
