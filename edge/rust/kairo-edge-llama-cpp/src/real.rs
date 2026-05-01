//! Real `llama-cpp-2` backend. Only compiled with `--features real-backend`.
//!
//! Holds the loaded model + backend; per-query state (context, sampler,
//! batch) is built inside `generate()`. A `Mutex<()>` guard serializes
//! concurrent `generate()` calls because llama_cpp_2 contexts are not
//! safe to share across threads even when constructed per-call from
//! the same model.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;

use kairo_edge_core::providers::ProviderError;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

pub(crate) struct Inner {
    backend: LlamaBackend,
    model: LlamaModel,
    inflight: Mutex<()>,
}

impl Inner {
    pub fn load(model_path: &Path) -> Result<Self, ProviderError> {
        let backend = LlamaBackend::init()
            .map_err(|e| ProviderError::Backend(format!("llama-cpp backend init: {e}")))?;

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| ProviderError::Backend(format!("llama-cpp model load: {e}")))?;

        Ok(Self {
            backend,
            model,
            inflight: Mutex::new(()),
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        n_ctx: u32,
        max_tokens: u32,
    ) -> Result<String, ProviderError> {
        let _guard = self
            .inflight
            .lock()
            .map_err(|_| ProviderError::Backend("inflight mutex poisoned".into()))?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| ProviderError::Backend(format!("llama-cpp context create: {e}")))?;

        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| ProviderError::Backend(format!("llama-cpp tokenize: {e}")))?;

        let prompt_len = tokens_list.len();
        if prompt_len as u32 >= n_ctx {
            return Err(ProviderError::Backend(format!(
                "prompt is too long for context: {prompt_len} tokens >= n_ctx {n_ctx}. \
                 Lower context_char_budget or increase n_ctx."
            )));
        }

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last_idx: i32 = (prompt_len - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.iter().copied()) {
            let is_last = i == last_idx;
            batch
                .add(token, i, &[0], is_last)
                .map_err(|e| ProviderError::Backend(format!("llama-cpp batch.add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| ProviderError::Backend(format!("llama-cpp decode prompt: {e}")))?;

        let mut sampler = LlamaSampler::greedy();
        let mut output = String::new();
        let mut n_cur: i32 = batch.n_tokens();
        let n_max: i32 = (prompt_len as i32) + (max_tokens as i32);

        while n_cur < n_max && (n_cur as u32) < n_ctx {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            // Multi-byte UTF-8 chars sometimes split across tokens —
            // skip non-utf8 fragments rather than crash.
            match self.model.token_to_bytes(token, Special::Tokenize) {
                Ok(bytes) => {
                    if let Ok(s) = std::str::from_utf8(&bytes) {
                        output.push_str(s);
                    }
                }
                Err(e) => {
                    return Err(ProviderError::Backend(format!(
                        "llama-cpp token_to_bytes failed: {e}"
                    )));
                }
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| ProviderError::Backend(format!("llama-cpp batch.add tail: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| ProviderError::Backend(format!("llama-cpp decode tail: {e}")))?;
            n_cur += 1;
        }

        Ok(output.trim().to_string())
    }
}
