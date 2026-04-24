//! Okapi BM25 — pure Rust, no external BM25 crate.
//!
//! Matches the math used by `rank_bm25::BM25Okapi` (the Python
//! `kairo-retrieval` dep): k1 = 1.2, b = 0.75, IDF uses
//! `ln((N - df + 0.5) / (df + 0.5))` without the `+ 1` clamp. That
//! choice aligns scores with the Python side so regression tests
//! compare across runtimes.

use std::collections::HashMap;

use crate::tokenize::tokenize;

pub const BM25_K1: f64 = 1.2;
pub const BM25_B: f64 = 0.75;

/// Pre-computed BM25 index for a fixed corpus.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    /// Document ids in the order they were indexed.
    doc_ids: Vec<String>,
    /// Document-frequency: number of docs containing each term.
    df: HashMap<String, u32>,
    /// Term-frequency per doc: tf[i][term] = count in doc i.
    tf: Vec<HashMap<String, u32>>,
    /// Length in tokens of doc i.
    doc_lens: Vec<u32>,
    /// Mean of `doc_lens`; precomputed so scoring is hot-path-free.
    avgdl: f64,
    /// Corpus size, kept so IDF calc stays O(1) per term.
    n: u32,
}

impl Bm25Index {
    /// Build the index from `(doc_id, tokens)` pairs.
    pub fn build<I, S>(corpus: I) -> Self
    where
        I: IntoIterator<Item = (S, Vec<String>)>,
        S: Into<String>,
    {
        let mut doc_ids = Vec::new();
        let mut tf: Vec<HashMap<String, u32>> = Vec::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut doc_lens = Vec::new();

        for (id, tokens) in corpus {
            let mut counts: HashMap<String, u32> = HashMap::new();
            for tok in &tokens {
                *counts.entry(tok.clone()).or_insert(0) += 1;
            }
            for term in counts.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            doc_lens.push(tokens.len() as u32);
            tf.push(counts);
            doc_ids.push(id.into());
        }

        let n = doc_ids.len() as u32;
        let avgdl = if n == 0 {
            0.0
        } else {
            doc_lens.iter().copied().map(f64::from).sum::<f64>() / f64::from(n)
        };

        Self {
            doc_ids,
            df,
            tf,
            doc_lens,
            avgdl,
            n,
        }
    }

    /// Convenience constructor that tokenizes each document's text.
    pub fn from_docs<I, S, T>(docs: I) -> Self
    where
        I: IntoIterator<Item = (S, T)>,
        S: Into<String>,
        T: AsRef<str>,
    {
        let pairs = docs
            .into_iter()
            .map(|(id, text)| (id, tokenize(text.as_ref())));
        Self::build(pairs)
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    pub fn doc_ids(&self) -> &[String] {
        &self.doc_ids
    }

    /// IDF following the Okapi variant used by `rank_bm25`:
    /// `ln((N - df + 0.5) / (df + 0.5))`. Can go negative for terms in
    /// > half the corpus — callers (see `Bm25Index::score`) keep all
    /// scores, positive or not, and let downstream truncation handle
    /// ranking cutoffs. The Python side made the same choice in
    /// Phase 11a, so scores align.
    fn idf(&self, term: &str) -> f64 {
        let df = f64::from(self.df.get(term).copied().unwrap_or(0));
        let n = f64::from(self.n);
        ((n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Score `query` against every indexed document; result is
    /// `[(doc_id, score); N]` in corpus order. Caller sorts / filters.
    pub fn score(&self, query: &str) -> Vec<(String, f64)> {
        self.score_tokens(&tokenize(query))
    }

    pub fn score_tokens(&self, query_tokens: &[String]) -> Vec<(String, f64)> {
        if self.n == 0 {
            return Vec::new();
        }
        let idfs: Vec<(String, f64)> = query_tokens
            .iter()
            .map(|t| (t.clone(), self.idf(t)))
            .collect();

        (0..self.doc_ids.len())
            .map(|i| {
                let dl = f64::from(self.doc_lens[i]);
                let norm = 1.0 - BM25_B + BM25_B * (dl / self.avgdl.max(f64::EPSILON));
                let mut score = 0.0;
                for (term, idf) in &idfs {
                    let tf = f64::from(self.tf[i].get(term).copied().unwrap_or(0));
                    if tf == 0.0 {
                        continue;
                    }
                    let num = tf * (BM25_K1 + 1.0);
                    let den = tf + BM25_K1 * norm;
                    score += idf * (num / den);
                }
                (self.doc_ids[i].clone(), score)
            })
            .collect()
    }

    /// Top-k by score, descending. Zero / negative scores are dropped
    /// iff at least one positive exists — identical discipline to
    /// `kairo_retrieval.local._bm25_ranking` after Phase 13a's
    /// zero-filter fix, so eval-harness parity holds across runtimes.
    pub fn top_k(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        let mut scored = self.score(query);
        let any_positive = scored.iter().any(|(_, s)| *s > 0.0);
        if any_positive {
            scored.retain(|(_, s)| *s > 0.0);
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
