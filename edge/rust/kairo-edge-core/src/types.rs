//! Wire-compatible types mirroring `kairo_core`'s Pydantic models.
//!
//! JSON serialization shape matches the Python side field-for-field so
//! a store written by either runtime is readable by the other. Extra
//! fields on the Python side that aren't modeled here (e.g. `steps` on
//! `QueryResponse`) are tolerated on deserialize thanks to serde's
//! default of ignoring unknown fields.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Metadata values follow Python's `Dict[str, Any]` shape. We use a
/// `BTreeMap` + `serde_json::Value` so stable key ordering matches the
/// Python side's canonical JSON ordering for deterministic hashing.
pub type Metadata = BTreeMap<String, serde_json::Value>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Document {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: Metadata::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Evidence {
    pub document_id: String,
    pub text: String,
    pub score: f64,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub filters: Metadata,
}

fn default_top_k() -> usize {
    5
}

impl QueryRequest {
    pub fn new(query: impl Into<String>, top_k: usize) -> Self {
        Self {
            query: query.into(),
            top_k,
            filters: Metadata::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryResponse {
    pub answer: String,
    pub evidences: Vec<Evidence>,
    /// Placeholder for future agent steps; empty on the edge runtime,
    /// kept for wire compatibility with the Python side's schema.
    #[serde(default)]
    pub steps: Vec<serde_json::Value>,
}
