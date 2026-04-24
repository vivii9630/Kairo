//! `EdgeStore` — JSON-backed lexical retrieval.
//!
//! Replaces Python's `kairo_edge.EdgeStore` (term-frequency scoring)
//! with a proper BM25 index built on the fly. Wire format of the
//! JSON file is stable with the Python side: a top-level
//! `{"documents": [...]}` object where each entry matches the
//! `Document` Pydantic schema. Either runtime can read a store
//! written by the other.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bm25::Bm25Index;
use crate::types::{Document, Evidence, QueryRequest, QueryResponse};

#[derive(Debug)]
pub enum StoreError {
    Io(io::Error),
    Serde(serde_json::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Serde(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct StoreFile {
    #[serde(default)]
    documents: Vec<Document>,
}

#[derive(Debug)]
pub struct EdgeStore {
    store_path: PathBuf,
    docs: BTreeMap<String, Document>,
    index: Option<Bm25Index>,
}

impl EdgeStore {
    /// Open the store at *path*, creating an empty one if missing.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let store_path = path.into();
        let mut store = Self {
            store_path,
            docs: BTreeMap::new(),
            index: None,
        };
        store.load()?;
        store.rebuild_index();
        Ok(store)
    }

    /// In-memory store; useful for tests and for runtimes (browser,
    /// IndexedDB-backed) that manage persistence themselves.
    pub fn in_memory() -> Self {
        Self {
            store_path: PathBuf::new(),
            docs: BTreeMap::new(),
            index: None,
        }
    }

    fn load(&mut self) -> Result<(), StoreError> {
        if self.store_path.as_os_str().is_empty() || !self.store_path.exists() {
            return Ok(());
        }
        let raw = fs::read_to_string(&self.store_path)?;
        if raw.trim().is_empty() {
            return Ok(());
        }
        let file: StoreFile = serde_json::from_str(&raw)?;
        for doc in file.documents {
            self.docs.insert(doc.id.clone(), doc);
        }
        Ok(())
    }

    fn save(&self) -> Result<(), StoreError> {
        if self.store_path.as_os_str().is_empty() {
            return Ok(());
        }
        let file = StoreFile {
            documents: self.docs.values().cloned().collect(),
        };
        if let Some(parent) = self.store_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let body = serde_json::to_string_pretty(&file)?;
        fs::write(&self.store_path, body)?;
        Ok(())
    }

    fn rebuild_index(&mut self) {
        if self.docs.is_empty() {
            self.index = None;
            return;
        }
        let pairs = self
            .docs
            .values()
            .map(|d| (d.id.clone(), d.text.clone()));
        self.index = Some(Bm25Index::from_docs(pairs));
    }

    /// Insert or replace *docs*; writes the store and rebuilds the
    /// index.  Matches the Python `EdgeStore.add` contract.
    pub fn add<I>(&mut self, docs: I) -> Result<(), StoreError>
    where
        I: IntoIterator<Item = Document>,
    {
        for doc in docs {
            self.docs.insert(doc.id.clone(), doc);
        }
        self.save()?;
        self.rebuild_index();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    pub fn query(&self, request: &QueryRequest) -> QueryResponse {
        let Some(index) = self.index.as_ref() else {
            return QueryResponse {
                answer: "No relevant evidence found.".to_string(),
                evidences: vec![],
                steps: vec![],
            };
        };

        let hits = index.top_k(&request.query, request.top_k);
        let evidences: Vec<Evidence> = hits
            .into_iter()
            .filter_map(|(doc_id, score)| {
                self.docs.get(&doc_id).map(|doc| Evidence {
                    document_id: doc.id.clone(),
                    text: doc.text.clone(),
                    score,
                    metadata: doc.metadata.clone(),
                })
            })
            .collect();

        let answer = if evidences.is_empty() {
            "No relevant evidence found.".to_string()
        } else {
            evidences
                .iter()
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
                .chars()
                .take(512)
                .collect()
        };

        QueryResponse {
            answer,
            evidences,
            steps: vec![],
        }
    }

    pub fn path(&self) -> &Path {
        &self.store_path
    }
}
