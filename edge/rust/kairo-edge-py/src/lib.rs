//! PyO3 bindings for `kairo-edge-core`.
//!
//! Everything here is a thin wrapper. The real logic lives in
//! `kairo-edge-core` — these classes exist so Python code can call
//! Rust in-process instead of via the `kairo-edge-cli` subprocess.
//!
//! Ownership model: each Python `EdgeStore` wraps a Rust `EdgeStore`
//! behind `RefCell` so query() (which takes `&self`) and add() (which
//! takes `&mut self`) can both be exposed as Python methods from a
//! single owning Python handle. PyO3's `#[pyclass]` uses GIL-based
//! aliasing guarantees; we never expose the inner store to multiple
//! Python handles at once, so no cross-thread locking is needed.

use std::cell::RefCell;
use std::path::PathBuf;

use kairo_edge_core::{
    self as core, Document as CoreDocument, EdgeStore as CoreStore, Evidence as CoreEvidence,
    QueryRequest as CoreRequest, QueryResponse as CoreResponse,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Metadata bridging — Python dict <-> BTreeMap<String, serde_json::Value>
// ---------------------------------------------------------------------------

fn py_dict_to_metadata(dict: &Bound<'_, PyDict>) -> PyResult<core::Metadata> {
    let mut out = core::Metadata::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let value_str = value.str()?.to_string();
        // For fidelity, try to parse as JSON first — so `{"age": 42}`
        // round-trips as an int rather than the string "42".
        let value_json = match serde_json::from_str::<serde_json::Value>(&value_str) {
            Ok(v) => v,
            Err(_) => serde_json::Value::String(value_str),
        };
        out.insert(key_str, value_json);
    }
    Ok(out)
}

fn metadata_to_py_dict<'py>(
    py: Python<'py>,
    metadata: &core::Metadata,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in metadata.iter() {
        dict.set_item(key, value.to_string())?;
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

#[pyclass(module = "kairo_edge_py")]
#[derive(Clone)]
pub struct Document {
    inner: CoreDocument,
}

#[pymethods]
impl Document {
    #[new]
    #[pyo3(signature = (id, text, metadata = None))]
    fn new(id: String, text: String, metadata: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let metadata = match metadata {
            Some(dict) => py_dict_to_metadata(&dict)?,
            None => core::Metadata::new(),
        };
        Ok(Self {
            inner: CoreDocument { id, text, metadata },
        })
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metadata_to_py_dict(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "Document(id={:?}, text_len={})",
            self.inner.id,
            self.inner.text.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Evidence (read-only view)
// ---------------------------------------------------------------------------

#[pyclass(module = "kairo_edge_py")]
#[derive(Clone)]
pub struct Evidence {
    inner: CoreEvidence,
}

#[pymethods]
impl Evidence {
    #[getter]
    fn document_id(&self) -> &str {
        &self.inner.document_id
    }

    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metadata_to_py_dict(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "Evidence(document_id={:?}, score={:.4})",
            self.inner.document_id, self.inner.score
        )
    }
}

// ---------------------------------------------------------------------------
// QueryResponse — exposes answer + evidences, steps stays empty on edge.
// ---------------------------------------------------------------------------

#[pyclass(module = "kairo_edge_py")]
#[derive(Clone)]
pub struct QueryResponse {
    inner: CoreResponse,
}

#[pymethods]
impl QueryResponse {
    #[getter]
    fn answer(&self) -> &str {
        &self.inner.answer
    }

    #[getter]
    fn evidences(&self) -> Vec<Evidence> {
        self.inner
            .evidences
            .iter()
            .cloned()
            .map(|e| Evidence { inner: e })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryResponse(answer_len={}, evidence_count={})",
            self.inner.answer.len(),
            self.inner.evidences.len()
        )
    }
}

// ---------------------------------------------------------------------------
// EdgeStore — the user-facing class
// ---------------------------------------------------------------------------

#[pyclass(module = "kairo_edge_py")]
pub struct EdgeStore {
    // RefCell so `query` can borrow immutably while `add` borrows
    // mutably on the same Python object. PyO3's GIL handles thread
    // safety for us.
    inner: RefCell<CoreStore>,
}

#[pymethods]
impl EdgeStore {
    /// Open a persistent EdgeStore at ``store_path`` (creates the file
    /// if absent). Pass ``None`` for an in-memory store — useful for
    /// tests and short-lived processes.
    #[new]
    #[pyo3(signature = (store_path = None))]
    fn new(store_path: Option<PathBuf>) -> PyResult<Self> {
        let store = match store_path {
            Some(path) => CoreStore::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?,
            None => CoreStore::in_memory(),
        };
        Ok(Self {
            inner: RefCell::new(store),
        })
    }

    /// Batch-insert documents. Accepts an iterable of ``Document``
    /// PyO3 objects.
    fn add(&self, docs: Vec<Document>) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .add(docs.into_iter().map(|d| d.inner))
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Retrieve the top-k evidences for *query*.
    #[pyo3(signature = (query, top_k = 5))]
    fn query(&self, query: &str, top_k: usize) -> QueryResponse {
        let request = CoreRequest {
            query: query.to_string(),
            top_k,
            filters: core::Metadata::new(),
        };
        let resp = self.inner.borrow().query(&request);
        QueryResponse { inner: resp }
    }

    /// Number of documents currently in the store.
    fn __len__(&self) -> usize {
        self.inner.borrow().len()
    }

    fn __repr__(&self) -> String {
        format!("EdgeStore(len={})", self.inner.borrow().len())
    }
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

#[pymodule]
fn kairo_edge_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Document>()?;
    m.add_class::<Evidence>()?;
    m.add_class::<QueryResponse>()?;
    m.add_class::<EdgeStore>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// Keep an unused-import reference so rustc doesn't remove serde_json
// when the bindings grow — the metadata bridge relies on it.
#[allow(dead_code)]
fn _force_serde_json_link() {
    let _ = serde_json::Value::Null;
}

// Defensive: surface the PyValueError type so downstream binding
// changes compile cleanly.
#[allow(dead_code)]
fn _pyvalue_error(msg: &str) -> PyErr {
    PyValueError::new_err(msg.to_string())
}
