//! Integration coverage for the full edge store path.

use std::collections::BTreeMap;

use kairo_edge_core::bm25::Bm25Index;
use kairo_edge_core::store::EdgeStore;
use kairo_edge_core::types::{Document, QueryRequest};

fn corpus() -> Vec<Document> {
    [
        ("d01", "Cats purr when they feel content and relaxed."),
        ("d02", "Felines are solitary hunters active at dusk."),
        ("d03", "A house cat's tongue has backward-facing papillae."),
        ("d04", "Dogs bark to alert owners of unfamiliar sounds."),
        ("d05", "Golden retrievers are friendly loyal family dogs."),
        ("d06", "Python is a high-level programming language."),
        ("d07", "Rust offers memory safety through ownership rules."),
        ("d08", "Transformers use self-attention across long contexts."),
        ("d09", "ETL pipelines move records into data warehouses."),
        ("d10", "Espresso is brewed by forcing hot water through grounds."),
    ]
    .into_iter()
    .map(|(id, text)| Document::new(id, text))
    .collect()
}

#[test]
fn bm25_top_hit_matches_query_topic() {
    let idx = Bm25Index::from_docs(corpus().into_iter().map(|d| (d.id, d.text)));
    let hits = idx.top_k("felines hunt dusk", 3);
    assert!(!hits.is_empty(), "expected at least one hit");
    assert_eq!(hits[0].0, "d02", "top hit should be about felines hunting");
    // All returned scores are strictly positive given zero-filter discipline.
    for (_, score) in &hits {
        assert!(*score > 0.0, "filtered results must be positive");
    }
}

#[test]
fn edge_store_roundtrip_serves_the_same_ranking() {
    use tempfile::tempdir;

    let tmp = tempdir().unwrap();
    let path = tmp.path().join("edge.json");

    let mut store = EdgeStore::open(&path).unwrap();
    store.add(corpus()).unwrap();
    assert_eq!(store.len(), 10);

    let response = store.query(&QueryRequest::new("rust memory safety", 3));
    let top_ids: Vec<&str> = response.evidences.iter().map(|e| e.document_id.as_str()).collect();
    assert!(top_ids.contains(&"d07"), "expected the rust doc in top-3");

    // Round-trip: re-open from disk and confirm the ranking is stable.
    drop(store);
    let reopened = EdgeStore::open(&path).unwrap();
    let again = reopened.query(&QueryRequest::new("rust memory safety", 3));
    let again_ids: Vec<&str> = again.evidences.iter().map(|e| e.document_id.as_str()).collect();
    assert_eq!(top_ids, again_ids);
}

#[test]
fn empty_store_returns_no_evidence_answer() {
    let store = EdgeStore::in_memory();
    let response = store.query(&QueryRequest::new("anything", 3));
    assert!(response.evidences.is_empty());
    assert_eq!(response.answer, "No relevant evidence found.");
}

#[test]
fn document_metadata_round_trips_through_the_store() {
    use tempfile::tempdir;

    let tmp = tempdir().unwrap();
    let path = tmp.path().join("edge.json");

    let mut meta: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    meta.insert("topic".to_string(), serde_json::json!("rust"));
    meta.insert("source".to_string(), serde_json::json!("test"));

    let doc = Document {
        id: "m1".to_string(),
        text: "Rust is a systems programming language.".to_string(),
        metadata: meta.clone(),
    };

    let mut store = EdgeStore::open(&path).unwrap();
    store.add([doc]).unwrap();

    let reopened = EdgeStore::open(&path).unwrap();
    let response = reopened.query(&QueryRequest::new("rust systems programming", 1));
    assert_eq!(response.evidences.len(), 1);
    assert_eq!(response.evidences[0].metadata, meta);
}

#[test]
fn wire_format_matches_python_shape() {
    // A Document JSON blob written by Python's Pydantic should
    // deserialize cleanly here. We mimic the exact shape — extra
    // fields are tolerated by serde's defaults.
    let python_json = r#"
        {
            "id": "p1",
            "text": "hello",
            "metadata": {"source": "pytest"}
        }
    "#;
    let doc: Document = serde_json::from_str(python_json).unwrap();
    assert_eq!(doc.id, "p1");
    assert_eq!(doc.metadata.get("source").unwrap(), &serde_json::json!("pytest"));
}
