//! Thin CLI binary around [`kairo_edge_core`].
//!
//! The Python emulator (`edge/emulator/scenarios/rust_smoke.py`)
//! subprocesses this binary to run Rust-side workloads through the
//! same profile budgets as the Python scenarios. Keeping the CLI
//! minimal — no argparse, just a JSON command on stdin and a JSON
//! result on stdout — matches the shape of how a real edge device
//! would invoke the Rust runtime from another language.
//!
//! Supported commands today:
//!
//! ```json
//! {
//!   "action": "smoke",
//!   "corpus": [{"id": "d01", "text": "..."}, ...],
//!   "query": "felines hunt at night",
//!   "top_k": 3
//! }
//! ```
//!
//! Response shape:
//!
//! ```json
//! {
//!   "ok": true,
//!   "ingest_ns": 123456,
//!   "query_ns": 4567,
//!   "retrieved_ids": ["d02"],
//!   "evidence_count": 1,
//!   "answer_preview": "Felines are solitary hunters..."
//! }
//! ```
//!
//! Errors are reported as `{"ok": false, "error": "..."}`.

use std::io::{self, Read};
use std::time::Instant;

use kairo_edge_core::{Document, EdgeStore, QueryRequest};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum Command {
    Smoke {
        corpus: Vec<Document>,
        query: String,
        #[serde(default = "default_top_k")]
        top_k: usize,
    },
}

fn default_top_k() -> usize {
    3
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Response {
    Ok(SmokeResponse),
    Err(ErrorResponse),
}

#[derive(Debug, Serialize)]
struct SmokeResponse {
    ok: bool,
    ingest_ns: u128,
    query_ns: u128,
    retrieved_ids: Vec<String>,
    evidence_count: usize,
    answer_preview: String,
    corpus_size: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    ok: bool,
    error: String,
}

fn main() {
    let mut input = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut input) {
        emit_error(format!("stdin read failed: {err}"));
        std::process::exit(1);
    }

    let command: Command = match serde_json::from_str(&input) {
        Ok(c) => c,
        Err(err) => {
            emit_error(format!("invalid command JSON: {err}"));
            std::process::exit(1);
        }
    };

    match run(command) {
        Ok(resp) => {
            let payload = Response::Ok(resp);
            println!("{}", serde_json::to_string(&payload).unwrap());
        }
        Err(err) => {
            emit_error(err);
            std::process::exit(1);
        }
    }
}

fn run(command: Command) -> Result<SmokeResponse, String> {
    match command {
        Command::Smoke {
            corpus,
            query,
            top_k,
        } => smoke(corpus, query, top_k),
    }
}

fn smoke(corpus: Vec<Document>, query: String, top_k: usize) -> Result<SmokeResponse, String> {
    let corpus_size = corpus.len();

    let ingest_start = Instant::now();
    let mut store = EdgeStore::in_memory();
    store.add(corpus).map_err(|e| format!("ingest failed: {e}"))?;
    let ingest_ns = ingest_start.elapsed().as_nanos();

    let query_start = Instant::now();
    let response = store.query(&QueryRequest {
        query,
        top_k,
        filters: Default::default(),
    });
    let query_ns = query_start.elapsed().as_nanos();

    let retrieved_ids: Vec<String> = response
        .evidences
        .iter()
        .map(|ev| ev.document_id.clone())
        .collect();
    let evidence_count = response.evidences.len();
    let answer_preview = response.answer.chars().take(80).collect();

    Ok(SmokeResponse {
        ok: true,
        ingest_ns,
        query_ns,
        retrieved_ids,
        evidence_count,
        answer_preview,
        corpus_size,
    })
}

fn emit_error(message: String) {
    let payload = Response::Err(ErrorResponse {
        ok: false,
        error: message,
    });
    println!("{}", serde_json::to_string(&payload).unwrap());
}
