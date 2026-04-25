# Kairo Protocol v0.1 — wire-format specification

**Status:** captured · **Stability:** unstable (will tighten in v0.2) · **Last updated:** 2026-04-25 · **Owner:** edge/

## 0. Why this exists

Every future runtime shell — Python, Rust, TypeScript, Kotlin, Swift —
must read and write the same on-disk artifacts so a `.kairo/` directory
built on a developer laptop is portable to a Raspberry Pi, an Android
phone, or a browser tab without translation. This document is the
contract those shells implement against.

It is deliberately a *capture* of what `kairo-edge-core` (Rust) and
`kairo_edge` (Python) already emit today, not an aspiration. Anything
not used by the shipped code is marked **future** and is not part of
v0.1.

The validator implementation lives in a follow-up commit (Phase 14e.1
deliverable: `edge/rust/kairo-edge-core/src/protocol.rs` plus a
roundtrip test that asserts every fixture in `edge/tests/fixtures/`
passes both Python and Rust validators). v0.1 is the doc you're
reading.

## 1. Versioning

- The protocol carries a single integer field: `protocol_version`.
- v0.1 is the current shape. Existing artifacts that *omit*
  `protocol_version` are treated as v0.1 — readers default to it on
  missing field.
- Breaking changes bump the major (v1.0, v2.0, …).
- Additive changes — new optional fields, new node kinds, new edge
  kinds — bump the minor (v0.2). Readers must tolerate unknown
  optional fields.
- Removal or rename is a breaking change.

```jsonc
// every top-level artifact carries this
{
  "protocol_version": 1,
  // ... rest of payload ...
}
```

## 2. Encoding rules

- **Format**: UTF-8 JSON. Pretty-printed by writers (2-space indent),
  parsed flexibly by readers (any whitespace).
- **No comments.** Production artifacts are strict JSON. Examples in
  this document use `// ...` to annotate but writers must not emit them.
- **Numbers**: integers fit in `i64`; floating-point fits in `f64`.
  Scores and confidences are always `f64` in `[0.0, 1.0]` unless
  otherwise documented.
- **Strings**: UTF-8, no normalization (NFC vs NFD is the producer's
  choice). Readers compare strings byte-for-byte.
- **Maps**: object keys are always strings. Implementations should
  use stable iteration order (alphabetical) when serializing for
  hashing — important for the temporal layer's snapshot ids.
- **Null vs missing**: an absent field defaults; an explicit `null`
  is a hard error. Readers must reject `{"id": null}` even though
  they accept `{}` with `id` defaulted.

## 3. Type definitions

### 3.1 `Document`

The atomic unit of corpus content.

```jsonc
{
  "id": "abc123",                 // required, unique within a store
  "text": "...",                  // required, the searchable body
  "metadata": {                   // optional, defaults to {}
    "source": "github",
    "page": 42
  }
}
```

| Field      | Type    | Required | Default | Notes |
|------------|---------|----------|---------|-------|
| `id`       | string  | yes      | —       | Caller-chosen. Treated as opaque. |
| `text`     | string  | yes      | —       | UTF-8. Empty string permitted; readers should treat it as a doc that scores zero on every query. |
| `metadata` | object  | no       | `{}`    | Free-form key/value. Values may be any JSON type. |

### 3.2 `Evidence`

A retrieval result — one document promoted into the response.

```jsonc
{
  "document_id": "abc123",        // required, must reference a Document.id
  "text": "...",                  // required, may be a snippet or full text
  "score": 1.7234,                // required, retriever-defined units
  "metadata": { ... }             // optional, defaults to {}
}
```

| Field         | Type    | Required | Default | Notes |
|---------------|---------|----------|---------|-------|
| `document_id` | string  | yes      | —       | The source `Document.id`. |
| `text`        | string  | yes      | —       | Content shown to the user. May be the full document text or a window. |
| `score`       | number  | yes      | —       | Retriever-defined scale. BM25 produces unbounded positive floats; cosine sits in `[-1, 1]`; RRF in `[0, ~0.05]`. Readers must not assume any specific range. |
| `metadata`    | object  | no       | `{}`    | Carried verbatim from `Document.metadata` by default; retrievers may add `kind`, `provenance`, `confidence`, etc. as additive fields (see §5). |

### 3.3 `QueryRequest`

The input envelope for `EdgeStore.query`.

```jsonc
{
  "query": "felines hunting at dusk",
  "top_k": 5,                     // optional, default 5
  "filters": { "topic": "data" }  // optional, default {}
}
```

| Field     | Type    | Required | Default | Notes |
|-----------|---------|----------|---------|-------|
| `query`   | string  | yes      | —       | Natural-language query. UTF-8. |
| `top_k`   | integer | no       | 5       | Must be ≥ 1. Readers must reject `0` or negative. |
| `filters` | object  | no       | `{}`    | Each `(key, value)` pair narrows the candidate pool to docs whose `metadata[key] == value`. v0.1 supports equality only. |

### 3.4 `QueryResponse`

The output envelope from `EdgeStore.query` and any future
`EdgePipeline.query`.

```jsonc
{
  "answer": "Felines hunt at dusk.",
  "evidences": [ /* Evidence */ ],
  "steps": [ /* future: agent step trace */ ]
}
```

| Field       | Type        | Required | Default | Notes |
|-------------|-------------|----------|---------|-------|
| `answer`    | string      | yes      | —       | The synthesized answer. Extractive runtimes return concatenated evidence text; LLM-backed runtimes return the model's generation. |
| `evidences` | Evidence[]  | yes      | —       | Top-k selected. Order is descending by relevance. May be empty. |
| `steps`     | any[]       | no       | `[]`    | Reserved for future agent-trace payloads. v0.1 readers must accept any shape and pass it through. |

## 4. Store file format

The on-disk artifact a single `EdgeStore` reads from and writes to.

### 4.1 `.kairo_edge.json` (v0.1 path-style)

```jsonc
{
  "protocol_version": 1,
  "documents": [
    { "id": "...", "text": "...", "metadata": {} },
    // ...
  ]
}
```

| Field              | Type        | Required | Default | Notes |
|--------------------|-------------|----------|---------|-------|
| `protocol_version` | integer     | no       | `1`     | Omission means v0.1 for backward compat with existing fixtures. Future writers must emit it explicitly. |
| `documents`        | Document[]  | yes      | —       | Order is the insertion order. v0.1 has no separate index file; the store rebuilds BM25 on load. |

### 4.2 `.kairo/` directory layout (forward-compatible)

The full temporal layout from `kairo-temporal` (Phase 3) is **future**
in this protocol — capturing the file shapes here so the edge runtime
knows what to round-trip when 14e.4 (`EdgeGraph`) lands. None of these
are required for v0.1 conformance.

```
.kairo/
├── config.toml                 # branch policy, embedding provider
├── HEAD                        # current branch ref, e.g. "refs/branches/main"
├── refs/branches/<name>        # branch -> snapshot_id
├── history.json                # DAG: snapshot_id -> { parents, timestamp, message }
├── snapshots/<snapshot_id>.json
└── embeddings/<snapshot_id>.npy
```

A v0.1 edge runtime that doesn't implement the temporal layer must
not corrupt these files when present — pass them through unchanged.

## 5. Tolerance and forward compatibility

Every reader implementation **must** follow these rules so future
shells don't fork the format:

1. **Unknown optional fields**: ignore on read, preserve on write
   when round-tripping. A reader that drops fields it doesn't
   understand is a non-conforming reader.
2. **Missing optional fields**: default per the table.
3. **Explicit `null` on a typed field**: hard error. Validators
   should report `{"id": null}` as a violation, not coerce to empty
   string.
4. **Type mismatch**: hard error. `{"top_k": "5"}` is not legal even
   though the integer is recoverable.
5. **Extra metadata**: tag-style fields like `provenance`,
   `confidence`, `kind`, `extractor`, `source_doc_id` (carried over
   from the desktop graph) are pass-through. Edge runtimes don't
   interpret them in v0.1 but must round-trip them losslessly.

## 6. Roundtrip contract

For every protocol type T:

```
deserialize(serialize(t)) == t
```

…must hold for any value `t` that the spec accepts. A test fixture in
`edge/tests/fixtures/` for each type seeds the cross-runtime CI: every
runtime shell must pass the same fixture set.

The Rust side already has this for `Document` / `Evidence` /
`QueryRequest` / `QueryResponse` via the integration test
`wire_format_matches_python_shape` in `kairo-edge-core/tests/`. Phase
14e.1's validator commit will:

1. Move the fixtures to a shared location.
2. Generate them from one canonical writer.
3. Add a Python pytest that asserts byte-identical round-trip.

## 7. Future shapes (declared, not v0.1 frozen)

These are reserved for the edge graph and provider work in 14e.2+.
Readers must accept the shapes when present but the *spec* may change
before v1.0; producers should emit them only when the consuming
runtime advertises support.

### 7.1 Graph node

```jsonc
{
  "id": "...",
  "kind": "document" | "concept" | "docstring" | "rationale" | ...,
  "label": "...",
  "attrs": { /* arbitrary key/value */ }
}
```

### 7.2 Graph edge

```jsonc
{
  "source": "...",
  "target": "...",
  "kind": "similar-to" | "contains" | "calls" | ...,
  "weight": 0.7,                  // optional
  "provenance": "structural" | "extracted" | "inferred" | "ambiguous",
  "confidence": 0.0,              // [0.0, 1.0]
  "attrs": { /* arbitrary */ }
}
```

The desktop graph already produces this shape via Phase 13b1. The
edge runtime must round-trip it unchanged.

### 7.3 Provider capability manifest

```jsonc
{
  "name": "extractive",
  "kind": "inference" | "embedding" | "storage",
  "min_ram_mb": 0,
  "needs_gpu": false,
  "needs_webgpu": false,
  "model_families": [],          // empty = no model required
  "version": "0.1.0"
}
```

Phase 14e.2 ships the first concrete instance of this for the
`ExtractiveProvider`. Other shells (TS, Swift) read manifests from
the same JSON files at runtime.

### 7.4 Device profile snapshot

```jsonc
{
  "name": "pi-4",
  "peak_ram_mb": 1024,
  "cpu_cores": 4,
  "cpu_ghz": 1.5,
  "has_gpu": false,
  "has_webgpu": false,
  "platform": "linux" | "windows" | "macos" | "android" | "ios" | "browser",
  "arch": "x86_64" | "aarch64" | "wasm32"
}
```

Captured here so 14e.2's `DeviceProfile.probe()` can serialize results
for log/audit. Not v0.1 frozen.

## 8. Open questions (deferred, will be answered before v1.0)

- **Binary format alternative.** JSON is fine for `.kairo_edge.json`
  at small scale; for embedding matrices and large graph snapshots,
  a binary format (CBOR or FlatBuffers) is likely needed. Decision
  postponed until a real ingest blows past 50 MB on disk.
- **Schema language.** This document is prose plus tables. A formal
  JSON Schema or Smithy/Cap'n Proto definition makes sense once
  multiple language runtimes exist (TS, Swift). v0.1 stays prose.
- **Snapshot id derivation.** The desktop temporal layer defines
  this; the edge runtime must reproduce the same derivation when it
  reads a `.kairo/`. Spec-aligned but not duplicated here yet.
- **Metadata value types.** v0.1 says "any JSON value." Practical
  use suggests we should restrict to `string | number | bool | null
  | array<string|number>` to keep cross-language type mapping
  trivial. Decision deferred.

## 9. Conformance

A runtime is **v0.1-conformant** when:

1. It can read every fixture in `edge/tests/fixtures/v0.1/` without
   error and `deserialize(serialize(x)) == x` for each.
2. It rejects every fixture in `edge/tests/fixtures/v0.1/invalid/`
   with a clear validator error.
3. It preserves unknown additive fields when round-tripping. A
   `provenance` tag on an Evidence written by a v0.2 producer must
   survive a v0.1 reader untouched.
4. It emits `protocol_version: 1` on every store-file write.
5. It declares its protocol version in any logs or error messages
   that reference store files, so debugging cross-runtime mismatches
   is unambiguous.

The fixture suite ships with the v14e.1 validator commit. Until then,
treat the existing `kairo-edge-core` integration tests as the
de-facto conformance reference.
