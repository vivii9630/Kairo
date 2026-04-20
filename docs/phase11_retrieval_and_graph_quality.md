# Phase 11 — Retrieval & Graph Quality

**Status:** planned
**Prerequisite for:** Phase 12 (STDP plasticity)
**Authored:** 2026-04-20

## 1. Context

Pre-Phase-11 state (what retrieval and the graph actually do today):

| Layer       | Current behavior                                                           | Where it lives                                             |
|-------------|----------------------------------------------------------------------------|------------------------------------------------------------|
| Retrieval   | Raw term-frequency sum, linear scan every document                         | `retrieval/src/kairo_retrieval/local.py:25-26`             |
| Embeddings  | `hash-stub` is default — byte-stable token hashing, **not semantic**       | `embeddings/src/kairo_embeddings/providers/hash_stub.py`   |
| Graph       | `DocumentGraphBuilder` creates **one node per doc, zero edges**            | `graph/src/kairo_graph/builders/document.py:20-31`         |
| Integration | `LocalPipeline` never consults the graph during retrieval                  | (disconnected)                                             |

Net effect: the graph is decorative. Retrieval quality is capped at lexical keyword matching. Any advanced graph-RAG feature (community summaries, STDP plasticity, multi-hop expansion) built on this substrate learns or traverses noise.

## 2. Goals

1. Hybrid retrieval (lexical + semantic) that actually uses embeddings.
2. A graph where edges carry meaning (similarity weights) so traversal is defensible.
3. A measurable baseline — an eval harness — so every further change can be justified numerically.
4. Stay CPU-only by default so local Ollama + small-LLM users get the full benefit without a GPU.

## 3. Sub-phases

### 11a — BM25 + hybrid scoring in retrieval

Replace raw TF with BM25 and add semantic cosine as a weighted second term.

- Dependency: `rank_bm25` (pure-Python, ~2 KB, battle-tested).
- `LocalPipeline.__init__` accepts an optional `EmbeddingProvider`; if set, the retriever pre-embeds all docs on ingest and computes `score = α · bm25_norm + β · cosine_norm` at query time. Defaults: `α = β = 0.5`.
- If no provider is configured, falls back to pure BM25 — the old hash-stub is not re-used as a similarity proxy (it's meaningless and misleading).
- Normalization: min-max across the current query's candidate pool, not global.

**Files touched:** `retrieval/src/kairo_retrieval/local.py`, `retrieval/pyproject.toml` (add `rank_bm25` to deps).

### 11b — Real embeddings as default

Flip `get_provider("default")` resolution from `hash-stub` to `sentence-transformers` with model `all-MiniLM-L6-v2`.

- 384-dim output, ~90 MB model, ~5 ms/sentence on CPU — the right size for a local default.
- `hash-stub` stays in the registry as an **explicit** opt-in for CI / offline-install / deterministic tests.
- If `sentence-transformers` isn't importable, `get_provider("default")` falls back to `hash-stub` with a one-line warning rather than raising — so `pip install kairo-core` without the `[embeddings]` extra still works.

**Files touched:** `embeddings/src/kairo_embeddings/registry.py`.

### 11c — Weighted edges in graph builders

Replace the current zero-edge `DocumentGraphBuilder` with one that writes similarity edges.

- After node creation, compute pairwise cosine over all doc embeddings.
- For each node, keep the top `k = 5` neighbors above threshold `τ = 0.3`.
- Edge attrs: `kind = "similar-to"`, `weight = float(cosine)`.
- Sparse by construction — on 10k docs this is ~50k edges, not 100M.
- For > ~5k docs, skip the O(n²) pass and use an approximate neighbors index (HNSW via `hnswlib`). Deferred to 11c.2 if we hit that scale.

Applies equally to `TabularGraphBuilder` and `CodeGraphBuilder` — they all currently emit edgeless graphs. Extract the pairwise-similarity pass into a shared `builders/_similarity.py` helper.

**Files touched:** `graph/src/kairo_graph/builders/{document,tabular,code}.py`, new `graph/src/kairo_graph/builders/_similarity.py`.

### 11d — Eval harness

The smallest thing that lets us tell if 11a–c actually helped.

- `retrieval/tests/eval_data.json` — 10–20 hand-labeled `{query, corpus_ids, relevant_ids}` triples.
- `retrieval/tests/eval_retrieval.py` — computes `recall@k`, `MRR`, `nDCG@k` for a given pipeline config.
- Print a comparison table across (TF, BM25, BM25+cosine, BM25+cosine+graph-expand).
- Run under `pytest` so CI catches regressions.

**Files touched:** new `retrieval/tests/eval_data.json`, new `retrieval/tests/eval_retrieval.py`.

## 4. Ship order

One commit per sub-phase, in order `11a → 11b → 11c → 11d`. 11d is intentionally last so it can measure the cumulative lift.

## 5. Expected wins (rough, from RAG literature)

| Metric             | TF baseline | BM25 only | BM25 + cosine | + graph-expand (Tier 2) |
|--------------------|-------------|-----------|---------------|-------------------------|
| recall@5           | ~0.30       | ~0.45     | ~0.65         | ~0.75                   |
| MRR                | ~0.25       | ~0.40     | ~0.60         | ~0.70                   |
| p95 query latency  | ~5 ms       | ~5 ms     | ~25 ms        | ~40 ms                  |

Numbers are indicative, not promises — the eval harness in 11d is what produces the actual per-corpus deltas.

## 6. Future work — STDP plasticity (Phase 12+)

Issue #3 proposes a Spike-Timing-Dependent-Plasticity layer on top of the graph. Edges learn from retrieval→reasoning→reward episodes instead of staying fixed at construction time. The full rule (eligibility traces with `τ+/τ-` windows, reward-modulated updates, goal-conditioned weights) is a multi-quarter program. What Phase 11 enables, and what Phase 12 would layer on:

### 12a — activation log + feedback bit (no learning)

- `kairo-retrieval` emits `{session_id, node_id, step_idx, ts, role}` per retrieved / cited / used node to `.kairo/activations.parquet`. Free data collection, no behavior change.
- `AskResponse` grows an `accepted: bool | None` field. UI captures thumbs-up / thumbs-down. This is the reward signal — without it, plasticity has nothing to gate on.

### 12b — simple co-activation counter

- For sessions with `accepted=True`, increment `helpful[A][B]` whenever nodes A and B co-activated within window W.
- Retrieval score gains one term: `μ · log(1 + helpful[query_neighbors][candidate])`.
- Zero math from issue #3 yet — no decay, no goal conditioning. Just "nodes that helped before help again."

### 12c — eligibility trace with timing

- Replace the counter with the exponentially-decayed trace from issue #3: `e_AB += exp(-Δt / τ)` on co-activation, multiplicatively decayed by λ per step.
- Still single-goal (one global weight table).

### 12d — goal-conditioned weights

- Classify each query into a goal bucket (initially rule-based: "diagnose", "explain", "plan", "compare"; later via a small classifier).
- Maintain `w_ij^g` per goal. Retrieval fetches the table for the current query's goal.
- This is where the formula `w_ij ← w_ij + η (u_g^⊤ x_ij) r_g e_ij` from issue #3 lands.

### 12e — staleness decay + correction spikes

- Use Kairo's existing `kairo-temporal` validity intervals. Edges attached to nodes whose `valid_to < now` decay at rate `β`.
- When later retrieval contradicts earlier reasoning (a "correction spike"), apply a negative update to the implicated edges.

### Where STDP hooks into the engine

| Concern                      | Package / file                              |
|------------------------------|---------------------------------------------|
| Activation log emission      | `kairo-retrieval` (hooks into scoring loop) |
| Feedback endpoint            | `kairo-ai/app.py` (new `POST /feedback`)    |
| Plasticity store             | New `kairo-plasticity` package              |
| Scoring integration          | `kairo-retrieval` (new term in score fn)    |
| Storage                      | `.kairo/activations.parquet`, `.kairo/plasticity.parquet` |

New package `kairo-plasticity` keeps the 11-package modular discipline (see `memory/project_package_structure.md`). STDP updates happen async post-session, not in the query hot path.

## 7. Why this matters especially for small LLMs

Kairo is designed to run against local Ollama-hosted small models (3B–8B class) as the default, not cloud giants. Small LLMs have three characterized weaknesses:

1. **Weak multi-hop synthesis** across retrieved chunks. A 70B model can chain "A mentions B, B implies C, so answer depends on C"; a 3B often cannot.
2. **Sharp degradation with distractor context.** Adding 5 irrelevant chunks harms a 3B model's answer much more than a 70B's.
3. **Smaller effective context.** Even if the window is 128k, attention quality at large offsets drops faster in smaller models.

Every Phase 11 and 12 item attacks exactly these weaknesses:

- **BM25 + semantic hybrid (11a-b) raises precision@k.** Fewer distractor docs reach the LLM. Big win #2 addressed.
- **Weighted edges (11c) let retrieval pre-synthesize multi-hop relations.** If the edge between doc A and doc C is already in the graph with a meaningful weight, the LLM doesn't have to infer the A→B→C chain — the graph did it at build time. Big win #1 addressed.
- **Graph 1-hop expansion (Tier 2) = the same multi-hop pre-synthesis at query time.** Pulls in semantically-linked evidence the pure vector search would miss.
- **STDP plasticity (Phase 12)** is where the real small-LLM leverage appears: over time, the retrieval pipeline learns *which specific nodes* answer *which specific goals*. A 3B model answering from 2 curated, learned-to-be-useful docs can match a 70B answering from 10 raw chunks. This shifts cost from expensive inference to cheap retrieval.

The general principle: **retrieval quality substitutes for model size**. Every token of noise removed from context is a token the small model doesn't have to reason past. Every multi-hop edge pre-computed in the graph is reasoning the small model doesn't have to perform. Phase 11 + 12 are the path to making a local 3B-on-Ollama Kairo behave like a 70B-on-cloud RAG system for most task-driven workloads.

## 8. Open questions

- **BM25 implementation.** `rank_bm25` package vs hand-roll (~30 lines). Recommending the package.
- **Eval corpus.** Synthetic 20-doc toy set for 11d, or label a real dataset? Synthetic first; swap to real when user has a concrete use case.
- **Commit bundling.** Four separate commits vs one Phase-11 squash. Four commits matches the existing 8a–8e style.
