"""Phase 13b3 — LLMExtractedGraphBuilder + ExtractorProvider protocol tests.

We use a FakeExtractor so tests don't need a live Ollama server. A
smoke test for the Ollama JSON-parsing path lives separately at the
end and can be run by hand.
"""

from __future__ import annotations

from typing import List, Sequence

from kairo_core import (
    Document,
    ExtractedConcept,
    ExtractedEdge,
    ExtractionResult,
)
from kairo_graph import (
    DocumentGraphBuilder,
    ExtractorProvider,
    KairoGraph,
    LLMExtractedGraphBuilder,
    OllamaExtractor,
)
from kairo_graph.extractors.ollama import _extract_json_block


# ---------------------------------------------------------------------------
# Fake extractor for builder tests
# ---------------------------------------------------------------------------

class FakeExtractor:
    name = "fake"

    def __init__(self, program: List[ExtractionResult]) -> None:
        self._program = list(program)
        self.seen_docs: List[str] = []

    def extract(
        self,
        document: Document,
        context: Sequence[Document] = (),
    ) -> ExtractionResult:
        self.seen_docs.append(document.id)
        if self._program:
            return self._program.pop(0)
        return ExtractionResult(extractor=self.name)


def test_fake_extractor_satisfies_protocol() -> None:
    assert isinstance(FakeExtractor([]), ExtractorProvider)


# ---------------------------------------------------------------------------
# Builder merges edges + concepts
# ---------------------------------------------------------------------------

def test_builder_merges_extracted_edges_onto_existing_graph() -> None:
    g = KairoGraph()
    g.add_node("d1", kind="document")
    g.add_node("d2", kind="document")

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[
                ExtractedEdge(
                    subject="d1",
                    relation="depends-on",
                    object="d2",
                    provenance="extracted",
                    confidence=0.9,
                    rationale="d1 imports from d2 in the prose",
                ),
            ],
        ),
    ])
    builder = LLMExtractedGraphBuilder(fake)
    results = builder.augment(g, [Document(id="d1", text="...")])

    assert len(results) == 1
    assert results[0].error is None
    edges = list(g.nx_graph.edges("d1", data=True))
    assert len(edges) == 1
    _, target, data = edges[0]
    assert target == "d2"
    assert data["kind"] == "depends-on"
    assert data["provenance"] == "extracted"
    assert data["confidence"] == 0.9
    assert data["rationale"].startswith("d1 imports")
    assert data["extractor"] == "fake"


def test_builder_creates_missing_concept_nodes_by_default() -> None:
    g = KairoGraph()
    g.add_node("d1", kind="document")

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[
                ExtractedEdge(
                    subject="d1",
                    relation="mentions",
                    object="xc:authentication",
                    provenance="extracted",
                    confidence=0.8,
                ),
            ],
        ),
    ])
    LLMExtractedGraphBuilder(fake).augment(g, [Document(id="d1", text="...")])

    assert "xc:authentication" in g.nx_graph.nodes
    assert g.nx_graph.nodes["xc:authentication"]["kind"] == "concept"


def test_builder_drops_edges_to_missing_nodes_when_create_disabled() -> None:
    g = KairoGraph()
    g.add_node("d1", kind="document")

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[
                ExtractedEdge(
                    subject="d1",
                    relation="mentions",
                    object="xc:authentication",
                    provenance="extracted",
                    confidence=0.8,
                ),
            ],
        ),
    ])
    LLMExtractedGraphBuilder(
        fake, create_missing_concepts=False
    ).augment(g, [Document(id="d1", text="...")])

    assert "xc:authentication" not in g.nx_graph.nodes
    assert g.nx_graph.number_of_edges() == 0


def test_builder_applies_relation_whitelist() -> None:
    g = KairoGraph()
    g.add_node("d1")
    g.add_node("d2")

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[
                ExtractedEdge(
                    subject="d1", relation="depends-on", object="d2",
                    provenance="extracted", confidence=0.9,
                ),
                ExtractedEdge(
                    subject="d1", relation="randomly-invented", object="d2",
                    provenance="extracted", confidence=0.9,
                ),
            ],
        ),
    ])
    LLMExtractedGraphBuilder(
        fake, relation_whitelist={"depends-on"}
    ).augment(g, [Document(id="d1", text="...")])

    kinds = [d["kind"] for _, _, d in g.nx_graph.edges(data=True)]
    assert kinds == ["depends-on"]


def test_builder_records_error_results_without_corrupting_graph() -> None:
    g = KairoGraph()
    g.add_node("d1")

    fake = FakeExtractor([
        ExtractionResult(extractor="fake", error="ollama timeout"),
    ])
    results = LLMExtractedGraphBuilder(fake).augment(
        g, [Document(id="d1", text="...")]
    )
    assert results[0].error == "ollama timeout"
    assert g.nx_graph.number_of_edges() == 0


def test_builder_persists_rationale_and_source_doc_id_on_edge() -> None:
    g = KairoGraph()
    g.add_node("d1")
    g.add_node("d2")

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[
                ExtractedEdge(
                    subject="d1", relation="cites", object="d2",
                    provenance="extracted", confidence=0.7,
                    rationale="paragraph 3 cites d2 by title",
                    source_doc_id="d1",
                ),
            ],
        ),
    ])
    LLMExtractedGraphBuilder(fake).augment(g, [Document(id="d1", text="...")])
    _, _, data = list(g.nx_graph.edges(data=True))[0]
    assert data["rationale"] == "paragraph 3 cites d2 by title"
    assert data["source_doc_id"] == "d1"


def test_stack_with_document_builder_preserves_structural_edges() -> None:
    """DocumentGraphBuilder edges come first; LLMExtractedGraphBuilder
    layers on top without invalidating them.

    KairoGraph wraps nx.DiGraph (one edge per directed pair), so the
    default collision_policy="skip" guards against the extractor
    overwriting structural/similarity edges. We pick node pairs that
    don't collide with the similarity layer (reverse direction +
    concept creation) so both edge types end up in the final graph.
    """
    from kairo_embeddings import get_provider

    docs = [
        Document(id="d1", text="cats purr felines hunt"),
        Document(id="d2", text="cats sleep felines bathe"),
    ]
    g = DocumentGraphBuilder(
        embedding_provider=get_provider("hash-stub", dim=64),
        similarity_k=1,
        similarity_threshold=0.0,
    ).build(docs)

    similar_before = sum(
        1 for _, _, d in g.nx_graph.edges(data=True)
        if d.get("kind") == "similar-to"
    )
    assert similar_before >= 1

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[
                # Reverse direction of an existing similar-to pair —
                # DiGraph allows this because (d2, d1) != (d1, d2).
                ExtractedEdge(
                    subject="d2", relation="cites", object="d1",
                    provenance="extracted", confidence=0.9,
                ),
                # Brand-new target (auto-created as a concept node).
                ExtractedEdge(
                    subject="d1", relation="mentions", object="xc:hunting",
                    provenance="extracted", confidence=0.8,
                ),
            ],
        ),
        ExtractionResult(extractor="fake"),
    ])
    LLMExtractedGraphBuilder(fake).augment(g, docs)

    similar_after = sum(
        1 for _, _, d in g.nx_graph.edges(data=True)
        if d.get("kind") == "similar-to"
    )
    extracted_after = sum(
        1 for _, _, d in g.nx_graph.edges(data=True)
        if d.get("kind") in {"cites", "mentions"}
    )
    assert similar_after == similar_before  # structural preserved
    assert extracted_after == 2  # both new edges landed
    assert g.nx_graph.nodes["xc:hunting"]["kind"] == "concept"


def test_collision_policy_skip_preserves_existing_edge_when_pair_collides() -> None:
    """When the extracted edge would land on the same directed pair as
    an existing edge, skip keeps the existing edge intact."""
    g = KairoGraph()
    g.add_node("d1")
    g.add_node("d2")
    g.add_edge(
        "d1", "d2",
        kind="similar-to",
        weight=0.7,
        provenance="inferred",
        confidence=0.7,
    )

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[ExtractedEdge(
                subject="d1", relation="related-to", object="d2",
                provenance="extracted", confidence=0.9,
            )],
        ),
    ])
    LLMExtractedGraphBuilder(fake, collision_policy="skip").augment(
        g, [Document(id="d1", text="...")]
    )

    data = g.nx_graph.get_edge_data("d1", "d2")
    assert data["kind"] == "similar-to"
    assert data["provenance"] == "inferred"


def test_collision_policy_overwrite_replaces_existing_edge() -> None:
    g = KairoGraph()
    g.add_node("d1")
    g.add_node("d2")
    g.add_edge(
        "d1", "d2",
        kind="similar-to",
        weight=0.7,
        provenance="inferred",
        confidence=0.7,
    )

    fake = FakeExtractor([
        ExtractionResult(
            extractor="fake",
            edges=[ExtractedEdge(
                subject="d1", relation="related-to", object="d2",
                provenance="extracted", confidence=0.9,
            )],
        ),
    ])
    LLMExtractedGraphBuilder(
        fake, collision_policy="overwrite"
    ).augment(g, [Document(id="d1", text="...")])

    data = g.nx_graph.get_edge_data("d1", "d2")
    assert data["kind"] == "related-to"
    assert data["provenance"] == "extracted"


def test_collision_policy_rejects_invalid_value() -> None:
    import pytest
    with pytest.raises(ValueError):
        LLMExtractedGraphBuilder(FakeExtractor([]), collision_policy="zap")


# ---------------------------------------------------------------------------
# Ollama JSON extractor parsing helpers
# ---------------------------------------------------------------------------

def test_extract_json_block_strips_fences() -> None:
    raw = "```json\n{\"edges\": []}\n```"
    assert _extract_json_block(raw) == '{"edges": []}'


def test_extract_json_block_carves_from_prose() -> None:
    raw = "Sure, here is the result:\n{\"edges\": [], \"concepts\": []}\nHope that helps."
    block = _extract_json_block(raw)
    assert block is not None
    assert block.startswith("{") and block.endswith("}")


def test_extract_json_block_none_on_garbage() -> None:
    assert _extract_json_block("no braces here") is None


def test_ollama_parse_response_tolerates_malformed_entries() -> None:
    ext = OllamaExtractor(model="test")
    # One good edge, one invalid (bad provenance), one good concept, one non-dict.
    raw = (
        '{"edges": ['
        '  {"subject": "a", "relation": "r", "object": "b", '
        '   "provenance": "extracted", "confidence": 0.9},'
        '  {"subject": "a", "relation": "r", "object": "b", '
        '   "provenance": "bogus", "confidence": 0.9}'
        '], "concepts": ['
        '  {"id": "c1", "label": "concept one", "confidence": 0.8},'
        '  "not a dict"'
        ']}'
    )
    result = ext._parse_response(raw, document_id="d1")
    assert result.error is None
    assert len(result.edges) == 1  # bogus provenance filtered
    assert len(result.concepts) == 1  # non-dict filtered


def test_ollama_parse_response_handles_top_level_non_object() -> None:
    ext = OllamaExtractor(model="test")
    result = ext._parse_response('["not", "an", "object"]', document_id="d1")
    assert result.error is not None
