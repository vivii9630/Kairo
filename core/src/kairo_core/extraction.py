"""Relationship-extraction data contracts.

Shared by ``kairo-graph`` (where extractors live) and any future
consumer (UI, agents) that needs to reason about extracted edges
without importing the extractor runtime. Follows the same pattern as
``analytics_models`` — zero heavy deps, pure Pydantic.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ExtractorName = str


class ExtractedConcept(BaseModel):
    """A concept node an extractor wants to introduce.

    Unlike community-derived concept nodes (see kairo_graph.communities),
    extracted concepts have provenance in the source material — the
    extractor claims to have read the concept out of a document rather
    than synthesized it from graph topology. ``source_doc_id`` points
    back at the document the concept was lifted from for auditing.
    """

    id: str = Field(..., description="Stable concept node id (e.g. 'xc:authentication').")
    label: str
    description: str = ""
    source_doc_id: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ExtractedEdge(BaseModel):
    """A (subject, relation, object) triple the extractor claims to
    have found in or inferred from a document.

    ``provenance`` lets the extractor distinguish "I read this in the
    text" (extracted) from "I inferred this from context" (inferred)
    from "I'm not sure, review this" (ambiguous). Structural is not a
    valid value here — that's reserved for AST/schema-derived edges.
    """

    subject: str = Field(..., description="Source node id.")
    relation: str = Field(..., description="Relation name, e.g. 'depends-on'.")
    object: str = Field(..., description="Target node id.")
    provenance: Literal["extracted", "inferred", "ambiguous"] = "extracted"
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""
    source_doc_id: Optional[str] = None


class ExtractionResult(BaseModel):
    """What an extractor returns for a single document (or batch).

    ``concepts`` can be empty — many extractors only produce edges
    between existing nodes. ``edges`` can reference node ids that
    don't yet exist in the target graph; the consuming builder is
    responsible for deciding whether to create missing nodes or drop
    the edge (see :class:`LLMExtractedGraphBuilder` for the default
    policy: create missing ``concept``-kind nodes, drop edges that
    point at nothing).
    """

    extractor: ExtractorName
    edges: List[ExtractedEdge] = Field(default_factory=list)
    concepts: List[ExtractedConcept] = Field(default_factory=list)
    error: Optional[str] = None
