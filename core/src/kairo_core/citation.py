"""Citation-verification data contracts.

Populated by :class:`~kairo_rag.citation_check.CitationVerifier` after
answer generation. The UI and any downstream audit tooling should
depend only on these types, never on the verifier implementation.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ClaimVerification(BaseModel):
    """Per-sentence verification result inside a generated answer."""

    sentence: str
    sentence_index: int = Field(..., ge=0)
    supported: bool
    support_score: float = Field(..., ge=0.0, le=1.0)
    best_evidence_id: Optional[str] = None
    best_evidence_score: float = Field(0.0, ge=0.0, le=1.0)


class CitationReport(BaseModel):
    """Answer-wide citation audit.

    ``overall_support`` is the fraction of claims that cleared the
    verifier's per-claim threshold. ``answer_downgraded`` is True when
    that fraction fell below the verifier's downgrade threshold — the
    UI can surface a warning badge or reduce the displayed confidence
    without rewriting the answer.
    """

    claims: List[ClaimVerification] = Field(default_factory=list)
    overall_support: float = Field(0.0, ge=0.0, le=1.0)
    answer_downgraded: bool = False
    strategy: str = Field(
        "token_overlap",
        description="Which verifier produced this report; useful when "
        "multiple strategies (token_overlap, embedding, llm) ship side by side.",
    )
