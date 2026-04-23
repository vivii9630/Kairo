"""Phase 13b1 — edge provenance and confidence conventions.

Every edge in a Kairo graph carries two attributes beyond the existing
``kind`` and (for similarity edges) ``weight``:

- ``provenance``: one of :data:`PROVENANCE_VALUES`. Tells the retrieval
  layer and downstream phases (Phase 12 STDP) how far to trust the edge.
- ``confidence`` ∈ ``[0.0, 1.0]``: how certain the builder is that this
  edge is real. Structural edges default to ``1.0``; inferred edges
  carry the score they were derived from (e.g. cosine similarity for
  ``similar-to``); extracted edges take the producer's self-reported
  confidence; ambiguous edges stay low until a human review bumps them.

Retrieval, graph-walk boosting, and future plasticity updates should
reference these attrs instead of inventing their own trust model. When
building a new edge type always pass both — defaults are only a
convenience for the common structural case.
"""

from __future__ import annotations

from typing import Any, Dict, Final, Tuple


PROVENANCE_STRUCTURAL: Final[str] = "structural"
PROVENANCE_EXTRACTED: Final[str] = "extracted"
PROVENANCE_INFERRED: Final[str] = "inferred"
PROVENANCE_AMBIGUOUS: Final[str] = "ambiguous"

PROVENANCE_VALUES: Final[Tuple[str, ...]] = (
    PROVENANCE_STRUCTURAL,
    PROVENANCE_EXTRACTED,
    PROVENANCE_INFERRED,
    PROVENANCE_AMBIGUOUS,
)


def edge_attrs(
    *,
    provenance: str = PROVENANCE_STRUCTURAL,
    confidence: float = 1.0,
    **extra: Any,
) -> Dict[str, Any]:
    """Assemble the canonical attribute dict for an edge.

    Raises ``ValueError`` on invalid provenance or out-of-range
    confidence — callers should never try to squash garbage past this
    helper; it's the single choke point for edge-trust discipline.
    """
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(
            f"Invalid provenance: {provenance!r}. "
            f"Expected one of {PROVENANCE_VALUES}."
        )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Invalid confidence: {confidence}. Must lie in [0.0, 1.0]."
        )
    return {"provenance": provenance, "confidence": float(confidence), **extra}
