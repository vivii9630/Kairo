"""Phase 13c — verify that a generated answer's claims are supported
by its cited evidences.

Default strategy is token_overlap: zero-dep, works without an LLM, and
catches the obvious hallucination case (the answer mentions an entity
that never appears in any evidence). Embedding- and LLM-based
verifiers can slot in behind the same interface later — every
verifier returns the same :class:`CitationReport`, so callers and
UI stay unchanged.

The verifier is intentionally conservative: short sentences and
highly-common tokens don't count as support evidence. A reader should
be able to trust that ``supported=True`` means at least one evidence
contains the *distinctive* content of the claim, not just shared stop
words.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from kairo_core import CitationReport, ClaimVerification, Evidence


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-_]{1,}")

# Small English stop list; keep it short so common content words stay
# eligible to carry the match signal.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "was",
    "one", "our", "has", "have", "had", "its", "from", "this", "that",
    "with", "been", "their", "they", "them", "were", "into", "than",
    "then", "what", "when", "your", "about", "would", "there", "could",
    "other", "after", "these", "those", "some", "more", "most", "very",
    "just", "also", "only", "over", "under", "which", "while", "who",
    "may", "can", "is", "it", "to", "in", "on", "of", "as", "an", "a",
    "be", "or", "we", "no", "so", "if", "at", "do",
})


def split_sentences(text: str) -> List[str]:
    """Split an answer into sentence-like units. Small and defensive —
    the model's output may omit terminal punctuation, so we also split
    on newlines and keep short fragments rather than dropping them."""
    text = text.strip()
    if not text:
        return []
    parts: List[str] = []
    for chunk in text.splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        for sent in _SENTENCE_RE.split(chunk):
            sent = sent.strip()
            if sent:
                parts.append(sent)
    return parts


def _content_tokens(text: str) -> set:
    return {
        tok.lower()
        for tok in _WORD_RE.findall(text)
        if tok.lower() not in _STOPWORDS and len(tok) > 2
    }


def _overlap_score(claim_tokens: set, evidence_tokens: set) -> float:
    """Asymmetric overlap: fraction of the *claim's* distinctive
    tokens also present in the evidence. This is the right metric for
    "is this claim supported?" — we don't punish long evidence for
    containing extra material.
    """
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


class CitationVerifier:
    """Token-overlap citation verifier.

    Parameters
    ----------
    claim_threshold :
        Minimum overlap fraction for a sentence to count as supported.
    downgrade_threshold :
        If the share of supported sentences falls below this, the
        overall answer is flagged as downgraded.
    min_content_tokens :
        Sentences with fewer distinctive tokens than this are treated
        as "trivial" and always pass (e.g. "Here is the summary:").
        Prevents transition phrases from tanking the support score.
    """

    name = "token_overlap"

    def __init__(
        self,
        *,
        claim_threshold: float = 0.35,
        downgrade_threshold: float = 0.6,
        min_content_tokens: int = 3,
    ) -> None:
        self.claim_threshold = claim_threshold
        self.downgrade_threshold = downgrade_threshold
        self.min_content_tokens = min_content_tokens

    def verify(self, answer: str, evidences: Sequence[Evidence]) -> CitationReport:
        sentences = split_sentences(answer)
        if not sentences:
            return CitationReport(strategy=self.name, overall_support=1.0)

        evidence_tokens = [(ev.document_id, _content_tokens(ev.text)) for ev in evidences]
        claims: List[ClaimVerification] = []
        supported_count = 0

        for idx, sent in enumerate(sentences):
            toks = _content_tokens(sent)
            if len(toks) < self.min_content_tokens:
                # Trivial / connective sentences — treat as supported
                # without evidence so they don't skew the overall score.
                claims.append(ClaimVerification(
                    sentence=sent,
                    sentence_index=idx,
                    supported=True,
                    support_score=1.0,
                ))
                supported_count += 1
                continue

            best_id = None
            best_score = 0.0
            for doc_id, ev_toks in evidence_tokens:
                score = _overlap_score(toks, ev_toks)
                if score > best_score:
                    best_score = score
                    best_id = doc_id

            supported = best_score >= self.claim_threshold
            if supported:
                supported_count += 1
            claims.append(ClaimVerification(
                sentence=sent,
                sentence_index=idx,
                supported=supported,
                support_score=best_score,
                best_evidence_id=best_id,
                best_evidence_score=best_score,
            ))

        overall = supported_count / len(claims)
        return CitationReport(
            strategy=self.name,
            claims=claims,
            overall_support=overall,
            answer_downgraded=overall < self.downgrade_threshold,
        )
