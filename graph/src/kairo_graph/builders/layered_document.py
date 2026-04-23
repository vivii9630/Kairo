"""Default 3-layer builder that decomposes documents into document / semantic / detail layers.

Layer 1 (document) — one node per document, structural edges (sequence).
Layer 2 (semantic) — key phrases / concepts extracted from each document,
    with intra-layer edges for co-occurrence within the same document.
Layer 3 (detail)   — individual sentences or data points, densely wired
    with similarity and reference edges.

Inter-layer edges connect each layer downward:
    document → semantic (contains), semantic → detail (grounds).
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Set, Tuple

from kairo_core import Document

from ..graph import KairoGraph
from ..layered import LayeredKairoGraph
from ..provenance import PROVENANCE_INFERRED, PROVENANCE_STRUCTURAL, edge_attrs


_STRUCTURAL = edge_attrs(provenance=PROVENANCE_STRUCTURAL, confidence=1.0)
# Co-occurrence / proximity signals are weak — phrases sharing a doc,
# sentences within N positions of each other. Keep the confidence low
# so downstream filtering can easily exclude them.
_COOCCUR = edge_attrs(provenance=PROVENANCE_INFERRED, confidence=0.4)
_NEAR = edge_attrs(provenance=PROVENANCE_INFERRED, confidence=0.3)


# ---------------------------------------------------------------------------
# Lightweight NLP helpers (zero external deps)
# ---------------------------------------------------------------------------

# Common English stop words — enough for phrase extraction without nltk.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "not", "no", "nor", "so", "if", "then",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "all", "also", "am", "as", "because", "before", "between", "both",
    "each", "few", "he", "her", "here", "him", "his", "how", "i", "into",
    "me", "more", "most", "my", "now", "only", "other", "our", "out",
    "over", "own", "s", "same", "she", "some", "such", "t", "there",
    "they", "through", "under", "up", "we", "what", "when", "where",
    "which", "while", "who", "whom", "why", "you", "your",
}

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_phrases(text: str, max_phrases: int = 12) -> List[str]:
    """Extract salient multi-word and single-word phrases from *text*.

    Uses simple frequency counting over non-stop-words.  Returns up to
    *max_phrases* phrases sorted by descending frequency.
    """
    words = [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP_WORDS]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: freq[w], reverse=True)
    return ranked[:max_phrases]


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences (simple regex-based)."""
    parts = _SENT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class DocumentLayeredBuilder:
    """Decomposes :class:`Document` objects into a 3-layer graph.

    Designed to work with any connector output — the decomposition is
    purely text-based so it applies to GitHub files, Slack messages,
    emails, Drive docs, etc.
    """

    name = "layered_document"

    def build(self, documents: Iterable[Document]) -> LayeredKairoGraph:
        lg = LayeredKairoGraph()
        doc_list = list(documents)
        prev_doc_id: str | None = None

        for doc in doc_list:
            # ---- Layer 1: document node -----------------------------------
            lg.document.add_node(
                doc.id,
                kind="document",
                label=doc.id,
                text_preview=doc.text[:200],
                **{f"meta_{k}": v for k, v in doc.metadata.items()},
            )
            if prev_doc_id is not None:
                lg.document.add_edge(prev_doc_id, doc.id, kind="sequence", **_STRUCTURAL)
            prev_doc_id = doc.id

            # ---- Layer 2: semantic phrase nodes ---------------------------
            phrases = _extract_phrases(doc.text)
            phrase_ids: List[str] = []
            for phrase in phrases:
                pid = f"sem:{doc.id}:{phrase}"
                lg.semantic.add_node(pid, kind="concept", label=phrase)
                phrase_ids.append(pid)
                # inter-layer: document → semantic
                lg.add_inter_edge(
                    doc.id, pid,
                    source_layer="document",
                    target_layer="semantic",
                    kind="contains",
                    **_STRUCTURAL,
                )

            # co-occurrence edges within this document's phrases
            for i, a in enumerate(phrase_ids):
                for b in phrase_ids[i + 1:]:
                    lg.semantic.add_edge(a, b, kind="co_occurs", **_COOCCUR)

            # ---- Layer 3: detail sentence nodes ---------------------------
            sentences = _split_sentences(doc.text)
            sent_ids: List[str] = []
            for idx, sent in enumerate(sentences):
                sid = f"det:{doc.id}:{idx}:{_short_hash(sent)}"
                lg.detail.add_node(
                    sid,
                    kind="sentence",
                    label=sent[:80],
                    full_text=sent,
                    position=idx,
                )
                sent_ids.append(sid)

                # ground each sentence to the phrases it mentions
                sent_lower = sent.lower()
                for pid, phrase in zip(phrase_ids, phrases):
                    if phrase in sent_lower:
                        lg.add_inter_edge(
                            pid, sid,
                            source_layer="semantic",
                            target_layer="detail",
                            kind="grounds",
                            **_STRUCTURAL,
                        )

            # sequence edges between sentences
            for i in range(len(sent_ids) - 1):
                lg.detail.add_edge(sent_ids[i], sent_ids[i + 1], kind="follows", **_STRUCTURAL)

            # similarity edges (window-based proximity for now;
            # embedding-based similarity lands in Phase 7b with KNN)
            window = 3
            for i, sid_a in enumerate(sent_ids):
                for sid_b in sent_ids[i + 2: i + 2 + window]:
                    lg.detail.add_edge(sid_a, sid_b, kind="near", **_NEAR)

        # Cross-document semantic links: shared phrases across docs
        _add_cross_doc_semantic_edges(lg, doc_list)

        return lg


def _add_cross_doc_semantic_edges(
    lg: LayeredKairoGraph, docs: List[Document]
) -> None:
    """Link semantic nodes from different documents that share the same phrase."""
    phrase_to_nodes: Dict[str, List[str]] = {}
    for node_id, data in lg.semantic.nx_graph.nodes(data=True):
        label = data.get("label", "")
        if label:
            phrase_to_nodes.setdefault(label, []).append(str(node_id))

    for phrase, node_ids in phrase_to_nodes.items():
        if len(node_ids) < 2:
            continue
        # star topology: first occurrence links to all others
        anchor = node_ids[0]
        for other in node_ids[1:]:
            lg.semantic.add_edge(anchor, other, kind="shared_concept", **_STRUCTURAL)
