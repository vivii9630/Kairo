"""Phase 13c — CitationVerifier token-overlap strategy."""

from __future__ import annotations

from kairo_core import Evidence
from kairo_rag import CitationVerifier, split_sentences


def _ev(doc_id: str, text: str) -> Evidence:
    return Evidence(document_id=doc_id, text=text, score=1.0, metadata={})


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_handles_periods_and_newlines() -> None:
    text = "First sentence. Second sentence! Third?\nFourth on a new line."
    parts = split_sentences(text)
    assert parts == [
        "First sentence.",
        "Second sentence!",
        "Third?",
        "Fourth on a new line.",
    ]


def test_split_sentences_empty_input_returns_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


# ---------------------------------------------------------------------------
# CitationVerifier
# ---------------------------------------------------------------------------

def test_claim_fully_supported_by_evidence_is_marked_supported() -> None:
    evidences = [_ev("e1", "Louvain and Leiden are community detection algorithms.")]
    answer = "Louvain is a community detection algorithm."
    report = CitationVerifier(claim_threshold=0.4).verify(answer, evidences)
    assert len(report.claims) == 1
    claim = report.claims[0]
    assert claim.supported is True
    assert claim.support_score >= 0.4
    assert claim.best_evidence_id == "e1"
    assert report.overall_support == 1.0
    assert report.answer_downgraded is False


def test_unsupported_claim_is_flagged_and_triggers_downgrade() -> None:
    evidences = [_ev("e1", "Cats purr when content.")]
    answer = "Python has list comprehensions and generator expressions."
    report = CitationVerifier(
        claim_threshold=0.4, downgrade_threshold=0.6
    ).verify(answer, evidences)
    assert len(report.claims) == 1
    assert report.claims[0].supported is False
    assert report.answer_downgraded is True
    assert report.overall_support == 0.0


def test_mixed_answer_produces_partial_support_score() -> None:
    evidences = [
        _ev("e1", "Leiden community detection improves over Louvain."),
        _ev("e2", "NetworkX provides graph algorithms in Python."),
    ]
    answer = (
        "Leiden improves over Louvain for community detection. "
        "Basketball requires five players per team."
    )
    report = CitationVerifier(claim_threshold=0.4).verify(answer, evidences)
    assert len(report.claims) == 2
    supported = [c for c in report.claims if c.supported]
    assert len(supported) == 1
    assert supported[0].best_evidence_id == "e1"
    assert 0.0 < report.overall_support < 1.0


def test_trivial_sentences_are_auto_supported() -> None:
    """Short connective sentences like 'Here is the summary:' shouldn't
    count against the support score — they carry no factual claim."""
    evidences = [_ev("e1", "Alpha beta gamma delta epsilon zeta eta theta.")]
    answer = "Here is the summary:"
    report = CitationVerifier(min_content_tokens=3).verify(answer, evidences)
    assert len(report.claims) == 1
    assert report.claims[0].supported is True
    assert report.claims[0].support_score == 1.0


def test_empty_answer_produces_empty_supported_report() -> None:
    report = CitationVerifier().verify("", [_ev("e1", "anything")])
    assert report.claims == []
    assert report.overall_support == 1.0
    assert report.answer_downgraded is False


def test_stopwords_do_not_carry_support() -> None:
    """A sentence made entirely of stopwords should end up as a trivial
    (auto-supported) claim, not a spurious high-overlap match against
    unrelated evidence."""
    evidences = [_ev("e1", "The cat is on the mat.")]
    answer = "The is of and or but."  # all stopwords
    report = CitationVerifier(min_content_tokens=3).verify(answer, evidences)
    assert report.claims[0].supported is True
    # ... because it was trivial, not because the overlap was high.
    assert report.claims[0].best_evidence_id is None


def test_threshold_tuning_changes_support_verdict() -> None:
    evidences = [_ev("e1", "One shared term: authentication.")]
    answer = "Authentication flows are tricky to implement correctly."
    loose = CitationVerifier(claim_threshold=0.1).verify(answer, evidences)
    strict = CitationVerifier(claim_threshold=0.9).verify(answer, evidences)
    assert loose.claims[0].supported is True
    assert strict.claims[0].supported is False


def test_report_records_strategy_name() -> None:
    report = CitationVerifier().verify("Some claim.", [_ev("e", "unrelated")])
    assert report.strategy == "token_overlap"
