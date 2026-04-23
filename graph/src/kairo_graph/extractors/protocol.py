"""The ``ExtractorProvider`` protocol.

Any object that exposes ``.name`` and an ``extract(document, context)``
returning an :class:`~kairo_core.ExtractionResult` plugs into
:class:`LLMExtractedGraphBuilder`. This is the seam that keeps the
builder agnostic to whether extraction runs locally via Ollama, against
Claude via Anthropic SDK, against GPT-4 via OpenAI, etc.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from kairo_core import Document, ExtractionResult


@runtime_checkable
class ExtractorProvider(Protocol):
    """Contract every extractor implements.

    Attributes
    ----------
    name : str
        Short identifier recorded on every produced edge / result so
        retrieval and audits can filter by which extractor produced an
        edge. Examples: ``"ollama:llama3.2"``, ``"claude:sonnet-4.6"``.

    Methods
    -------
    extract(document, context=()) -> ExtractionResult
        Run the backend over *document*. ``context`` optionally carries
        sibling documents the extractor should be aware of (e.g. already
        detected concept summaries from community detection, or prior
        docs in the corpus for reference resolution). Returning an
        :class:`ExtractionResult` with ``error`` set should be preferred
        over raising — the builder logs and continues so one bad doc
        doesn't fail the whole batch.
    """

    name: str

    def extract(
        self,
        document: Document,
        context: Sequence[Document] = (),
    ) -> ExtractionResult:
        ...
