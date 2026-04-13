from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from kairo_core import Document

from .graph import KairoGraph


@runtime_checkable
class GraphBuilder(Protocol):
    """Turns an iterable of content into a :class:`KairoGraph`.

    Per-content-type builders live under ``kairo_graph.builders``.
    """

    name: str

    def build(self, documents: Iterable[Document]) -> KairoGraph: ...
