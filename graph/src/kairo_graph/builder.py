from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from kairo_core import Document

from .graph import KairoGraph
from .layered import LayeredKairoGraph


@runtime_checkable
class GraphBuilder(Protocol):
    """Turns an iterable of content into a :class:`KairoGraph`.

    Per-content-type builders live under ``kairo_graph.builders``.
    """

    name: str

    def build(self, documents: Iterable[Document]) -> KairoGraph: ...


@runtime_checkable
class LayeredGraphBuilder(Protocol):
    """Turns content into a 3-layer :class:`LayeredKairoGraph`.

    Layer 1 (document): structural/section-level view.
    Layer 2 (semantic): concept/relationship view.
    Layer 3 (detail): fine-grained data-point view.

    Builders are responsible for populating all three layers and the
    inter-layer edges that connect them.
    """

    name: str

    def build(self, documents: Iterable[Document]) -> LayeredKairoGraph: ...
