"""Kairo graph: construction primitives and per-content-type builders."""

from .builder import GraphBuilder, LayeredGraphBuilder
from .builders.document import DocumentGraphBuilder
from .builders.layered_document import DocumentLayeredBuilder
from .communities import (
    add_concept_nodes,
    detect_communities,
    leiden_available,
    summarize_community,
)
from .graph import KairoGraph
from .layered import LayeredKairoGraph
from .provenance import (
    PROVENANCE_AMBIGUOUS,
    PROVENANCE_EXTRACTED,
    PROVENANCE_INFERRED,
    PROVENANCE_STRUCTURAL,
    PROVENANCE_VALUES,
    edge_attrs,
)
from .store import load_graph, save_graph

__all__ = [
    "GraphBuilder",
    "LayeredGraphBuilder",
    "KairoGraph",
    "LayeredKairoGraph",
    "DocumentGraphBuilder",
    "DocumentLayeredBuilder",
    "detect_communities",
    "add_concept_nodes",
    "summarize_community",
    "leiden_available",
    "edge_attrs",
    "PROVENANCE_STRUCTURAL",
    "PROVENANCE_EXTRACTED",
    "PROVENANCE_INFERRED",
    "PROVENANCE_AMBIGUOUS",
    "PROVENANCE_VALUES",
    "save_graph",
    "load_graph",
]
