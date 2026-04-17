from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from kairo_core import (
    GraphData,
    InterLayerEdge,
    LayeredGraphData,
)

from .graph import KairoGraph

# Canonical layer order — used for iteration and hashing.
LAYER_ORDER = ("document", "semantic", "detail")


class LayeredKairoGraph:
    """Composite 3-layer graph with inter-layer edges.

    Wraps three :class:`KairoGraph` instances (one per layer) and maintains
    a separate list of inter-layer edges that link nodes across adjacent
    layers.  The whole structure snapshots as a single unit — one snapshot
    = all 3 layers frozen at time *t*.
    """

    def __init__(
        self,
        *,
        document: Optional[KairoGraph] = None,
        semantic: Optional[KairoGraph] = None,
        detail: Optional[KairoGraph] = None,
        inter_layer_edges: Optional[List[InterLayerEdge]] = None,
    ) -> None:
        self._layers: Dict[str, KairoGraph] = {
            "document": document or KairoGraph(),
            "semantic": semantic or KairoGraph(),
            "detail": detail or KairoGraph(),
        }
        self._inter: List[InterLayerEdge] = list(inter_layer_edges or [])

    # -- layer access -------------------------------------------------------

    @property
    def document(self) -> KairoGraph:
        return self._layers["document"]

    @property
    def semantic(self) -> KairoGraph:
        return self._layers["semantic"]

    @property
    def detail(self) -> KairoGraph:
        return self._layers["detail"]

    def layer(self, name: str) -> KairoGraph:
        if name not in self._layers:
            raise KeyError(f"Unknown layer {name!r}; expected one of {LAYER_ORDER}")
        return self._layers[name]

    @property
    def inter_layer_edges(self) -> List[InterLayerEdge]:
        return self._inter

    # -- mutation -----------------------------------------------------------

    def add_inter_edge(
        self,
        source: str,
        target: str,
        source_layer: str,
        target_layer: str,
        *,
        kind: str = "refines",
        **attrs: Any,
    ) -> None:
        self._inter.append(
            InterLayerEdge(
                source=source,
                target=target,
                source_layer=source_layer,  # type: ignore[arg-type]
                target_layer=target_layer,  # type: ignore[arg-type]
                kind=kind,
                attrs=attrs,
            )
        )

    # -- stats --------------------------------------------------------------

    def num_nodes(self, layer: Optional[str] = None) -> int:
        if layer:
            return self.layer(layer).num_nodes()
        return sum(g.num_nodes() for g in self._layers.values())

    def num_edges(self, layer: Optional[str] = None) -> int:
        if layer:
            return self.layer(layer).num_edges()
        intra = sum(g.num_edges() for g in self._layers.values())
        return intra + len(self._inter)

    # -- neighbors across layers --------------------------------------------

    def inter_neighbors(
        self, node_id: str, *, direction: str = "down"
    ) -> List[InterLayerEdge]:
        """Return inter-layer edges touching *node_id*.

        ``direction="down"`` returns edges where *node_id* is the source
        (going from document→semantic or semantic→detail).
        ``direction="up"`` returns edges where *node_id* is the target.
        ``direction="both"`` returns all.
        """
        result: List[InterLayerEdge] = []
        for e in self._inter:
            if direction in ("down", "both") and e.source == node_id:
                result.append(e)
            if direction in ("up", "both") and e.target == node_id:
                result.append(e)
        return result

    # -- serialization ------------------------------------------------------

    def to_data(self) -> LayeredGraphData:
        return LayeredGraphData(
            document=self._layers["document"].to_data(),
            semantic=self._layers["semantic"].to_data(),
            detail=self._layers["detail"].to_data(),
            inter_layer_edges=list(self._inter),
        )

    @classmethod
    def from_data(cls, data: LayeredGraphData) -> "LayeredKairoGraph":
        return cls(
            document=KairoGraph.from_data(data.document),
            semantic=KairoGraph.from_data(data.semantic),
            detail=KairoGraph.from_data(data.detail),
            inter_layer_edges=list(data.inter_layer_edges),
        )

    def hash(self) -> str:
        """SHA-256 over canonical serialization of all 3 layers + inter edges."""
        data = self.to_data()
        payload = {
            layer_name: {
                "nodes": sorted(
                    (n.model_dump() for n in getattr(data, layer_name).nodes),
                    key=lambda n: n["id"],
                ),
                "edges": sorted(
                    (e.model_dump() for e in getattr(data, layer_name).edges),
                    key=lambda e: (e["source"], e["target"], e["kind"]),
                ),
            }
            for layer_name in LAYER_ORDER
        }
        payload["inter_layer_edges"] = sorted(
            (e.model_dump() for e in data.inter_layer_edges),
            key=lambda e: (e["source_layer"], e["source"], e["target"]),
        )
        blob = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
