from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)
    time_hint: Optional[str] = None


class RetrievalResult(BaseModel):
    evidences: List[Evidence]
    plan: Optional[str] = None


class AgentStep(BaseModel):
    name: str
    input: str
    output: str


class QueryResponse(BaseModel):
    answer: str
    evidences: List[Evidence]
    steps: List[AgentStep] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    kind: str = "node"
    label: str = ""
    attrs: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: str = "edge"
    attrs: Dict[str, Any] = Field(default_factory=dict)


class GraphData(BaseModel):
    """Serializable wire format for a Kairo graph."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class GraphDiff(BaseModel):
    added_nodes: List[str] = Field(default_factory=list)
    removed_nodes: List[str] = Field(default_factory=list)
    changed_nodes: List[str] = Field(default_factory=list)
    added_edges: List[List[str]] = Field(default_factory=list)
    removed_edges: List[List[str]] = Field(default_factory=list)


class Snapshot(BaseModel):
    """Frozen, addressable version of a KairoGraph at a point in time.

    Branching is native: ``parents`` is a list so merges and forks are a
    schema-level concept, not a later migration.
    """

    id: str
    history_graph_id: str
    parents: List[str] = Field(default_factory=list)
    timestamp: datetime
    message: str = ""
    graph_hash: str
    builder: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoryNode(BaseModel):
    """Lightweight view of a snapshot's position in the history DAG."""

    snapshot_id: str
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    timestamp: datetime


class BranchRef(BaseModel):
    name: str
    snapshot_id: str


class TemporalQuery(BaseModel):
    query: str
    top_k: int = 5
    at: Optional[datetime] = None
    branch: Optional[str] = None
    mode: Literal["current", "rollback", "walk", "diff"] = "current"
    filters: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3-Layer graph types
# ---------------------------------------------------------------------------

LayerKind = Literal["document", "semantic", "detail"]
"""The three decomposition layers of a Kairo layered graph.

- **document**: High-level structural view (sections, headings, files).
- **semantic**: Mid-level meaning view (concepts, relationships, themes).
- **detail**: Fine-grained dense view (data points, parameters, evidence).
"""


class InterLayerEdge(BaseModel):
    """Edge connecting a node in one layer to a node in an adjacent layer.

    Kept separate from intra-layer ``GraphEdge`` so traversal can
    distinguish "going deeper" from "going sideways".
    """

    source: str
    target: str
    source_layer: LayerKind
    target_layer: LayerKind
    kind: str = "refines"
    attrs: Dict[str, Any] = Field(default_factory=dict)


class LayeredGraphData(BaseModel):
    """Serializable wire format for a 3-layer Kairo graph.

    Each layer is a standard ``GraphData`` (nodes + intra-layer edges).
    ``inter_layer_edges`` connect nodes across adjacent layers.
    """

    document: GraphData = Field(default_factory=GraphData)
    semantic: GraphData = Field(default_factory=GraphData)
    detail: GraphData = Field(default_factory=GraphData)
    inter_layer_edges: List[InterLayerEdge] = Field(default_factory=list)


class TraversalStep(BaseModel):
    """Single step in an agent's traversal through the layered graph."""

    agent_id: str
    node_id: str
    layer: LayerKind
    action: Literal["visit", "expand", "cross_layer", "cluster", "score"]
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraversalTrace(BaseModel):
    """Full record of how agents traversed the 3-layer graph for a query.

    Used to drive the 3D interactive visualization on the frontend.
    """

    query: str
    steps: List[TraversalStep] = Field(default_factory=list)
    visited_nodes: Dict[LayerKind, List[str]] = Field(default_factory=dict)
    crossed_edges: List[InterLayerEdge] = Field(default_factory=list)
    clusters: Dict[str, List[str]] = Field(default_factory=dict)
