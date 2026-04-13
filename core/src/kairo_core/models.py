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
