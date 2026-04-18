"""Pydantic request/response models for the Kairo AI HTTP API.

Kept in one file while the surface is small. Split by route once any
single section grows past its own screen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from kairo_core.models import LayeredGraphData, TraversalTrace


class PluginSummary(BaseModel):
    """Flattened ``PluginManifest`` shaped for the plugin picker UI."""

    name: str
    label: str
    description: str
    uri_example: str
    icon: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    auth_required: bool = False
    auth_methods: List[str] = Field(default_factory=list)
    auth_scopes: List[str] = Field(default_factory=list)
    auth_instructions: str = ""


class Citation(BaseModel):
    source: str
    title: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


TraceKind = Literal["plan", "retrieve", "synthesize", "tool", "note"]


class TraceStep(BaseModel):
    """One entry in the agent's trace panel.

    ``snapshot_id`` points into the temporal history DAG when the step
    touched the graph — the UI renders it as a node on the timeline.
    """

    step: int
    kind: TraceKind
    summary: str
    snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    query: str
    plugin: Optional[str] = None
    thread_id: Optional[str] = None
    ingest_id: Optional[str] = None


class IngestUrlRequest(BaseModel):
    source_url: str


class IngestResponse(BaseModel):
    """Summary of a completed ingest.

    ``source`` is the raw input (URL or filename); ``label`` is the
    short chip string the UI renders (``"owner/repo"`` for GitHub,
    filename for uploads).
    """

    ingest_id: str
    source: str
    label: str
    doc_count: int
    kind: Literal["github", "csv", "xlsx"]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class AskResponse(BaseModel):
    thread_id: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    trace: List[TraceStep] = Field(default_factory=list)
    stubbed: bool = True
    graph_data: Optional[LayeredGraphData] = None
    traversal_trace: Optional[TraversalTrace] = None


class Thread(BaseModel):
    id: str
    title: str
    created_at: datetime
    plugin: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)


class ThreadCreateRequest(BaseModel):
    title: Optional[str] = None
    plugin: Optional[str] = None
