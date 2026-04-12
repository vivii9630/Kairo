from __future__ import annotations

from typing import Any, Dict, List, Optional

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
