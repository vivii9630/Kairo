from __future__ import annotations

from typing import Optional

from .local import LocalPipeline
from .models import QueryRequest, QueryResponse


class MultiAgentOrchestrator:
    """Very small orchestration façade for multi-step retrieval + answer."""

    def __init__(self, pipeline: Optional[LocalPipeline] = None):
        self.pipeline = pipeline or LocalPipeline()

    def run(self, query: str, top_k: int = 5) -> QueryResponse:
        request = QueryRequest(query=query, top_k=top_k)
        return self.pipeline.query(request)

