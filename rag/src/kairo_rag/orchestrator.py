from __future__ import annotations

from typing import Optional

from kairo_core import QueryRequest, QueryResponse
from kairo_retrieval import LocalPipeline


class MultiAgentOrchestrator:
    """Small orchestration façade for multi-step retrieval + answer.

    This is a minimal RAG pattern; richer patterns (hybrid, temporal, tool-using
    agents with LLM think_fns) can be added alongside it in this package.
    """

    def __init__(self, pipeline: Optional[LocalPipeline] = None):
        self.pipeline = pipeline or LocalPipeline()

    def run(self, query: str, top_k: int = 5) -> QueryResponse:
        request = QueryRequest(query=query, top_k=top_k)
        return self.pipeline.query(request)
