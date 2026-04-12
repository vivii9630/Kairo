from __future__ import annotations

from typing import Callable, Dict, List, Optional

from kairo_core import QueryRequest, QueryResponse
from kairo_retrieval import LocalPipeline


class FlowNode:
    def __init__(
        self,
        name: str,
        run: Callable[[QueryRequest, LocalPipeline], Optional[QueryResponse]],
        next_nodes: Optional[List[str]] = None,
    ):
        self.name = name
        self.run = run
        self.next_nodes = next_nodes or []


class FlowGraph:
    """Directed graph of agent/tool steps; allows cycles by design."""

    def __init__(self, pipeline: Optional[LocalPipeline] = None):
        self.nodes: Dict[str, FlowNode] = {}
        self.pipeline = pipeline or LocalPipeline()

    def add_node(self, node: FlowNode) -> None:
        self.nodes[node.name] = node

    def run(self, start: str, request: QueryRequest, max_steps: int = 16) -> QueryResponse:
        current = start
        steps = 0
        last_response: Optional[QueryResponse] = None
        while steps < max_steps and current in self.nodes:
            node = self.nodes[current]
            resp = node.run(request, self.pipeline)
            if resp:
                last_response = resp
            if not node.next_nodes:
                break
            current = node.next_nodes[0]
            steps += 1
        if last_response is None:
            raise RuntimeError("Flow executed with no responses produced")
        return last_response


def retrieval_node(name: str = "retriever") -> FlowNode:
    def _run(req: QueryRequest, pipeline: LocalPipeline) -> QueryResponse:
        return pipeline.query(req)

    return FlowNode(name=name, run=_run)


def identity_node(name: str = "identity") -> FlowNode:
    def _run(req: QueryRequest, pipeline: LocalPipeline) -> QueryResponse:
        return pipeline.query(req)

    return FlowNode(name=name, run=_run)
