from kairo_core import QueryRequest
from kairo_flow import FlowGraph, retrieval_node


def test_flow_retriever_smoke():
    graph = FlowGraph()
    graph.add_node(retrieval_node("retriever"))
    resp = graph.run(start="retriever", request=QueryRequest(query="Hybrid retrieval", top_k=3))
    assert resp.answer is not None
    assert isinstance(resp.answer, str)
