"""Phase 13b4 — docstring and rationale-comment extraction on CodeGraphBuilder."""

from __future__ import annotations

from kairo_core import Document
from kairo_graph.builders.code import CodeGraphBuilder, _extract_rationales


# ---------------------------------------------------------------------------
# _extract_rationales unit tests
# ---------------------------------------------------------------------------

def test_extract_rationales_picks_up_marker_variants() -> None:
    src = (
        "x = 1\n"
        "# WHY: zero is not a safe default here\n"
        "y = 2\n"
        "# rationale: the algorithm requires a seed\n"
        "z = 3\n"
        "# BECAUSE: legacy callers pass strings\n"
        "# note: only runs on Windows\n"
    )
    found = _extract_rationales(src)
    texts = [t for _, t in found]
    assert texts == [
        "zero is not a safe default here",
        "the algorithm requires a seed",
        "legacy callers pass strings",
        "only runs on Windows",
    ]
    # Lines are 1-indexed.
    lines = [ln for ln, _ in found]
    assert lines == [2, 4, 6, 7]


def test_extract_rationales_ignores_plain_comments() -> None:
    src = "# just a comment\n# TODO: not a rationale\n"
    assert _extract_rationales(src) == []


# ---------------------------------------------------------------------------
# Docstring attachment
# ---------------------------------------------------------------------------

def test_module_docstring_becomes_node() -> None:
    src = '"""Module-level description of foo."""\n\ndef bar(): pass\n'
    graph = CodeGraphBuilder().build(
        [Document(id="foo.py", text=src, metadata={"path": "foo.py"})]
    )
    doc_nodes = [
        (nid, d) for nid, d in graph.nx_graph.nodes(data=True)
        if d.get("kind") == "docstring"
    ]
    assert len(doc_nodes) == 1
    nid, data = doc_nodes[0]
    assert nid == "foo.py::__doc__"
    assert "Module-level description of foo" in data["text"]

    # Linked to the module via 'documents' edge.
    edges = [
        (u, v, d) for u, v, d in graph.nx_graph.edges("foo.py", data=True)
        if d.get("kind") == "documents"
    ]
    assert len(edges) == 1


def test_function_and_class_docstrings_attach_to_own_nodes() -> None:
    src = (
        "def foo():\n"
        '    """foo explains itself."""\n'
        "    return 1\n"
        "\n"
        "class Bar:\n"
        '    """Bar is a class."""\n'
        "    def m(self):\n"
        '        """m is a method."""\n'
        "        return 2\n"
    )
    graph = CodeGraphBuilder().build([Document(id="m.py", text=src)])
    labels_by_owner = {}
    for u, v, data in graph.nx_graph.edges(data=True):
        if data.get("kind") == "documents":
            labels_by_owner[u] = graph.nx_graph.nodes[v]["text"]

    assert "foo explains itself" in labels_by_owner["m.py::foo"]
    assert "Bar is a class" in labels_by_owner["m.py::Bar"]
    assert "m is a method" in labels_by_owner["m.py::Bar.m"]


def test_missing_docstring_does_not_create_empty_node() -> None:
    src = "def foo():\n    return 1\n"
    graph = CodeGraphBuilder().build([Document(id="m.py", text=src)])
    assert not any(
        d.get("kind") == "docstring"
        for _, d in graph.nx_graph.nodes(data=True)
    )


# ---------------------------------------------------------------------------
# Rationale attachment
# ---------------------------------------------------------------------------

def test_rationale_inside_function_attaches_to_function() -> None:
    src = (
        "def foo():\n"
        "    # WHY: early-return avoids the expensive branch\n"
        "    return 1\n"
    )
    graph = CodeGraphBuilder().build([Document(id="m.py", text=src)])
    rationale_edges = [
        (u, v, d) for u, v, d in graph.nx_graph.edges(data=True)
        if d.get("kind") == "explains"
    ]
    assert len(rationale_edges) == 1
    u, v, data = rationale_edges[0]
    assert u == "m.py::foo"
    assert data["provenance"] == "structural"
    assert "early-return" in graph.nx_graph.nodes[v]["text"]


def test_rationale_nested_in_method_does_not_double_attach() -> None:
    """A rationale line that falls inside both a class and a method
    range should attach only to the innermost owner (the method)."""
    src = (
        "class K:\n"
        "    def m(self):\n"
        "        # RATIONALE: locals are faster than attribute lookups\n"
        "        x = self.y\n"
    )
    graph = CodeGraphBuilder().build([Document(id="m.py", text=src)])
    owners = [
        u for u, _v, d in graph.nx_graph.edges(data=True)
        if d.get("kind") == "explains"
    ]
    assert owners == ["m.py::K.m"]


def test_rationale_at_module_level_attaches_to_module() -> None:
    src = (
        "# WHY: we import this eagerly to trigger the side effect\n"
        "import json\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
    )
    graph = CodeGraphBuilder().build([Document(id="m.py", text=src)])
    owners = [
        u for u, _v, d in graph.nx_graph.edges(data=True)
        if d.get("kind") == "explains"
    ]
    assert owners == ["m.py"]


def test_every_docstring_and_rationale_edge_is_structural() -> None:
    src = (
        '"""Module doc."""\n'
        "\n"
        "def foo():\n"
        '    """foo doc."""\n'
        "    # WHY: x\n"
        "    return 1\n"
    )
    graph = CodeGraphBuilder().build([Document(id="m.py", text=src)])
    for _u, _v, data in graph.nx_graph.edges(data=True):
        assert data.get("provenance") == "structural"
        assert data.get("confidence") == 1.0
