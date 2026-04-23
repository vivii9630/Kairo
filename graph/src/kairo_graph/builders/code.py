from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Optional, Set

from kairo_core import Document

from ..graph import KairoGraph
from ..provenance import PROVENANCE_STRUCTURAL, edge_attrs


_STRUCTURAL = edge_attrs(provenance=PROVENANCE_STRUCTURAL, confidence=1.0)


class CodeGraphBuilder:
    """Builds a code-structure graph from Python source documents.

    Each ``Document.text`` is parsed with :mod:`ast`; the builder extracts
    modules, top-level functions, classes, their methods, import targets,
    and intra-module call edges. Non-Python or syntactically invalid
    documents are skipped rather than raising — ingestion pipelines often
    include README text alongside source, and a single broken file should
    not fail the whole build.

    Node kinds:
      - ``module``   — one per Python document, id = ``Document.id``
      - ``function`` — free functions and methods
      - ``class``    — class definitions
      - ``external_module`` — import targets not present in the input set

    Edge kinds:
      - ``contains``  module → function / class, class → method
      - ``imports``   module → external_module
      - ``calls``     function → function (only when callee resolves
                      statically within the same module)
    """

    name = "code"

    def build(self, documents: Iterable[Document]) -> KairoGraph:
        graph = KairoGraph()
        docs = list(documents)
        module_ids = {doc.id for doc in docs}

        for doc in docs:
            try:
                tree = ast.parse(doc.text, filename=doc.id)
            except SyntaxError:
                continue
            _emit_module(graph, doc, tree, module_ids)
        return graph


def _emit_module(graph: KairoGraph, doc: Document, tree: ast.Module, module_ids: Set[str]) -> None:
    module_id = doc.id
    graph.add_node(
        module_id,
        kind="module",
        label=module_id,
        path=doc.metadata.get("path", ""),
    )

    defined_functions: Dict[str, str] = {}  # bare_name -> node_id

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            fn_id = f"{module_id}::{stmt.name}"
            graph.add_node(fn_id, kind="function", label=stmt.name)
            graph.add_edge(module_id, fn_id, kind="contains", **_STRUCTURAL)
            defined_functions[stmt.name] = fn_id

        elif isinstance(stmt, ast.ClassDef):
            cls_id = f"{module_id}::{stmt.name}"
            graph.add_node(cls_id, kind="class", label=stmt.name)
            graph.add_edge(module_id, cls_id, kind="contains", **_STRUCTURAL)
            for inner in stmt.body:
                if isinstance(inner, ast.FunctionDef):
                    method_id = f"{cls_id}.{inner.name}"
                    graph.add_node(method_id, kind="function", label=inner.name)
                    graph.add_edge(cls_id, method_id, kind="contains", **_STRUCTURAL)
                    defined_functions[f"{stmt.name}.{inner.name}"] = method_id
                    defined_functions.setdefault(inner.name, method_id)

        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for target in _import_targets(stmt):
                if target in module_ids:
                    graph.add_edge(module_id, target, kind="imports", **_STRUCTURAL)
                else:
                    ext_id = f"ext::{target}"
                    if ext_id not in graph.nx_graph:
                        graph.add_node(ext_id, kind="external_module", label=target)
                    graph.add_edge(module_id, ext_id, kind="imports", **_STRUCTURAL)

    # Second pass: resolve intra-module call edges.
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            caller_id = defined_functions[stmt.name]
            for callee in _call_names(stmt):
                target_id = defined_functions.get(callee)
                if target_id is not None and target_id != caller_id:
                    graph.add_edge(caller_id, target_id, kind="calls", **_STRUCTURAL)
        elif isinstance(stmt, ast.ClassDef):
            for inner in stmt.body:
                if isinstance(inner, ast.FunctionDef):
                    caller_id = defined_functions[f"{stmt.name}.{inner.name}"]
                    for callee in _call_names(inner):
                        target_id = defined_functions.get(callee)
                        if target_id is not None and target_id != caller_id:
                            graph.add_edge(caller_id, target_id, kind="calls", **_STRUCTURAL)


def _import_targets(stmt: ast.AST) -> List[str]:
    if isinstance(stmt, ast.Import):
        return [alias.name.split(".")[0] for alias in stmt.names]
    if isinstance(stmt, ast.ImportFrom) and stmt.module:
        return [stmt.module.split(".")[0]]
    return []


def _call_names(fn: ast.FunctionDef) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call(node.func)
        if resolved is not None:
            names.add(resolved)
    return names


def _resolve_call(node: ast.AST) -> Optional[str]:
    """Best-effort static resolution of a call target to a bare name.

    Handles ``foo()`` and ``self.foo()`` / ``Class.foo()``; returns None
    for dynamic or deeply-nested expressions so we don't invent false edges.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            return node.attr
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
    return None
