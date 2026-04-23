from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from kairo_core import Document

from ..graph import KairoGraph
from ..provenance import PROVENANCE_STRUCTURAL, edge_attrs


_STRUCTURAL = edge_attrs(provenance=PROVENANCE_STRUCTURAL, confidence=1.0)

# Rationale comments: free-form "# WHY: ..." / "# RATIONALE: ..." /
# "# BECAUSE: ..." lines that the author left to explain intent. These
# are stripped by ``ast`` so we scan the raw source separately and
# attach each to the nearest enclosing def/class by line range.
_RATIONALE_RE = re.compile(
    r"^\s*#\s*(?:WHY|RATIONALE|BECAUSE|NOTE)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class CodeGraphBuilder:
    """Builds a code-structure graph from Python source documents.

    Each ``Document.text`` is parsed with :mod:`ast`; the builder extracts
    modules, top-level functions, classes, their methods, import targets,
    intra-module call edges, plus — as of 13b4 — docstrings and rationale
    comments as first-class nodes. Non-Python or syntactically invalid
    documents are skipped rather than raising.

    Node kinds:
      - ``module``   — one per Python document, id = ``Document.id``
      - ``function`` — free functions and methods
      - ``class``    — class definitions
      - ``external_module`` — import targets not present in the input set
      - ``docstring`` — the docstring attached to a module/class/function
      - ``rationale`` — a ``# WHY:`` / ``# RATIONALE:`` / ``# BECAUSE:``
        / ``# NOTE:`` comment, attached to the nearest enclosing def

    Edge kinds:
      - ``contains``  module → function / class, class → method
      - ``imports``   module → external_module
      - ``calls``     function → function (only when callee resolves
                      statically within the same module)
      - ``documents`` node → docstring
      - ``explains``  node → rationale
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


def _attach_docstring(graph: KairoGraph, owner_id: str, docstring: Optional[str]) -> None:
    if not docstring:
        return
    dstr = docstring.strip()
    if not dstr:
        return
    doc_node_id = f"{owner_id}::__doc__"
    graph.add_node(
        doc_node_id,
        kind="docstring",
        label=dstr[:80],
        text=dstr,
    )
    graph.add_edge(owner_id, doc_node_id, kind="documents", **_STRUCTURAL)


def _extract_rationales(source: str) -> List[Tuple[int, str]]:
    """Return ``(lineno, text)`` tuples for every rationale comment.

    ``lineno`` is 1-indexed to match ast.AST.lineno conventions so a
    later pass can match comments to the nearest enclosing def/class
    by line range.
    """
    rationales: List[Tuple[int, str]] = []
    for match in _RATIONALE_RE.finditer(source):
        line_no = source[: match.start()].count("\n") + 1
        rationales.append((line_no, match.group(1).strip()))
    return rationales


def _stmt_range(stmt: ast.AST) -> Tuple[int, int]:
    start = getattr(stmt, "lineno", 0)
    end = getattr(stmt, "end_lineno", start)
    return start, end or start


def _attach_rationales(
    graph: KairoGraph,
    owner_id: str,
    rationales: List[Tuple[int, str]],
    *,
    line_start: int,
    line_end: int,
    available: Set[int],
) -> List[int]:
    """Attach rationales whose line falls in [line_start, line_end] and
    whose index is still in *available*. Returns the indices consumed.
    """
    consumed: List[int] = []
    for idx, (line, text) in enumerate(rationales):
        if idx not in available:
            continue
        if line_start <= line <= line_end:
            rid = f"{owner_id}::__why__::{idx}"
            graph.add_node(
                rid,
                kind="rationale",
                label=text[:80],
                text=text,
                line=line,
            )
            graph.add_edge(owner_id, rid, kind="explains", **_STRUCTURAL)
            consumed.append(idx)
    return consumed


def _emit_module(graph: KairoGraph, doc: Document, tree: ast.Module, module_ids: Set[str]) -> None:
    module_id = doc.id
    graph.add_node(
        module_id,
        kind="module",
        label=module_id,
        path=doc.metadata.get("path", ""),
    )
    _attach_docstring(graph, module_id, ast.get_docstring(tree))

    all_rationales = _extract_rationales(doc.text)
    available = set(range(len(all_rationales)))

    defined_functions: Dict[str, str] = {}  # bare_name -> node_id

    # Pass 1 — attach rationales to the innermost owning def/class.
    # Methods first (most specific), then classes, then top-level defs.
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            fn_id = f"{module_id}::{stmt.name}"
            graph.add_node(fn_id, kind="function", label=stmt.name)
            graph.add_edge(module_id, fn_id, kind="contains", **_STRUCTURAL)
            _attach_docstring(graph, fn_id, ast.get_docstring(stmt))
            defined_functions[stmt.name] = fn_id
            start, end = _stmt_range(stmt)
            for i in _attach_rationales(graph, fn_id, all_rationales,
                                        line_start=start, line_end=end,
                                        available=available):
                available.discard(i)

        elif isinstance(stmt, ast.ClassDef):
            cls_id = f"{module_id}::{stmt.name}"
            graph.add_node(cls_id, kind="class", label=stmt.name)
            graph.add_edge(module_id, cls_id, kind="contains", **_STRUCTURAL)
            _attach_docstring(graph, cls_id, ast.get_docstring(stmt))
            cls_start, cls_end = _stmt_range(stmt)

            for inner in stmt.body:
                if isinstance(inner, ast.FunctionDef):
                    method_id = f"{cls_id}.{inner.name}"
                    graph.add_node(method_id, kind="function", label=inner.name)
                    graph.add_edge(cls_id, method_id, kind="contains", **_STRUCTURAL)
                    _attach_docstring(graph, method_id, ast.get_docstring(inner))
                    defined_functions[f"{stmt.name}.{inner.name}"] = method_id
                    defined_functions.setdefault(inner.name, method_id)
                    m_start, m_end = _stmt_range(inner)
                    for i in _attach_rationales(graph, method_id, all_rationales,
                                                line_start=m_start, line_end=m_end,
                                                available=available):
                        available.discard(i)

            # Class-body rationales outside any method (idx still in available)
            for i in _attach_rationales(graph, cls_id, all_rationales,
                                        line_start=cls_start, line_end=cls_end,
                                        available=available):
                available.discard(i)

        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for target in _import_targets(stmt):
                if target in module_ids:
                    graph.add_edge(module_id, target, kind="imports", **_STRUCTURAL)
                else:
                    ext_id = f"ext::{target}"
                    if ext_id not in graph.nx_graph:
                        graph.add_node(ext_id, kind="external_module", label=target)
                    graph.add_edge(module_id, ext_id, kind="imports", **_STRUCTURAL)

    # Module-level rationales — anything that fell outside all defs/classes.
    module_lines = doc.text.count("\n") + 1
    for i in _attach_rationales(graph, module_id, all_rationales,
                                line_start=1, line_end=module_lines,
                                available=available):
        available.discard(i)

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
