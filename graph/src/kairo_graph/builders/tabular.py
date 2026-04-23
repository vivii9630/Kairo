from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from kairo_core import Document

from ..graph import KairoGraph
from ..provenance import PROVENANCE_STRUCTURAL, edge_attrs


_STRUCTURAL = edge_attrs(provenance=PROVENANCE_STRUCTURAL, confidence=1.0)


class TabularGraphBuilder:
    """Builds a graph from tabular row documents (CSV / Excel rows).

    Input shape matches :func:`kairo_ingest.loaders.ingest_csv`: one
    ``Document`` per row, ``doc.text`` is a JSON object of the row's
    cells, and ``doc.metadata["path"]`` (or ``"sheet"``) identifies the
    table the row belongs to. Rows without a recognizable table key are
    grouped under ``"<unknown-table>"``.

    Node kinds:
      - ``table``   — one per distinct source file/sheet
      - ``column``  — one per unique column in a table
      - ``row``     — one per input document; cell values live in ``attrs``

    Edge kinds:
      - ``has_column``  table → column
      - ``has_row``     table → row

    We intentionally do not emit a ``row → column`` edge per cell — that
    would quadruple the node count without adding traversal value, since
    the row's cell values are already on the row node.
    """

    name = "tabular"

    def build(self, documents: Iterable[Document]) -> KairoGraph:
        graph = KairoGraph()
        grouped: Dict[str, List[Document]] = defaultdict(list)
        row_payloads: Dict[str, dict] = {}

        for doc in documents:
            table_id = _table_id(doc)
            try:
                row = json.loads(doc.text) if doc.text else {}
            except json.JSONDecodeError:
                row = {"_raw": doc.text}
            if not isinstance(row, dict):
                row = {"_raw": row}
            grouped[table_id].append(doc)
            row_payloads[doc.id] = row

        for table_id, docs in grouped.items():
            graph.add_node(table_id, kind="table", label=table_id)

            columns: Set[str] = set()
            for doc in docs:
                columns.update(row_payloads[doc.id].keys())
            for col in sorted(columns):
                col_id = f"{table_id}::col::{col}"
                graph.add_node(col_id, kind="column", label=col)
                graph.add_edge(table_id, col_id, kind="has_column", **_STRUCTURAL)

            for doc in docs:
                row = row_payloads[doc.id]
                cell_attrs = {f"cell_{k}": v for k, v in row.items()}
                graph.add_node(
                    doc.id,
                    kind="row",
                    label=doc.id,
                    **cell_attrs,
                )
                graph.add_edge(table_id, doc.id, kind="has_row", **_STRUCTURAL)

        return graph


def _table_id(doc: Document) -> str:
    sheet = doc.metadata.get("sheet")
    path = doc.metadata.get("path")
    if path:
        name = Path(path).name
        return f"{name}::{sheet}" if sheet else name
    return "<unknown-table>"
