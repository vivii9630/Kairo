from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from kairo_core import GraphData

from .graph import KairoGraph


def save_graph(graph: KairoGraph, path: Union[Path, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.to_data().model_dump()
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path: Union[Path, str]) -> KairoGraph:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    data = GraphData.model_validate(raw)
    return KairoGraph.from_data(data)
