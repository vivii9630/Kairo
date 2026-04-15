from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

from kairo_core.models import HistoryNode, Snapshot


class HistoryGraph:
    """Branching DAG of snapshots for a single ``history_graph_id``.

    Each node is a ``HistoryNode`` (snapshot_id + parent/child links + timestamp).
    The DAG is explicit rather than derived so that merges (multiple parents)
    and forks (multiple children) are first-class and free to traverse in
    either direction.
    """

    def __init__(self, history_graph_id: str):
        self.history_graph_id = history_graph_id
        self._nodes: Dict[str, HistoryNode] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, snapshot_id: str) -> bool:
        return snapshot_id in self._nodes

    def add(self, snapshot: Snapshot) -> HistoryNode:
        if snapshot.history_graph_id != self.history_graph_id:
            raise ValueError(
                f"snapshot belongs to {snapshot.history_graph_id!r}, "
                f"not {self.history_graph_id!r}"
            )
        if snapshot.id in self._nodes:
            return self._nodes[snapshot.id]

        for parent_id in snapshot.parents:
            if parent_id not in self._nodes:
                raise KeyError(
                    f"parent {parent_id!r} not found in history "
                    f"{self.history_graph_id!r}"
                )

        node = HistoryNode(
            snapshot_id=snapshot.id,
            parents=list(snapshot.parents),
            children=[],
            timestamp=snapshot.timestamp,
        )
        self._nodes[snapshot.id] = node
        for parent_id in snapshot.parents:
            parent = self._nodes[parent_id]
            if snapshot.id not in parent.children:
                parent.children.append(snapshot.id)
        return node

    def get(self, snapshot_id: str) -> HistoryNode:
        return self._nodes[snapshot_id]

    def nodes(self) -> List[HistoryNode]:
        return list(self._nodes.values())

    def roots(self) -> List[str]:
        return [sid for sid, n in self._nodes.items() if not n.parents]

    def leaves(self) -> List[str]:
        return [sid for sid, n in self._nodes.items() if not n.children]

    def latest(self) -> Optional[str]:
        """Snapshot id with the newest timestamp, or None if empty."""
        if not self._nodes:
            return None
        return max(self._nodes.values(), key=lambda n: n.timestamp).snapshot_id

    def to_dict(self) -> dict:
        return {
            "history_graph_id": self.history_graph_id,
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryGraph":
        hg = cls(data["history_graph_id"])
        for raw in data.get("nodes", []):
            node = HistoryNode(**raw)
            hg._nodes[node.snapshot_id] = node
        return hg

    def iter_snapshot_ids(self) -> Iterable[str]:
        return iter(self._nodes.keys())
