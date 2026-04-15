from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List, Optional

from kairo_core.models import Snapshot
from kairo_graph.graph import KairoGraph


def snapshot_id_from_hash(history_graph_id: str, graph_hash: str, timestamp: datetime) -> str:
    """Deterministic snapshot id from (history_graph_id, graph_hash, timestamp).

    Including the timestamp means two byte-identical graphs snapshotted at
    different moments still get distinct ids — the history DAG is about
    *when* a state was observed, not only *what* state it was.
    """
    payload = f"{history_graph_id}|{graph_hash}|{timestamp.isoformat()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class SnapshotBuilder:
    """Build a ``Snapshot`` metadata record from a live KairoGraph.

    Does not persist anything — persistence is ``KairoStore``'s job. Keeping
    these separate lets callers build a snapshot in memory and decide later
    whether to write it to disk.
    """

    def __init__(self, *, history_graph_id: str, builder_name: str = ""):
        self.history_graph_id = history_graph_id
        self.builder_name = builder_name

    def build(
        self,
        graph: KairoGraph,
        *,
        parents: Optional[List[str]] = None,
        message: str = "",
        timestamp: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> Snapshot:
        ts = timestamp or datetime.now(timezone.utc)
        graph_hash = graph.hash()
        sid = snapshot_id_from_hash(self.history_graph_id, graph_hash, ts)
        return Snapshot(
            id=sid,
            history_graph_id=self.history_graph_id,
            parents=list(parents or []),
            timestamp=ts,
            message=message,
            graph_hash=graph_hash,
            builder=self.builder_name,
            metadata=dict(metadata or {}),
        )
