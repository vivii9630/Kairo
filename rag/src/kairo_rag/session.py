"""In-memory session store for temporal graph snapshots.

Each session (typically one user conversation) accumulates snapshots of the
3-layer graph + embeddings indexed by timestamp.  When a follow-up query
arrives, the engine checks this store first — if a relevant snapshot exists
it reconstructs the graph from saved state instead of rebuilding from
scratch.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from kairo_core import LayeredGraphData
from kairo_graph import LayeredKairoGraph

from .layered_store import LayerEmbeddingStore


@dataclass
class GraphSnapshot:
    """A frozen 3-layer graph + embeddings at a specific point in time."""

    id: str
    timestamp: datetime
    query: str
    graph_data: LayeredGraphData
    embeddings_dict: Dict  # serialized LayerEmbeddingStore
    graph_hash: str
    metadata: Dict = field(default_factory=dict)

    def reconstruct_graph(self) -> LayeredKairoGraph:
        """Rebuild the live graph from saved wire data."""
        return LayeredKairoGraph.from_data(self.graph_data)

    def reconstruct_embeddings(self) -> LayerEmbeddingStore:
        """Rebuild the layer embedding stores from saved data."""
        return LayerEmbeddingStore.from_dict(self.embeddings_dict)


class SessionStore:
    """In-memory store of graph snapshots for a single session.

    Thread-safe.  Snapshots are indexed by timestamp so follow-up queries
    can locate the nearest prior state.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._snapshots: List[GraphSnapshot] = []
        self._lock = threading.Lock()

    def save_snapshot(
        self,
        graph: LayeredKairoGraph,
        embeddings: LayerEmbeddingStore,
        query: str,
        *,
        metadata: Optional[Dict] = None,
    ) -> GraphSnapshot:
        """Snapshot the current graph + embeddings state."""
        snap = GraphSnapshot(
            id=uuid.uuid4().hex[:16],
            timestamp=datetime.now(timezone.utc),
            query=query,
            graph_data=graph.to_data(),
            embeddings_dict=embeddings.to_dict(),
            graph_hash=graph.hash(),
            metadata=metadata or {},
        )
        with self._lock:
            self._snapshots.append(snap)
        return snap

    def find_nearest(
        self,
        timestamp: Optional[datetime] = None,
    ) -> Optional[GraphSnapshot]:
        """Find the snapshot closest to *timestamp* (or the latest)."""
        with self._lock:
            if not self._snapshots:
                return None
            if timestamp is None:
                return self._snapshots[-1]
            best: Optional[GraphSnapshot] = None
            best_delta = float("inf")
            for snap in self._snapshots:
                delta = abs((snap.timestamp - timestamp).total_seconds())
                if delta < best_delta:
                    best_delta = delta
                    best = snap
            return best

    def find_by_query(self, query: str) -> Optional[GraphSnapshot]:
        """Find the most recent snapshot that was built for *query*."""
        with self._lock:
            for snap in reversed(self._snapshots):
                if snap.query == query:
                    return snap
        return None

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._snapshots)

    def list_snapshots(self) -> List[GraphSnapshot]:
        with self._lock:
            return list(self._snapshots)

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()
