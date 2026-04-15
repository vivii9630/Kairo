from __future__ import annotations

from collections import deque
from typing import Iterator, List, Optional

from .history import HistoryGraph


class HistoryWalker:
    """Traverse a ``HistoryGraph`` in either direction.

    Walks are BFS-ordered so the result is stable and the nearest snapshots
    to the origin come first. Cycles are impossible (history is a DAG by
    construction), but we still guard against revisits so the walker is safe
    over any structure.
    """

    def __init__(self, history: HistoryGraph):
        self.history = history

    def ancestors(self, snapshot_id: str, *, max_depth: Optional[int] = None) -> List[str]:
        return list(self._walk(snapshot_id, direction="parents", max_depth=max_depth))

    def descendants(self, snapshot_id: str, *, max_depth: Optional[int] = None) -> List[str]:
        return list(self._walk(snapshot_id, direction="children", max_depth=max_depth))

    def path_to_root(self, snapshot_id: str) -> List[str]:
        """One linear chain back to a root — follows the first parent at
        each step, mirroring ``git log --first-parent``.
        """
        chain: List[str] = []
        current: Optional[str] = snapshot_id
        seen = set()
        while current is not None and current not in seen:
            seen.add(current)
            chain.append(current)
            node = self.history.get(current)
            current = node.parents[0] if node.parents else None
        return chain

    def _walk(
        self,
        start: str,
        *,
        direction: str,
        max_depth: Optional[int],
    ) -> Iterator[str]:
        if start not in self.history:
            raise KeyError(f"snapshot {start!r} not in history")
        visited = {start}
        queue: deque = deque([(start, 0)])
        while queue:
            sid, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            neighbors = getattr(self.history.get(sid), direction)
            for nxt in neighbors:
                if nxt in visited:
                    continue
                visited.add(nxt)
                yield nxt
                queue.append((nxt, depth + 1))
