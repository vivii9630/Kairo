from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from kairo_core.models import Snapshot
from kairo_embeddings.store import EmbeddingStore
from kairo_graph.graph import KairoGraph
from kairo_graph.store import load_graph, save_graph

from .history import HistoryGraph


class KairoStore:
    """Filesystem source of truth for a Kairo history.

    Owns the ``.kairo/`` directory layout::

        .kairo/
        ├── HEAD
        ├── refs/branches/<name>
        ├── history.json
        └── snapshots/<snapshot_id>/
            ├── snapshot.json
            ├── graph.json
            └── embeddings/{matrix.npy,index.json}

    HEAD is either ``ref: refs/branches/<name>`` (attached) or a bare
    snapshot id (detached, e.g. after rollback). Nothing here knows what a
    graph *means* — that's the job of kairo-graph builders and, later,
    kairo-rag's TemporalRAG orchestrator.
    """

    HEAD_FILE = "HEAD"
    HISTORY_FILE = "history.json"
    SNAPSHOTS_DIR = "snapshots"
    BRANCHES_DIR = "refs/branches"

    def __init__(self, root: Union[Path, str], history_graph_id: str):
        self.root = Path(root)
        self.history_graph_id = history_graph_id

    @classmethod
    def init(
        cls,
        project_root: Union[Path, str],
        *,
        history_graph_id: str,
        default_branch: str = "main",
    ) -> "KairoStore":
        store = cls(Path(project_root) / ".kairo", history_graph_id)
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / cls.SNAPSHOTS_DIR).mkdir(exist_ok=True)
        (store.root / cls.BRANCHES_DIR).mkdir(parents=True, exist_ok=True)
        if not (store.root / cls.HISTORY_FILE).exists():
            store._write_history(HistoryGraph(history_graph_id))
        if not (store.root / cls.HEAD_FILE).exists():
            store._write_head(f"ref: {cls.BRANCHES_DIR}/{default_branch}")
        return store

    # ---- history -----------------------------------------------------

    def load_history(self) -> HistoryGraph:
        path = self.root / self.HISTORY_FILE
        if not path.exists():
            return HistoryGraph(self.history_graph_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return HistoryGraph.from_dict(data)

    def _write_history(self, history: HistoryGraph) -> None:
        path = self.root / self.HISTORY_FILE
        path.write_text(
            json.dumps(history.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- HEAD + branches --------------------------------------------

    def _write_head(self, value: str) -> None:
        (self.root / self.HEAD_FILE).write_text(value, encoding="utf-8")

    def read_head(self) -> str:
        return (self.root / self.HEAD_FILE).read_text(encoding="utf-8").strip()

    def resolve_head(self) -> Optional[str]:
        """Return the snapshot id HEAD points at, or None if unborn."""
        head = self.read_head()
        if head.startswith("ref: "):
            ref_path = self.root / head[len("ref: "):]
            if not ref_path.exists():
                return None
            return ref_path.read_text(encoding="utf-8").strip() or None
        return head or None

    def set_branch(self, name: str, snapshot_id: str) -> None:
        path = self.root / self.BRANCHES_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snapshot_id, encoding="utf-8")

    def get_branch(self, name: str) -> Optional[str]:
        path = self.root / self.BRANCHES_DIR / name
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def list_branches(self) -> List[str]:
        bdir = self.root / self.BRANCHES_DIR
        if not bdir.exists():
            return []
        return sorted(p.name for p in bdir.iterdir() if p.is_file())

    def checkout_branch(self, name: str) -> None:
        if self.get_branch(name) is None:
            raise KeyError(f"branch {name!r} does not exist")
        self._write_head(f"ref: {self.BRANCHES_DIR}/{name}")

    def checkout_snapshot(self, snapshot_id: str) -> None:
        """Detached checkout — HEAD points directly at a snapshot id."""
        if not self._snapshot_dir(snapshot_id).exists():
            raise KeyError(f"snapshot {snapshot_id!r} not found on disk")
        self._write_head(snapshot_id)

    # ---- snapshots ---------------------------------------------------

    def _snapshot_dir(self, snapshot_id: str) -> Path:
        return self.root / self.SNAPSHOTS_DIR / snapshot_id

    def commit(
        self,
        snapshot: Snapshot,
        graph: KairoGraph,
        embeddings: Optional[EmbeddingStore] = None,
        *,
        advance_branch: Optional[str] = None,
    ) -> None:
        """Persist a snapshot to disk and wire it into history.

        If ``advance_branch`` is given, that branch ref moves to the new
        snapshot. Otherwise, if HEAD is attached to a branch, that branch
        advances automatically.
        """
        sdir = self._snapshot_dir(snapshot.id)
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "snapshot.json").write_text(
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_graph(graph, sdir / "graph.json")
        if embeddings is not None:
            edir = sdir / "embeddings"
            edir.mkdir(exist_ok=True)
            embeddings.save(edir / "matrix.npy", edir / "index.json")

        history = self.load_history()
        history.add(snapshot)
        self._write_history(history)

        target_branch = advance_branch or self._current_branch()
        if target_branch is not None:
            self.set_branch(target_branch, snapshot.id)

    def load_snapshot(self, snapshot_id: str) -> Tuple[Snapshot, KairoGraph, Optional[EmbeddingStore]]:
        sdir = self._snapshot_dir(snapshot_id)
        if not sdir.exists():
            raise KeyError(f"snapshot {snapshot_id!r} not found")
        snap = Snapshot.model_validate_json(
            (sdir / "snapshot.json").read_text(encoding="utf-8")
        )
        graph = load_graph(sdir / "graph.json")
        edir = sdir / "embeddings"
        embeddings: Optional[EmbeddingStore] = None
        if (edir / "index.json").exists() and (edir / "matrix.npy").exists():
            embeddings = EmbeddingStore.load(edir / "matrix.npy", edir / "index.json")
        return snap, graph, embeddings

    def rollback(self, snapshot_id: str) -> Tuple[Snapshot, KairoGraph, Optional[EmbeddingStore]]:
        """Load a prior snapshot and point HEAD at it (detached)."""
        result = self.load_snapshot(snapshot_id)
        self.checkout_snapshot(snapshot_id)
        return result

    # ---- helpers -----------------------------------------------------

    def _current_branch(self) -> Optional[str]:
        head = self.read_head()
        prefix = f"ref: {self.BRANCHES_DIR}/"
        if head.startswith(prefix):
            return head[len(prefix):]
        return None
