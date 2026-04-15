from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class PluginManifest:
    """Declarative description of a connector plugin.

    Used by three consumers: the Kairo UI plugin picker, the agent tool-use
    layer (so agents can call connectors as tools), and the CLI ``kairo init``
    flow. Keeping it a plain dataclass means all three can read it without
    importing the connector implementation.
    """

    name: str
    label: str
    description: str
    uri_example: str
    icon: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class FetchResult:
    """What a connector returns after materializing a source.

    ``root`` is the local directory where content was written (usually a
    temp dir). ``files`` is the pre-walked list of regular files under
    ``root`` that downstream ingest should consider. ``cleanup`` is an
    optional callable the caller invokes once it is done with the files —
    splitting materialization from cleanup lets the caller pipeline
    ingestion without racing a ``__del__``.
    """

    root: Path
    files: List[Path]
    source_uri: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup: Optional[Callable[[], None]] = None

    def __len__(self) -> int:
        return len(self.files)


@runtime_checkable
class ConnectorPlugin(Protocol):
    """Uniform contract for every Kairo data-source plugin.

    Implementations live under ``kairo_connectors/plugins/<name>.py`` and
    are registered via ``kairo_connectors.registry.register_connector``.
    Each plugin is self-describing (``manifest``) and self-contained
    (``fetch``); the rest of Kairo never imports the plugin directly.
    """

    manifest: PluginManifest

    def fetch(self, uri: str, **kwargs: Any) -> FetchResult:
        """Materialize ``uri`` locally and return a ``FetchResult``."""
        ...
