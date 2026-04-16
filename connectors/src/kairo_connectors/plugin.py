from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from kairo_core import Document

from .auth import AuthMethod, AuthRequirement


@dataclass
class FetchSpec:
    """Base fetch specification — connectors subclass this with their own fields.

    Two shared fields matter across every connector:
      - ``since`` — pull only records changed/created after this timestamp.
        Required for temporal snapshots to stay cheap as history grows.
      - ``limit`` — cap the number of records returned in one call.
    """

    since: Optional[datetime] = None
    limit: Optional[int] = None


@dataclass
class PluginManifest:
    """Declarative description of a connector plugin.

    Read by the UI plugin picker, the agent tool-use layer, the CLI
    ``kairo init`` flow, and the credential store (to know what auth
    to prompt for). Keeping it a plain dataclass means every consumer
    can read it without importing the plugin implementation.
    """

    name: str
    label: str
    description: str
    uri_example: str
    auth: Optional[AuthRequirement] = None
    icon: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class FetchResult:
    """What a connector returns after materializing a source.

    A plugin fills whichever container matches its shape:

      - **File-shaped** plugins (GitHub, Drive) set ``root`` and ``files``.
        Downstream ingest walks ``files`` and parses them into Documents.
      - **Record-shaped** plugins (Slack, Gmail, Jira) set ``records``
        directly as ``kairo_core.Document`` instances — no ingest step
        needed, the content is already structured.

    Plugins may fill both when sensible (e.g. Drive returning file paths
    plus metadata records). ``cleanup`` is an optional callable the caller
    invokes once it's done; splitting materialization from cleanup lets
    callers pipeline ingestion without racing a ``__del__``.
    """

    source_uri: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    root: Optional[Path] = None
    files: List[Path] = field(default_factory=list)
    records: List[Document] = field(default_factory=list)
    cleanup: Optional[Callable[[], None]] = None

    def __len__(self) -> int:
        return len(self.files) + len(self.records)


@runtime_checkable
class ConnectorPlugin(Protocol):
    """Uniform contract for every Kairo data-source plugin.

    Implementations live under ``kairo_connectors/plugins/<name>.py`` and
    are registered via ``kairo_connectors.registry.register_connector``.
    Each plugin is self-describing (``manifest``) and self-contained
    (``fetch``); the rest of Kairo never imports the plugin directly.
    """

    manifest: PluginManifest

    def fetch(
        self,
        spec: FetchSpec,
        *,
        auth: Optional[AuthMethod] = None,
        **kwargs: Any,
    ) -> FetchResult:
        """Materialize ``spec`` locally and return a ``FetchResult``.

        Plugins that need auth MUST validate ``auth`` against
        ``self.manifest.auth`` and raise a ``ValueError`` if it is
        missing or of the wrong type. Plugins that declare
        ``manifest.auth = None`` ignore the ``auth`` argument entirely.
        """
        ...
