from __future__ import annotations

from typing import Any, Dict, List, Type

from .plugin import ConnectorPlugin, PluginManifest


_BUILTIN: Dict[str, Type] = {}


def _register_builtins() -> None:
    """Import built-in connector plugins lazily so optional deps stay optional."""
    try:
        from .plugins.github import GitHubConnector

        _BUILTIN["github"] = GitHubConnector
    except ImportError:
        pass

    try:
        from .plugins.slack import SlackConnector

        _BUILTIN["slack"] = SlackConnector
    except ImportError:
        pass

    try:
        from .plugins.gdrive import GoogleDriveConnector

        _BUILTIN["google_drive"] = GoogleDriveConnector
    except ImportError:
        pass

    try:
        from .plugins.gmail import GmailConnector

        _BUILTIN["gmail"] = GmailConnector
    except ImportError:
        pass


_register_builtins()


def get_connector(name: str, **kwargs: Any) -> ConnectorPlugin:
    """Resolve a connector by name and construct an instance."""
    if name not in _BUILTIN:
        available = ", ".join(sorted(_BUILTIN)) or "<none>"
        raise ValueError(
            f"Unknown connector: {name!r}. Available: {available}."
        )
    return _BUILTIN[name](**kwargs)


def register_connector(name: str, cls: Type) -> None:
    """Register a user-defined connector plugin into the registry."""
    _BUILTIN[name] = cls


def available_connectors() -> List[str]:
    return sorted(_BUILTIN)


def list_manifests() -> List[PluginManifest]:
    """Return manifests for every registered connector.

    This is what a UI plugin picker or agent tool-lister would call — no
    instances are constructed, only class-level ``manifest`` attributes.
    """
    manifests: List[PluginManifest] = []
    for cls in _BUILTIN.values():
        manifest = getattr(cls, "manifest", None)
        if isinstance(manifest, PluginManifest):
            manifests.append(manifest)
    return manifests
