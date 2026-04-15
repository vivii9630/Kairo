"""Kairo connectors: adapters for external data sources.

Every connector implements the :class:`ConnectorPlugin` protocol and is
registered in :mod:`kairo_connectors.registry`. The legacy
:class:`BaseConnector` ABC is kept for backward compatibility and for
connectors that only yield ``Document`` streams (no local materialization).

Quick start::

    from kairo_connectors import get_connector
    plugin = get_connector("github")
    result = plugin.fetch("https://github.com/owner/repo")
    try:
        for path in result.files:
            ...  # hand off to kairo-ingest
    finally:
        if result.cleanup:
            result.cleanup()
"""

from .base import BaseConnector
from .plugin import ConnectorPlugin, FetchResult, PluginManifest
from .registry import (
    available_connectors,
    get_connector,
    list_manifests,
    register_connector,
)

__all__ = [
    "BaseConnector",
    "ConnectorPlugin",
    "FetchResult",
    "PluginManifest",
    "available_connectors",
    "get_connector",
    "list_manifests",
    "register_connector",
]
