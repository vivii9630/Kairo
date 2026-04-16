"""Kairo connectors: adapters for external data sources.

Every connector implements the :class:`ConnectorPlugin` protocol and is
registered in :mod:`kairo_connectors.registry`. The legacy
:class:`BaseConnector` ABC is kept for connectors that only yield
``Document`` streams (no local materialization).

Quick start — public GitHub repo::

    from kairo_connectors import get_connector
    from kairo_connectors.plugins.github import GitHubFetchSpec

    plugin = get_connector("github")
    result = plugin.fetch(GitHubFetchSpec(uri="https://github.com/owner/repo"))
    try:
        for path in result.files:
            ...  # hand off to kairo-ingest
    finally:
        if result.cleanup:
            result.cleanup()

Quick start — private GitHub repo with a PAT from the env store::

    from kairo_connectors import EnvCredentialStore
    from kairo_connectors.auth import BearerTokenAuth
    import os

    store = EnvCredentialStore({
        "github": lambda env: (
            BearerTokenAuth(token=env["GITHUB_TOKEN"]) if env.get("GITHUB_TOKEN") else None
        ),
    })
    result = plugin.fetch(
        GitHubFetchSpec(uri="https://github.com/me/private-repo"),
        auth=store.get("github"),
    )
"""

from .auth import (
    ApiKeyAuth,
    AuthMethod,
    AuthRequirement,
    BasicAuth,
    BearerTokenAuth,
    OAuth2Auth,
)
from .base import BaseConnector
from .credentials import (
    ChainedCredentialStore,
    CredentialStore,
    EnvCredentialStore,
    FileCredentialStore,
)
from .plugin import ConnectorPlugin, FetchResult, FetchSpec, PluginManifest
from .registry import (
    available_connectors,
    get_connector,
    list_manifests,
    register_connector,
)

__all__ = [
    # auth
    "ApiKeyAuth",
    "AuthMethod",
    "AuthRequirement",
    "BasicAuth",
    "BearerTokenAuth",
    "OAuth2Auth",
    # credential stores
    "ChainedCredentialStore",
    "CredentialStore",
    "EnvCredentialStore",
    "FileCredentialStore",
    # plugin protocol
    "BaseConnector",
    "ConnectorPlugin",
    "FetchResult",
    "FetchSpec",
    "PluginManifest",
    # registry
    "available_connectors",
    "get_connector",
    "list_manifests",
    "register_connector",
]
