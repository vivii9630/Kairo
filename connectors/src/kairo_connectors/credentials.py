"""Credential stores for Kairo connectors.

Two concrete stores are shipped:

- :class:`EnvCredentialStore` — reads secrets from well-known environment
  variables per plugin. Zero setup, perfect for dev machines and CI.
- :class:`FileCredentialStore` — reads and writes a JSON file on disk
  (gitignored by default). Right for persisting OAuth refresh tokens
  between sessions.

Both implement the :class:`CredentialStore` Protocol, and
:class:`ChainedCredentialStore` lets callers layer them (env first, file
fallback) without plugins caring which store produced the credential.

When ``kairo-auth`` lands, it implements the same Protocol and becomes a
drop-in replacement — plugins never import a concrete store directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Type, runtime_checkable

from .auth import (
    ApiKeyAuth,
    AuthMethod,
    BasicAuth,
    BearerTokenAuth,
    OAuth2Auth,
)


@runtime_checkable
class CredentialStore(Protocol):
    """Uniform Protocol every credential source implements.

    Plugins call ``get(plugin_name)`` and never import a concrete store
    themselves, so the store can be swapped (env -> file -> kairo-auth)
    without touching plugin code.
    """

    def get(self, plugin_name: str) -> Optional[AuthMethod]: ...

    def put(self, plugin_name: str, auth: AuthMethod) -> None: ...


class EnvCredentialStore:
    """Reads credentials from a caller-supplied mapping of env-var factories.

    Each plugin registers a small factory that knows how to build its own
    auth from ``os.environ``. Keeping the factory with the plugin (rather
    than hardcoding env var names here) means adding a new connector does
    not require editing this file.
    """

    def __init__(
        self,
        factories: Optional[Dict[str, Callable[[Dict[str, str]], Optional[AuthMethod]]]] = None,
    ):
        self._factories: Dict[str, Callable[[Dict[str, str]], Optional[AuthMethod]]] = (
            dict(factories) if factories else {}
        )

    def register(
        self,
        plugin_name: str,
        factory: Callable[[Dict[str, str]], Optional[AuthMethod]],
    ) -> None:
        self._factories[plugin_name] = factory

    def get(self, plugin_name: str) -> Optional[AuthMethod]:
        factory = self._factories.get(plugin_name)
        if factory is None:
            return None
        return factory(dict(os.environ))

    def put(self, plugin_name: str, auth: AuthMethod) -> None:
        raise NotImplementedError(
            "EnvCredentialStore is read-only; use FileCredentialStore for persistence"
        )


class FileCredentialStore:
    """JSON-backed persistent credential store.

    Layout on disk::

        {
          "slack": {"__type__": "BearerTokenAuth", "token": "xoxb-..."},
          "gmail": {"__type__": "OAuth2Auth", "access_token": "...",
                    "refresh_token": "...", "expires_at": "2026-01-01T00:00:00+00:00",
                    "scopes": ["gmail.readonly"]}
        }

    The file is written with restrictive permissions (``0o600``) on POSIX
    systems. Callers are responsible for keeping the path out of version
    control (add ``.kairo/credentials.json`` to ``.gitignore``).
    """

    _TYPE_MAP: Dict[str, Type] = {
        "ApiKeyAuth": ApiKeyAuth,
        "BearerTokenAuth": BearerTokenAuth,
        "BasicAuth": BasicAuth,
        "OAuth2Auth": OAuth2Auth,
    }

    def __init__(self, path: Path):
        self.path = Path(path)

    def get(self, plugin_name: str) -> Optional[AuthMethod]:
        data = self._load()
        payload = data.get(plugin_name)
        if payload is None:
            return None
        return self._deserialize(payload)

    def put(self, plugin_name: str, auth: AuthMethod) -> None:
        data = self._load()
        data[plugin_name] = self._serialize(auth)
        self._save(data)

    def delete(self, plugin_name: str) -> None:
        data = self._load()
        if plugin_name in data:
            del data[plugin_name]
            self._save(data)

    def _load(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass  # chmod is a no-op on Windows; that's fine

    @staticmethod
    def _serialize(auth: AuthMethod) -> dict:
        if not is_dataclass(auth):
            raise TypeError(f"cannot serialize non-dataclass auth: {type(auth).__name__}")
        payload: Dict[str, object] = {"__type__": type(auth).__name__}
        for f in fields(auth):
            value = getattr(auth, f.name)
            if isinstance(value, datetime):
                value = value.astimezone(timezone.utc).isoformat()
            payload[f.name] = value
        return payload

    @classmethod
    def _deserialize(cls, payload: dict) -> AuthMethod:
        type_name = payload.get("__type__")
        auth_cls = cls._TYPE_MAP.get(type_name) if type_name else None
        if auth_cls is None:
            raise ValueError(f"unknown auth type in credentials file: {type_name!r}")
        kwargs = {k: v for k, v in payload.items() if k != "__type__"}
        if auth_cls is OAuth2Auth and isinstance(kwargs.get("expires_at"), str):
            kwargs["expires_at"] = datetime.fromisoformat(kwargs["expires_at"])
        return auth_cls(**kwargs)


class ChainedCredentialStore:
    """Tries each underlying store in order and returns the first hit.

    ``put`` always writes to the first writable store (typically the
    :class:`FileCredentialStore`) so persistence behavior is predictable.
    """

    def __init__(self, stores: List[CredentialStore]):
        if not stores:
            raise ValueError("ChainedCredentialStore requires at least one store")
        self._stores = list(stores)

    def get(self, plugin_name: str) -> Optional[AuthMethod]:
        for store in self._stores:
            auth = store.get(plugin_name)
            if auth is not None:
                return auth
        return None

    def put(self, plugin_name: str, auth: AuthMethod) -> None:
        last_error: Optional[Exception] = None
        for store in self._stores:
            try:
                store.put(plugin_name, auth)
                return
            except NotImplementedError as exc:
                last_error = exc
                continue
        raise RuntimeError(
            "no store in chain accepted put()"
        ) from last_error
