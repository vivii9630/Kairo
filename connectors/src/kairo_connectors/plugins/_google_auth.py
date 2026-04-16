"""Shared Google OAuth scaffold for Drive and Gmail.

Keeps the google-auth / googleapiclient imports behind a module so each
connector can opt in without forcing every user of kairo-connectors to
install the Google client libraries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

try:
    from google.oauth2.credentials import Credentials  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    _HAS_GOOGLE = True
except ImportError:  # pragma: no cover
    Credentials = None  # type: ignore
    Request = None  # type: ignore
    build = None  # type: ignore
    _HAS_GOOGLE = False

from ..auth import OAuth2Auth


DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"


def require_google() -> None:
    if not _HAS_GOOGLE:
        raise ImportError(
            "Google connectors require google-api-python-client and google-auth. "
            "Install with: pip install kairo-connectors[google]"
        )


def to_google_credentials(auth: OAuth2Auth) -> Any:
    """Convert a Kairo ``OAuth2Auth`` to a ``google.oauth2.credentials.Credentials``.

    google-auth stores ``expiry`` as a naive UTC datetime (their convention);
    we convert our tz-aware ``expires_at`` to match.
    """
    require_google()
    expiry = None
    if auth.expires_at is not None:
        utc = auth.expires_at.astimezone(timezone.utc)
        expiry = utc.replace(tzinfo=None)
    return Credentials(
        token=auth.access_token,
        refresh_token=auth.refresh_token,
        token_uri=auth.token_uri or DEFAULT_TOKEN_URI,
        client_id=auth.client_id,
        client_secret=auth.client_secret,
        scopes=list(auth.scopes) if auth.scopes else None,
        expiry=expiry,
    )


def from_google_credentials(creds: Any, *, token_type: str = "Bearer") -> OAuth2Auth:
    """Snapshot a refreshed ``Credentials`` object back into Kairo's ``OAuth2Auth``."""
    expires_at: Optional[datetime] = None
    if getattr(creds, "expiry", None) is not None:
        expires_at = creds.expiry.replace(tzinfo=timezone.utc)
    scopes: List[str] = list(getattr(creds, "scopes", None) or [])
    return OAuth2Auth(
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        expires_at=expires_at,
        scopes=scopes,
        token_type=token_type,
        client_id=getattr(creds, "client_id", None),
        client_secret=getattr(creds, "client_secret", None),
        token_uri=getattr(creds, "token_uri", None),
    )


def maybe_refresh(auth: OAuth2Auth) -> OAuth2Auth:
    """Refresh ``auth`` in place if its access token is expired; no-op otherwise.

    Returns the (possibly new) ``OAuth2Auth``. Callers that want to persist
    refreshed tokens should write the returned value back to their
    :class:`CredentialStore`.
    """
    if not auth.is_expired():
        return auth
    if not auth.refresh_token or not auth.client_id or not auth.client_secret:
        return auth  # can't refresh without the full triple — let the API call 401
    creds = to_google_credentials(auth)
    creds.refresh(Request())
    return from_google_credentials(creds, token_type=auth.token_type)


def build_service(api: str, version: str, auth: OAuth2Auth) -> Any:
    """Build a googleapiclient service (``drive``/``gmail``) from Kairo auth."""
    require_google()
    creds = to_google_credentials(auth)
    return build(api, version, credentials=creds, cache_discovery=False)
