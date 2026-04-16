"""Authentication primitives for Kairo connectors.

These live in kairo-connectors today, but the types are designed so a
future `kairo-auth` package can take over credential brokering without
any plugin needing to change. Plugins declare what they need via
:class:`AuthRequirement` on their manifest; plugins receive a concrete
:data:`AuthMethod` instance when ``fetch`` is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Type, Union


@dataclass
class ApiKeyAuth:
    """Flat API key — e.g. SerpAPI, Notion, most REST APIs with a single secret."""

    api_key: str
    header_name: str = "Authorization"
    prefix: str = ""

    def header_value(self) -> str:
        return f"{self.prefix}{self.api_key}" if self.prefix else self.api_key


@dataclass
class BearerTokenAuth:
    """Bearer token — e.g. Slack bot tokens, GitHub PATs, Jira Cloud tokens."""

    token: str

    def header_value(self) -> str:
        return f"Bearer {self.token}"


@dataclass
class BasicAuth:
    """HTTP Basic — e.g. Jira Server with username + API token."""

    username: str
    password: str


@dataclass
class OAuth2Auth:
    """OAuth 2.0 access token, optionally with refresh material.

    ``expires_at`` is stored as a UTC datetime; ``is_expired`` compares
    against ``datetime.now(timezone.utc)`` with a small skew buffer so
    callers don't race token expiry.
    """

    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)
    token_type: str = "Bearer"

    def header_value(self) -> str:
        return f"{self.token_type} {self.access_token}"

    def is_expired(self, skew_seconds: int = 30) -> bool:
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc)
        return (self.expires_at - now).total_seconds() <= skew_seconds


AuthMethod = Union[ApiKeyAuth, BearerTokenAuth, BasicAuth, OAuth2Auth]


@dataclass
class AuthRequirement:
    """Declares what authentication a connector needs.

    ``methods`` lists acceptable auth classes (a plugin may accept either
    a bot token or OAuth, for example). ``scopes`` is OAuth-specific and
    ignored for key/token auth. ``instructions`` is a short human-readable
    blurb the UI plugin picker shows while prompting for credentials.
    """

    methods: List[Type] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    instructions: str = ""

    def accepts(self, auth: object) -> bool:
        return any(isinstance(auth, m) for m in self.methods)
