"""Slack connector — record-shaped, no filesystem materialization.

Pulls messages out of one or more channels via the Slack Web API and emits
them directly as :class:`kairo_core.Document` records. No temp dir, no
file walk — Slack messages are already structured, so we skip the ingest
pipeline's parsing stage.

Auth: a Slack bot token (``xoxb-...``) with ``channels:history`` and
``groups:history`` scopes at minimum; add ``users:read`` if you want the
plugin to resolve user IDs into display names.

Incremental sync: pass ``since`` on the fetch spec (or ``oldest``
explicitly) and the plugin converts it to Slack's unix-timestamp string
format. Pagination is handled internally via ``next_cursor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from slack_sdk import WebClient  # type: ignore
    from slack_sdk.errors import SlackApiError  # type: ignore

    _HAS_SLACK_SDK = True
except ImportError:  # pragma: no cover - exercised via optional-extras path
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore
    _HAS_SLACK_SDK = False

from kairo_core import Document

from ..auth import AuthMethod, AuthRequirement, BearerTokenAuth
from ..plugin import FetchResult, FetchSpec, PluginManifest


@dataclass
class SlackFetchSpec(FetchSpec):
    """Fetch spec for the Slack connector.

    ``channels`` is a list of channel IDs (``C0123456``) — resolving names
    to IDs is the caller's job; the plugin does not list workspaces.
    ``oldest`` / ``latest`` override the base ``since`` field when you
    need a closed window rather than "everything after".
    """

    channels: List[str] = field(default_factory=list)
    oldest: Optional[datetime] = None
    latest: Optional[datetime] = None
    resolve_users: bool = True


class SlackConnector:
    """Pulls channel history into ``Document`` records.

    Inject a ``WebClient`` instance via the ``client`` kwarg for testing;
    in production the plugin builds one from the ``BearerTokenAuth``
    supplied to :meth:`fetch`.
    """

    manifest = PluginManifest(
        name="slack",
        label="Slack",
        description="Ingest Slack channel history via the Web API (bot token).",
        uri_example="slack://workspace",
        auth=AuthRequirement(
            methods=[BearerTokenAuth],
            scopes=["channels:history", "groups:history", "users:read"],
            instructions=(
                "Provide a Slack bot token (xoxb-...) with channels:history "
                "and groups:history. Add users:read to resolve display names."
            ),
        ),
        icon="slack",
        tags=["chat", "messaging"],
    )

    def __init__(self, *, client: Optional[Any] = None, page_size: int = 200):
        if not _HAS_SLACK_SDK and client is None:
            raise ImportError(
                "SlackConnector requires the 'slack-sdk' package. "
                "Install with: pip install kairo-connectors[slack]"
            )
        self._client_override = client
        self.page_size = int(page_size)

    def fetch(
        self,
        spec: FetchSpec,
        *,
        auth: Optional[AuthMethod] = None,
        **kwargs: Any,
    ) -> FetchResult:
        if not isinstance(spec, SlackFetchSpec):
            raise TypeError(
                f"SlackConnector expects SlackFetchSpec, got {type(spec).__name__}"
            )
        if not spec.channels:
            raise ValueError("SlackFetchSpec.channels must list at least one channel ID")

        client = self._client_override
        if client is None:
            if not isinstance(auth, BearerTokenAuth):
                raise ValueError(
                    "SlackConnector requires BearerTokenAuth; "
                    f"got {type(auth).__name__ if auth else 'None'}"
                )
            client = WebClient(token=auth.token)

        oldest_ts = _to_slack_ts(spec.oldest or spec.since)
        latest_ts = _to_slack_ts(spec.latest)

        records: List[Document] = []
        user_cache: Dict[str, str] = {}
        fetched = 0
        limit = spec.limit

        for channel in spec.channels:
            for message in _iter_channel_history(
                client,
                channel,
                oldest=oldest_ts,
                latest=latest_ts,
                page_size=self.page_size,
            ):
                doc = _message_to_document(
                    message,
                    channel=channel,
                    client=client if spec.resolve_users else None,
                    user_cache=user_cache,
                )
                records.append(doc)
                fetched += 1
                if limit is not None and fetched >= limit:
                    break
            if limit is not None and fetched >= limit:
                break

        metadata = {
            "connector": "slack",
            "channels": list(spec.channels),
            "oldest": oldest_ts,
            "latest": latest_ts,
            "message_count": len(records),
        }
        return FetchResult(
            source_uri=f"slack://channels/{','.join(spec.channels)}",
            metadata=metadata,
            records=records,
        )


def _to_slack_ts(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to Slack's unix-timestamp string (``"1700000000.000000"``)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return f"{dt.timestamp():.6f}"


def _iter_channel_history(
    client: Any,
    channel: str,
    *,
    oldest: Optional[str],
    latest: Optional[str],
    page_size: int,
):
    """Yield every message in a channel, paginating via ``next_cursor``."""
    cursor: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"channel": channel, "limit": page_size}
        if oldest is not None:
            kwargs["oldest"] = oldest
        if latest is not None:
            kwargs["latest"] = latest
        if cursor:
            kwargs["cursor"] = cursor

        response = client.conversations_history(**kwargs)
        data = _unwrap(response)
        for msg in data.get("messages", []):
            yield msg

        metadata = data.get("response_metadata") or {}
        cursor = metadata.get("next_cursor") or None
        if not cursor:
            break


def _message_to_document(
    message: Dict[str, Any],
    *,
    channel: str,
    client: Optional[Any],
    user_cache: Dict[str, str],
) -> Document:
    ts = str(message.get("ts", ""))
    user_id = message.get("user") or message.get("bot_id") or ""
    user_name = _resolve_user(client, user_id, user_cache) if client and user_id else None

    doc_id = f"slack:{channel}:{ts}"
    text = str(message.get("text", ""))

    metadata: Dict[str, Any] = {
        "source": "slack",
        "channel": channel,
        "ts": ts,
        "user": user_id,
        "type": message.get("type"),
    }
    if user_name:
        metadata["user_name"] = user_name
    if message.get("thread_ts"):
        metadata["thread_ts"] = message["thread_ts"]
    if message.get("subtype"):
        metadata["subtype"] = message["subtype"]

    return Document(id=doc_id, text=text, metadata=metadata)


def _resolve_user(client: Any, user_id: str, cache: Dict[str, str]) -> Optional[str]:
    if user_id in cache:
        return cache[user_id] or None
    try:
        resp = client.users_info(user=user_id)
    except SlackApiError:
        cache[user_id] = ""
        return None
    data = _unwrap(resp)
    profile = (data.get("user") or {}).get("profile") or {}
    name = (
        profile.get("display_name")
        or profile.get("real_name")
        or (data.get("user") or {}).get("name")
        or ""
    )
    cache[user_id] = name
    return name or None


def _unwrap(response: Any) -> Dict[str, Any]:
    """Slack SDK responses behave like dicts; test doubles may just be dicts."""
    if isinstance(response, dict):
        return response
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return data
    return dict(response)  # type: ignore[arg-type]
