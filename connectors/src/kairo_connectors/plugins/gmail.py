"""Gmail connector — record-shaped.

Pulls messages matching a Gmail search query and emits them as
:class:`kairo_core.Document` records. The message body is assembled by
walking the MIME payload and concatenating ``text/plain`` parts (falling
back to ``text/html`` stripped of tags if no plain text exists).

Auth: OAuth2Auth with ``gmail.readonly`` (or broader). Needs the same
refresh triple as the Drive connector to survive long-running sessions.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from kairo_core import Document

from ..auth import AuthMethod, AuthRequirement, OAuth2Auth
from ..plugin import FetchResult, FetchSpec, PluginManifest
from . import _google_auth


_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class GmailFetchSpec(FetchSpec):
    """Fetch spec for Gmail.

    ``query`` is raw Gmail search syntax (same as the web UI: ``from:foo
    after:2024/01/01``). ``label_ids`` filters to specific labels.
    ``include_spam_trash`` defaults off to match the UX users expect.
    """

    query: Optional[str] = None
    label_ids: Optional[List[str]] = None
    include_spam_trash: bool = False
    user_id: str = "me"


class GmailConnector:
    """Reads Gmail messages into ``Document`` records via the Gmail v1 API."""

    manifest = PluginManifest(
        name="gmail",
        label="Gmail",
        description="Ingest Gmail messages matching a search query.",
        uri_example="gmail:?q=from:foo+after:2024/01/01",
        auth=AuthRequirement(
            methods=[OAuth2Auth],
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            instructions=(
                "Provide OAuth2Auth with gmail.readonly (or broader) scope "
                "and client_id/client_secret/refresh_token for refresh."
            ),
        ),
        icon="gmail",
        tags=["email", "google"],
    )

    def __init__(self, *, service: Optional[Any] = None):
        self._service_override = service

    def fetch(
        self,
        spec: FetchSpec,
        *,
        auth: Optional[AuthMethod] = None,
        **kwargs: Any,
    ) -> FetchResult:
        if not isinstance(spec, GmailFetchSpec):
            raise TypeError(
                f"GmailConnector expects GmailFetchSpec, got {type(spec).__name__}"
            )

        service = self._service_override
        if service is None:
            if not isinstance(auth, OAuth2Auth):
                raise ValueError(
                    "GmailConnector requires OAuth2Auth; "
                    f"got {type(auth).__name__ if auth else 'None'}"
                )
            auth = _google_auth.maybe_refresh(auth)
            service = _google_auth.build_service("gmail", "v1", auth)

        query = _effective_query(spec)
        records: List[Document] = []
        for msg_id in _iter_message_ids(
            service,
            user_id=spec.user_id,
            query=query,
            label_ids=spec.label_ids,
            include_spam_trash=spec.include_spam_trash,
            limit=spec.limit,
        ):
            full = (
                service.users()
                .messages()
                .get(userId=spec.user_id, id=msg_id, format="full")
                .execute()
            )
            records.append(_message_to_document(full))

        return FetchResult(
            source_uri=f"gmail:?q={query or ''}",
            metadata={
                "connector": "gmail",
                "query": query,
                "message_count": len(records),
            },
            records=records,
        )


def _effective_query(spec: GmailFetchSpec) -> Optional[str]:
    parts: List[str] = []
    if spec.query:
        parts.append(spec.query)
    if spec.since is not None:
        # Gmail search only has day-level granularity; use seconds epoch for precision.
        parts.append(f"after:{int(spec.since.timestamp())}")
    return " ".join(parts) if parts else None


def _iter_message_ids(
    service: Any,
    *,
    user_id: str,
    query: Optional[str],
    label_ids: Optional[List[str]],
    include_spam_trash: bool,
    limit: Optional[int],
) -> Iterable[str]:
    page_token: Optional[str] = None
    fetched = 0
    while True:
        kwargs: Dict[str, Any] = {
            "userId": user_id,
            "includeSpamTrash": include_spam_trash,
        }
        if query:
            kwargs["q"] = query
        if label_ids:
            kwargs["labelIds"] = label_ids
        if page_token:
            kwargs["pageToken"] = page_token

        response = service.users().messages().list(**kwargs).execute()
        for m in response.get("messages", []) or []:
            yield m["id"]
            fetched += 1
            if limit is not None and fetched >= limit:
                return
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def _message_to_document(msg: Dict[str, Any]) -> Document:
    payload = msg.get("payload") or {}
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    body = _extract_body(payload)
    subject = headers.get("subject", "")
    text = f"Subject: {subject}\n\n{body}" if subject else body

    metadata: Dict[str, Any] = {
        "source": "gmail",
        "id": msg.get("id"),
        "threadId": msg.get("threadId"),
        "labelIds": msg.get("labelIds", []),
        "internalDate": msg.get("internalDate"),
        "from": headers.get("from"),
        "to": headers.get("to"),
        "cc": headers.get("cc"),
        "subject": subject,
        "date": headers.get("date"),
    }
    return Document(id=f"gmail:{msg.get('id')}", text=text, metadata=metadata)


def _extract_body(payload: Dict[str, Any]) -> str:
    """Walk the MIME tree, preferring text/plain and falling back to stripped text/html."""
    plain: List[str] = []
    html: List[str] = []

    def walk(part: Dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data:
            decoded = _decode_base64url(data)
            if mime == "text/plain":
                plain.append(decoded)
            elif mime == "text/html":
                html.append(decoded)
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    if plain:
        return "\n\n".join(plain).strip()
    if html:
        return _HTML_TAG_RE.sub("", "\n\n".join(html)).strip()
    return ""


def _decode_base64url(data: str) -> str:
    padding = "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(data + padding)
    except (ValueError, TypeError):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")
