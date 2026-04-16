"""Google Drive connector — file-shaped.

Pulls files out of Drive into a temp directory and returns their paths.
Google-native documents (Docs/Sheets/Slides) are exported to plain
formats (text, CSV, text) so downstream ingest can parse them without
needing Google-specific readers.

Auth: OAuth2Auth with ``drive.readonly`` scope (or broader) and
``client_id``/``client_secret``/``refresh_token`` populated so the
scaffold can refresh an expired access token.
"""

from __future__ import annotations

import io
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore

    _HAS_GOOGLE = True
except ImportError:  # pragma: no cover
    MediaIoBaseDownload = None  # type: ignore
    _HAS_GOOGLE = False

from ..auth import AuthMethod, AuthRequirement, OAuth2Auth
from ..plugin import FetchResult, FetchSpec, PluginManifest
from . import _google_auth


_GOOGLE_EXPORTS: Dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}

_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class GDriveFetchSpec(FetchSpec):
    """Fetch spec for Google Drive.

    Either ``query`` (raw Drive search syntax) or ``folder_id`` (shorthand
    for ``"'<id>' in parents"``) must be set. ``mime_types`` narrows by
    MIME; ``export_google_docs`` enables converting native Google formats
    to plain text/CSV on download.
    """

    query: Optional[str] = None
    folder_id: Optional[str] = None
    mime_types: Optional[List[str]] = None
    export_google_docs: bool = True
    page_size: int = 100


class GoogleDriveConnector:
    """Downloads Drive files to a temp directory via the Drive v3 API."""

    manifest = PluginManifest(
        name="google_drive",
        label="Google Drive",
        description="Ingest files from Google Drive using OAuth 2.0.",
        uri_example="gdrive://folder/<folder_id>",
        auth=AuthRequirement(
            methods=[OAuth2Auth],
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
            instructions=(
                "Provide OAuth2Auth with drive.readonly (or broader) scope "
                "and client_id/client_secret/refresh_token so the connector "
                "can keep the access token fresh."
            ),
        ),
        icon="google-drive",
        tags=["docs", "files", "google"],
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
        if not isinstance(spec, GDriveFetchSpec):
            raise TypeError(
                f"GoogleDriveConnector expects GDriveFetchSpec, got {type(spec).__name__}"
            )
        if not spec.query and not spec.folder_id:
            raise ValueError("GDriveFetchSpec requires either query or folder_id")

        service = self._service_override
        if service is None:
            if not isinstance(auth, OAuth2Auth):
                raise ValueError(
                    "GoogleDriveConnector requires OAuth2Auth; "
                    f"got {type(auth).__name__ if auth else 'None'}"
                )
            auth = _google_auth.maybe_refresh(auth)
            service = _google_auth.build_service("drive", "v3", auth)

        query = _build_query(spec)
        tmp_root = Path(tempfile.mkdtemp(prefix="kairo_gdrive_"))
        downloaded: List[Path] = []
        metadata_index: List[Dict[str, Any]] = []

        try:
            for meta in _iter_drive_files(service, query, page_size=spec.page_size):
                if spec.limit is not None and len(downloaded) >= spec.limit:
                    break
                local_path = self._download_file(
                    service,
                    meta,
                    dest=tmp_root,
                    export_google_docs=spec.export_google_docs,
                )
                if local_path is not None:
                    downloaded.append(local_path)
                    metadata_index.append(
                        {
                            "id": meta.get("id"),
                            "name": meta.get("name"),
                            "mimeType": meta.get("mimeType"),
                            "modifiedTime": meta.get("modifiedTime"),
                            "path": str(local_path),
                        }
                    )
        except Exception:
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise

        def cleanup() -> None:
            shutil.rmtree(tmp_root, ignore_errors=True)

        return FetchResult(
            source_uri=f"gdrive:?q={query}",
            metadata={
                "connector": "google_drive",
                "query": query,
                "file_count": len(downloaded),
                "files": metadata_index,
            },
            root=tmp_root,
            files=downloaded,
            cleanup=cleanup,
        )

    def _download_file(
        self,
        service: Any,
        meta: Dict[str, Any],
        *,
        dest: Path,
        export_google_docs: bool,
    ) -> Optional[Path]:
        mime = meta.get("mimeType", "")
        name = _safe_filename(meta.get("name") or meta.get("id") or "unnamed")
        file_id = meta["id"]

        if mime in _GOOGLE_EXPORTS:
            if not export_google_docs:
                return None
            export_mime, ext = _GOOGLE_EXPORTS[mime]
            target = dest / f"{name}{ext}"
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        elif mime.startswith("application/vnd.google-apps."):
            # Unsupported native type (e.g. Forms, Sites) — skip.
            return None
        else:
            target = dest / name
            request = service.files().get_media(fileId=file_id)

        _write_media(request, target)
        return target


def _build_query(spec: GDriveFetchSpec) -> str:
    parts: List[str] = []
    if spec.query:
        parts.append(f"({spec.query})")
    if spec.folder_id:
        parts.append(f"'{spec.folder_id}' in parents")
    if spec.mime_types:
        mime_clause = " or ".join(f"mimeType = '{m}'" for m in spec.mime_types)
        parts.append(f"({mime_clause})")
    if spec.since is not None:
        ts = spec.since.isoformat()
        parts.append(f"modifiedTime > '{ts}'")
    parts.append("trashed = false")
    return " and ".join(parts)


def _iter_drive_files(service: Any, query: str, *, page_size: int) -> Iterable[Dict[str, Any]]:
    page_token: Optional[str] = None
    fields = "nextPageToken, files(id, name, mimeType, modifiedTime, parents, size)"
    while True:
        kwargs: Dict[str, Any] = {
            "q": query,
            "pageSize": page_size,
            "fields": fields,
        }
        if page_token:
            kwargs["pageToken"] = page_token
        response = service.files().list(**kwargs).execute()
        for f in response.get("files", []):
            yield f
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def _write_media(request: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Real googleapiclient HttpRequest objects carry .uri; when they don't
    # (test stubs, some export helpers), fall back to .execute() which
    # returns raw bytes or text.
    if MediaIoBaseDownload is not None and hasattr(request, "uri"):
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        target.write_bytes(buf.getvalue())
        return
    payload = request.execute()
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        target.write_text(str(payload), encoding="utf-8")


def _safe_filename(name: str) -> str:
    cleaned = _INVALID_PATH_CHARS.sub("_", name).strip(" .")
    return cleaned or "unnamed"
