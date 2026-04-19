"""Ingestion sources for Kairo AI.

Two source kinds feed the engine today:

* **GitHub URL** — public repos via the GitHub + ``raw.githubusercontent.com``
  HTTP APIs. Smart-shallow: README split by headings, entry points,
  one level deep into canonical source dirs, per-file size + line caps.
  Stdlib only (``urllib``); authenticated GitHub tokens would only
  raise the rate-limit ceiling, not change the shape.
* **Attachment** — CSV (stdlib ``csv``) and XLSX (optional ``openpyxl``).
  One row ≈ one ``Document`` so tabular Q&A gets grounded per-row.

An :class:`IngestStore` keeps the resulting ``Document`` lists alive in
memory keyed by an opaque ``ingest_id`` that ``/ask`` forwards to the
engine. Nothing persists across process restarts — the demo does not
need it, and adding persistence is a swap of the in-memory dict.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from kairo_core import Document


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


@dataclass
class Ingest:
    id: str
    source: str
    label: str
    documents: List[Document]
    kind: str = "other"  # "github" | "csv" | "xlsx" | "other"
    dataframe: Optional[pd.DataFrame] = None  # populated for tabular kinds only
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IngestStore:
    """Thread-safe in-memory ingest cache.

    Keyed by an opaque ``ingest_id``. Swapping for disk persistence is
    a one-class change — the rest of the API only depends on this shape.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, Ingest] = {}
        self._lock = threading.Lock()

    def save(
        self,
        source: str,
        label: str,
        docs: List[Document],
        *,
        kind: str = "other",
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Ingest:
        ingest = Ingest(
            id=uuid.uuid4().hex,
            source=source,
            label=label,
            documents=docs,
            kind=kind,
            dataframe=dataframe,
        )
        with self._lock:
            self._by_id[ingest.id] = ingest
        return ingest

    def get(self, ingest_id: str) -> Optional[Ingest]:
        with self._lock:
            return self._by_id.get(ingest_id)


# ---------------------------------------------------------------------------
# GitHub smart-shallow ingestion
# ---------------------------------------------------------------------------


# File types we keep. Anything else (images, lockfiles, build artifacts,
# binaries) is noise for graph-building.
_TEXT_EXTS = {
    ".md", ".rst", ".txt",
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".java", ".kt", ".rb",
    ".c", ".h", ".cc", ".cpp", ".hpp",
    ".cs", ".swift", ".php",
    ".json", ".yaml", ".yml", ".toml",
}

# Entry-point filenames that deserve inclusion even one level deep.
_ENTRYPOINT_BASES = {
    "main", "index", "app", "__init__", "__main__",
    "cli", "server", "client", "api",
}

# Canonical source dirs we walk one level into (beyond the top level).
_SOURCE_DIRS = {"src", "lib", "app", "apps", "packages", "cmd",
                "pkg", "internal", "core", "server", "client"}

# Per-file caps (shallow ≠ thin, but still shallow).
_MAX_FILE_BYTES = 50 * 1024
_MAX_FILE_LINES = 200
_MAX_FILES = 30


@dataclass
class _TreeEntry:
    path: str
    size: int
    blob_type: str  # "blob" or "tree"


class GitHubFetchError(Exception):
    """Raised when a GitHub URL can't be turned into a document set."""


def _parse_github_url(url: str) -> Tuple[str, str]:
    """Return (owner, repo) from a GitHub URL. Raises on anything else."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise GitHubFetchError(f"not a github.com URL: {url!r}")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise GitHubFetchError(
            f"URL missing owner/repo: {url!r} — expected "
            "https://github.com/<owner>/<repo>"
        )
    owner, repo = parts[0], parts[1]
    # Strip a trailing .git if present.
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _gh_get_json(url: str, *, timeout: float = 15.0) -> dict:
    """GET a JSON URL against GitHub's API.

    Uses ``GITHUB_TOKEN`` if set to raise the rate-limit ceiling.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "kairo-ai-ingest",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        raise GitHubFetchError(
            f"GitHub API {exc.code} at {url}: {body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GitHubFetchError(f"network error fetching {url}: {exc}") from exc


def _gh_get_raw(
    owner: str, repo: str, ref: str, path: str, *, timeout: float = 15.0
) -> str:
    """Fetch a single raw file. Returns the text (decoded as UTF-8, replace)."""
    url = (
        "https://raw.githubusercontent.com/"
        f"{owner}/{repo}/{ref}/{urllib.parse.quote(path)}"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "kairo-ai-ingest"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(_MAX_FILE_BYTES + 1)  # read one extra to detect overflow
    return raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")


def _pick_files(tree: List[_TreeEntry]) -> List[_TreeEntry]:
    """Smart-shallow selection.

    Priority:
      1. ``README*`` at top level
      2. Any ``*.md`` at top level
      3. Top-level entry-point files (main.*, index.*, etc.)
      4. Any top-level supported source file
      5. Entry points + ``*.md`` one level deep into canonical source dirs
    Capped at ``_MAX_FILES`` total.
    """
    def _ext(p: str) -> str:
        idx = p.rfind(".")
        return p[idx:].lower() if idx >= 0 else ""

    def _base(p: str) -> str:
        name = p.rsplit("/", 1)[-1]
        idx = name.rfind(".")
        return name[:idx].lower() if idx >= 0 else name.lower()

    top_level = [e for e in tree if "/" not in e.path]
    one_deep = [
        e for e in tree
        if e.path.count("/") == 1
        and e.path.split("/", 1)[0] in _SOURCE_DIRS
    ]

    picked: List[_TreeEntry] = []
    seen: set[str] = set()

    def _add(entry: _TreeEntry) -> None:
        if entry.path in seen:
            return
        if entry.size > _MAX_FILE_BYTES:
            return
        if entry.blob_type != "blob":
            return
        seen.add(entry.path)
        picked.append(entry)

    # 1. README
    for e in top_level:
        if e.path.lower().startswith("readme"):
            _add(e)

    # 2. Top-level *.md
    for e in top_level:
        if _ext(e.path) == ".md":
            _add(e)

    # 3. Top-level entrypoints
    for e in top_level:
        if _base(e.path) in _ENTRYPOINT_BASES and _ext(e.path) in _TEXT_EXTS:
            _add(e)

    # 4. Remaining top-level text files (stable order)
    for e in top_level:
        if _ext(e.path) in _TEXT_EXTS:
            _add(e)
        if len(picked) >= _MAX_FILES:
            return picked

    # 5. One-deep: entrypoints + *.md
    for e in one_deep:
        if _ext(e.path) not in _TEXT_EXTS:
            continue
        if _base(e.path) in _ENTRYPOINT_BASES or _ext(e.path) == ".md":
            _add(e)
        if len(picked) >= _MAX_FILES:
            return picked

    # 6. Any remaining one-deep text files to fill up to the cap
    for e in one_deep:
        if _ext(e.path) in _TEXT_EXTS:
            _add(e)
        if len(picked) >= _MAX_FILES:
            return picked

    return picked


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


def _split_readme(text: str, doc_id_prefix: str) -> List[Document]:
    """Split a README on ``##``/``###`` headings.

    Produces one ``Document`` per section so the semantic layer can
    connect section-themes rather than treating the whole README as
    one opaque blob.
    """
    if not text.strip():
        return []

    # Find headings (level 2+); level 1 is typically the repo title.
    cuts: List[Tuple[int, str]] = [(0, "preamble")]
    for m in _HEADING_RE.finditer(text):
        level = len(m.group(1))
        if level < 2:
            continue
        cuts.append((m.start(), m.group(2).strip()))

    if len(cuts) == 1:
        # No sub-headings — keep the whole thing as one doc.
        return [
            Document(
                id=f"{doc_id_prefix}:full",
                text=text[:_MAX_FILE_BYTES],
                metadata={"kind": "readme", "section": "full"},
            )
        ]

    docs: List[Document] = []
    for i, (start, title) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        slug = re.sub(r"\W+", "_", title).strip("_").lower() or f"s{i}"
        docs.append(
            Document(
                id=f"{doc_id_prefix}:{slug}",
                text=body[:_MAX_FILE_BYTES],
                metadata={"kind": "readme", "section": title},
            )
        )
    return docs


def _truncate_source(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= _MAX_FILE_LINES:
        return text
    return "\n".join(lines[:_MAX_FILE_LINES]) + f"\n\n# … truncated at {_MAX_FILE_LINES} lines …"


def ingest_github(url: str) -> Tuple[List[Document], str]:
    """Pull a smart-shallow document set from a public GitHub repo.

    Returns
    -------
    (documents, label)
        ``label`` is ``"<owner>/<repo>"`` for the UI chip.
    """
    owner, repo = _parse_github_url(url)
    label = f"{owner}/{repo}"

    repo_meta = _gh_get_json(
        f"https://api.github.com/repos/{owner}/{repo}"
    )
    default_branch = repo_meta.get("default_branch") or "main"

    tree_meta = _gh_get_json(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
        f"{urllib.parse.quote(default_branch)}?recursive=1"
    )
    if tree_meta.get("truncated"):
        # Big repos — still OK, we only pick ≤30 files anyway.
        pass

    tree = [
        _TreeEntry(
            path=e.get("path", ""),
            size=int(e.get("size", 0)),
            blob_type=e.get("type", ""),
        )
        for e in tree_meta.get("tree", [])
        if e.get("path")
    ]

    picks = _pick_files(tree)
    if not picks:
        raise GitHubFetchError(
            f"no text files found in {label} (default branch {default_branch!r})"
        )

    docs: List[Document] = []
    for entry in picks:
        try:
            raw = _gh_get_raw(owner, repo, default_branch, entry.path)
        except urllib.error.HTTPError as exc:
            # Skip individual-file misses rather than failing the whole ingest.
            continue
        except urllib.error.URLError:
            continue

        doc_id_prefix = f"{label}:{entry.path}"
        if entry.path.lower().endswith(".md") or entry.path.lower().startswith("readme"):
            docs.extend(_split_readme(raw, doc_id_prefix))
        else:
            trimmed = _truncate_source(raw)
            docs.append(
                Document(
                    id=doc_id_prefix,
                    text=trimmed,
                    metadata={
                        "kind": "source",
                        "path": entry.path,
                        "repo": label,
                    },
                )
            )

    if not docs:
        raise GitHubFetchError(f"fetched 0 documents from {label}")
    return docs, label


# ---------------------------------------------------------------------------
# Tabular ingestion (CSV + XLSX)
# ---------------------------------------------------------------------------


class TabularFetchError(Exception):
    """Raised when a CSV/XLSX can't be turned into a document set."""


_MAX_ROWS = 500


def _row_to_text(row: Dict[str, object]) -> str:
    """Render a dict row as ``key: value`` lines.

    Dicts flatten nicely into an LLM prompt and are the same shape the
    embedding hash-stub can work with.
    """
    parts: List[str] = []
    for k, v in row.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def ingest_csv(data: bytes, filename: str) -> Tuple[List[Document], str, pd.DataFrame]:
    """Parse a CSV into row-documents and a DataFrame.

    First row is treated as a header; subsequent rows become one
    ``Document`` each. Empty/all-blank rows are skipped. The DataFrame
    is returned alongside so the analytics pipeline can operate on the
    same row set without re-parsing.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TabularFetchError(f"{filename!r} has no header row")
    fieldnames = list(reader.fieldnames)

    rows: List[Dict[str, Any]] = list(reader)[:_MAX_ROWS]

    docs: List[Document] = []
    for idx, row in enumerate(rows):
        body = _row_to_text(row)
        if not body:
            continue
        docs.append(
            Document(
                id=f"{filename}:row{idx}",
                text=body,
                metadata={
                    "kind": "csv_row",
                    "row_index": idx,
                    "source_file": filename,
                },
            )
        )

    if not docs:
        raise TabularFetchError(f"{filename!r} had no usable rows")

    schema_doc = Document(
        id=f"{filename}:schema",
        text="columns: " + ", ".join(fieldnames),
        metadata={"kind": "csv_schema", "source_file": filename},
    )
    df = pd.DataFrame(rows, columns=fieldnames)
    df = _coerce_numeric(df)
    return [schema_doc] + docs, filename, df


def ingest_xlsx(data: bytes, filename: str) -> Tuple[List[Document], str, pd.DataFrame]:
    """Parse an XLSX into row-documents and a concatenated DataFrame.

    Rows from every sheet are merged into a single DataFrame (with a
    ``_sheet`` column) so the analytics pipeline can operate uniformly.
    Per-sheet Documents keep the ``sheet`` metadata so RAG retrieval
    still distinguishes them.
    """
    try:
        from openpyxl import load_workbook  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TabularFetchError(
            "XLSX ingestion requires openpyxl. Install it with "
            "`pip install openpyxl` and retry."
        ) from exc

    wb = load_workbook(
        io.BytesIO(data), read_only=True, data_only=True
    )
    docs: List[Document] = []
    all_columns: List[str] = []
    merged_rows: List[Dict[str, Any]] = []

    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration:
            continue
        columns = [str(h).strip() if h is not None else f"col{i}"
                   for i, h in enumerate(header)]
        all_columns.extend(columns)

        for idx, row in enumerate(rows):
            if idx >= _MAX_ROWS:
                break
            mapping = {columns[i]: v for i, v in enumerate(row) if i < len(columns)}
            body = _row_to_text(mapping)
            if not body:
                continue
            docs.append(
                Document(
                    id=f"{filename}:{ws.title}:row{idx}",
                    text=body,
                    metadata={
                        "kind": "xlsx_row",
                        "row_index": idx,
                        "sheet": ws.title,
                        "source_file": filename,
                    },
                )
            )
            merged_rows.append({"_sheet": ws.title, **mapping})

    if not docs:
        raise TabularFetchError(f"{filename!r} had no usable rows")

    schema_doc = Document(
        id=f"{filename}:schema",
        text="columns: " + ", ".join(all_columns),
        metadata={"kind": "xlsx_schema", "source_file": filename},
    )
    df = pd.DataFrame(merged_rows)
    df = _coerce_numeric(df)
    return [schema_doc] + docs, filename, df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Promote string columns to numeric where every non-null value parses.

    csv.DictReader yields strings for every cell; pd.DataFrame preserves
    them as object-dtype. Without this pass, numeric columns like
    ``"42"`` stay as strings and the planner misclassifies them.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna() & (df[col] != "")
        if non_null.any() and (coerced.notna() & non_null).sum() / non_null.sum() >= 0.95:
            df[col] = coerced
    return df
