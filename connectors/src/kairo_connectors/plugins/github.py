from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List

from ..plugin import FetchResult, PluginManifest


_GITHUB_URL_RE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)"
    r"(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+?)(?:\.git)?/?$"
)


class GitHubConnector:
    """Shallow-clones a public GitHub repository into a temp dir.

    Uses the ``git`` CLI via subprocess rather than a Python binding so we
    don't add a new dependency for a feature every developer already has.
    Private repos will work if the user's local git is already authed
    (SSH key, credential helper, PAT in remote URL) — the connector stays
    out of credentials, which will flow through ``kairo-auth`` later.
    """

    manifest = PluginManifest(
        name="github",
        label="GitHub",
        description="Ingest a public GitHub repository by URL.",
        uri_example="https://github.com/owner/repo",
        icon="github",
        tags=["code", "vcs"],
    )

    def __init__(self, *, depth: int = 1, timeout: float = 120.0):
        self.depth = int(depth)
        self.timeout = float(timeout)

    def fetch(self, uri: str, **kwargs: Any) -> FetchResult:
        owner, repo = _parse_github_url(uri)
        tmp_root = Path(tempfile.mkdtemp(prefix=f"kairo_github_{owner}_{repo}_"))
        try:
            self._clone(uri, tmp_root)
        except Exception:
            _force_rmtree(tmp_root)
            raise

        files = _walk_files(tmp_root)
        metadata = {
            "connector": "github",
            "owner": owner,
            "repo": repo,
            "depth": self.depth,
            "file_count": len(files),
        }

        def cleanup() -> None:
            _force_rmtree(tmp_root)

        return FetchResult(
            root=tmp_root,
            files=files,
            source_uri=uri,
            metadata=metadata,
            cleanup=cleanup,
        )

    def _clone(self, uri: str, dest: Path) -> None:
        cmd = [
            "git",
            "clone",
            "--depth",
            str(self.depth),
            "--quiet",
            uri,
            str(dest),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {uri!r}: {result.stderr.strip() or 'unknown error'}"
            )


def _parse_github_url(uri: str) -> tuple[str, str]:
    match = _GITHUB_URL_RE.match(uri.strip())
    if not match:
        raise ValueError(
            f"Not a recognizable GitHub URL: {uri!r}. "
            f"Expected https://github.com/owner/repo or git@github.com:owner/repo.git"
        )
    return match.group("owner"), match.group("repo")


def _force_rmtree(path: Path) -> None:
    """``shutil.rmtree`` that clears read-only bits.

    Git on Windows writes pack/object files with the read-only attribute,
    which makes the default ``rmtree`` fail with ``PermissionError``. The
    fix (used by pip, cookiecutter, and git itself) is to chmod and retry.
    """
    if not path.exists():
        return

    def _onerror(func, target, exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_onerror)


def _walk_files(root: Path, *, skip_git_dir: bool = True) -> List[Path]:
    """Return all regular files under ``root``, skipping ``.git/`` by default."""
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if skip_git_dir and ".git" in p.relative_to(root).parts:
            continue
        out.append(p)
    out.sort()
    return out
