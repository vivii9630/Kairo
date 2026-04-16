from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from ..auth import AuthMethod, AuthRequirement, BearerTokenAuth
from ..plugin import FetchResult, FetchSpec, PluginManifest


_GITHUB_URL_RE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)"
    r"(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+?)(?:\.git)?/?$"
)


@dataclass
class GitHubFetchSpec(FetchSpec):
    """Fetch spec for the GitHub connector.

    ``uri`` is required. ``ref`` selects a branch, tag, or commit; if
    omitted, git's default HEAD is cloned. ``depth`` is passed straight
    to ``git clone --depth`` (1 is usually what you want for snapshotting).
    """

    uri: str = ""
    ref: Optional[str] = None
    depth: int = 1


class GitHubConnector:
    """Shallow-clones a GitHub repository into a temp dir.

    Uses the ``git`` CLI via subprocess rather than a Python binding so we
    don't add a new dependency for a feature every developer already has.

    Public repos: no auth. Private repos: pass a :class:`BearerTokenAuth`
    holding a personal-access token (or fine-grained token) — the plugin
    injects it into the clone URL as ``x-access-token``. Callers can also
    pre-auth their local git (SSH key, credential helper) and skip the
    ``auth`` argument entirely.
    """

    manifest = PluginManifest(
        name="github",
        label="GitHub",
        description="Ingest a GitHub repository by URL (public, or private with a PAT).",
        uri_example="https://github.com/owner/repo",
        auth=AuthRequirement(
            methods=[BearerTokenAuth],
            instructions=(
                "Optional. Provide a GitHub Personal Access Token with 'repo' "
                "scope to clone private repositories. Public repos need no auth."
            ),
        ),
        icon="github",
        tags=["code", "vcs"],
    )

    def __init__(self, *, timeout: float = 120.0):
        self.timeout = float(timeout)

    def fetch(
        self,
        spec: FetchSpec,
        *,
        auth: Optional[AuthMethod] = None,
        **kwargs: Any,
    ) -> FetchResult:
        if not isinstance(spec, GitHubFetchSpec):
            raise TypeError(
                f"GitHubConnector expects GitHubFetchSpec, got {type(spec).__name__}"
            )
        if not spec.uri:
            raise ValueError("GitHubFetchSpec.uri is required")
        if auth is not None and not isinstance(auth, BearerTokenAuth):
            raise ValueError(
                "GitHubConnector accepts BearerTokenAuth only; "
                f"got {type(auth).__name__}"
            )

        owner, repo = _parse_github_url(spec.uri)
        tmp_root = Path(tempfile.mkdtemp(prefix=f"kairo_github_{owner}_{repo}_"))
        try:
            self._clone(spec, auth, tmp_root)
        except Exception:
            _force_rmtree(tmp_root)
            raise

        files = _walk_files(tmp_root)
        metadata = {
            "connector": "github",
            "owner": owner,
            "repo": repo,
            "ref": spec.ref,
            "depth": spec.depth,
            "file_count": len(files),
            "authenticated": auth is not None,
        }

        def cleanup() -> None:
            _force_rmtree(tmp_root)

        return FetchResult(
            source_uri=spec.uri,
            metadata=metadata,
            root=tmp_root,
            files=files,
            cleanup=cleanup,
        )

    def _clone(
        self,
        spec: GitHubFetchSpec,
        auth: Optional[BearerTokenAuth],
        dest: Path,
    ) -> None:
        clone_url = _inject_token(spec.uri, auth)
        cmd: List[str] = [
            "git",
            "clone",
            "--depth",
            str(spec.depth),
            "--quiet",
        ]
        if spec.ref:
            cmd.extend(["--branch", spec.ref])
        cmd.extend([clone_url, str(dest)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {spec.uri!r}: "
                f"{result.stderr.strip() or 'unknown error'}"
            )


def _inject_token(uri: str, auth: Optional[BearerTokenAuth]) -> str:
    """Return a clone URL with an auth token injected, if one was supplied.

    Only HTTPS URLs get rewritten; SSH URLs are assumed to rely on the
    user's existing SSH key setup.
    """
    if auth is None or not uri.startswith("http"):
        return uri
    return uri.replace("https://", f"https://x-access-token:{auth.token}@", 1)


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
