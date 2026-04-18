"""Tavily search adapter for kairo-agents.

Provides a stdlib-only client for Tavily's `/search` endpoint.  A free
API key (tvly-...) is required and must be passed explicitly via
:class:`TavilyConfig` or placed in the ``TAVILY_API_KEY`` environment
variable (read by callers, not here).

We intentionally avoid the ``tavily-python`` SDK — urllib keeps the
package dependency-free and matches the philosophy of
:mod:`kairo_agents.providers.ollama`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TavilyConfig:
    """Configuration for the Tavily search provider.

    Parameters
    ----------
    api_key : str
        Tavily API key (``tvly-...``).
    base_url : str
        Tavily API endpoint.
    max_results : int
        Cap on result count per query.  Free tier supports up to ~10.
    search_depth : str
        ``"basic"`` (fast, shallow) or ``"advanced"`` (slower, deeper
        content snippets).
    include_answer : bool
        Ask Tavily to synthesize a short answer — we ignore the answer
        text but it costs no extra credits.
    timeout : float
        HTTP timeout in seconds.
    """

    api_key: str
    base_url: str = "https://api.tavily.com"
    max_results: int = 5
    search_depth: str = "basic"
    include_answer: bool = False
    timeout: float = 20.0
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TavilyResult:
    """One normalized search hit."""

    url: str
    title: str
    content: str
    score: float = 0.0


class TavilyError(RuntimeError):
    """Raised when Tavily's API is unreachable or returns a non-2xx."""


class TavilyClient:
    """Minimal Tavily HTTP client — only the ``/search`` endpoint.

    Call sites use :meth:`search` to get a normalized list of hits; the
    web-enrichment pass translates each hit into a detail-layer graph
    node so the 3D viz shows web-sourced context alongside local data.
    """

    def __init__(self, config: TavilyConfig) -> None:
        if not config.api_key:
            raise TavilyError("TavilyConfig.api_key is required")
        self.config = config

    def search(self, query: str) -> List[TavilyResult]:
        """Return a list of :class:`TavilyResult` for *query*.

        Empty queries return an empty list; network failures raise
        :class:`TavilyError` so the caller can downgrade gracefully.
        """
        q = (query or "").strip()
        if not q:
            return []

        payload: Dict[str, Any] = {
            "api_key": self.config.api_key,
            "query": q,
            "max_results": self.config.max_results,
            "search_depth": self.config.search_depth,
            "include_answer": self.config.include_answer,
        }
        payload.update(self.config.options)

        url = f"{self.config.base_url.rstrip('/')}/search"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise TavilyError(
                f"Tavily HTTP {exc.code}: {exc.reason} {detail}".strip()
            ) from exc
        except urllib.error.URLError as exc:
            raise TavilyError(f"Tavily unreachable: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise TavilyError(
                f"Tavily unexpected error: {type(exc).__name__}: {exc}"
            ) from exc

        hits_raw = body.get("results") or []
        results: List[TavilyResult] = []
        for h in hits_raw:
            results.append(
                TavilyResult(
                    url=str(h.get("url", "")),
                    title=str(h.get("title", "")).strip() or str(h.get("url", "")),
                    content=str(h.get("content", "")).strip(),
                    score=float(h.get("score", 0.0) or 0.0),
                )
            )
        return results


def tavily_client_from_env(
    *,
    env_var: str = "TAVILY_API_KEY",
    **overrides: Any,
) -> Optional[TavilyClient]:
    """Build a :class:`TavilyClient` from an env var, or return ``None``.

    Returns ``None`` when the env var is unset so callers can skip
    web-enrichment cleanly in offline mode.
    """
    import os

    key = os.environ.get(env_var, "").strip()
    if not key:
        return None
    cfg = TavilyConfig(api_key=key, **overrides)
    return TavilyClient(cfg)
