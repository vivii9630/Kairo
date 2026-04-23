"""Ollama LLM adapter for kairo-agents.

Produces a :data:`ThinkFn` that calls a local Ollama instance via its
HTTP API (``/api/generate`` or ``/api/chat``).  No SDK dependency —
uses stdlib ``urllib`` so kairo-agents stays lightweight.

Usage::

    from kairo_agents.providers.ollama import ollama_think_fn, OllamaConfig

    think = ollama_think_fn(OllamaConfig(model="qwen2.5:1.5b"))
    agent = Agent("analyst", "analyst", "you are...", think_fn=think)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class OllamaConfig:
    """Configuration for the Ollama provider.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. ``"qwen2.5:1.5b"``, ``"llama3.2:1b"``.
    base_url : str
        Ollama HTTP endpoint (default ``http://localhost:11434``).
    temperature : float
        Sampling temperature (0.0 = deterministic).
    max_tokens : int
        Maximum tokens in the response.
    system_prefix : str
        Optional text prepended to every system message.
    options : dict
        Extra Ollama API options (``num_ctx``, ``top_p``, etc.).
    """

    model: str = "qwen2.5:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 2048
    system_prefix: str = ""
    options: Dict[str, Any] = field(default_factory=dict)


def ollama_think_fn(config: Optional[OllamaConfig] = None):
    """Return a :data:`ThinkFn` that calls Ollama's ``/api/chat`` endpoint.

    The returned callable has signature ``(role, instruction, user_input) -> str``
    matching :class:`kairo_agents.Agent`'s ``think_fn`` contract.

    Temperature defaults to 0.0 for deterministic output.
    """
    cfg = config or OllamaConfig()

    def _think(role: str, instruction: str, user_input: str) -> str:
        system_msg = cfg.system_prefix
        if system_msg:
            system_msg += "\n\n"
        system_msg += f"You are a {role}.\n{instruction}"

        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_input},
            ],
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                **cfg.options,
            },
        }

        url = f"{cfg.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("message", {}).get("content", "")
        except urllib.error.URLError as exc:
            return (
                f"[Ollama error] Could not reach {cfg.base_url}: {exc}. "
                f"Is Ollama running? Try: ollama serve"
            )
        except Exception as exc:
            return f"[Ollama error] {type(exc).__name__}: {exc}"

    return _think


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is reachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_ollama_models(
    base_url: str = "http://localhost:11434",
) -> list:
    """Return list of locally available model names."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Streaming (Phase 13d)
# ---------------------------------------------------------------------------


def ollama_stream_chat(
    messages: List[Dict[str, str]],
    *,
    config: Optional[OllamaConfig] = None,
    timeout: int = 120,
) -> Iterator[Tuple[str, str]]:
    """Stream a chat completion from Ollama as ``(kind, content)`` tuples.

    ``kind`` is one of:
      - ``"token"`` — an incremental piece of assistant text. Emit to
        the client as it arrives for the progressive-render UX.
      - ``"done"`` — terminal marker with the full concatenated answer
        as ``content``. Consumers should stop iterating after this.
      - ``"error"`` — transport or parse failure. ``content`` is the
        human-readable message.

    Emits roughly one ``token`` event per Ollama JSON chunk; Ollama
    tends to send a few characters per chunk so the client can render
    at a natural cadence without additional buffering.
    """
    cfg = config or OllamaConfig()
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            **cfg.options,
        },
    }
    url = f"{cfg.base_url}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    accumulated: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines — Ollama occasionally sends
                    # keepalives that aren't valid JSON.
                    continue
                piece = str(
                    (chunk.get("message") or {}).get("content", "")
                )
                if piece:
                    accumulated.append(piece)
                    yield ("token", piece)
                if chunk.get("done"):
                    break
    except urllib.error.URLError as exc:
        yield (
            "error",
            f"Ollama transport error: {exc}. Is Ollama running at "
            f"{cfg.base_url}?",
        )
        return
    except Exception as exc:  # noqa: BLE001 — surface to caller
        yield ("error", f"{type(exc).__name__}: {exc}")
        return

    yield ("done", "".join(accumulated))
