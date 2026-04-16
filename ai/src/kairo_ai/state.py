"""In-memory thread store — placeholder until persistence lands.

Thread state lives here while the engine side is still being built.
Swapping for a ``KairoStore``-backed implementation is a one-class
change; the rest of the app only depends on :class:`ThreadStore`.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .models import Message, Thread


class ThreadStore:
    def __init__(self) -> None:
        self._threads: Dict[str, Thread] = {}
        self._lock = threading.Lock()

    def create(self, *, title: Optional[str] = None, plugin: Optional[str] = None) -> Thread:
        thread = Thread(
            id=uuid.uuid4().hex,
            title=title or "New thread",
            created_at=datetime.now(timezone.utc),
            plugin=plugin,
        )
        with self._lock:
            self._threads[thread.id] = thread
        return thread

    def get(self, thread_id: str) -> Optional[Thread]:
        with self._lock:
            return self._threads.get(thread_id)

    def list(self) -> List[Thread]:
        with self._lock:
            return sorted(
                self._threads.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )

    def append_message(self, thread_id: str, message: Message) -> Thread:
        with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            thread.messages.append(message)
            if thread.title == "New thread" and message.role == "user":
                thread.title = _summarize(message.content)
            return thread


def _summarize(text: str, *, limit: int = 60) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet or "New thread"
    return snippet[: limit - 1].rstrip() + "…"
