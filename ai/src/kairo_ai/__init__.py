"""Kairo AI: backend orchestrator for the chat-native UI.

Thin FastAPI shell that exposes connector manifests, a stubbed ``/ask``
endpoint, and in-memory thread storage. The real answer path lands once
the Phase 7 ``TemporalRAG`` orchestrator is in place; until then the
stubs return realistic shapes so the frontend can be built against
stable contracts.
"""

from .app import create_app

__all__ = ["create_app"]
