"""Kairo embeddings: pluggable providers and a numpy-backed store."""

from .credentials import Credentials
from .provider import EmbeddingProvider
from .registry import available_providers, get_provider, register_provider
from .store import EmbeddingStore

__all__ = [
    "EmbeddingProvider",
    "EmbeddingStore",
    "Credentials",
    "get_provider",
    "register_provider",
    "available_providers",
]
