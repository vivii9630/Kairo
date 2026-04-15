from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .credentials import Credentials
from .provider import EmbeddingProvider
from .providers.hash_stub import HashStubProvider


_BUILTIN: Dict[str, Type] = {
    "hash-stub": HashStubProvider,
}


def _register_optional_providers() -> None:
    """Lazy-register providers whose deps come from optional extras."""
    try:
        from .providers.sentence_transformers import SentenceTransformersProvider

        _BUILTIN["sentence-transformers"] = SentenceTransformersProvider
    except ImportError:
        pass


_register_optional_providers()


def get_provider(
    name: str,
    *,
    credentials: Optional[Credentials] = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Resolve an embedding provider by name.

    Examples::

        get_provider("hash-stub", dim=64)
        get_provider("sentence-transformers", model="all-MiniLM-L6-v2")
        get_provider("openai", model="text-embedding-3-small",
                     credentials=Credentials(api_key=os.environ["OPENAI_API_KEY"]))
    """
    if name not in _BUILTIN:
        available = ", ".join(sorted(_BUILTIN)) or "<none>"
        raise ValueError(
            f"Unknown embedding provider: {name!r}. Available: {available}. "
            f"Providers requiring optional extras (sentence-transformers, openai, "
            f"cohere, voyage, bedrock) must have their extras installed first, e.g. "
            f'pip install -e "./embeddings[sentence-transformers]"'
        )
    cls = _BUILTIN[name]
    return cls(credentials=credentials, **kwargs)


def register_provider(name: str, cls: Type) -> None:
    """Register a user-defined embedding provider into the registry."""
    _BUILTIN[name] = cls


def available_providers() -> List[str]:
    """Return the names of all currently registered providers."""
    return sorted(_BUILTIN)
