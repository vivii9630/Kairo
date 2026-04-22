from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .credentials import Credentials
from .provider import EmbeddingProvider
from .providers.hash_stub import HashStubProvider


_BUILTIN: Dict[str, Type] = {
    "hash-stub": HashStubProvider,
}

# Priority list used to resolve name="default" — first importable wins.
# Kept narrow on purpose: "default" means "the best real semantic embedding
# available locally", never the hash-stub (which carries no semantic signal
# and is only there for CI / offline smoke tests).
_DEFAULT_PRIORITY: List[str] = ["sentence-transformers"]


def _register_optional_providers() -> None:
    """Lazy-register providers whose deps come from optional extras.

    A broad except is intentional: optional providers can fail to import
    not just on missing-package (ImportError) but on broken transitive
    deps (e.g. protobuf TypeError, accelerate/wandb chain errors). The
    contract is "if this provider can't load cleanly, pretend it isn't
    installed" — we'd rather degrade to hash-stub than crash registry
    initialization for every consumer.
    """
    try:
        from .providers.sentence_transformers import SentenceTransformersProvider

        _BUILTIN["sentence-transformers"] = SentenceTransformersProvider
    except Exception:  # noqa: BLE001 - see docstring
        pass


_register_optional_providers()


def _resolve_default() -> str:
    for name in _DEFAULT_PRIORITY:
        if name in _BUILTIN:
            return name
    raise ImportError(
        "No real semantic embedding provider is installed. "
        'Install one with: pip install -e "./embeddings[sentence-transformers]" '
        '(or pass name="hash-stub" explicitly for CI/offline plumbing).'
    )


def get_provider(
    name: str,
    *,
    credentials: Optional[Credentials] = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Resolve an embedding provider by name.

    Passing ``name="default"`` returns the best real semantic provider
    currently importable (priority: sentence-transformers). It never
    silently falls back to the hash-stub — ask for ``"hash-stub"`` by
    name if that's what you actually want.

    Examples::

        get_provider("default")                              # preferred
        get_provider("hash-stub", dim=64)                    # CI/offline
        get_provider("sentence-transformers", model="all-MiniLM-L6-v2")
        get_provider("openai", model="text-embedding-3-small",
                     credentials=Credentials(api_key=os.environ["OPENAI_API_KEY"]))
    """
    if name == "default":
        name = _resolve_default()
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
