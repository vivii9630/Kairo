"""Tests for the `name="default"` resolution in get_provider."""

from __future__ import annotations

import pytest

from kairo_embeddings import available_providers, get_provider
from kairo_embeddings.registry import _BUILTIN, _DEFAULT_PRIORITY


def test_default_never_resolves_to_hash_stub() -> None:
    """`default` must not silently degrade to hash-stub."""
    assert "hash-stub" not in _DEFAULT_PRIORITY


def test_default_prefers_sentence_transformers_when_available() -> None:
    """If ST is importable, `default` should hand it back."""
    if "sentence-transformers" not in available_providers():
        pytest.skip("sentence-transformers not installed")
    provider = get_provider("default")
    assert provider.name == "sentence-transformers"
    assert provider.dim == 384  # all-MiniLM-L6-v2


def test_default_raises_when_no_semantic_provider_installed() -> None:
    """Simulate the no-ST case by temporarily stripping it from the registry."""
    original = _BUILTIN.copy()
    for name in _DEFAULT_PRIORITY:
        _BUILTIN.pop(name, None)
    try:
        with pytest.raises(ImportError, match="No real semantic embedding provider"):
            get_provider("default")
    finally:
        _BUILTIN.clear()
        _BUILTIN.update(original)


def test_hash_stub_still_available_by_explicit_name() -> None:
    """Hash-stub is still usable for CI / plumbing — just not as `default`."""
    provider = get_provider("hash-stub", dim=32)
    assert provider.name == "hash-stub"
    assert provider.dim == 32
