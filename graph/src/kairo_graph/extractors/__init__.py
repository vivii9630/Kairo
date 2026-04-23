"""Pluggable relationship extractors.

The :class:`ExtractorProvider` protocol keeps :class:`LLMExtractedGraphBuilder`
agnostic to which backend runs. Default is Ollama (local-first). Cloud
API providers (Claude / OpenAI / Anthropic SDK) can ship as separate
subclasses without touching the builder.
"""

from .ollama import OllamaExtractor
from .protocol import ExtractorProvider

__all__ = ["ExtractorProvider", "OllamaExtractor"]
