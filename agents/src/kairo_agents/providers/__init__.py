"""LLM provider adapters for kairo-agents ThinkFn."""

from .ollama import ollama_think_fn, OllamaConfig

__all__ = ["ollama_think_fn", "OllamaConfig"]
