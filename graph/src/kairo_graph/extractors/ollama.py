"""Ollama-backed relationship extractor.

Talks to a local Ollama server over HTTP (stdlib ``urllib`` — no extra
deps beyond what's already in the monorepo). Prompts the model for a
strict JSON envelope of edges + concepts, parses with Pydantic, and
returns an :class:`ExtractionResult`.

The extractor is tolerant: malformed JSON, missing fields, or HTTP
failures produce an :class:`ExtractionResult` with ``error`` set and
empty ``edges`` / ``concepts`` — the calling builder logs and keeps
going rather than failing the whole corpus on one bad document.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from pydantic import ValidationError

from kairo_core import (
    Document,
    ExtractedConcept,
    ExtractedEdge,
    ExtractionResult,
)


DEFAULT_MODEL = "llama3.2"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 60


_SYSTEM_PROMPT = (
    "You are a careful knowledge-graph extractor. From the provided "
    "document, identify meaningful concepts and relationships. "
    "Respond with a single JSON object. Do not wrap it in markdown "
    "fences. The object has two keys: 'concepts' (list) and 'edges' "
    "(list). Each concept: {id, label, description, confidence}. "
    "Each edge: {subject, relation, object, provenance, confidence, "
    "rationale}. 'provenance' is one of 'extracted' (stated directly in "
    "text), 'inferred' (reasonable inference), or 'ambiguous' (unsure). "
    "'confidence' is a float in [0,1]. Use concept ids for subject/"
    "object when applicable. Output ONLY the JSON."
)


def _extract_json_block(text: str) -> Optional[str]:
    """Best-effort recovery when the model wraps JSON in prose or fences."""
    text = text.strip()
    if not text:
        return None
    # Strip code fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Otherwise carve out the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


class OllamaExtractor:
    """Local relationship extractor using an Ollama model."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        system_prompt: str = _SYSTEM_PROMPT,
        context_char_budget: int = 4000,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("KAIRO_OLLAMA_HOST", DEFAULT_HOST)
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.context_char_budget = context_char_budget
        self.name = f"ollama:{model}"

    # ------------------------------------------------------------------
    # ExtractorProvider interface
    # ------------------------------------------------------------------

    def extract(
        self,
        document: Document,
        context: Sequence[Document] = (),
    ) -> ExtractionResult:
        user_prompt = self._build_user_prompt(document, context)
        try:
            raw = self._ollama_chat(self.system_prompt, user_prompt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ExtractionResult(
                extractor=self.name,
                error=f"ollama transport error: {exc}",
            )

        parsed = self._parse_response(raw, document_id=document.id)
        return parsed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        document: Document,
        context: Sequence[Document],
    ) -> str:
        body = document.text[: self.context_char_budget]
        parts = [f"Document id: {document.id}", "", body]
        if context:
            ctx_lines = ["", "Sibling concept context (for reference):"]
            budget = self.context_char_budget
            for sibling in context:
                snippet = sibling.text[:200]
                line = f"- {sibling.id}: {snippet}"
                if budget - len(line) < 0:
                    break
                budget -= len(line)
                ctx_lines.append(line)
            parts.extend(ctx_lines)
        return "\n".join(parts)

    def _ollama_chat(self, system: str, user: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        message = data.get("message") or {}
        return str(message.get("content", ""))

    def _parse_response(
        self,
        raw: str,
        *,
        document_id: str,
    ) -> ExtractionResult:
        block = _extract_json_block(raw)
        if block is None:
            return ExtractionResult(
                extractor=self.name,
                error="no JSON object found in model output",
            )
        try:
            obj = json.loads(block)
        except json.JSONDecodeError as exc:
            return ExtractionResult(
                extractor=self.name,
                error=f"invalid JSON: {exc}",
            )
        if not isinstance(obj, dict):
            return ExtractionResult(
                extractor=self.name,
                error="top-level JSON was not an object",
            )

        concepts: List[ExtractedConcept] = []
        for raw_concept in obj.get("concepts") or []:
            if not isinstance(raw_concept, dict):
                continue
            raw_concept.setdefault("source_doc_id", document_id)
            try:
                concepts.append(ExtractedConcept(**raw_concept))
            except ValidationError:
                continue

        edges: List[ExtractedEdge] = []
        for raw_edge in obj.get("edges") or []:
            if not isinstance(raw_edge, dict):
                continue
            raw_edge.setdefault("source_doc_id", document_id)
            try:
                edges.append(ExtractedEdge(**raw_edge))
            except ValidationError:
                continue

        return ExtractionResult(
            extractor=self.name,
            concepts=concepts,
            edges=edges,
        )
