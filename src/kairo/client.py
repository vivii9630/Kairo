from __future__ import annotations

from typing import List, Optional

import httpx

from .models import Document, QueryRequest, QueryResponse


class HostedClient:
    """Thin HTTP client for a hosted Kairo service.

    Replace endpoint paths with your deployed API.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def ingest(self, documents: List[Document]) -> None:
        payload = [doc.model_dump() for doc in documents]
        resp = self._client.post(f"{self.base_url}/v1/ingest", json=payload, headers=self._headers())
        resp.raise_for_status()

    def query(self, request: QueryRequest) -> QueryResponse:
        resp = self._client.post(
            f"{self.base_url}/v1/query",
            json=request.model_dump(),
            headers=self._headers(),
        )
        resp.raise_for_status()
        return QueryResponse.model_validate(resp.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

