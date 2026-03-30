from __future__ import annotations

from typing import Any

import httpx

from localmelo.melo.contracts.providers import BaseEmbeddingProvider


class OpenAICompatEmbedding(BaseEmbeddingProvider):
    """OpenAI-compatible embedding provider.

    Works with any backend that exposes ``POST /embeddings``
    in the OpenAI format:

    - **Local**: MLC LLM, Ollama, vLLM, LMStudio
    - **Cloud**: OpenAI, Together AI, Voyage AI, etc.

    Parameters
    ----------
    base_url : str
        Server base URL including ``/v1``.
    model : str
        Model identifier sent in the ``"model"`` field.
    api_key : str | None
        Bearer token for authenticated endpoints.
    timeout : float
        HTTP timeout in seconds (default 120).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        resp = await self._http.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self) -> None:
        await self._http.aclose()
