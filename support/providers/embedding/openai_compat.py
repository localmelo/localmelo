from __future__ import annotations

from typing import Any

import httpx

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.support.providers._http import malformed_body, post_json


class OpenAICompatEmbedding(BaseEmbeddingProvider):
    """OpenAI-compatible embedding provider.

    Works with any backend that exposes ``POST /embeddings``
    in the OpenAI format:

    - **Local**: MLC LLM, Ollama, vLLM, LMStudio
    - **Cloud**: OpenAI, Together AI, Voyage AI, etc.

    A failed request raises
    :class:`~localmelo.melo.contracts.providers.ProviderError`, mapped the
    same way as for the chat providers.

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
    transport : httpx.AsyncBaseTransport | None
        Optional HTTP transport, e.g. ``httpx.MockTransport`` in tests.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
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
            transport=transport,
            follow_redirects=False,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        path = "/embeddings"
        resp, data = await post_json(self._http, path, payload)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or not all(
            isinstance(item, dict) and "embedding" in item for item in items
        ):
            raise malformed_body(resp, path, "no data[].embedding list")
        return [item["embedding"] for item in items]

    async def close(self) -> None:
        await self._http.aclose()
