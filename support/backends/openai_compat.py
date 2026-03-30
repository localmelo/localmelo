"""Shared helpers for constructing OpenAI-compatible providers."""

from __future__ import annotations

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def normalize_url(url: str) -> str:
    """Strip trailing slash and append ``/v1`` if not already present.

    >>> normalize_url("http://localhost:11434/")
    'http://localhost:11434/v1'
    >>> normalize_url("http://localhost:11434/v1")
    'http://localhost:11434/v1'
    """
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def build_openai_chat(
    base_url: str,
    model: str,
    api_key: str | None = None,
) -> BaseLLMProvider:
    """Build an :class:`OpenAICompatLLM` with consistent URL handling."""
    return OpenAICompatLLM(base_url=base_url, model=model, api_key=api_key)


def build_openai_embedding(
    base_url: str,
    model: str,
    api_key: str | None = None,
) -> BaseEmbeddingProvider:
    """Build an :class:`OpenAICompatEmbedding` with consistent URL handling."""
    return OpenAICompatEmbedding(base_url=base_url, model=model, api_key=api_key)
