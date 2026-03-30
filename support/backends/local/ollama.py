"""Ollama backend adapter -- external OpenAI-compatible endpoint connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.support.backends.base import BaseBackend
from localmelo.support.backends.openai_compat import (
    build_openai_embedding,
    normalize_url,
)
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config


class OllamaBackend(BaseBackend):
    """Backend adapter for Ollama (external OpenAI-compatible endpoint)."""

    @property
    def key(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    def validate(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.ollama.chat_url:
            errors.append(
                "[ollama] chat_url is empty. "
                "Set it to your Ollama server URL (e.g. http://localhost:11434)."
            )
        if not cfg.ollama.chat_model:
            errors.append(
                "[ollama] chat_model is empty. "
                "Specify the Ollama model name (e.g. qwen3:8b)."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.ollama.embedding_url:
            errors.append(
                "[ollama] embedding_url is empty. "
                "Set it to your Ollama server URL for embedding."
            )
        if not cfg.ollama.embedding_model:
            errors.append(
                "[ollama] embedding_model is empty. "
                "Set an Ollama embedding model name to use Ollama for embedding."
            )
        return errors

    def has_embedding(self, cfg: Config) -> bool:
        return bool(cfg.ollama.embedding_url and cfg.ollama.embedding_model)

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        # Use the native Ollama /api/chat endpoint (not OpenAI-compat)
        # so that extended-thinking (think: true) works natively.
        # The URL must NOT include /v1.
        url = cfg.ollama.chat_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return OllamaNativeChat(
            base_url=url,
            model=cfg.ollama.chat_model,
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        if cfg.ollama.embedding_url and cfg.ollama.embedding_model:
            return build_openai_embedding(
                base_url=normalize_url(cfg.ollama.embedding_url),
                model=cfg.ollama.embedding_model,
            )
        return None
