"""vLLM backend adapter -- external OpenAI-compatible endpoint connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.support.backends.base import BaseBackend
from localmelo.support.backends.openai_compat import (
    build_openai_chat,
    build_openai_embedding,
    normalize_url,
)

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config


class VllmBackend(BaseBackend):
    """Backend adapter for vLLM (external OpenAI-compatible endpoint)."""

    @property
    def key(self) -> str:
        return "vllm"

    @property
    def display_name(self) -> str:
        return "vLLM"

    def validate(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.vllm.chat_url:
            errors.append(
                "[vllm] chat_url is empty. "
                "Set it to your vLLM server URL (e.g. http://localhost:8000)."
            )
        if not cfg.vllm.chat_model:
            errors.append("[vllm] chat_model is empty. " "Specify the vLLM model name.")
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.vllm.embedding_url:
            errors.append(
                "[vllm] embedding_url is empty. "
                "Set it to your vLLM server URL for embedding."
            )
        if not cfg.vllm.embedding_model:
            errors.append(
                "[vllm] embedding_model is empty. " "Set a vLLM embedding model name."
            )
        return errors

    def has_embedding(self, cfg: Config) -> bool:
        return bool(cfg.vllm.embedding_url and cfg.vllm.embedding_model)

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        return build_openai_chat(
            base_url=normalize_url(cfg.vllm.chat_url),
            model=cfg.vllm.chat_model,
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        if cfg.vllm.embedding_url and cfg.vllm.embedding_model:
            return build_openai_embedding(
                base_url=normalize_url(cfg.vllm.embedding_url),
                model=cfg.vllm.embedding_model,
            )
        return None
