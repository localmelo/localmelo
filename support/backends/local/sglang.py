"""SGLang backend adapter -- external OpenAI-compatible endpoint connector."""

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


class SglangBackend(BaseBackend):
    """Backend adapter for SGLang (external OpenAI-compatible endpoint)."""

    @property
    def key(self) -> str:
        return "sglang"

    @property
    def display_name(self) -> str:
        return "SGLang"

    def validate(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.sglang.chat_url:
            errors.append(
                "[sglang] chat_url is empty. "
                "Set it to your SGLang server URL (e.g. http://localhost:30000)."
            )
        if not cfg.sglang.chat_model:
            errors.append(
                "[sglang] chat_model is empty. " "Specify the SGLang model name."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.sglang.embedding_url:
            errors.append(
                "[sglang] embedding_url is empty. "
                "Set it to your SGLang server URL for embedding."
            )
        if not cfg.sglang.embedding_model:
            errors.append(
                "[sglang] embedding_model is empty. "
                "Set an SGLang embedding model name."
            )
        return errors

    def has_embedding(self, cfg: Config) -> bool:
        return bool(cfg.sglang.embedding_url and cfg.sglang.embedding_model)

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        return build_openai_chat(
            base_url=normalize_url(cfg.sglang.chat_url),
            model=cfg.sglang.chat_model,
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        if cfg.sglang.embedding_url and cfg.sglang.embedding_model:
            return build_openai_embedding(
                base_url=normalize_url(cfg.sglang.embedding_url),
                model=cfg.sglang.embedding_model,
            )
        return None
