"""MLC backend adapter -- external OpenAI-compatible endpoint connector."""

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


class MlcBackend(BaseBackend):
    """Backend adapter for MLC-LLM (external OpenAI-compatible endpoint)."""

    @property
    def key(self) -> str:
        return "mlc"

    @property
    def display_name(self) -> str:
        return "MLC-LLM"

    def validate(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.mlc.chat_url:
            errors.append(
                "[mlc] chat_url is empty. "
                "Set it to your MLC-LLM server URL (e.g. http://127.0.0.1:8400)."
            )
        if not cfg.mlc.chat_model:
            errors.append(
                "[mlc] chat_model is empty. "
                "Run 'melo --reconfigure' to select a model."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.mlc.embedding_url:
            errors.append(
                "[mlc] embedding_url is empty. "
                "Set it to your MLC-LLM server URL for embedding."
            )
        if not cfg.mlc.embedding_model:
            errors.append(
                "[mlc] embedding_model is empty. "
                "Set an MLC-LLM embedding model name."
            )
        return errors

    def has_embedding(self, cfg: Config) -> bool:
        return bool(cfg.mlc.embedding_url and cfg.mlc.embedding_model)

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        return build_openai_chat(
            base_url=normalize_url(cfg.mlc.chat_url),
            model=cfg.mlc.chat_model,
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        if cfg.mlc.embedding_url and cfg.mlc.embedding_model:
            return build_openai_embedding(
                base_url=normalize_url(cfg.mlc.embedding_url),
                model=cfg.mlc.embedding_model,
            )
        return None
