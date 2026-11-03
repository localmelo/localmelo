"""OpenAI cloud backend adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.support import hosted
from localmelo.support.backends.base import BaseBackend
from localmelo.support.backends.openai_compat import build_openai_chat

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIBackend(BaseBackend):
    """Backend adapter for the OpenAI API."""

    @property
    def key(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        """The service this section's address names: DeepSeek, OpenAI or its host."""
        return hosted.label_for("openai", cfg.openai)

    def validate(self, cfg: Config) -> list[str]:
        errors = hosted.key_errors("openai", cfg.openai, "OPENAI_API_KEY")
        if not cfg.openai.chat_model:
            errors.append(
                "[openai] chat_model is empty. " "Specify the model name (e.g. gpt-4o)."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        return ["openai is a chat-only backend and cannot be used for embedding."]

    def has_embedding(self, cfg: Config) -> bool:
        return False

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        base_url = cfg.openai.base_url or _DEFAULT_BASE_URL
        api_key = hosted.api_key(cfg.openai)
        return build_openai_chat(
            base_url=base_url,
            model=cfg.openai.chat_model,
            api_key=api_key,
            extra_body=hosted.request_body("openai", cfg.openai),
            thinking_off_body=hosted.retry_body("openai", cfg.openai),
            timeout=cfg.request_timeout_s,
            reasoning_field=getattr(cfg.openai, "reasoning_field", "") or None,
            requires_reasoning_replay=getattr(
                cfg.openai, "requires_reasoning_replay", False
            ),
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None
