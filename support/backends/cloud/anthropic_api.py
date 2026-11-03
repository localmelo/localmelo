"""Anthropic cloud backend adapter."""

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

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


class AnthropicBackend(BaseBackend):
    """Backend adapter for the Anthropic API (OpenAI-compatible bridge)."""

    @property
    def key(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic"

    def validate(self, cfg: Config) -> list[str]:
        errors = hosted.key_errors("anthropic", cfg.anthropic, "ANTHROPIC_API_KEY")
        if not cfg.anthropic.chat_model:
            errors.append(
                "[anthropic] chat_model is empty. "
                "Specify the model name (e.g. claude-sonnet-4-20250514)."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        return ["anthropic is a chat-only backend and cannot be used for embedding."]

    def has_embedding(self, cfg: Config) -> bool:
        return False

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        base_url = cfg.anthropic.base_url or _DEFAULT_BASE_URL
        api_key = hosted.api_key(cfg.anthropic)
        return build_openai_chat(
            base_url=base_url,
            model=cfg.anthropic.chat_model,
            api_key=api_key,
            extra_body=hosted.request_body("anthropic", cfg.anthropic),
            thinking_off_body=hosted.retry_body("anthropic", cfg.anthropic),
            timeout=cfg.request_timeout_s,
            reasoning_field=getattr(cfg.anthropic, "reasoning_field", "") or None,
            requires_reasoning_replay=getattr(
                cfg.anthropic, "requires_reasoning_replay", False
            ),
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None
