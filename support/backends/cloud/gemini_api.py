"""Gemini cloud backend adapter."""

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

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class GeminiBackend(BaseBackend):
    """Backend adapter for the Gemini API (OpenAI-compatible bridge)."""

    @property
    def key(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Gemini"

    def validate(self, cfg: Config) -> list[str]:
        errors = hosted.key_errors("gemini", cfg.gemini, "GEMINI_API_KEY")
        if not cfg.gemini.chat_model:
            errors.append(
                "[gemini] chat_model is empty. "
                "Specify the model name (e.g. gemini-2.0-flash)."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        return ["gemini is a chat-only backend and cannot be used for embedding."]

    def has_embedding(self, cfg: Config) -> bool:
        return False

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        base_url = cfg.gemini.base_url or _DEFAULT_BASE_URL
        api_key = hosted.api_key(cfg.gemini)
        return build_openai_chat(
            base_url=base_url,
            model=cfg.gemini.chat_model,
            api_key=api_key,
            extra_body=hosted.request_body("gemini", cfg.gemini),
            thinking_off_body=hosted.retry_body("gemini", cfg.gemini),
            timeout=cfg.request_timeout_s,
            reasoning_field=getattr(cfg.gemini, "reasoning_field", "") or None,
            requires_reasoning_replay=getattr(
                cfg.gemini, "requires_reasoning_replay", False
            ),
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None
