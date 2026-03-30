"""Anthropic cloud backend adapter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

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
        errors: list[str] = []
        if not cfg.anthropic.api_key_env:
            errors.append(
                "[anthropic] api_key_env is empty. "
                "Set the environment variable name that holds your API key "
                "(e.g. ANTHROPIC_API_KEY)."
            )
        elif not os.environ.get(cfg.anthropic.api_key_env):
            errors.append(
                f"[anthropic] environment variable ${cfg.anthropic.api_key_env} "
                "is not set. Export it before starting the gateway."
            )
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
        api_key = os.environ.get(cfg.anthropic.api_key_env, "")
        return build_openai_chat(
            base_url=base_url, model=cfg.anthropic.chat_model, api_key=api_key
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None
