"""NVIDIA cloud backend adapter."""

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

_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaBackend(BaseBackend):
    """Backend adapter for the NVIDIA API (OpenAI-compatible)."""

    @property
    def key(self) -> str:
        return "nvidia"

    @property
    def display_name(self) -> str:
        return "NVIDIA"

    def validate(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        if not cfg.nvidia.api_key_env:
            errors.append(
                "[nvidia] api_key_env is empty. "
                "Set the environment variable name that holds your API key "
                "(e.g. NVIDIA_API_KEY)."
            )
        elif not os.environ.get(cfg.nvidia.api_key_env):
            errors.append(
                f"[nvidia] environment variable ${cfg.nvidia.api_key_env} "
                "is not set. Export it before starting the gateway."
            )
        if not cfg.nvidia.chat_model:
            errors.append(
                "[nvidia] chat_model is empty. "
                "Specify the model name (e.g. meta/llama-3.1-70b-instruct)."
            )
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        return ["nvidia is a chat-only backend and cannot be used for embedding."]

    def has_embedding(self, cfg: Config) -> bool:
        return False

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        base_url = cfg.nvidia.base_url or _DEFAULT_BASE_URL
        api_key = os.environ.get(cfg.nvidia.api_key_env, "")
        return build_openai_chat(
            base_url=base_url, model=cfg.nvidia.chat_model, api_key=api_key
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None
