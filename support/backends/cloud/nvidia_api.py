"""NVIDIA cloud backend adapter."""

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
        errors = hosted.key_errors("nvidia", cfg.nvidia, "NVIDIA_API_KEY")
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
        api_key = hosted.api_key(cfg.nvidia)
        return build_openai_chat(
            base_url=base_url,
            model=cfg.nvidia.chat_model,
            api_key=api_key,
            extra_body=hosted.request_body("nvidia", cfg.nvidia),
            thinking_off_body=hosted.retry_body("nvidia", cfg.nvidia),
            timeout=cfg.request_timeout_s,
            reasoning_field=getattr(cfg.nvidia, "reasoning_field", "") or None,
            requires_reasoning_replay=getattr(
                cfg.nvidia, "requires_reasoning_replay", False
            ),
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        return None
