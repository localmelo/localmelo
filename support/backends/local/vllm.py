"""OpenAI-compatible backend adapter (key ``vllm``): oMLX, vLLM or any such server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from localmelo.support.backends import discovery
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
    """Backend adapter for an external OpenAI-compatible server.

    The key ``vllm`` stands for oMLX, vLLM and every other server that speaks
    the OpenAI API; ``chat_backend = "omlx"`` is the same backend, spelled so.
    """

    @property
    def key(self) -> str:
        return "vllm"

    @property
    def display_name(self) -> str:
        return discovery.OPENAI_COMPATIBLE_NAME

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        """``oMLX`` when the config says so or a listing of the server did."""
        if embedding:
            url = cfg.vllm.embedding_url or cfg.vllm.chat_url
            declared = cfg.embedding_alias
        else:
            url, declared = cfg.vllm.chat_url, cfg.chat_alias
        return discovery.server_name(url, declared=declared == "omlx")

    def validate(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        name = self.label(cfg)
        if not cfg.vllm.chat_url:
            errors.append(
                "[vllm] chat_url is empty. "
                f"Set it to your {name} URL (e.g. http://localhost:8000)."
            )
        if not cfg.vllm.chat_model:
            errors.append(f"[vllm] chat_model is empty. Specify the {name} model name.")
        return errors

    def validate_embedding(self, cfg: Config) -> list[str]:
        errors: list[str] = []
        name = self.label(cfg, embedding=True)
        if not cfg.vllm.embedding_url:
            errors.append(
                "[vllm] embedding_url is empty. "
                f"Set it to your {name} URL for embedding."
            )
        if not cfg.vllm.embedding_model:
            errors.append(
                f"[vllm] embedding_model is empty. Set an {name} embedding model name."
            )
        return errors

    def has_embedding(self, cfg: Config) -> bool:
        return bool(cfg.vllm.embedding_url and cfg.vllm.embedding_model)

    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        return build_openai_chat(
            base_url=normalize_url(cfg.vllm.chat_url),
            model=cfg.vllm.chat_model,
            extra_body=cfg.vllm.extra_body,
            timeout=cfg.request_timeout_s,
            reasoning_field=getattr(cfg.vllm, "reasoning_field", "") or None,
            requires_reasoning_replay=getattr(
                cfg.vllm, "requires_reasoning_replay", False
            ),
        )

    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        if cfg.vllm.embedding_url and cfg.vllm.embedding_model:
            return build_openai_embedding(
                base_url=normalize_url(cfg.vllm.embedding_url),
                model=cfg.vllm.embedding_model,
            )
        return None
