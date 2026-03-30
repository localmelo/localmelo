"""BaseBackend -- abstract contract for every backend adapter."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from localmelo.support.backends.tokenization import count_tokens

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config


class BaseBackend(abc.ABC):
    """Contract that every backend adapter must implement.

    Backends are pure connectors: they validate configuration and build
    provider objects.  They never own or start runtimes.

    Every backend inherits :meth:`count_tokens` — a shared, deterministic
    token counter used for normalized cross-backend comparison metrics.
    """

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """Config-level backend identifier, e.g. 'mlc', 'ollama', 'openai'."""

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """Human-readable label for UI / onboarding."""

    # ── Shared tokenizer (non-abstract) ────────────────────────

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens using the shared deterministic tokenizer.

        All backends share the same implementation so that normalized
        token metrics are directly comparable.  See
        :mod:`localmelo.support.backends.tokenization` for details.
        """
        return count_tokens(text)

    @abc.abstractmethod
    def validate(self, cfg: Config) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""

    @abc.abstractmethod
    def validate_embedding(self, cfg: Config) -> list[str]:
        """Return embedding-specific validation errors."""

    @abc.abstractmethod
    def has_embedding(self, cfg: Config) -> bool:
        """Whether embedding is available for this backend + config combination."""

    @abc.abstractmethod
    def build_chat_provider(self, cfg: Config) -> BaseLLMProvider:
        """Construct the chat LLM provider."""

    @abc.abstractmethod
    def build_embedding_provider(self, cfg: Config) -> BaseEmbeddingProvider | None:
        """Construct the embedding provider, or None if unavailable."""
