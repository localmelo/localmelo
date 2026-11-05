"""BaseBackend -- abstract contract for every backend adapter."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

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
    """

    #: Config-level backend identifier, e.g. 'mlc', 'ollama', 'openai'.
    key: str
    #: Human-readable label for UI / onboarding.
    display_name: str

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        """What a banner or a status row calls the server *cfg* names.

        :attr:`display_name` unless the backend can tell more from *cfg*;
        *embedding* asks for the embedding server's label.
        """
        return self.display_name

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
