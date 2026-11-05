"""BaseBackend -- abstract contract for every backend adapter."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config


def configured_model(cfg: Config, key: str, *, embedding: bool = False) -> str:
    """The chat (or *embedding*) model the config's section *key* names; empty for none."""
    field = "embedding_model" if embedding else "chat_model"
    return str(getattr(getattr(cfg, key, None), field, ""))


class BaseBackend(abc.ABC):
    """Contract that every backend adapter must implement.

    Backends are pure connectors: they validate configuration and build
    provider objects.  They never own or start runtimes.
    """

    #: Config-level backend identifier, e.g. 'mlc', 'ollama', 'openai'.
    key: str
    #: Human-readable label for UI / onboarding.
    display_name: str
    #: Whether the config must name a model (melo_serve serves its own).
    model_required: ClassVar[bool] = True

    def label(self, cfg: Config, *, embedding: bool = False) -> str:
        """What a banner or a status row calls the server *cfg* names.

        :attr:`display_name` unless the backend can tell more from *cfg*;
        *embedding* asks for the embedding server's label.
        """
        return self.display_name

    def welcome_note(self, label: str) -> str:
        """Words the terminal's welcome row shows beside the model; empty for none.

        *label* is what :meth:`in_use` answered for the chat.
        """
        return ""

    def model_name(self, cfg: Config, *, embedding: bool = False) -> str:
        """The model *cfg* names for this backend; empty when it names none."""
        return configured_model(cfg, self.key, embedding=embedding)

    def in_use(self, cfg: Config, *, embedding: bool = False) -> tuple[str, str]:
        """``(label, model)``: what a banner or a request calls the server and model now.

        The label and the model are answered together, so a backend whose
        model is not in the config (melo_serve serves what it serves now)
        gives both from one reading of its source.
        """
        return self.label(cfg, embedding=embedding), self.model_name(
            cfg, embedding=embedding
        )

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
