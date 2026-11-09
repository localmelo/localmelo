"""What a model server offers for chat, and whether the configured model is on it.

One reading of a server's model list for every surface: Settings, the terminal's
``/model`` and the provider that meets a model the server does not have. The
list is the server's own (:func:`~localmelo.support.backends.discovery.list_models`);
this module only splits it and compares one model name with it.

:func:`offer` puts the chat models first and the embedding models apart.
:func:`check_chat_model` says what to do about the model the config names:
nothing (``ok``), use the one chat model the server offers (``switch``), or ask
the person which of several (``missing``). Its :meth:`~ModelCheck.line` is the
one sentence every surface shows, worded by the console's catalogs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from localmelo.support.backends import discovery
from localmelo.support.words import t_in

CheckState = Literal["ok", "switch", "missing"]

# Seconds the provider waits for the listing it asks for after a refusal.
CHECK_TIMEOUT_S: Final = 2.0


@dataclass(frozen=True)
class ModelOffer:
    """A server's models as a chat picks from them: ``chat`` first, ``embedding`` apart."""

    chat: tuple[str, ...]
    embedding: tuple[str, ...]


def offer(models: Sequence[str]) -> ModelOffer:
    """Split a server's listing into its chat models and its embedding models.

    Both keep the listing's order; a reranking model is in neither.
    """
    return ModelOffer(
        chat=tuple(m for m in models if discovery.model_kind(m) == "chat"),
        embedding=tuple(discovery.embedding_models(models)),
    )


@dataclass(frozen=True)
class ModelCheck:
    """The configured chat model against what the server offers.

    ``ok`` says nothing is wrong, or that there is nothing to compare. ``switch``
    names the one chat model on offer in ``use``. ``missing`` is a model that is
    not on the server while it offers several chat models (``chat``).
    """

    state: CheckState
    model: str
    use: str = ""
    chat: tuple[str, ...] = ()

    def line(self, language: str = "en") -> str:
        """The one sentence that says it, in *language*; ``""`` for ``ok``."""
        if self.state == "switch":
            return t_in(language, "models.switched", model=self.model, use=self.use)
        if self.state == "missing":
            return t_in(language, "models.missing", model=self.model, n=len(self.chat))
        return ""


def check_chat_model(configured: str, models: Sequence[str]) -> ModelCheck:
    """Whether *configured* is among *models*, and what to do when it is not.

    A server that lists no chat model, or a config that names no model, leaves
    nothing to compare: ``ok``.
    """
    chat = offer(models).chat
    if not configured or configured in models or not chat:
        return ModelCheck("ok", configured)
    if len(chat) == 1:
        return ModelCheck("switch", configured, use=chat[0], chat=chat)
    return ModelCheck("missing", configured, chat=chat)
