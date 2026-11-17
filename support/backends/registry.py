"""Backend registry -- register and look up backend adapters by key."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from localmelo.support.backends.base import BaseBackend, configured_model

if TYPE_CHECKING:
    from localmelo.melo.contracts.providers import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
    )
    from localmelo.support.config import Config

_BACKENDS: dict[str, BaseBackend] = {}
_DEFAULT_BACKENDS_REGISTERED = False
_REGISTER_LOCK = threading.Lock()


def register(backend: BaseBackend) -> None:
    """Register a backend adapter instance.

    Silently skips if a backend with the same key is already registered,
    making repeated calls safe.
    """
    if backend.key not in _BACKENDS:
        _BACKENDS[backend.key] = backend


def ensure_defaults_registered() -> None:
    """Register the built-in backends: the rows of :mod:`openai_family`.

    Safe to call repeatedly -- the actual registration runs at most once.
    This is called automatically by :func:`get_backend` and
    :func:`list_backends` so that programmatic code paths (e.g.
    :func:`providers_for`) work without the CLI entrypoint.
    """
    global _DEFAULT_BACKENDS_REGISTERED  # noqa: PLW0603
    if _DEFAULT_BACKENDS_REGISTERED:
        return
    with _REGISTER_LOCK:
        if _DEFAULT_BACKENDS_REGISTERED:
            return  # double-check after acquiring lock

        from localmelo.support.backends.openai_family import (
            HOSTED_SERVICES,
            LOCAL_SERVERS,
        )

        for backend in (*LOCAL_SERVERS, *HOSTED_SERVICES):
            register(backend)
        _DEFAULT_BACKENDS_REGISTERED = True


def get_backend(name: str) -> BaseBackend:
    """Look up a registered backend by key. Raises KeyError if not found."""
    ensure_defaults_registered()
    try:
        return _BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise KeyError(f"Unknown backend '{name}'. Available: {available}") from None


def find(key: str) -> BaseBackend | None:
    """The registered backend *key*, or None when nothing registered it."""
    try:
        return get_backend(key)
    except KeyError:
        return None


def in_use(cfg: Config, key: str, *, embedding: bool = False) -> tuple[str, str]:
    """``(label, model)`` of the backend *key* as *cfg* has it used now.

    The one lookup every banner, identity and cache key asks. A backend nothing
    registered is called by its key and has the model its config section names.
    """
    backend = find(key)
    if backend is None:
        return key, configured_model(cfg, key, embedding=embedding)
    return backend.in_use(cfg, embedding=embedding)


def list_backends() -> list[BaseBackend]:
    """Return all registered backends in insertion order."""
    ensure_defaults_registered()
    return list(_BACKENDS.values())


def providers_for(
    cfg: Config, reasoning: str = "legacy"
) -> tuple[BaseLLMProvider, BaseEmbeddingProvider | None]:
    """Build the chat and embedding providers that *cfg* names.

    Uses the split backend model: ``cfg.chat_backend`` selects the chat
    provider and ``cfg.embedding_backend`` selects the embedding provider.
    The two may refer to different backend adapters.

    *reasoning* is the reasoning mode of the Agent that gets them. The chat
    section's reasoning field is only for replay, which is ``compress`` on a
    provider that requires it: in every other mode the provider sends none.
    Ollama keeps its fixed ``thinking`` field, whose native wire legacy
    already uses.
    """
    llm = get_backend(cfg.chat_backend).build_chat_provider(cfg)
    replay = reasoning == "compress+replay" or (
        reasoning == "compress" and llm.requires_reasoning_replay
    )
    if not replay and cfg.chat_backend != "ollama":
        llm.reasoning_field = None
    embedding = None
    if cfg.has_embedding:
        embedding = get_backend(cfg.embedding_backend).build_embedding_provider(cfg)
    return llm, embedding


def _clear() -> None:
    """For testing only. Resets both the registry and the initialization flag."""
    global _DEFAULT_BACKENDS_REGISTERED  # noqa: PLW0603
    _BACKENDS.clear()
    _DEFAULT_BACKENDS_REGISTERED = False
