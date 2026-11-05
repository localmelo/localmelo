"""Backend registry -- register and look up backend adapters by key."""

from __future__ import annotations

import threading

from localmelo.support.backends.base import BaseBackend

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
    ``Agent(config=...)``) work without the CLI entrypoint.
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


def list_backends() -> list[BaseBackend]:
    """Return all registered backends in insertion order."""
    ensure_defaults_registered()
    return list(_BACKENDS.values())


def _clear() -> None:
    """For testing only. Resets both the registry and the initialization flag."""
    global _DEFAULT_BACKENDS_REGISTERED  # noqa: PLW0603
    _BACKENDS.clear()
    _DEFAULT_BACKENDS_REGISTERED = False
