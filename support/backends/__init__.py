"""Backend adapter registry."""

from localmelo.support.backends.base import BaseBackend
from localmelo.support.backends.registry import (
    ensure_defaults_registered,
    get_backend,
    list_backends,
    providers_for,
    register,
)

__all__ = [
    "BaseBackend",
    "ensure_defaults_registered",
    "get_backend",
    "list_backends",
    "providers_for",
    "register",
]
