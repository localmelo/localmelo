"""Contracts (abstract interfaces) that support/ implementations must satisfy."""

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool

__all__ = [
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "BaseToolProvider",
    "ProvidedTool",
]
