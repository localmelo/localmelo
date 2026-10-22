"""request_timeout_s: every chat backend passes it to its chat provider.

Unset, each provider keeps its own default: 120 s for the OpenAI-compatible
chat provider, 300 s for Ollama's native chat. Embedding providers keep
theirs (120 s) either way.
"""

from __future__ import annotations

import httpx
import pytest

from localmelo.support.backends.base import BaseBackend
from localmelo.support.backends.registry import get_backend, list_backends
from localmelo.support.config import (
    VALID_CHAT_BACKENDS,
    VALID_EMBEDDING_BACKENDS,
    Config,
    LocalBackendConfig,
)
from localmelo.support.providers.embedding.openai_compat import OpenAICompatEmbedding
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def _cfg(key: str, request_timeout_s: float | None = None) -> Config:
    """A config whose chat backend is *key*, with a URL or key env and a model."""
    cfg = Config(
        chat_backend=key, embedding_backend="none", request_timeout_s=request_timeout_s
    )
    section = getattr(cfg, key)
    if isinstance(section, LocalBackendConfig):
        section.chat_url = "http://127.0.0.1:9/v1"
    else:
        section.api_key_env = "LOCALMELO_TEST_KEY"
    section.chat_model = "m"
    return cfg


def _builtin_backends() -> list[BaseBackend]:
    """The backends ensure_defaults_registered() registers.

    Other tests register fakes next to them, defined outside the package.
    """
    return [
        backend
        for backend in list_backends()
        if type(backend).__module__.startswith("localmelo.support.backends.")
    ]


def _chat_timeout(provider: object) -> httpx.Timeout:
    assert isinstance(provider, OpenAICompatLLM | OllamaNativeChat)
    return provider._http.timeout


def test_every_registered_chat_backend_passes_request_timeout_s() -> None:
    backends = _builtin_backends()
    assert sorted(backend.key for backend in backends) == sorted(VALID_CHAT_BACKENDS)
    for backend in backends:
        key = backend.key
        provider_type = OllamaNativeChat if key == "ollama" else OpenAICompatLLM
        default = 300.0 if key == "ollama" else 120.0

        unset = backend.build_chat_provider(_cfg(key))
        assert isinstance(unset, provider_type), key
        assert _chat_timeout(unset) == httpx.Timeout(default), key

        for value in (600, 42.5):
            provider = backend.build_chat_provider(_cfg(key, value))
            assert isinstance(provider, provider_type), key
            assert _chat_timeout(provider) == httpx.Timeout(value), (key, value)


@pytest.mark.parametrize(
    "key", [key for key in VALID_EMBEDDING_BACKENDS if key != "none"]
)
def test_embedding_requests_keep_their_timeout(key: str) -> None:
    cfg = _cfg(key, 600)
    cfg.embedding_backend = key
    section = getattr(cfg, key)
    section.embedding_url = "http://127.0.0.1:9/v1"
    section.embedding_model = "e"
    provider = get_backend(key).build_embedding_provider(cfg)
    assert isinstance(provider, OpenAICompatEmbedding)
    assert provider._http.timeout == httpx.Timeout(120.0)
