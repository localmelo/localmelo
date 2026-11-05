"""The OpenAI-compatible backends are rows of data: one test of the one place.

``support/backends/openai_family.py`` holds the rows. These tests run every row
through the same checks, so a new row is covered without a new test.
"""

from __future__ import annotations

import os
from unittest import mock

import httpx
import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider
from localmelo.support import hosted
from localmelo.support.backends.openai_family import (
    CLOUD_BACKENDS,
    LOCAL_BACKENDS,
    HostedBackend,
    LocalBackend,
)
from localmelo.support.backends.registry import get_backend, list_backends
from localmelo.support.config import (
    VALID_CHAT_BACKENDS,
    VALID_EMBEDDING_BACKENDS,
    CloudVendorConfig,
    Config,
    LocalBackendConfig,
)
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

OPENAI_FORMAT_LOCAL = [key for key in LOCAL_BACKENDS if key != "ollama"]


def _local(key: str, **fields: str) -> Config:
    cfg = Config(chat_backend=key)
    for name, value in fields.items():
        setattr(getattr(cfg, key), name, value)
    return cfg


def test_the_rows_are_the_registered_backends_and_the_config_sections() -> None:
    registered = [
        b.key for b in list_backends() if isinstance(b, LocalBackend | HostedBackend)
    ]
    assert registered == [*LOCAL_BACKENDS, *CLOUD_BACKENDS]
    assert list(VALID_CHAT_BACKENDS) == registered
    assert list(VALID_EMBEDDING_BACKENDS) == [*LOCAL_BACKENDS, "none"]
    cfg = Config()
    for key in LOCAL_BACKENDS:
        assert isinstance(getattr(cfg, key), LocalBackendConfig), key
    for key in CLOUD_BACKENDS:
        assert isinstance(getattr(cfg, key), CloudVendorConfig), key


# -- a model server on this computer --


@pytest.mark.parametrize("key", LOCAL_BACKENDS)
def test_an_empty_local_section_says_what_to_set(key: str) -> None:
    backend = get_backend(key)
    assert isinstance(backend, LocalBackend)
    name = backend.label(Config(chat_backend=key))
    assert backend.validate(Config(chat_backend=key)) == [
        f"[{key}] chat_url is empty. Set it to your {name} URL (e.g. {backend.url}).",
        f"[{key}] chat_model is empty. Specify the {name} model name"
        + (f" (e.g. {backend.chat_model})." if backend.chat_model else "."),
    ]
    assert backend.validate_embedding(Config(chat_backend=key)) == [
        f"[{key}] embedding_url is empty. Set it to your {name} URL for embedding.",
        f"[{key}] embedding_model is empty. Set the {name} embedding model name.",
    ]
    filled = _local(key, chat_url="http://h:1", chat_model="m")
    assert backend.validate(filled) == []


@pytest.mark.parametrize("key", LOCAL_BACKENDS)
def test_embedding_needs_both_a_url_and_a_model(key: str) -> None:
    backend = get_backend(key)
    for fields in ({}, {"embedding_url": "http://h:1"}, {"embedding_model": "e"}):
        cfg = _local(key, **fields)
        assert backend.has_embedding(cfg) is False
        assert backend.build_embedding_provider(cfg) is None
    cfg = _local(key, embedding_url="http://h:1", embedding_model="e")
    assert backend.has_embedding(cfg) is True
    assert backend.validate_embedding(cfg) == []
    provider = backend.build_embedding_provider(cfg)
    assert isinstance(provider, BaseEmbeddingProvider)
    assert str(provider._http.base_url) == "http://h:1/v1/"  # type: ignore[attr-defined]


@pytest.mark.parametrize("key", OPENAI_FORMAT_LOCAL)
def test_a_local_chat_provider_is_built_from_its_section(key: str) -> None:
    cfg = _local(key, chat_url="http://h:1/", chat_model="m")
    getattr(cfg, key).extra_body = {"x": 1}
    cfg.request_timeout_s = 42.5
    provider = get_backend(key).build_chat_provider(cfg)
    assert isinstance(provider, OpenAICompatLLM)
    assert str(provider._http.base_url) == "http://h:1/v1/"
    assert provider.model == "m"
    assert provider.extra_body == {"x": 1}
    assert provider._http.timeout == httpx.Timeout(42.5)


def test_ollama_chats_on_its_native_api_without_v1() -> None:
    cfg = _local("ollama", chat_url="http://h:1/v1", chat_model="m")
    provider = get_backend("ollama").build_chat_provider(cfg)
    assert isinstance(provider, OllamaNativeChat)
    assert str(provider._http.base_url).rstrip("/") == "http://h:1"


def test_a_row_is_all_a_new_local_backend_needs() -> None:
    row = LocalBackend("acme", "Acme", "Acme (local endpoint)", "http://localhost:9/v1")
    cfg = Config(chat_backend="acme")
    cfg.acme = LocalBackendConfig(chat_url="http://h:1", chat_model="m")  # type: ignore[attr-defined]
    assert row.validate(cfg) == []
    assert row.detect_url == "http://127.0.0.1:9/v1"


# -- a hosted service --


@pytest.mark.parametrize("key", CLOUD_BACKENDS)
def test_a_hosted_section_says_what_to_set(key: str) -> None:
    backend = get_backend(key)
    preset = hosted.BY_ID[key]
    assert backend.display_name == preset.label
    with mock.patch.dict(os.environ, {}, clear=True):
        errors = backend.validate(Config(chat_backend=key))
    assert f"[{key}] api_key_env is empty." in errors[0]
    assert f"(e.g. {preset.key_env})" in errors[0]
    assert errors[1] == (
        f"[{key}] chat_model is empty. Specify the model name (e.g. {preset.models[0]})."
    )
    cfg = Config(chat_backend=key)
    section = getattr(cfg, key)
    section.api_key_env, section.chat_model = "TEST_KEY", "m"
    with mock.patch.dict(os.environ, {"TEST_KEY": "k"}):
        assert backend.validate(cfg) == []


@pytest.mark.parametrize("key", CLOUD_BACKENDS)
def test_a_hosted_service_is_chat_only(key: str) -> None:
    backend = get_backend(key)
    cfg = Config(chat_backend=key)
    assert backend.has_embedding(cfg) is False
    assert backend.build_embedding_provider(cfg) is None
    assert backend.validate_embedding(cfg) == [
        f"{key} is a chat-only backend and cannot be used for embedding."
    ]


@pytest.mark.parametrize("key", CLOUD_BACKENDS)
def test_a_hosted_chat_provider_uses_the_preset_address_unless_the_section_has_one(
    key: str,
) -> None:
    cfg = Config(chat_backend=key)
    section = getattr(cfg, key)
    section.api_key_env, section.chat_model = "TEST_KEY", "m"
    with mock.patch.dict(os.environ, {"TEST_KEY": "secret"}):
        provider = get_backend(key).build_chat_provider(cfg)
        assert isinstance(provider, OpenAICompatLLM)
        assert str(provider._http.base_url).rstrip("/") == hosted.BY_ID[key].base_url
        assert provider._http.headers["authorization"] == "Bearer secret"
        section.base_url = "https://proxy.example/v1"
        proxied = get_backend(key).build_chat_provider(cfg)
        assert isinstance(proxied, OpenAICompatLLM)
        assert str(proxied._http.base_url).rstrip("/") == "https://proxy.example/v1"


def test_the_label_of_a_hosted_service_follows_its_address() -> None:
    cfg = Config(chat_backend="openai")
    assert get_backend("openai").label(cfg) == "OpenAI"
    cfg.openai.base_url = "https://api.deepseek.com/v1"
    assert get_backend("openai").label(cfg) == "DeepSeek"
