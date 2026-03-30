"""Tests for local backends (Ollama, vLLM, SGLang), cloud backends, and shared helpers."""

from __future__ import annotations

import os
from unittest import mock

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.support.backends.cloud.anthropic_api import AnthropicBackend
from localmelo.support.backends.cloud.gemini_api import GeminiBackend
from localmelo.support.backends.cloud.nvidia_api import NvidiaBackend
from localmelo.support.backends.cloud.openai_api import OpenAIBackend
from localmelo.support.backends.local.ollama import OllamaBackend
from localmelo.support.backends.local.sglang import SglangBackend
from localmelo.support.backends.local.vllm import VllmBackend
from localmelo.support.backends.openai_compat import (
    build_openai_chat,
    build_openai_embedding,
    normalize_url,
)
from localmelo.support.config import CloudVendorConfig, Config, LocalBackendConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_cfg(
    backend_key: str = "ollama",
    chat_url: str = "http://localhost:11434",
    chat_model: str = "qwen3:8b",
    embedding_url: str = "",
    embedding_model: str = "",
) -> Config:
    cfg = Config(chat_backend=backend_key)
    setattr(
        cfg,
        backend_key,
        LocalBackendConfig(
            chat_url=chat_url,
            chat_model=chat_model,
            embedding_url=embedding_url,
            embedding_model=embedding_model,
        ),
    )
    return cfg


def _cloud_cfg(
    backend_key: str = "openai",
    api_key_env: str = "OPENAI_API_KEY",
    chat_model: str = "gpt-4o",
    base_url: str = "",
) -> Config:
    cfg = Config(chat_backend=backend_key)
    setattr(
        cfg,
        backend_key,
        CloudVendorConfig(
            api_key_env=api_key_env,
            chat_model=chat_model,
            base_url=base_url,
        ),
    )
    return cfg


# ===========================================================================
# normalize_url
# ===========================================================================


class TestNormalizeUrl:
    def test_appends_v1(self) -> None:
        assert normalize_url("http://localhost:11434") == "http://localhost:11434/v1"

    def test_strips_trailing_slash_then_appends(self) -> None:
        assert normalize_url("http://localhost:11434/") == "http://localhost:11434/v1"

    def test_keeps_existing_v1(self) -> None:
        assert normalize_url("http://localhost:11434/v1") == "http://localhost:11434/v1"

    def test_strips_trailing_slash_after_v1(self) -> None:
        assert (
            normalize_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"
        )


# ===========================================================================
# build_openai_chat / build_openai_embedding
# ===========================================================================


class TestBuildOpenaiChat:
    def test_returns_llm_provider(self) -> None:
        provider = build_openai_chat(
            base_url="http://localhost:8400/v1", model="test-model"
        )
        assert isinstance(provider, BaseLLMProvider)

    def test_passes_api_key(self) -> None:
        provider = build_openai_chat(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        assert isinstance(provider, BaseLLMProvider)


class TestBuildOpenaiEmbedding:
    def test_returns_embedding_provider(self) -> None:
        provider = build_openai_embedding(
            base_url="http://localhost:11434/v1", model="nomic-embed"
        )
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_passes_api_key(self) -> None:
        provider = build_openai_embedding(
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
            api_key="sk-test",
        )
        assert isinstance(provider, BaseEmbeddingProvider)


# ===========================================================================
# OllamaBackend
# ===========================================================================


class TestOllamaBackendProperties:
    def test_key(self) -> None:
        assert OllamaBackend().key == "ollama"

    def test_display_name(self) -> None:
        assert OllamaBackend().display_name == "Ollama"


class TestOllamaValidation:
    def test_valid_config(self) -> None:
        cfg = _local_cfg("ollama")
        assert OllamaBackend().validate(cfg) == []

    def test_missing_chat_url(self) -> None:
        cfg = _local_cfg("ollama", chat_url="")
        errors = OllamaBackend().validate(cfg)
        assert len(errors) == 1
        assert "chat_url" in errors[0]

    def test_missing_chat_model(self) -> None:
        cfg = _local_cfg("ollama", chat_model="")
        errors = OllamaBackend().validate(cfg)
        assert len(errors) == 1
        assert "chat_model" in errors[0]

    def test_missing_both(self) -> None:
        cfg = _local_cfg("ollama", chat_url="", chat_model="")
        errors = OllamaBackend().validate(cfg)
        assert len(errors) == 2


class TestOllamaValidateEmbedding:
    def test_with_url_and_model(self) -> None:
        cfg = _local_cfg(
            "ollama",
            embedding_url="http://localhost:11434",
            embedding_model="nomic-embed-text",
        )
        assert OllamaBackend().validate_embedding(cfg) == []

    def test_missing_embedding_model(self) -> None:
        cfg = _local_cfg(
            "ollama",
            embedding_url="http://localhost:11434",
            embedding_model="",
        )
        errors = OllamaBackend().validate_embedding(cfg)
        assert any("embedding_model" in e for e in errors)

    def test_missing_embedding_url(self) -> None:
        cfg = _local_cfg(
            "ollama",
            embedding_url="",
            embedding_model="nomic-embed-text",
        )
        errors = OllamaBackend().validate_embedding(cfg)
        assert any("embedding_url" in e for e in errors)


class TestOllamaHasEmbedding:
    def test_with_url_and_model(self) -> None:
        cfg = _local_cfg(
            "ollama",
            embedding_url="http://localhost:11434",
            embedding_model="nomic-embed-text",
        )
        assert OllamaBackend().has_embedding(cfg) is True

    def test_without_model_returns_false(self) -> None:
        cfg = _local_cfg(
            "ollama", embedding_url="http://localhost:11434", embedding_model=""
        )
        assert OllamaBackend().has_embedding(cfg) is False

    def test_without_url_returns_false(self) -> None:
        cfg = _local_cfg("ollama", embedding_url="", embedding_model="nomic-embed-text")
        assert OllamaBackend().has_embedding(cfg) is False


class TestOllamaBuildChatProvider:
    def test_returns_llm_provider(self) -> None:
        cfg = _local_cfg("ollama")
        provider = OllamaBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)


class TestOllamaBuildEmbeddingProvider:
    def test_with_url_and_model(self) -> None:
        cfg = _local_cfg(
            "ollama",
            embedding_url="http://localhost:11434",
            embedding_model="nomic-embed-text",
        )
        provider = OllamaBackend().build_embedding_provider(cfg)
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_no_model_returns_none(self) -> None:
        cfg = _local_cfg(
            "ollama", embedding_url="http://localhost:11434", embedding_model=""
        )
        provider = OllamaBackend().build_embedding_provider(cfg)
        assert provider is None

    def test_no_url_returns_none(self) -> None:
        cfg = _local_cfg("ollama", embedding_url="", embedding_model="nomic-embed-text")
        provider = OllamaBackend().build_embedding_provider(cfg)
        assert provider is None


# ===========================================================================
# VllmBackend
# ===========================================================================


class TestVllmBackendProperties:
    def test_key(self) -> None:
        assert VllmBackend().key == "vllm"

    def test_display_name(self) -> None:
        assert VllmBackend().display_name == "vLLM"


class TestVllmValidation:
    def test_valid_config(self) -> None:
        cfg = _local_cfg(
            "vllm", chat_url="http://localhost:8000", chat_model="meta-llama/Llama-3-8B"
        )
        assert VllmBackend().validate(cfg) == []

    def test_missing_chat_url(self) -> None:
        cfg = _local_cfg("vllm", chat_url="", chat_model="model")
        errors = VllmBackend().validate(cfg)
        assert any("chat_url" in e for e in errors)

    def test_missing_chat_model(self) -> None:
        cfg = _local_cfg("vllm", chat_url="http://localhost:8000", chat_model="")
        errors = VllmBackend().validate(cfg)
        assert any("chat_model" in e for e in errors)


class TestVllmBuildChatProvider:
    def test_returns_llm_provider(self) -> None:
        cfg = _local_cfg("vllm", chat_url="http://localhost:8000", chat_model="model")
        provider = VllmBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)


# ===========================================================================
# SglangBackend
# ===========================================================================


class TestSglangBackendProperties:
    def test_key(self) -> None:
        assert SglangBackend().key == "sglang"

    def test_display_name(self) -> None:
        assert SglangBackend().display_name == "SGLang"


class TestSglangValidation:
    def test_valid_config(self) -> None:
        cfg = _local_cfg(
            "sglang",
            chat_url="http://localhost:30000",
            chat_model="meta-llama/Llama-3-8B",
        )
        assert SglangBackend().validate(cfg) == []

    def test_missing_chat_url(self) -> None:
        cfg = _local_cfg("sglang", chat_url="", chat_model="model")
        errors = SglangBackend().validate(cfg)
        assert any("chat_url" in e for e in errors)

    def test_missing_chat_model(self) -> None:
        cfg = _local_cfg("sglang", chat_url="http://localhost:30000", chat_model="")
        errors = SglangBackend().validate(cfg)
        assert any("chat_model" in e for e in errors)


class TestSglangBuildChatProvider:
    def test_returns_llm_provider(self) -> None:
        cfg = _local_cfg(
            "sglang", chat_url="http://localhost:30000", chat_model="model"
        )
        provider = SglangBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)


# ===========================================================================
# OpenAIBackend
# ===========================================================================


class TestOpenAIBackendProperties:
    def test_key(self) -> None:
        assert OpenAIBackend().key == "openai"

    def test_display_name(self) -> None:
        assert OpenAIBackend().display_name == "OpenAI"


class TestOpenAIValidation:
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_valid_config(self) -> None:
        cfg = _cloud_cfg("openai")
        assert OpenAIBackend().validate(cfg) == []

    def test_missing_api_key_env(self) -> None:
        cfg = _cloud_cfg("openai", api_key_env="")
        errors = OpenAIBackend().validate(cfg)
        assert any("api_key_env" in e and "empty" in e for e in errors)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_unset_env_var(self) -> None:
        cfg = _cloud_cfg("openai", api_key_env="MISSING_KEY")
        errors = OpenAIBackend().validate(cfg)
        assert any("$MISSING_KEY" in e for e in errors)

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_missing_chat_model(self) -> None:
        cfg = _cloud_cfg("openai", chat_model="")
        errors = OpenAIBackend().validate(cfg)
        assert any("chat_model" in e and "empty" in e for e in errors)


class TestOpenAIValidateEmbedding:
    def test_always_returns_error(self) -> None:
        cfg = _cloud_cfg("openai")
        errors = OpenAIBackend().validate_embedding(cfg)
        assert len(errors) == 1
        assert "chat-only" in errors[0]


class TestOpenAIHasEmbedding:
    def test_always_false(self) -> None:
        cfg = _cloud_cfg("openai")
        assert OpenAIBackend().has_embedding(cfg) is False


class TestOpenAIBuildChatProvider:
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_returns_llm_provider(self) -> None:
        cfg = _cloud_cfg("openai")
        provider = OpenAIBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)


class TestOpenAIBuildEmbeddingProvider:
    def test_always_returns_none(self) -> None:
        cfg = _cloud_cfg("openai")
        assert OpenAIBackend().build_embedding_provider(cfg) is None


# ===========================================================================
# GeminiBackend
# ===========================================================================


class TestGeminiBackendProperties:
    def test_key(self) -> None:
        assert GeminiBackend().key == "gemini"

    def test_display_name(self) -> None:
        assert GeminiBackend().display_name == "Gemini"


class TestGeminiValidation:
    @mock.patch.dict(os.environ, {"GEMINI_KEY": "gem-test"})
    def test_valid_config(self) -> None:
        cfg = _cloud_cfg(
            "gemini", api_key_env="GEMINI_KEY", chat_model="gemini-2.0-flash"
        )
        assert GeminiBackend().validate(cfg) == []

    @mock.patch.dict(os.environ, {"GEMINI_KEY": "gem-test"})
    def test_missing_chat_model(self) -> None:
        cfg = _cloud_cfg("gemini", api_key_env="GEMINI_KEY", chat_model="")
        errors = GeminiBackend().validate(cfg)
        assert any("chat_model" in e for e in errors)


class TestGeminiBuildChatProvider:
    @mock.patch.dict(os.environ, {"GEMINI_KEY": "gem-test"})
    def test_returns_llm_provider(self) -> None:
        cfg = _cloud_cfg(
            "gemini", api_key_env="GEMINI_KEY", chat_model="gemini-2.0-flash"
        )
        provider = GeminiBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)


# ===========================================================================
# AnthropicBackend
# ===========================================================================


class TestAnthropicBackendProperties:
    def test_key(self) -> None:
        assert AnthropicBackend().key == "anthropic"

    def test_display_name(self) -> None:
        assert AnthropicBackend().display_name == "Anthropic"


class TestAnthropicValidation:
    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
    def test_valid_config(self) -> None:
        cfg = _cloud_cfg(
            "anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            chat_model="claude-sonnet-4-20250514",
        )
        assert AnthropicBackend().validate(cfg) == []


class TestAnthropicBuildChatProvider:
    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
    def test_returns_llm_provider(self) -> None:
        cfg = _cloud_cfg(
            "anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            chat_model="claude-sonnet-4-20250514",
        )
        provider = AnthropicBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)


# ===========================================================================
# NvidiaBackend
# ===========================================================================


class TestNvidiaBackendProperties:
    def test_key(self) -> None:
        assert NvidiaBackend().key == "nvidia"

    def test_display_name(self) -> None:
        assert NvidiaBackend().display_name == "NVIDIA"


class TestNvidiaValidation:
    @mock.patch.dict(os.environ, {"NVIDIA_KEY": "nvapi-test"})
    def test_valid_config(self) -> None:
        cfg = _cloud_cfg(
            "nvidia", api_key_env="NVIDIA_KEY", chat_model="meta/llama-3.1-70b-instruct"
        )
        assert NvidiaBackend().validate(cfg) == []


class TestNvidiaBuildChatProvider:
    @mock.patch.dict(os.environ, {"NVIDIA_KEY": "nvapi-test"})
    def test_returns_llm_provider(self) -> None:
        cfg = _cloud_cfg(
            "nvidia", api_key_env="NVIDIA_KEY", chat_model="meta/llama-3.1-70b-instruct"
        )
        provider = NvidiaBackend().build_chat_provider(cfg)
        assert isinstance(provider, BaseLLMProvider)
