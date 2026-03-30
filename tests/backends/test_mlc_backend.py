"""Tests for the MLC backend adapter (local OpenAI-compatible connector)."""

from __future__ import annotations

import pytest

from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.support.backends.local.mlc import MlcBackend
from localmelo.support.config import Config, LocalBackendConfig


@pytest.fixture()
def backend() -> MlcBackend:
    return MlcBackend()


@pytest.fixture()
def valid_cfg() -> Config:
    cfg = Config(chat_backend="mlc", embedding_backend="mlc")
    cfg.mlc = LocalBackendConfig(
        chat_url="http://127.0.0.1:8400",
        chat_model="Qwen3-4B",
        embedding_url="http://127.0.0.1:8400",
        embedding_model="Qwen3-Embedding-0.6B",
    )
    return cfg


# -- Properties --


class TestProperties:
    def test_key(self, backend: MlcBackend) -> None:
        assert backend.key == "mlc"

    def test_display_name(self, backend: MlcBackend) -> None:
        assert backend.display_name == "MLC-LLM"


# -- Validation --


class TestValidate:
    def test_valid_config(self, backend: MlcBackend, valid_cfg: Config) -> None:
        errors = backend.validate(valid_cfg)
        assert errors == []

    def test_missing_chat_url(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(chat_url="", chat_model="Qwen3-4B")
        errors = backend.validate(cfg)
        assert len(errors) == 1
        assert "chat_url" in errors[0]

    def test_missing_chat_model(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(chat_url="http://127.0.0.1:8400", chat_model="")
        errors = backend.validate(cfg)
        assert len(errors) == 1
        assert "chat_model is empty" in errors[0]

    def test_missing_both(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(chat_url="", chat_model="")
        errors = backend.validate(cfg)
        assert len(errors) == 2


# -- validate_embedding --


class TestValidateEmbedding:
    def test_valid_with_explicit_url_and_model(
        self, backend: MlcBackend, valid_cfg: Config
    ) -> None:
        errors = backend.validate_embedding(valid_cfg)
        assert errors == []

    def test_missing_embedding_url(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            chat_url="http://127.0.0.1:8400",
            chat_model="Qwen3-4B",
            embedding_url="",
            embedding_model="Qwen3-Embedding-0.6B",
        )
        errors = backend.validate_embedding(cfg)
        assert len(errors) == 1
        assert "embedding_url" in errors[0]

    def test_missing_embedding_model(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            chat_url="http://127.0.0.1:8400",
            chat_model="Qwen3-4B",
            embedding_url="http://127.0.0.1:8400",
            embedding_model="",
        )
        errors = backend.validate_embedding(cfg)
        assert len(errors) == 1
        assert "embedding_model" in errors[0]

    def test_missing_both(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            chat_url="http://127.0.0.1:8400",
            chat_model="Qwen3-4B",
            embedding_url="",
            embedding_model="",
        )
        errors = backend.validate_embedding(cfg)
        assert len(errors) == 2


# -- has_embedding --


class TestHasEmbedding:
    def test_true_when_both_set(self, backend: MlcBackend, valid_cfg: Config) -> None:
        assert backend.has_embedding(valid_cfg) is True

    def test_false_when_url_missing(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            embedding_url="", embedding_model="Qwen3-Embedding-0.6B"
        )
        assert backend.has_embedding(cfg) is False

    def test_false_when_model_missing(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            embedding_url="http://127.0.0.1:8400", embedding_model=""
        )
        assert backend.has_embedding(cfg) is False


# -- build_chat_provider --


class TestBuildChatProvider:
    def test_returns_llm_provider(self, backend: MlcBackend, valid_cfg: Config) -> None:
        provider = backend.build_chat_provider(valid_cfg)
        assert isinstance(provider, BaseLLMProvider)


# -- build_embedding_provider --


class TestBuildEmbeddingProvider:
    def test_returns_embedding_provider(
        self, backend: MlcBackend, valid_cfg: Config
    ) -> None:
        provider = backend.build_embedding_provider(valid_cfg)
        assert isinstance(provider, BaseEmbeddingProvider)

    def test_returns_none_without_url(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            chat_url="http://127.0.0.1:8400",
            chat_model="Qwen3-4B",
            embedding_url="",
            embedding_model="Qwen3-Embedding-0.6B",
        )
        assert backend.build_embedding_provider(cfg) is None

    def test_returns_none_without_model(self, backend: MlcBackend) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="mlc")
        cfg.mlc = LocalBackendConfig(
            chat_url="http://127.0.0.1:8400",
            chat_model="Qwen3-4B",
            embedding_url="http://127.0.0.1:8400",
            embedding_model="",
        )
        assert backend.build_embedding_provider(cfg) is None
