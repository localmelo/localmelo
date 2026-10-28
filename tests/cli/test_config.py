"""Tests for the support layer: config roundtrip, validation, embedding."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.support.backends.registry import ensure_defaults_registered
from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    ConfigError,
    GatewayConfig,
    LocalBackendConfig,
    SchedulerConfig,
    load,
    save,
)

# Ensure default backends are registered so that has_embedding delegation works.
ensure_defaults_registered()

# -- Config roundtrip --


class TestConfigRoundtrip:
    """save() then load() should preserve all fields."""

    def test_mlc_roundtrip(self, tmp_path: Path) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:9000",
                chat_model="Qwen3-8B",
                embedding_url="http://127.0.0.1:9000",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
            gateway=GatewayConfig(port=9999, host="0.0.0.0"),
        )
        config_path = str(tmp_path / "config.toml")
        with (
            mock.patch("localmelo.support.config.CONFIG_PATH", config_path),
            mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
        ):
            save(cfg)
            loaded = load()

        assert loaded.chat_backend == "mlc"
        assert loaded.embedding_backend == "mlc"
        assert loaded.mlc.chat_url == "http://127.0.0.1:9000"
        assert loaded.mlc.chat_model == "Qwen3-8B"
        assert loaded.mlc.embedding_url == "http://127.0.0.1:9000"
        assert loaded.mlc.embedding_model == "Qwen3-Embedding-0.6B"
        assert loaded.gateway.port == 9999
        assert loaded.gateway.host == "0.0.0.0"

    def test_ollama_roundtrip(self, tmp_path: Path) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://myhost:11434",
                chat_model="llama3.2",
                embedding_url="http://myhost:11434",
                embedding_model="nomic-embed",
            ),
        )
        config_path = str(tmp_path / "config.toml")
        with (
            mock.patch("localmelo.support.config.CONFIG_PATH", config_path),
            mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
        ):
            save(cfg)
            loaded = load()

        assert loaded.chat_backend == "ollama"
        assert loaded.embedding_backend == "ollama"
        assert loaded.ollama.chat_url == "http://myhost:11434"
        assert loaded.ollama.chat_model == "llama3.2"
        assert loaded.ollama.embedding_url == "http://myhost:11434"
        assert loaded.ollama.embedding_model == "nomic-embed"

    def test_openai_roundtrip(self, tmp_path: Path) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
            ),
        )
        config_path = str(tmp_path / "config.toml")
        with (
            mock.patch("localmelo.support.config.CONFIG_PATH", config_path),
            mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
        ):
            save(cfg)
            loaded = load()

        assert loaded.chat_backend == "openai"
        assert loaded.embedding_backend == "none"
        assert loaded.openai.api_key_env == "OPENAI_API_KEY"
        assert loaded.openai.chat_model == "gpt-4o"

    def test_vllm_roundtrip(self, tmp_path: Path) -> None:
        cfg = Config(
            chat_backend="vllm",
            embedding_backend="vllm",
            vllm=LocalBackendConfig(
                chat_url="http://localhost:8000",
                chat_model="meta-llama/Llama-3-8B",
                embedding_url="http://localhost:8000",
                embedding_model="BAAI/bge-small-en",
            ),
        )
        config_path = str(tmp_path / "config.toml")
        with (
            mock.patch("localmelo.support.config.CONFIG_PATH", config_path),
            mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
        ):
            save(cfg)
            loaded = load()

        assert loaded.chat_backend == "vllm"
        assert loaded.embedding_backend == "vllm"
        assert loaded.vllm.chat_url == "http://localhost:8000"
        assert loaded.vllm.chat_model == "meta-llama/Llama-3-8B"

    def test_load_missing_file_returns_default(self, tmp_path: Path) -> None:
        config_path = str(tmp_path / "nonexistent.toml")
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == ""
        assert loaded.embedding_backend == ""
        assert not loaded.is_configured


# -- Config validation --


class TestConfigValidation:
    """Config.validate() should catch misconfigurations."""

    def test_empty_chat_backend(self) -> None:
        cfg = Config(chat_backend="", embedding_backend="none")
        errors = cfg.validate()
        assert len(errors) == 1
        assert "chat_backend is not set" in errors[0]

    def test_invalid_chat_backend(self) -> None:
        cfg = Config(chat_backend="not-a-backend", embedding_backend="none")
        errors = cfg.validate()
        assert any("not recognised" in e for e in errors)

    def test_empty_embedding_backend(self) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="")
        errors = cfg.validate()
        assert any("embedding_backend is not set" in e for e in errors)

    def test_invalid_embedding_backend(self) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="bad")
        errors = cfg.validate()
        assert any("not recognised" in e for e in errors)

    def test_none_not_valid_as_chat_backend(self) -> None:
        """'none' is not a valid chat_backend."""
        cfg = Config(chat_backend="none", embedding_backend="none")
        errors = cfg.validate()
        assert any("not recognised" in e for e in errors)

    def test_cloud_vendor_not_valid_as_embedding_backend(self) -> None:
        """Cloud vendors are not valid embedding backends."""
        cfg = Config(chat_backend="openai", embedding_backend="openai")
        errors = cfg.validate()
        assert any("not recognised" in e for e in errors)

    def test_mlc_missing_url(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(chat_url="", chat_model="Qwen3-1.7B"),
        )
        errors = cfg.validate()
        assert any("chat_url" in e for e in errors)

    def test_mlc_missing_model(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(chat_url="http://127.0.0.1:8400", chat_model=""),
        )
        errors = cfg.validate()
        assert any("chat_model" in e for e in errors)

    def test_mlc_valid(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
        )
        assert cfg.validate() == []

    def test_ollama_missing_url(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="none",
            ollama=LocalBackendConfig(chat_url="", chat_model="llama3"),
        )
        errors = cfg.validate()
        assert any("chat_url" in e for e in errors)

    def test_ollama_missing_model(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="none",
            ollama=LocalBackendConfig(chat_url="http://localhost:11434", chat_model=""),
        )
        errors = cfg.validate()
        assert any("chat_model" in e for e in errors)

    def test_ollama_valid(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="none",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434", chat_model="qwen3:8b"
            ),
        )
        assert cfg.validate() == []

    def test_openai_missing_api_key_env(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(api_key_env="", chat_model="gpt-4o"),
        )
        errors = cfg.validate()
        assert any("api_key_env" in e for e in errors)

    def test_openai_env_var_not_set(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY_NONEXISTENT"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_var, None)
            cfg = Config(
                chat_backend="openai",
                embedding_backend="none",
                openai=CloudVendorConfig(api_key_env=env_var, chat_model="gpt-4o"),
            )
            errors = cfg.validate()
            assert any(env_var in e for e in errors)

    def test_openai_env_var_set(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "sk-test123"}):
            cfg = Config(
                chat_backend="openai",
                embedding_backend="none",
                openai=CloudVendorConfig(api_key_env=env_var, chat_model="gpt-4o"),
            )
            assert cfg.validate() == []

    def test_openai_missing_model(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "sk-test123"}):
            cfg = Config(
                chat_backend="openai",
                embedding_backend="none",
                openai=CloudVendorConfig(api_key_env=env_var, chat_model=""),
            )
            errors = cfg.validate()
            assert any("chat_model" in e for e in errors)

    def test_ollama_embedding_empty_model_validation_error(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_url="http://localhost:11434",
                embedding_model="",
            ),
        )
        errors = cfg.validate()
        assert any("embedding_model" in e for e in errors)

    def test_split_ollama_embedding_no_url_validation_error(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "sk-test123"}):
            cfg = Config(
                chat_backend="openai",
                embedding_backend="ollama",
                openai=CloudVendorConfig(
                    api_key_env=env_var,
                    chat_model="gpt-4o",
                ),
                ollama=LocalBackendConfig(
                    chat_url="",
                    chat_model="",
                    embedding_url="",
                    embedding_model="nomic-embed-text",
                ),
            )
            errors = cfg.validate()
            assert any("embedding_url" in e for e in errors)

    def test_bad_gateway_port(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
            gateway=GatewayConfig(port=99999),
        )
        errors = cfg.validate()
        assert any("gateway" in e and "port" in e for e in errors)

    def test_validate_or_raise(self) -> None:
        cfg = Config(chat_backend="", embedding_backend="")
        with pytest.raises(ConfigError, match="Configuration errors"):
            cfg.validate_or_raise()

    def test_validate_or_raise_passes(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
        )
        cfg.validate_or_raise()  # should not raise


# -- has_embedding property --


class TestHasEmbedding:
    def test_mlc_embedding(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400",
                chat_model="Qwen3-4B",
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
        )
        assert cfg.has_embedding is True

    def test_ollama_embedding(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_url="http://localhost:11434",
                embedding_model="nomic-embed",
            ),
        )
        assert cfg.has_embedding is True

    def test_ollama_embedding_empty_model(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_url="http://localhost:11434",
                embedding_model="",
            ),
        )
        assert cfg.has_embedding is False

    def test_none_embedding(self) -> None:
        cfg = Config(chat_backend="openai", embedding_backend="none")
        assert cfg.has_embedding is False

    def test_empty_embedding(self) -> None:
        cfg = Config(chat_backend="mlc", embedding_backend="")
        assert cfg.has_embedding is False

    def test_openai_chat_with_mlc_embedding(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
        )
        assert cfg.has_embedding is True

    def test_openai_chat_with_ollama_embedding(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="",
                embedding_url="http://localhost:11434",
                embedding_model="nomic-embed",
            ),
        )
        assert cfg.has_embedding is True

    def test_split_ollama_embedding_empty_urls_no_embedding(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="",
                chat_model="",
                embedding_url="",
                embedding_model="nomic-embed-text",
            ),
        )
        assert cfg.has_embedding is False


# -- Migration tests --


class TestMigration:
    """load() should migrate old configs to the new format."""

    def test_migrate_online_without_local_embedding(self, tmp_path: Path) -> None:
        """Old backend='online' without local_embedding -> openai + none."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'backend = "online"\n\n'
            "[online]\n"
            'provider = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
            'chat_model = "gpt-4o"\n'
            "local_embedding = false\n"
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "openai"
        assert loaded.embedding_backend == "none"
        assert loaded.openai.api_key_env == "OPENAI_API_KEY"
        assert loaded.openai.chat_model == "gpt-4o"

    def test_migrate_online_with_local_embedding(self, tmp_path: Path) -> None:
        """Old backend='online' with local_embedding=true -> openai + mlc."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'backend = "online"\n\n'
            "[online]\n"
            'provider = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
            'chat_model = "gpt-4o"\n'
            "local_embedding = true\n"
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "openai"
        assert loaded.embedding_backend == "mlc"

    def test_migrate_online_gemini(self, tmp_path: Path) -> None:
        """Old backend='online' with provider='gemini' -> gemini vendor key."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'backend = "online"\n\n'
            "[online]\n"
            'provider = "gemini"\n'
            'api_key_env = "GEMINI_API_KEY"\n'
            'chat_model = "gemini-2.0-flash"\n'
            "local_embedding = false\n"
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "gemini"
        assert loaded.embedding_backend == "none"
        assert loaded.gemini.api_key_env == "GEMINI_API_KEY"
        assert loaded.gemini.chat_model == "gemini-2.0-flash"

    def test_migrate_mlc_llm_legacy(self, tmp_path: Path) -> None:
        """Old backend='mlc-llm' -> mlc + mlc."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'backend = "mlc-llm"\n\n'
            "[mlc]\n"
            'chat_model = "Qwen3-1.7B"\n'
            "chat_port = 8400\n"
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "mlc"
        assert loaded.embedding_backend == "mlc"
        assert loaded.mlc.chat_model == "Qwen3-1.7B"
        assert loaded.mlc.chat_url == "http://127.0.0.1:8400/v1"

    def test_migrate_ollama_without_embedding(self, tmp_path: Path) -> None:
        """Old backend='ollama' without embedding_model -> ollama + none."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'backend = "ollama"\n\n'
            "[ollama]\n"
            'chat_url = "http://localhost:11434"\n'
            'chat_model = "qwen3:8b"\n'
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "ollama"
        assert loaded.embedding_backend == "none"

    def test_migrate_ollama_with_embedding(self, tmp_path: Path) -> None:
        """Old backend='ollama' with embedding_model -> ollama + ollama."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'backend = "ollama"\n\n'
            "[ollama]\n"
            'chat_url = "http://localhost:11434"\n'
            'chat_model = "qwen3:8b"\n'
            'embedding_model = "nomic-embed-text"\n'
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "ollama"
        assert loaded.embedding_backend == "ollama"
        assert loaded.ollama.embedding_model == "nomic-embed-text"

    def test_migrate_chat_backend_mlc_llm(self, tmp_path: Path) -> None:
        """New-ish format with chat_backend='mlc-llm' -> migrated to 'mlc'."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'chat_backend = "mlc-llm"\n'
            'embedding_backend = "mlc-llm"\n\n'
            "[mlc]\n"
            'chat_model = "Qwen3-4B"\n'
            "chat_port = 8400\n"
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "mlc"
        assert loaded.embedding_backend == "mlc"
        assert loaded.mlc.chat_url == "http://127.0.0.1:8400/v1"

    def test_migrate_chat_backend_cloud_api(self, tmp_path: Path) -> None:
        """New-ish format with chat_backend='cloud_api' -> migrated to vendor key."""
        config_path = str(tmp_path / "config.toml")
        content = (
            'chat_backend = "cloud_api"\n'
            'embedding_backend = "none"\n\n'
            "[cloud_api]\n"
            'provider = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
            'chat_model = "gpt-4o"\n'
        )
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "openai"
        assert loaded.embedding_backend == "none"
        assert loaded.openai.api_key_env == "OPENAI_API_KEY"
        assert loaded.openai.chat_model == "gpt-4o"

    def test_new_format_not_migrated(self, tmp_path: Path) -> None:
        """New format with vendor-specific keys should NOT trigger migration."""
        config_path = str(tmp_path / "config.toml")
        content = 'chat_backend = "openai"\n' 'embedding_backend = "ollama"\n'
        (tmp_path / "config.toml").write_text(content)
        with mock.patch("localmelo.support.config.CONFIG_PATH", config_path):
            loaded = load()
        assert loaded.chat_backend == "openai"
        assert loaded.embedding_backend == "ollama"


# -- Fail-fast config validation --


class TestFailFastValidation:
    def test_multiple_errors_collected(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(api_key_env="", chat_model=""),
        )
        errors = cfg.validate()
        # Should catch: missing api_key_env, missing chat_model
        assert len(errors) >= 2

    def test_validate_or_raise_includes_all_errors(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(api_key_env="", chat_model=""),
        )
        with pytest.raises(ConfigError) as exc_info:
            cfg.validate_or_raise()
        msg = str(exc_info.value)
        assert "api_key_env" in msg
        assert "chat_model" in msg

    def test_empty_chat_backend_returns_early(self) -> None:
        cfg = Config(chat_backend="", embedding_backend="none")
        errors = cfg.validate()
        assert len(errors) == 1
        assert "chat_backend is not set" in errors[0]

    def test_invalid_chat_backend_returns_early(self) -> None:
        cfg = Config(chat_backend="nope", embedding_backend="none")
        errors = cfg.validate()
        assert len(errors) == 1
        assert "not recognised" in errors[0]

    def test_valid_config_returns_no_errors(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
            gateway=GatewayConfig(port=8401, host="127.0.0.1"),
        )
        assert cfg.validate() == []
        cfg.validate_or_raise()


# -- Deployment matrix validation --


class TestDeploymentMatrix:
    """Validate the target deployment combinations."""

    def test_ollama_chat_ollama_embedding(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_url="http://localhost:11434",
                embedding_model="nomic-embed",
            ),
        )
        assert cfg.is_configured
        assert cfg.has_embedding

    def test_openai_chat_ollama_embedding(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "sk-test123"}):
            cfg = Config(
                chat_backend="openai",
                embedding_backend="ollama",
                openai=CloudVendorConfig(
                    api_key_env=env_var,
                    chat_model="gpt-4o",
                ),
                ollama=LocalBackendConfig(
                    chat_url="http://localhost:11434",
                    chat_model="",
                    embedding_url="http://localhost:11434",
                    embedding_model="nomic-embed",
                ),
            )
            assert cfg.is_configured
            assert cfg.has_embedding

    def test_mlc_chat_mlc_embedding(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400",
                chat_model="Qwen3-8B",
                embedding_url="http://127.0.0.1:8400",
                embedding_model="Qwen3-Embedding-0.6B",
            ),
        )
        assert cfg.is_configured
        assert cfg.has_embedding

    def test_openai_chat_mlc_embedding(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "sk-test123"}):
            cfg = Config(
                chat_backend="openai",
                embedding_backend="mlc",
                openai=CloudVendorConfig(
                    api_key_env=env_var,
                    chat_model="gpt-4o",
                ),
                mlc=LocalBackendConfig(
                    embedding_url="http://127.0.0.1:8400",
                    embedding_model="Qwen3-Embedding-0.6B",
                ),
            )
            assert cfg.is_configured
            assert cfg.has_embedding

    def test_openai_chat_no_embedding(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "sk-test123"}):
            cfg = Config(
                chat_backend="openai",
                embedding_backend="none",
                openai=CloudVendorConfig(
                    api_key_env=env_var,
                    chat_model="gpt-4o",
                ),
            )
            assert cfg.is_configured
            assert cfg.has_embedding is False

    def test_vllm_chat_vllm_embedding(self) -> None:
        cfg = Config(
            chat_backend="vllm",
            embedding_backend="vllm",
            vllm=LocalBackendConfig(
                chat_url="http://localhost:8000",
                chat_model="meta-llama/Llama-3-8B",
                embedding_url="http://localhost:8001",
                embedding_model="BAAI/bge-small-en",
            ),
        )
        assert cfg.is_configured
        assert cfg.has_embedding

    def test_sglang_chat_sglang_embedding(self) -> None:
        cfg = Config(
            chat_backend="sglang",
            embedding_backend="sglang",
            sglang=LocalBackendConfig(
                chat_url="http://localhost:30000",
                chat_model="meta-llama/Llama-3-8B",
                embedding_url="http://localhost:30001",
                embedding_model="BAAI/bge-small-en",
            ),
        )
        assert cfg.is_configured
        assert cfg.has_embedding

    def test_gemini_chat_no_embedding(self) -> None:
        env_var = "_LOCALMELO_TEST_KEY"
        with mock.patch.dict(os.environ, {env_var: "gem-test"}):
            cfg = Config(
                chat_backend="gemini",
                embedding_backend="none",
                gemini=CloudVendorConfig(
                    api_key_env=env_var,
                    chat_model="gemini-2.0-flash",
                ),
            )
            assert cfg.is_configured
            assert cfg.has_embedding is False


# -- Chat request timeout (request_timeout_s) --


def _timeout_cfg(request_timeout_s: Any = None) -> Config:
    """A valid vllm config, with *request_timeout_s* as given."""
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        request_timeout_s=request_timeout_s,
        vllm=LocalBackendConfig(
            chat_url="http://127.0.0.1:8000/v1",
            chat_model="Qwen3-8B",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        openai=CloudVendorConfig(api_key_env="OPENAI_API_KEY", chat_model="gpt-4o"),
    )


# What save() wrote for _timeout_cfg() before request_timeout_s existed. A
# config without the key must still be saved byte for byte like this.
_SAVED_BEFORE_REQUEST_TIMEOUT = """\
chat_backend = "vllm"
embedding_backend = "none"

[mlc]
chat_url = ""
chat_model = ""
embedding_url = ""
embedding_model = ""

[ollama]
chat_url = ""
chat_model = ""
embedding_url = ""
embedding_model = ""

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B"
embedding_url = ""
embedding_model = ""
extra_body = {chat_template_kwargs = {enable_thinking = false}}

[sglang]
chat_url = ""
chat_model = ""
embedding_url = ""
embedding_model = ""

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"
base_url = ""

[gemini]
api_key_env = ""
chat_model = ""
base_url = ""

[anthropic]
api_key_env = ""
chat_model = ""
base_url = ""

[nvidia]
api_key_env = ""
chat_model = ""
base_url = ""

[gateway]
port = 8401
host = "127.0.0.1"
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Iterator[Path]:
    """config.toml in *tmp_path*, which save() and load() use meanwhile."""
    path = tmp_path / "config.toml"
    with (
        mock.patch("localmelo.support.config.CONFIG_PATH", str(path)),
        mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
    ):
        yield path


class TestRequestTimeoutConfig:
    def test_unset_by_default(self, config_path: Path) -> None:
        assert Config().request_timeout_s is None
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n', encoding="utf-8"
        )
        assert load().request_timeout_s is None

    def test_a_config_without_it_is_saved_as_before(self, config_path: Path) -> None:
        save(_timeout_cfg())
        assert config_path.read_bytes() == _SAVED_BEFORE_REQUEST_TIMEOUT.encode()
        loaded = load()
        assert loaded.request_timeout_s is None
        assert loaded.validate() == []

    @pytest.mark.parametrize(
        ("value", "written"), [(600, "600"), (450.5, "450.5"), (0.25, "0.25")]
    )
    def test_it_is_saved_before_the_first_table_and_loaded_back(
        self, config_path: Path, value: float, written: str
    ) -> None:
        save(_timeout_cfg(value))
        # One more line in the top-level block, after embedding_backend.
        assert config_path.read_text(encoding="utf-8") == (
            _SAVED_BEFORE_REQUEST_TIMEOUT.replace(
                'embedding_backend = "none"\n',
                f'embedding_backend = "none"\nrequest_timeout_s = {written}\n',
                1,
            )
        )
        loaded = load()
        assert loaded.request_timeout_s == value
        assert type(loaded.request_timeout_s) is type(value)  # 600 stays an int
        assert loaded.validate() == []
        # Saving what was loaded writes the same file again.
        before = config_path.read_bytes()
        save(loaded)
        assert config_path.read_bytes() == before

    def test_a_legacy_file_keeps_it(self, config_path: Path) -> None:
        config_path.write_text(
            'backend = "ollama"\n'
            "request_timeout_s = 900\n\n"
            "[ollama]\n"
            'chat_url = "http://localhost:11434"\n'
            'chat_model = "qwen3:8b"\n',
            encoding="utf-8",
        )
        loaded = load()
        assert (loaded.chat_backend, loaded.request_timeout_s) == ("ollama", 900)

    def test_a_key_inside_a_table_is_not_the_top_level_key(
        self, config_path: Path
    ) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\n'
            'embedding_backend = "none"\n\n'
            "[gateway]\n"
            "request_timeout_s = 900\n",
            encoding="utf-8",
        )
        assert load().request_timeout_s is None

    def test_a_bad_value_is_loaded_for_validate_to_report(
        self, config_path: Path
    ) -> None:
        save(_timeout_cfg())
        text = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            text.replace("\n\n[mlc]", '\nrequest_timeout_s = "600"\n\n[mlc]', 1),
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.request_timeout_s == "600"
        assert loaded.validate() == [
            "request_timeout_s must be a finite number of seconds > 0, got '600'"
        ]

    @pytest.mark.parametrize(
        "value",
        ["600", True, False, 0, 0.0, -1, -0.5, math.inf, -math.inf, math.nan, 10**400],
        ids=[
            "string",
            "true",
            "false",
            "zero",
            "zero-float",
            "negative",
            "negative-float",
            "inf",
            "minus-inf",
            "nan",
            "too-large-for-a-float",
        ],
    )
    def test_validate_reports_a_value_that_is_not_a_positive_finite_number(
        self, value: Any
    ) -> None:
        cfg = _timeout_cfg(value)
        errors = cfg.validate()
        assert len(errors) == 1
        assert errors[0].startswith(
            "request_timeout_s must be a finite number of seconds > 0, got "
        )
        with pytest.raises(ConfigError, match="request_timeout_s"):
            cfg.validate_or_raise()

    @pytest.mark.parametrize("value", [600, 0.5, 1e-3, 86_400.0])
    def test_validate_accepts_a_positive_finite_number(self, value: float) -> None:
        assert _timeout_cfg(value).validate() == []

    def test_it_is_reported_with_the_other_errors(self) -> None:
        cfg = _timeout_cfg(-1)
        cfg.gateway.port = 0
        errors = cfg.validate()
        assert any(e.startswith("request_timeout_s ") for e in errors)
        assert any(e.startswith("[gateway] port=0") for e in errors)

    def test_reconfigure_keeps_it(self, config_path: Path) -> None:
        """melo --reconfigure starts from the saved config and saves it again."""
        save(_timeout_cfg(600))
        seen: list[Any] = []

        def setup_ollama(cfg: Config) -> bool:
            seen.append(cfg.request_timeout_s)
            cfg.ollama.chat_url = "http://localhost:11434"
            cfg.ollama.chat_model = "qwen3:8b"
            return True

        import localmelo.support.onboard as onboard

        with (
            # 2 = ollama in CHAT_BACKENDS, 5 = none in EMBEDDING_BACKENDS
            mock.patch.object(onboard, "_ask_choice", side_effect=[2, 5]),
            mock.patch.dict(onboard._CHAT_SETUP, {"ollama": setup_ollama}),
            mock.patch.object(onboard, "_run_probes", return_value=True),
            mock.patch("builtins.print"),
        ):
            result = onboard.run_backend_setup()

        assert seen == [600]  # the setup started from the saved config
        assert result is not None and result.request_timeout_s == 600
        loaded = load()
        assert (loaded.chat_backend, loaded.request_timeout_s) == ("ollama", 600)
        assert "\nrequest_timeout_s = 600\n" in config_path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("value", [0, "600"], ids=["zero", "string"])
    def test_reconfigure_reports_a_bad_value_as_a_failed_probe(
        self, config_path: Path, capsys: pytest.CaptureFixture[str], value: Any
    ) -> None:
        """melo --reconfigure names the key instead of probing with the value.

        The setup does not validate the config before it probes, so the
        chat probe checks the value itself, before it builds a provider.
        """
        save(_timeout_cfg(value))
        before = config_path.read_bytes()

        import localmelo.support.onboard as onboard

        with (
            # 3 = vllm in CHAT_BACKENDS, 5 = none in EMBEDDING_BACKENDS
            mock.patch.object(onboard, "_ask_choice", side_effect=[3, 5]),
            # Keep the saved [vllm] section as it is.
            mock.patch.dict(onboard._CHAT_SETUP, {"vllm": lambda cfg: True}),
            # "[r]etry, [s]ave anyway, [N]o" is answered with its default, No.
            mock.patch("builtins.input", return_value="") as ask,
            # No provider is built, so nothing is sent to the configured URL.
            mock.patch(
                "localmelo.support.onboard_probe.get_backend",
                side_effect=AssertionError("the probe built a provider"),
            ),
        ):
            result = onboard.run_backend_setup()

        assert result is None
        ask.assert_called_once_with("  [r]etry, [s]ave anyway, [N]o [N]: ")
        out = capsys.readouterr().out
        assert (
            "Probe failed: chat probe failed: request_timeout_s must be a finite "
            f"number of seconds > 0, got {value!r}\n" in out
        )
        assert "config not saved" in out
        assert config_path.read_bytes() == before


def test_the_mcp_config_key_round_trips(tmp_path: Path) -> None:
    from localmelo.support import config

    path = tmp_path / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        cfg = config.Config(chat_backend="vllm", embedding_backend="none")
        cfg.mcp_config = "~/mcp.json"
        config.save(cfg)
        assert config.load().mcp_config == "~/mcp.json"
        cfg.mcp_config = ""
        config.save(cfg)
        assert "mcp_config" not in path.read_text(encoding="utf-8")


# -- The scheduler's settings (S2) --


class TestSchedulerConfig:
    def test_scheduler_defaults_are_not_written(self, config_path: Path) -> None:
        save(_timeout_cfg())
        # The exact file of a config without scheduler settings is unchanged.
        assert config_path.read_text(encoding="utf-8") == _SAVED_BEFORE_REQUEST_TIMEOUT

    def test_a_non_default_scheduler_section_round_trips(
        self, config_path: Path
    ) -> None:
        cfg = _timeout_cfg()
        cfg.scheduler = SchedulerConfig(
            policy="class",
            n_srv=2,
            r_f=0,
            n_fg=0,
            call_log=False,
            call_log_dir="/tmp/x",
        )
        save(cfg)
        assert "[scheduler]" in config_path.read_text(encoding="utf-8")
        loaded = load()
        assert loaded.scheduler.policy == "class"
        assert loaded.scheduler.n_srv == 2
        assert loaded.scheduler.r_f == 0
        assert loaded.scheduler.n_fg == 0
        assert loaded.scheduler.call_log is False
        assert loaded.scheduler.call_log_dir == "/tmp/x"
        assert loaded.validate() == []

    @pytest.mark.parametrize(
        "changes",
        [
            {"policy": "x"},
            {"n_srv": 0},
            {"n_srv": True},
            {"r_f": -1},
            {"n_fg": -1},
            {"call_log": "yes"},
            {"call_log_dir": 3},
        ],
        ids=["policy", "n_srv", "n_srv-bool", "r_f", "n_fg", "call_log", "dir"],
    )
    def test_a_bad_scheduler_value_is_a_config_error(
        self, changes: dict[str, Any]
    ) -> None:
        cfg = _timeout_cfg()
        cfg.scheduler = SchedulerConfig(**changes)
        errors = cfg.validate()
        assert any(error.startswith("[scheduler]") for error in errors)
        with pytest.raises(ConfigError, match=r"\[scheduler\]"):
            cfg.validate_or_raise()

    def test_a_non_table_scheduler_is_ignored_on_load(self, config_path: Path) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\nscheduler = "off"\n',
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.scheduler == SchedulerConfig()

    def test_an_unknown_scheduler_key_is_ignored_on_load(
        self, config_path: Path
    ) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[scheduler]\npolicy = "fcfs"\nn_fg = 0\nfuture_key = 1\n',
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.scheduler.policy == "fcfs"
        assert loaded.scheduler.n_fg == 0
        assert not hasattr(loaded.scheduler, "future_key")
