"""Tests for the support layer: config roundtrip, validation, embedding."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from localmelo.support.backends.registry import ensure_defaults_registered
from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    ConfigError,
    GatewayConfig,
    LocalBackendConfig,
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
