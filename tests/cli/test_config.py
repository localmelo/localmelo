"""Tests for the support layer: config roundtrip, validation, embedding."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.support import config
from localmelo.support.backends.registry import ensure_defaults_registered
from localmelo.support.config import (
    CliConfig,
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


def _changed(value: Any) -> Any:
    """A value of the same kind as *value* that is not *value*."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, list):
        return ["a"]
    if isinstance(value, dict):
        return {"k": "v"}
    return 7  # a number that is unset by default


def test_every_setting_is_saved_alone_and_loaded_back(config_path: Path) -> None:
    """Each field of the config, changed on its own, is the only key the file gains."""
    settings_by_key = config.flat(Config())
    bad = []
    for key, default in settings_by_key.items():
        if key == "pricing":  # tables of its own, tested with the prices
            continue
        new = _changed(default)
        cfg = Config()
        config.set_key(cfg, key, new)
        save(cfg)
        section, dot, name = key.rpartition(".")
        want: dict[str, Any] = {"chat_backend": "", "embedding_backend": ""}
        if dot:
            want[section] = {name: new}
        else:
            want[name] = new
        written = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if written != want or config.get_key(load(), key) != new:
            bad.append(key)
    assert bad == []
    assert len(settings_by_key) > 100  # the walk found the tables


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

    def test_a_file_in_the_old_single_backend_format_has_no_backend(
        self, config_path: Path
    ) -> None:
        config_path.write_text(
            'backend = "ollama"\n\n[ollama]\nchat_model = "qwen3:8b"\n',
            encoding="utf-8",
        )
        assert "chat_backend is not set" in load().validate()[0]

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


# What save() writes for _timeout_cfg(): the two backends, then only what is not
# at its default, a table at a time.
_SAVED = """\
chat_backend = "vllm"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B"
extra_body = {chat_template_kwargs = {enable_thinking = false}}

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"
"""

# What the version before wrote for the same config: every table, and the keys
# that were always written. It must load to the same config.
_SAVED_BEFORE_DEFAULTS_WERE_LEFT_OUT = """\
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


class TestSavedFile:
    def test_only_what_differs_from_the_defaults_is_written(
        self, config_path: Path
    ) -> None:
        save(_timeout_cfg())
        assert config_path.read_text(encoding="utf-8") == _SAVED

    def test_a_file_the_earlier_version_wrote_loads_to_the_same_config(
        self, config_path: Path
    ) -> None:
        config_path.write_text(_SAVED_BEFORE_DEFAULTS_WERE_LEFT_OUT, encoding="utf-8")
        loaded = load()
        assert loaded == _timeout_cfg()
        save(loaded)
        assert config_path.read_text(encoding="utf-8") == _SAVED

    def test_blanks_around_a_model_name_or_an_address_are_dropped_when_read(
        self, config_path: Path
    ) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "vllm"\n\n'
            '[vllm]\nchat_url = "http://h:1/v1 "\nchat_model = " m"\n'
            'embedding_url = "http://h:2/v1"\n'
            'embedding_model = "Qwen3-Embedding-0.6B "\n\n'
            '[openai]\napi_key_env = "KEY "\napi_key = " keychain:localmelo.k "\n',
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.vllm.chat_url == "http://h:1/v1"
        assert loaded.vllm.chat_model == "m"
        assert loaded.vllm.embedding_model == "Qwen3-Embedding-0.6B"
        assert loaded.openai.api_key_env == "KEY"
        assert loaded.openai.api_key == " keychain:localmelo.k "  # never the key


class TestWrongKindOfValue:
    def test_a_port_in_quotes_is_a_sentence_and_not_a_traceback(
        self, config_path: Path
    ) -> None:
        save(_timeout_cfg())
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + '\n[gateway]\nport = "8401"\n',
            encoding="utf-8",
        )
        assert load().validate() == [
            "[gateway] port must be a whole number, got '8401'"
        ]

    def test_a_quoted_false_in_approvals_stops_the_check(
        self, config_path: Path
    ) -> None:
        # "false" in quotes would read as true and turn on what the person
        # meant to turn off: [approvals] changes behaviour, so it is checked.
        save(_timeout_cfg())
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + '\n[approvals]\nauto_asks = "false"\n',
            encoding="utf-8",
        )
        assert load().validate() == [
            "[approvals] auto_asks must be true or false, got 'false'"
        ]

    def test_a_table_that_is_not_a_table_is_ignored(self, config_path: Path) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\ngateway = 5\n',
            encoding="utf-8",
        )
        assert load().gateway == GatewayConfig()

    def test_every_field_of_every_table_is_checked_and_nothing_raises(self) -> None:
        silent = []
        for key, default in config.flat(Config()).items():
            section, dot, name = key.rpartition(".")
            if dot:
                cfg = _timeout_cfg()
                config.set_key(cfg, key, 12345 if isinstance(default, str) else "oops")
                errors = [*cfg.validate(), *cfg.setting_errors(), *cfg.cli_errors()]
                if not any(
                    e.startswith(f"[{section}] {name} must be ") for e in errors
                ):
                    silent.append(key)
        assert silent == []


class TestRequestTimeoutConfig:
    def test_unset_by_default(self, config_path: Path) -> None:
        assert Config().request_timeout_s is None
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n', encoding="utf-8"
        )
        assert load().request_timeout_s is None

    @pytest.mark.parametrize(
        ("value", "written"), [(600, "600"), (450.5, "450.5"), (0.25, "0.25")]
    )
    def test_it_is_saved_before_the_first_table_and_loaded_back(
        self, config_path: Path, value: float, written: str
    ) -> None:
        save(_timeout_cfg(value))
        # One more line in the top-level block, after embedding_backend.
        assert config_path.read_text(encoding="utf-8") == _SAVED.replace(
            'embedding_backend = "none"\n',
            f'embedding_backend = "none"\nrequest_timeout_s = {written}\n',
            1,
        )
        loaded = load()
        assert loaded.request_timeout_s == value
        assert type(loaded.request_timeout_s) is type(value)  # 600 stays an int
        assert loaded.validate() == []
        # Saving what was loaded writes the same file again.
        before = config_path.read_bytes()
        save(loaded)
        assert config_path.read_bytes() == before

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
            text.replace("\n\n[vllm]", '\nrequest_timeout_s = "600"\n\n[vllm]', 1),
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

    @pytest.mark.parametrize(
        "changes",
        [
            {"policy": "x"},
            {"n_srv": 0},
            {"n_srv": True},
            {"r_f": -1},
            {"n_fg": -1},
            {"focus_s": 0},
            {"focus_s": True},
            {"focus_s": "10"},
            {"focus_s": float("inf")},
            {"call_log": "yes"},
            {"call_log_dir": 3},
        ],
        ids=[
            "policy",
            "n_srv",
            "n_srv-bool",
            "r_f",
            "n_fg",
            "focus-zero",
            "focus-bool",
            "focus-text",
            "focus-inf",
            "call_log",
            "dir",
        ],
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

    @pytest.mark.parametrize(
        "changes",
        [
            {"lock_gate": "x"},
            {"gpu_lock_file": 3},
            {"lock_self": 3},
            {"gpu_lock_file": "/tmp/a\x00b"},
            {"lock_self": "a\x00b"},
        ],
        ids=["gate", "path", "self", "path-nul", "self-nul"],
    )
    def test_a_bad_lock_setting_is_a_config_error(
        self, changes: dict[str, Any]
    ) -> None:
        cfg = _timeout_cfg()
        cfg.scheduler = SchedulerConfig(**changes)
        errors = cfg.validate()
        assert any(error.startswith("[scheduler]") for error in errors)
        with pytest.raises(ConfigError, match=r"\[scheduler\]"):
            cfg.validate_or_raise()

    def test_an_unknown_lock_key_is_ignored_on_load(self, config_path: Path) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[scheduler]\nlock_gate = "wait"\nlock_future = 1\n',
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.scheduler.lock_gate == "wait"
        assert not hasattr(loaded.scheduler, "lock_future")


class TestGatewayWorkspaceAndAtomicSave:

    def test_every_string_field_round_trips_quotes_backslashes_brackets_and_non_bmp(
        self, config_path: Path
    ) -> None:
        tricky = 'a"b\\c]d\U0001f680\x7f'
        cfg = Config(
            chat_backend="vllm",
            embedding_backend="ollama",
            mcp_config=tricky,
            vllm=LocalBackendConfig(
                chat_url=tricky,
                chat_model=tricky,
                embedding_url=tricky,
                embedding_model=tricky,
            ),
            openai=CloudVendorConfig(
                api_key_env=tricky,
                chat_model=tricky,
                base_url=tricky,
            ),
            gateway=GatewayConfig(port=8401, host=tricky, workspace=tricky),
        )
        save(cfg)
        loaded = load()
        assert loaded.mcp_config == tricky
        assert loaded.vllm.chat_url == tricky
        assert loaded.vllm.chat_model == tricky
        assert loaded.vllm.embedding_url == tricky
        assert loaded.vllm.embedding_model == tricky
        assert loaded.openai.api_key_env == tricky
        assert loaded.openai.chat_model == tricky
        assert loaded.openai.base_url == tricky
        assert loaded.gateway.host == tricky
        assert loaded.gateway.workspace == tricky

    def test_save_replaces_atomically_and_leaves_old_file_on_failure(
        self, config_path: Path
    ) -> None:
        save(_timeout_cfg())
        before = config_path.read_bytes()
        changed = _timeout_cfg(900)
        with (
            mock.patch(
                "localmelo.support.config.os.replace",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            save(changed)
        assert config_path.read_bytes() == before
        assert list(config_path.parent.glob("*.tmp")) == []

    def test_save_preserves_existing_file_mode_and_follows_symlink(
        self, tmp_path: Path
    ) -> None:
        real_dir = tmp_path / "dotfiles"
        real_dir.mkdir()
        real_file = real_dir / "config.toml"
        real_file.write_text('chat_backend = "vllm"\n', encoding="utf-8")
        os.chmod(real_file, 0o644)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        link_path = cache_dir / "config.toml"
        os.symlink(real_file, link_path)

        with (
            mock.patch("localmelo.support.config.CONFIG_PATH", str(link_path)),
            mock.patch("localmelo.support.config.CONFIG_DIR", str(cache_dir)),
        ):
            save(_timeout_cfg())
            loaded = load()

        assert link_path.is_symlink()
        assert os.path.realpath(link_path) == str(real_file.resolve())
        assert (real_file.stat().st_mode & 0o777) == 0o644
        assert loaded.chat_backend == "vllm"

    @pytest.mark.parametrize(
        "bad_ws",
        [5, "relative/path", "~/bad\nline", "~/bad\x85line", "~/bad\x00"],
        ids=["int", "relative", "newline", "nel-c1", "nul"],
    )
    def test_validate_rejects_invalid_gateway_workspace(
        self, config_path: Path, bad_ws: Any
    ) -> None:
        cfg = _timeout_cfg()
        cfg.gateway.workspace = bad_ws
        errors = cfg.validate()
        assert any(err.startswith("[gateway] workspace") for err in errors)
        with pytest.raises(ConfigError, match=r"\[gateway\] workspace"):
            cfg.validate_or_raise()

        if isinstance(bad_ws, int):
            config_path.write_text(
                'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
                '[vllm]\nchat_url = "http://127.0.0.1:8000/v1"\nchat_model = "m"\n\n'
                f"[gateway]\nworkspace = {bad_ws}\n",
                encoding="utf-8",
            )
            loaded = load()
            assert any(
                err.startswith("[gateway] workspace") for err in loaded.validate()
            )

    def test_gateway_workspace_rejects_root_home_and_symlinks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        link = tmp_path / "home-link"
        link.symlink_to(home, target_is_directory=True)
        cfg = _timeout_cfg()
        for path in ("/", "~/", str(home), str(link)):
            cfg.gateway.workspace = path
            assert any("[gateway] workspace" in error for error in cfg.validate()), path
        cfg.gateway.workspace = str(home / "project")
        assert not any("[gateway] workspace" in error for error in cfg.validate())


class TestCliStatusLineConfig:

    @pytest.mark.parametrize("value", ['"sometimes"', "true", "1", '["auto"]'])
    def test_a_bad_value_is_reported_and_is_not_a_config_error(
        self, config_path: Path, value: str
    ) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[vllm]\nchat_url = "http://127.0.0.1:8000/v1"\nchat_model = "m"\n\n'
            f"[cli]\nstatus_line = {value}\n",
            encoding="utf-8",
        )
        loaded = load()
        assert any("status_line" in error for error in loaded.cli_errors())
        loaded.validate_or_raise()


class TestCliColorConfig:

    def test_a_bare_number_reads_as_the_depth(self, config_path: Path) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[vllm]\nchat_url = "http://127.0.0.1:8000/v1"\nchat_model = "m"\n\n'
            "[cli]\ncolors = 16\n",
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.cli.colors == "16" and loaded.cli_errors() == []

    @pytest.mark.parametrize(
        ("key", "value"),
        [("colors", '"88"'), ("colors", "true"), ("background", '"grey"')],
    )
    def test_a_bad_value_is_reported_and_is_not_a_config_error(
        self, config_path: Path, key: str, value: str
    ) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[vllm]\nchat_url = "http://127.0.0.1:8000/v1"\nchat_model = "m"\n\n'
            f"[cli]\n{key} = {value}\n",
            encoding="utf-8",
        )
        loaded = load()
        assert any(f"[cli] {key}" in error for error in loaded.cli_errors())
        loaded.validate_or_raise()


class TestCliStreamConfig:

    def test_a_reasoning_tags_subtable_loads(self, config_path: Path) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[vllm]\nchat_url = "http://127.0.0.1:8000/v1"\nchat_model = "m"\n\n'
            "[cli]\nstream = true\n\n[cli.reasoning_tags]\n"
            '"stub-model" = "closing_only"\n',
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.cli.reasoning_tags == {"stub-model": "closing_only"}
        assert loaded.validate() == []

    @pytest.mark.parametrize(
        ("stream", "tags"),
        [
            ("yes", {}),
            (False, {"stub-model": "maybe"}),
            (False, {"": "none"}),
            (False, {"bad\nmodel": "none"}),
        ],
        ids=["stream-not-bool", "unknown-hint", "empty-model", "control-model"],
    )
    def test_invalid_cli_values_are_reported(
        self, stream: Any, tags: dict[str, str]
    ) -> None:
        cfg = _timeout_cfg()
        cfg.cli = CliConfig(reasoning_tags=tags)
        cfg.cli.stream = stream
        assert cfg.cli_errors()
        # A display preference: the UI warns and falls back, everything else
        # (direct mode, the gateway) never fails on it.
        assert not any(error.startswith("[cli]") for error in cfg.validate())

    @pytest.mark.parametrize(
        "section",
        [
            'stream = "no"',
            'reasoning_tags = "stub-model"',
            '[cli.reasoning_tags]\n"stub-model" = "sometimes"',
        ],
        ids=["stream-string", "tags-string", "unknown-hint"],
    )
    def test_a_bad_cli_table_is_loaded_for_validation(
        self, config_path: Path, section: str
    ) -> None:
        config_path.write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            '[vllm]\nchat_url = "http://127.0.0.1:8000/v1"\nchat_model = "m"\n\n'
            f"[cli]\n{section}\n",
            encoding="utf-8",
        )
        loaded = load()
        assert loaded.cli_errors()
        assert not any(error.startswith("[cli]") for error in loaded.validate())
        loaded.validate_or_raise()  # a bad [cli] is not a configuration error


# ── oMLX as another spelling of vllm ──

_OMLX_FILE = """\
chat_backend = "omlx"
embedding_backend = "omlx"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"
embedding_url = "http://127.0.0.1:8000/v1"
embedding_model = "Qwen3-Embedding-0.6B"
"""


class TestOmlxSpelling:
    """``omlx`` names the ``vllm`` backend: one adapter, one ``[vllm]`` table."""

    def test_it_loads_as_the_vllm_backend(self, config_path: Path) -> None:
        config_path.write_text(_OMLX_FILE, encoding="utf-8")
        cfg = load()
        assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "vllm")
        assert (cfg.chat_alias, cfg.embedding_alias) == ("omlx", "omlx")
        assert cfg.vllm.chat_model == "Qwen3-8B-4bit"
        assert cfg.validate() == []
        assert cfg.has_embedding

    def test_it_is_the_same_config_as_the_vllm_spelling(
        self, config_path: Path
    ) -> None:
        config_path.write_text(_OMLX_FILE, encoding="utf-8")
        as_omlx = load()
        config_path.write_text(_OMLX_FILE.replace('"omlx"', '"vllm"'), encoding="utf-8")
        as_vllm = load()
        assert as_omlx == as_vllm and as_vllm.chat_alias == ""

    def test_it_is_saved_as_written(self, config_path: Path) -> None:
        config_path.write_text(_OMLX_FILE, encoding="utf-8")
        cfg = load()
        cfg.vllm.chat_model = "Qwen3-32B-4bit"  # what /model does
        save(cfg)
        text = config_path.read_text(encoding="utf-8")
        assert text.startswith('chat_backend = "omlx"\nembedding_backend = "omlx"\n')
        again = load()
        assert (again.chat_alias, again.embedding_alias) == ("omlx", "omlx")
        assert again.vllm.chat_model == "Qwen3-32B-4bit"

    def test_each_backend_keeps_its_own_spelling(self, config_path: Path) -> None:
        config_path.write_text(
            _OMLX_FILE.replace('chat_backend = "omlx"', 'chat_backend = "vllm"'),
            encoding="utf-8",
        )
        cfg = load()
        assert (cfg.chat_alias, cfg.embedding_alias) == ("", "omlx")
        save(cfg)
        text = config_path.read_text(encoding="utf-8")
        assert text.startswith('chat_backend = "vllm"\nembedding_backend = "omlx"\n')

    def test_another_backend_is_saved_under_its_own_key(
        self, config_path: Path
    ) -> None:
        config_path.write_text(_OMLX_FILE, encoding="utf-8")
        cfg = load()
        cfg.chat_backend = cfg.embedding_backend = "ollama"
        save(cfg)
        text = config_path.read_text(encoding="utf-8")
        assert text.startswith(
            'chat_backend = "ollama"\nembedding_backend = "ollama"\n'
        )

    def test_the_vllm_spelling_is_still_saved_as_vllm(self, config_path: Path) -> None:
        config_path.write_text(_OMLX_FILE.replace('"omlx"', '"vllm"'), encoding="utf-8")
        save(load())
        text = config_path.read_text(encoding="utf-8")
        assert text.startswith('chat_backend = "vllm"\nembedding_backend = "vllm"\n')

    def test_use_chat_server_puts_the_server_in_the_config(self) -> None:
        cfg = Config(chat_backend="vllm", embedding_backend="vllm")
        cfg.chat_alias = cfg.embedding_alias = "omlx"
        cfg.vllm.chat_url = "http://h:1/v1"
        cfg.use_chat_server("http://h:1/v1", "m")  # the same server keeps its spelling
        assert (cfg.chat_alias, cfg.vllm.chat_model) == ("omlx", "m")
        cfg.use_chat_server("http://h:2/v1", "n", embedding_model="e")
        assert (cfg.chat_alias, cfg.embedding_alias) == ("", "")
        assert (cfg.embedding_backend, cfg.vllm.embedding_url) == (
            "vllm",
            "http://h:2/v1",
        )
        assert (cfg.vllm.chat_model, cfg.vllm.embedding_model) == ("n", "e")
        cfg.use_chat_server("http://h:3", "o", ollama=True)
        assert (cfg.chat_backend, cfg.ollama.chat_url, cfg.ollama.chat_model) == (
            "ollama",
            "http://h:3",
            "o",
        )

    @pytest.mark.parametrize("spelling", ["OMLX", "omlx ", "omlx2", "oMLX"])
    def test_another_spelling_is_not_recognised(
        self, config_path: Path, spelling: str
    ) -> None:
        config_path.write_text(
            _OMLX_FILE.replace('chat_backend = "omlx"', f'chat_backend = "{spelling}"'),
            encoding="utf-8",
        )
        errors = load().validate()
        assert len(errors) == 1 and "not recognised" in errors[0]
        assert errors[0].startswith(f"chat_backend {spelling!r}")

    def test_a_misspelt_embedding_backend_is_not_recognised(
        self, config_path: Path
    ) -> None:
        config_path.write_text(
            _OMLX_FILE.replace(
                'embedding_backend = "omlx"', 'embedding_backend = "omlxx"'
            ),
            encoding="utf-8",
        )
        errors = load().validate()
        assert len(errors) == 1 and "not recognised" in errors[0]
        assert errors[0].startswith("embedding_backend 'omlxx'")

    def test_the_vllm_table_is_still_checked(self, config_path: Path) -> None:
        config_path.write_text(
            'chat_backend = "omlx"\nembedding_backend = "none"\n\n[vllm]\n',
            encoding="utf-8",
        )
        assert load().validate() == [
            "[vllm] chat_url is empty. "
            "Set it to your oMLX URL (e.g. http://localhost:8000/v1).",
            "[vllm] chat_model is empty. Specify the oMLX model name.",
        ]


class TestSettingOverrides:
    """``-X`` values live in a layer ``load()`` applies and ``save()`` never writes."""

    @pytest.fixture(autouse=True)
    def _file(self, tmp_path: Path) -> Iterator[Path]:
        path = tmp_path / "config.toml"
        with (
            mock.patch("localmelo.support.config.CONFIG_PATH", str(path)),
            mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
        ):
            yield path

    def test_load_applies_them_and_save_writes_what_the_file_had(
        self, _file: Path
    ) -> None:
        saved = Config(
            chat_backend="ollama", ollama=LocalBackendConfig(chat_model="m1")
        )
        saved.retrieval.k = 5
        save(saved)
        config.set_overrides({"cli.stream": False, "retrieval.k": 9})
        cfg = load()
        assert (cfg.cli.stream, cfg.retrieval.k) == (False, 9)
        cfg.ollama.chat_model = "m2"
        save(cfg)
        text = _file.read_text(encoding="utf-8")
        assert 'chat_model = "m2"' in text and "k = 5" in text
        assert "stream" not in text and "[cli]" not in text and "k = 9" not in text
        assert cfg.retrieval.k == 9  # the caller's config is not touched
        assert load().cli.stream is False  # and the next load applies them again
        config.set_overrides({})
        assert (load().cli.stream, load().retrieval.k) == (True, 5)

    def test_a_value_the_caller_changed_is_theirs(self, _file: Path) -> None:
        save(Config(chat_backend="ollama"))
        config.set_overrides({"cli.stream": True, "retrieval.k": 9})
        cfg = load()
        cfg.retrieval.k = 6
        cfg.cli.stream = False
        save(cfg)
        config.set_overrides({})
        assert (load().retrieval.k, load().cli.stream) == (6, False)

    def test_a_setting_the_config_has_no_field_for_is_kept_and_ignored(self) -> None:
        config.set_overrides({"memory.dir": "/x", "agent.attempt_policy": "none"})
        assert load() == Config()
        assert config.overrides() == {
            "memory.dir": "/x",
            "agent.attempt_policy": "none",
        }

    def test_set_overrides_replaces_the_last_ones(self) -> None:
        config.set_overrides({"cli.stream": False})
        config.set_overrides({"scheduler.n_srv": 2})
        assert (load().cli.stream, load().scheduler.n_srv) == (True, 2)

    def test_setting_errors_need_no_backend(self) -> None:
        cfg = Config()
        assert cfg.setting_errors() == []
        cfg.cli.status_line = "loud"
        cfg.request_timeout_s = -1
        assert len(cfg.setting_errors()) == 2
        assert cfg.validate() == ["chat_backend is not set (run 'melo setup')"] + [
            e for e in cfg.validate()[1:]
        ]


class TestApprovalsConfig:
    """``[approvals] mode``: how a new chat starts (#339)."""

    def test_another_mode_is_a_setting_error(self) -> None:
        cfg = Config()
        cfg.approvals.mode = "yolo"
        assert any("[approvals] mode" in e for e in cfg.setting_errors())
