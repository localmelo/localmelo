"""extra_body: provider merge, backend wiring, and config load/save/migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

from localmelo.melo.schema import Message, ToolDef
from localmelo.support.backends.registry import ensure_defaults_registered, get_backend
from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    LocalBackendConfig,
    load,
    save,
)
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

ensure_defaults_registered()

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
LOW_EFFORT = {"chat_template_kwargs": {"reasoning_effort": "low"}}

ECHO = ToolDef(
    name="echo",
    description="echo",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)


class Recorder:
    """Fake transport that records JSON bodies."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self.reply)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _openai_reply() -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def _ollama_reply() -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": "ok"}}


# ── Provider merge ─────────────────────────────────────────────────────────


class TestOpenAIExtraBody:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("extra", [NO_THINKING, LOW_EFFORT])
    async def test_merged_into_payload(self, extra: dict[str, Any]) -> None:
        rec = Recorder(_openai_reply())
        llm = OpenAICompatLLM(
            base_url="http://omlx.test/v1",
            model="m",
            extra_body=extra,
            transport=rec.transport,
        )
        await llm.chat([Message(role="user", content="hi")], tools=[ECHO])
        body = rec.bodies[0]
        assert body["chat_template_kwargs"] == extra["chat_template_kwargs"]
        assert body["model"] == "m"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["tool_choice"] == "auto"
        await llm.close()

    @pytest.mark.asyncio
    async def test_sampling_fields_pass_through(self) -> None:
        rec = Recorder(_openai_reply())
        llm = OpenAICompatLLM(
            base_url="http://omlx.test/v1",
            model="m",
            extra_body={**NO_THINKING, "temperature": 0.2, "max_tokens": 512},
            transport=rec.transport,
        )
        await llm.chat([Message(role="user", content="hi")])
        assert rec.bodies[0]["temperature"] == 0.2
        assert rec.bodies[0]["max_tokens"] == 512
        await llm.close()

    @pytest.mark.asyncio
    async def test_no_extra_body_by_default(self) -> None:
        rec = Recorder(_openai_reply())
        llm = OpenAICompatLLM(
            base_url="http://omlx.test/v1", model="m", transport=rec.transport
        )
        await llm.chat([Message(role="user", content="hi")])
        assert set(rec.bodies[0]) == {"model", "messages"}
        await llm.close()

    @pytest.mark.asyncio
    async def test_caller_mutation_does_not_leak(self) -> None:
        extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        rec = Recorder(_openai_reply())
        llm = OpenAICompatLLM(
            base_url="http://omlx.test/v1",
            model="m",
            extra_body=extra,
            transport=rec.transport,
        )
        extra["chat_template_kwargs"]["enable_thinking"] = True
        extra["temperature"] = 1.0
        await llm.chat([Message(role="user", content="a")], tools=[ECHO])
        await llm.chat([Message(role="user", content="b")])
        for body in rec.bodies:
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            assert "temperature" not in body
        # The tools of the first request did not stick to later requests.
        assert "tools" not in rec.bodies[1]
        await llm.close()

    @pytest.mark.parametrize(
        "key", ["model", "messages", "tools", "tool_choice", "stream"]
    )
    def test_reserved_keys_rejected(self, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            OpenAICompatLLM(base_url="http://x/v1", model="m", extra_body={key: 1})

    def test_not_json_serializable_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            OpenAICompatLLM(
                base_url="http://x/v1", model="m", extra_body={"bad": object()}
            )


class TestOllamaExtraBody:
    @pytest.mark.asyncio
    async def test_can_turn_thinking_off(self) -> None:
        rec = Recorder(_ollama_reply())
        llm = OllamaNativeChat(
            base_url="http://ollama.test",
            model="qwen3:8b",
            extra_body={"think": False, "options": {"temperature": 0}},
            transport=rec.transport,
        )
        await llm.chat([Message(role="user", content="hi")])
        body = rec.bodies[0]
        assert body["think"] is False
        assert body["options"] == {"temperature": 0}
        assert body["stream"] is False
        assert body["model"] == "qwen3:8b"
        await llm.close()

    @pytest.mark.asyncio
    async def test_default_keeps_thinking_on(self) -> None:
        rec = Recorder(_ollama_reply())
        llm = OllamaNativeChat(
            base_url="http://ollama.test", model="qwen3:8b", transport=rec.transport
        )
        await llm.chat([Message(role="user", content="hi")])
        assert rec.bodies[0]["think"] is True
        await llm.close()

    def test_reserved_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="stream"):
            OllamaNativeChat(
                base_url="http://x", model="m", extra_body={"stream": True}
            )


# ── Backend wiring ─────────────────────────────────────────────────────────


class TestBackendsPassExtraBody:
    @pytest.mark.parametrize("key", ["mlc", "vllm", "sglang"])
    def test_local_openai_compatible(self, key: str) -> None:
        cfg = Config(chat_backend=key)
        setattr(
            cfg,
            key,
            LocalBackendConfig(
                chat_url="http://127.0.0.1:8000/v1",
                chat_model="Qwen3.8-27B-8bit",
                extra_body=NO_THINKING,
            ),
        )
        provider = get_backend(key).build_chat_provider(cfg)
        assert isinstance(provider, OpenAICompatLLM)
        assert provider.extra_body == NO_THINKING

    def test_ollama(self) -> None:
        cfg = Config(
            chat_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                extra_body={"think": False},
            ),
        )
        provider = get_backend("ollama").build_chat_provider(cfg)
        assert isinstance(provider, OllamaNativeChat)
        assert provider.extra_body == {"think": False}

    @pytest.mark.parametrize("key", ["openai", "gemini", "anthropic", "nvidia"])
    def test_cloud(self, key: str) -> None:
        cfg = Config(chat_backend=key)
        setattr(
            cfg,
            key,
            CloudVendorConfig(
                api_key_env="LOCALMELO_TEST_KEY",
                chat_model="m",
                extra_body=LOW_EFFORT,
            ),
        )
        provider = get_backend(key).build_chat_provider(cfg)
        assert isinstance(provider, OpenAICompatLLM)
        assert provider.extra_body == LOW_EFFORT


# ── Config ─────────────────────────────────────────────────────────────────


def _load_text(tmp_path: Path, content: str) -> Config:
    config_path = tmp_path / "config.toml"
    config_path.write_text(content)
    with mock.patch("localmelo.support.config.CONFIG_PATH", str(config_path)):
        return load()


def _save_and_load(tmp_path: Path, cfg: Config) -> tuple[Config, str]:
    config_path = str(tmp_path / "config.toml")
    with (
        mock.patch("localmelo.support.config.CONFIG_PATH", config_path),
        mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        loaded = load()
    return loaded, (tmp_path / "config.toml").read_text()


class TestExtraBodyConfig:
    def test_defaults_are_empty(self) -> None:
        assert LocalBackendConfig().extra_body == {}
        assert CloudVendorConfig().extra_body == {}
        # Each instance gets its own dict.
        assert LocalBackendConfig().extra_body is not LocalBackendConfig().extra_body

    def test_old_file_without_extra_body_loads(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'chat_backend = "vllm"\n'
            'embedding_backend = "none"\n\n'
            "[vllm]\n"
            'chat_url = "http://localhost:8000/v1"\n'
            'chat_model = "m"\n',
        )
        assert cfg.vllm.chat_model == "m"
        assert cfg.vllm.extra_body == {}
        assert cfg.validate() == []

    def test_table_form_loads(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'chat_backend = "vllm"\n'
            'embedding_backend = "none"\n\n'
            "[vllm]\n"
            'chat_url = "http://localhost:8000/v1"\n'
            'chat_model = "m"\n\n'
            "[vllm.extra_body]\n"
            "temperature = 0.6\n\n"
            "[vllm.extra_body.chat_template_kwargs]\n"
            "enable_thinking = false\n",
        )
        assert cfg.vllm.extra_body == {
            "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_roundtrip_local_and_cloud(self, tmp_path: Path) -> None:
        cfg = Config(
            chat_backend="vllm",
            embedding_backend="none",
            vllm=LocalBackendConfig(
                chat_url="http://127.0.0.1:8000/v1",
                chat_model="Qwen3.8-27B-8bit",
                extra_body={**NO_THINKING, "temperature": 0.7, "stop": ["</s>"]},
            ),
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
                extra_body=LOW_EFFORT,
            ),
        )
        loaded, _ = _save_and_load(tmp_path, cfg)
        assert loaded.vllm.extra_body == cfg.vllm.extra_body
        assert loaded.openai.extra_body == LOW_EFFORT
        assert loaded.mlc.extra_body == {}

    def test_save_without_extra_body_writes_no_line(self, tmp_path: Path) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="none",
            mlc=LocalBackendConfig(chat_url="http://127.0.0.1:8400/v1", chat_model="m"),
        )
        _, text = _save_and_load(tmp_path, cfg)
        assert "extra_body" not in text

    def test_flat_chat_template_kwargs_migrates(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'chat_backend = "sglang"\n'
            'embedding_backend = "none"\n\n'
            "[sglang]\n"
            'chat_url = "http://localhost:30000/v1"\n'
            'chat_model = "m"\n'
            "chat_template_kwargs = { enable_thinking = false }\n",
        )
        assert cfg.sglang.extra_body == NO_THINKING

        # Saving writes the canonical form; the flat key is gone.
        loaded, text = _save_and_load(tmp_path, cfg)
        assert loaded.sglang.extra_body == NO_THINKING
        assert "extra_body = {chat_template_kwargs = {enable_thinking = false}}" in (
            text
        )
        assert "\nchat_template_kwargs" not in text

    def test_explicit_extra_body_wins_over_flat_key(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'chat_backend = "vllm"\n'
            'embedding_backend = "none"\n\n'
            "[vllm]\n"
            'chat_url = "http://localhost:8000/v1"\n'
            'chat_model = "m"\n'
            "chat_template_kwargs = { enable_thinking = true }\n"
            "extra_body = { chat_template_kwargs = { enable_thinking = false } }\n",
        )
        assert cfg.vllm.extra_body == NO_THINKING

    def test_legacy_single_backend_file_keeps_extra_body(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'backend = "mlc-llm"\n\n'
            "[mlc]\n"
            'chat_model = "Qwen3-4B"\n'
            "chat_port = 8400\n"
            "chat_template_kwargs = { enable_thinking = false }\n",
        )
        assert cfg.chat_backend == "mlc"
        assert cfg.mlc.chat_url == "http://127.0.0.1:8400/v1"
        assert cfg.mlc.extra_body == NO_THINKING

    def test_legacy_cloud_api_section_keeps_extra_body(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'chat_backend = "cloud_api"\n'
            'embedding_backend = "none"\n\n'
            "[cloud_api]\n"
            'provider = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n'
            'chat_model = "gpt-4o"\n'
            'chat_template_kwargs = { reasoning_effort = "low" }\n',
        )
        assert cfg.chat_backend == "openai"
        assert cfg.openai.extra_body == LOW_EFFORT

    def test_validate_rejects_reserved_key(self) -> None:
        cfg = Config(
            chat_backend="vllm",
            embedding_backend="none",
            vllm=LocalBackendConfig(
                chat_url="http://localhost:8000/v1",
                chat_model="m",
                extra_body={"model": "other"},
            ),
        )
        errors = cfg.validate()
        assert any(e.startswith("[vllm]") and "'model'" in e for e in errors)

    def test_validate_rejects_non_table(self, tmp_path: Path) -> None:
        cfg = _load_text(
            tmp_path,
            'chat_backend = "vllm"\n'
            'embedding_backend = "none"\n\n'
            "[vllm]\n"
            'chat_url = "http://localhost:8000/v1"\n'
            'chat_model = "m"\n'
            'extra_body = "enable_thinking=false"\n',
        )
        errors = cfg.validate()
        assert any("extra_body must be a table" in e for e in errors)

    def test_validate_accepts_chat_template_kwargs(self) -> None:
        cfg = Config(
            chat_backend="vllm",
            embedding_backend="none",
            vllm=LocalBackendConfig(
                chat_url="http://localhost:8000/v1",
                chat_model="m",
                extra_body=LOW_EFFORT,
            ),
        )
        assert cfg.validate() == []
