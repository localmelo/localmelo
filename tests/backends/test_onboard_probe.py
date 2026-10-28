"""Tests for the onboarding connectivity probes."""

from __future__ import annotations

import math
from typing import Any
from unittest import mock

import pytest

from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.schema import Message, ToolDef
from localmelo.support import config as _config_mod
from localmelo.support.backends.base import BaseBackend
from localmelo.support.onboard import _run_probes
from localmelo.support.onboard_probe import (
    ProbeError,
    probe_chat,
    probe_embedding,
)

# ── Fakes ────────────────────────────────────────────────


class _FakeChat(BaseLLMProvider):
    def __init__(self, *, fail: bool = False, return_none: bool = False) -> None:
        self.fail = fail
        self.return_none = return_none
        self.calls: list[list[Message]] = []
        self.closed = False

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("connection refused")
        if self.return_none:
            return None  # type: ignore[return-value]
        return Message(role="assistant", content="pong")

    async def close(self) -> None:
        self.closed = True


class _FakeEmbedding(BaseEmbeddingProvider):
    def __init__(
        self,
        *,
        fail: bool = False,
        empty: bool = False,
    ) -> None:
        self.fail = fail
        self.empty = empty
        self.calls: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("auth error")
        if self.empty:
            return [[]]
        return [[0.1, 0.2, 0.3]]

    async def close(self) -> None:
        self.closed = True


class _FakeBackend(BaseBackend):
    """Backend stub registered into the registry for probe tests."""

    def __init__(
        self,
        key: str,
        *,
        chat: BaseLLMProvider | None = None,
        embedding: BaseEmbeddingProvider | None = None,
    ) -> None:
        self._key = key
        self._chat = chat
        self._embedding = embedding

    @property
    def key(self) -> str:
        return self._key

    @property
    def display_name(self) -> str:
        return f"fake:{self._key}"

    def validate(self, cfg: Any) -> list[str]:
        return []

    def validate_embedding(self, cfg: Any) -> list[str]:
        return []

    def has_embedding(self, cfg: Any) -> bool:
        return self._embedding is not None

    def build_chat_provider(self, cfg: Any) -> BaseLLMProvider:
        assert self._chat is not None, "chat fake not configured"
        return self._chat

    def build_embedding_provider(self, cfg: Any) -> BaseEmbeddingProvider | None:
        return self._embedding


# ── Helpers ──────────────────────────────────────────────


def _register(backend: _FakeBackend) -> None:
    """Force-register a fake backend, overwriting any existing entry."""
    from localmelo.support.backends import registry as reg

    reg._BACKENDS[backend.key] = backend  # noqa: SLF001 — test-only


def _cfg(chat: str, emb: str) -> _config_mod.Config:
    cfg = _config_mod.Config()
    cfg.chat_backend = chat
    cfg.embedding_backend = emb
    return cfg


# ── Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_chat_succeeds_with_valid_response() -> None:
    chat = _FakeChat()
    _register(_FakeBackend("probe-ok-chat", chat=chat))
    cfg = _cfg("probe-ok-chat", "none")

    await probe_chat(cfg)

    assert chat.calls and chat.calls[0][0].content == "ping"
    assert chat.closed


@pytest.mark.asyncio
async def test_probe_chat_wraps_provider_failure() -> None:
    chat = _FakeChat(fail=True)
    _register(_FakeBackend("probe-fail-chat", chat=chat))
    cfg = _cfg("probe-fail-chat", "none")

    with pytest.raises(ProbeError, match="chat probe failed"):
        await probe_chat(cfg)
    assert chat.closed  # provider must be closed even on failure


@pytest.mark.asyncio
async def test_probe_chat_rejects_non_message_response() -> None:
    chat = _FakeChat(return_none=True)
    _register(_FakeBackend("probe-bad-chat", chat=chat))
    cfg = _cfg("probe-bad-chat", "none")

    with pytest.raises(ProbeError, match="no message"):
        await probe_chat(cfg)


def _invalid_extra_body_cfg() -> _config_mod.Config:
    """A config whose chat provider cannot be built (real vllm backend)."""
    cfg = _cfg("vllm", "none")
    cfg.vllm = _config_mod.LocalBackendConfig(
        chat_url="http://127.0.0.1:8000/v1",
        chat_model="m",
        extra_body={"model": "other"},
    )
    return cfg


@pytest.mark.asyncio
async def test_probe_chat_wraps_provider_build_failure() -> None:
    """An invalid extra_body fails the probe instead of escaping it."""
    with pytest.raises(
        ProbeError, match=r"chat probe failed: \[vllm\] extra_body must not set"
    ):
        await probe_chat(_invalid_extra_body_cfg())


def test_run_probes_reports_invalid_extra_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Setup reports the failure and asks whether to save; no traceback."""
    with mock.patch("builtins.input", return_value="N") as ask:
        assert _run_probes(_invalid_extra_body_cfg()) is False

    ask.assert_called_once_with("  [s]ave anyway, [N]o [N]: ")
    out = capsys.readouterr().out
    assert "Probe failed: chat probe failed: [vllm] extra_body must not set" in out
    assert "config not saved" in out


def _timeout_cfg(
    key: str, chat: _FakeChat, request_timeout_s: Any
) -> _config_mod.Config:
    """A config for a fake chat backend *key*, with *request_timeout_s*."""
    _register(_FakeBackend(key, chat=chat))
    cfg = _cfg(key, "none")
    cfg.request_timeout_s = request_timeout_s
    return cfg


@pytest.mark.parametrize(
    "value",
    [0, -1, "600", True, math.inf, math.nan],
    ids=["zero", "negative", "string", "true", "inf", "nan"],
)
@pytest.mark.asyncio
async def test_probe_chat_reports_an_invalid_request_timeout(value: Any) -> None:
    """The key is named before a provider is built or a request is sent.

    As the provider's HTTP timeout such a value would fail the ping with an
    error that does not name it (a timeout, or a TypeError), or, as
    ``True``, let it pass.
    """
    chat = _FakeChat()
    cfg = _timeout_cfg("probe-timeout-chat", chat, value)

    with pytest.raises(ProbeError) as raised:
        await probe_chat(cfg)

    assert str(raised.value) == (
        "chat probe failed: request_timeout_s must be a finite number of "
        f"seconds > 0, got {value!r}"
    )
    assert chat.calls == []  # no ping
    assert not chat.closed  # no provider was built


@pytest.mark.parametrize("value", [None, 600, 0.5])
@pytest.mark.asyncio
async def test_probe_chat_pings_with_a_valid_request_timeout(
    value: float | None,
) -> None:
    chat = _FakeChat()
    cfg = _timeout_cfg("probe-timeout-ok-chat", chat, value)

    await probe_chat(cfg)

    assert [[m.content for m in call] for call in chat.calls] == [["ping"]]
    assert chat.closed


def test_run_probes_reports_an_invalid_request_timeout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Setup names the key and asks whether to save, as for extra_body."""
    chat = _FakeChat()
    cfg = _timeout_cfg("probe-timeout-chat", chat, 0)

    with mock.patch("builtins.input", return_value="N") as ask:
        assert _run_probes(cfg) is False

    ask.assert_called_once_with("  [s]ave anyway, [N]o [N]: ")
    out = capsys.readouterr().out
    assert (
        "Probe failed: chat probe failed: request_timeout_s must be a finite "
        "number of seconds > 0, got 0\n" in out
    )
    assert "config not saved" in out
    assert chat.calls == []


@pytest.mark.asyncio
async def test_probe_embedding_skipped_for_none() -> None:
    """``embedding_backend == 'none'`` short-circuits without provider build."""
    cfg = _cfg("probe-skip-chat", "none")
    # No backend registered for embedding — call must not touch the registry.
    await probe_embedding(cfg)


@pytest.mark.asyncio
async def test_probe_embedding_succeeds() -> None:
    emb = _FakeEmbedding()
    _register(
        _FakeBackend(
            "probe-ok-emb",
            chat=_FakeChat(),
            embedding=emb,
        )
    )
    cfg = _cfg("probe-ok-emb", "probe-ok-emb")

    await probe_embedding(cfg)
    assert emb.calls == [["ping"]]
    assert emb.closed


@pytest.mark.asyncio
async def test_probe_embedding_wraps_failure() -> None:
    emb = _FakeEmbedding(fail=True)
    _register(
        _FakeBackend(
            "probe-fail-emb",
            chat=_FakeChat(),
            embedding=emb,
        )
    )
    cfg = _cfg("probe-fail-emb", "probe-fail-emb")

    with pytest.raises(ProbeError, match="embedding probe failed"):
        await probe_embedding(cfg)
    assert emb.closed


@pytest.mark.asyncio
async def test_probe_embedding_rejects_empty_vector() -> None:
    emb = _FakeEmbedding(empty=True)
    _register(
        _FakeBackend(
            "probe-empty-emb",
            chat=_FakeChat(),
            embedding=emb,
        )
    )
    cfg = _cfg("probe-empty-emb", "probe-empty-emb")

    with pytest.raises(ProbeError, match="empty vector"):
        await probe_embedding(cfg)


def test_backend_defaults_matches_onboarding_tables() -> None:
    from localmelo.support import onboard

    defaults = onboard.backend_defaults()
    assert set(defaults) == set(onboard._LOCAL_DEFAULTS) | set(onboard._CLOUD_DEFAULTS)
    for key, (chat_url, chat_model) in onboard._LOCAL_DEFAULTS.items():
        emb_url, emb_model = onboard._EMBEDDING_DEFAULTS[key]
        assert defaults[key] == {
            "chat_url": chat_url,
            "chat_model": chat_model,
            "embedding_url": emb_url,
            "embedding_model": emb_model,
        }
    for key, (api_key_env, chat_model) in onboard._CLOUD_DEFAULTS.items():
        assert defaults[key] == {
            "api_key_env": api_key_env,
            "chat_model": chat_model,
        }
