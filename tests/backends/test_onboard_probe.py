"""Tests for the onboarding connectivity probes."""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from localmelo.melo.schema import Message, ToolDef
from localmelo.support import config as _config_mod
from localmelo.support.backends.base import BaseBackend
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
