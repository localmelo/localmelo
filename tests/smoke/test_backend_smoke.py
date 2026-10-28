"""Pytest-discoverable per-backend smoke for local backends.

These tests build a real :class:`localmelo.melo.agent.Agent` against the
configured local backend and run a single chat round-trip. They are opt-in:
without ``LOCALMELO_SMOKE=1`` they are skipped, and collecting this file never
opens a socket. With the switch on, each test probes its backend when it runs
and skips it when the server does not answer, so the file can still be
collected on a dev laptop or CI without any local model server running.

Online cloud backends are intentionally out of scope here — see issue
``[Maintenance][Tests][Backend] Add pytest-discoverable smoke for online
cloud backends``.

Run the smoke for both local backends with:

    LOCALMELO_SMOKE=1 pytest -m smoke_backend tests/smoke/test_backend_smoke.py -q

Or skip them globally with:

    pytest -m 'not smoke_backend' -q
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from localmelo.melo.agent import Agent
from localmelo.support.config import Config, LocalBackendConfig

_BACKENDS_JSON = Path(__file__).resolve().parent / "data" / "backends.json"

_SMOKE_ENV = "LOCALMELO_SMOKE"
_SMOKE_ENABLED = os.environ.get(_SMOKE_ENV) == "1"
_SKIP_REASON = (
    "real-backend smoke tests are opt-in; run "
    f"{_SMOKE_ENV}=1 pytest -m smoke_backend tests/smoke/test_backend_smoke.py"
)

pytestmark = [
    pytest.mark.smoke_backend,
    pytest.mark.skipif(not _SMOKE_ENABLED, reason=_SKIP_REASON),
]


def _load_backends() -> dict[str, dict[str, Any]]:
    return json.loads(_BACKENDS_JSON.read_text("utf-8"))  # type: ignore[no-any-return]


def _can_reach(url: str, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to *url*'s host:port succeeds."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host is None:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_BACKENDS = _load_backends()


def _build_config(backend_key: str) -> Config:
    """Build a chat-only :class:`Config` from ``backends.json``.

    Embedding is set to ``"none"`` so the smoke does not require an
    embedding endpoint. The chat side goes through the registered
    backend exactly as the agent runtime does.
    """
    spec = _BACKENDS[backend_key]
    section = LocalBackendConfig(
        chat_url=spec["chat_url"],
        chat_model=spec["chat_model"],
    )
    cfg = Config(chat_backend=backend_key, embedding_backend="none")
    setattr(cfg, backend_key, section)
    return cfg


_OLLAMA_HEALTH = _BACKENDS["ollama"]["health_url"]
_MLC_HEALTH = _BACKENDS["mlc"]["health_url"]


def _skip_when_unreachable(backend: str, url: str) -> None:
    """Skip at run time, so collection never opens a socket."""
    if not _can_reach(url):
        pytest.skip(f"{backend} not reachable at {url}")


async def test_smoke_ollama_chat_round_trip() -> None:
    """ollama: agent.run('What is 6*7?') returns an answer containing '42'."""
    _skip_when_unreachable("ollama", _OLLAMA_HEALTH)
    agent = Agent(config=_build_config("ollama"))
    try:
        answer = await agent.run("What is 6*7? Reply with just the number.")
    finally:
        await agent.close()

    assert isinstance(answer, str)
    assert "42" in answer, f"expected '42' in ollama answer, got: {answer!r}"


async def test_smoke_mlc_chat_round_trip() -> None:
    """mlc: agent.run('What is 6*7?') returns an answer containing '42'."""
    _skip_when_unreachable("mlc-llm", _MLC_HEALTH)
    agent = Agent(config=_build_config("mlc"))
    try:
        answer = await agent.run("What is 6*7? Reply with just the number.")
    finally:
        await agent.close()

    assert isinstance(answer, str)
    assert "42" in answer, f"expected '42' in mlc answer, got: {answer!r}"
