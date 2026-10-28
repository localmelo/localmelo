"""The console's store open, its 503 and its retry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localmelo.support.gateway.console as console_mod
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.sessions.store import SessionStore

from .conftest import FakeBuilder, FakeProbe, make_cfg


def _client(console: Console) -> TestClient:
    app = FastAPI()
    console.mount(app)
    return TestClient(app, base_url="http://127.0.0.1:8401")


async def test_the_open_is_retried_after_the_owner_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder: FakeBuilder,
    probe: FakeProbe,
) -> None:
    path = str(tmp_path / "sessions.db")
    owner = SessionStore(path, writer="gateway")
    await owner.open()
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(store_path=path, builder=builder, probe=probe),
    )
    monkeypatch.setattr(console_mod, "STORE_RETRY_S", 0.01)
    await console.start()
    try:
        assert console.store_state() == "unavailable"
        client = _client(console)
        assert client.get("/api/sessions").status_code == 503
        await owner.aclose()
        for _ in range(200):
            if console.store_state() == "ok":
                break
            await asyncio.sleep(0.01)
        assert console.store_state() == "ok"
        assert client.get("/api/sessions").status_code == 200
    finally:
        await console.aclose()


async def test_aclose_releases_the_owner_lock(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    path = str(tmp_path / "sessions.db")
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(store_path=path, builder=builder, probe=probe),
    )
    await console.start()
    assert console.store_state() == "ok"
    await console.aclose()
    again = SessionStore(path, writer="gateway")
    await again.open()
    await again.aclose()


async def test_begin_shutdown_sets_the_event(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), builder=builder, probe=probe
        ),
    )
    await console.start()
    try:
        assert not console.shutdown.is_set()
        console.begin_shutdown()
        assert console.shutdown.is_set()
    finally:
        await console.aclose()


async def test_aclose_is_idempotent(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"), builder=builder, probe=probe
        ),
    )
    await console.start()
    await console.aclose()
    await console.aclose()
    assert probe.closed is True
