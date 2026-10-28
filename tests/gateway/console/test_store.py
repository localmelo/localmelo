"""The console's store open, its 503 and its retry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localmelo.support.gateway.console as console_mod
from localmelo.support.gateway import GatewayServer
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.sessions.store import SessionStore, SessionStoreError

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


async def test_an_owned_store_never_names_its_holder(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    path = str(tmp_path / "sessions.db")
    owner = SessionStore(path, writer="gateway")
    await owner.open()
    console = Console(
        make_cfg(),
        port=8401,
        options=ConsoleOptions(store_path=path, builder=builder, probe=probe),
    )
    await console.start()
    try:
        assert console.store_state() == "unavailable"
        message = console.store_error()
        assert "Another melo gateway is using" in message
        assert "pid" not in message
        resp = _client(console).get("/api/sessions")
        assert resp.status_code == 503
        error = resp.json()["error"]
        assert error["code"] == "store_unavailable"
        assert "Another melo gateway is using" in error["message"]
        assert "pid" not in error["message"]
    finally:
        await console.aclose()
        await owner.aclose()


async def test_a_store_message_labels_the_home_folder(
    console: Console, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The re-read after a failed store call can itself fail; that message
    # must still write the home folder as "~" (docs/api.md).
    async def broken(session_id: str) -> None:
        raise SessionStoreError("reread failed")

    monkeypatch.setattr(console.store, "get_session", broken)
    error = await console.store_api_error(
        SessionStoreError(f"{home}/gateway/sessions.db: disk I/O error"), "sess"
    )
    assert error.status_code == 503
    assert error.code == "store_unavailable"
    assert error.message.startswith("~")
    assert str(home) not in error.message


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


class _QuietGate:
    async def close(self) -> None:
        pass


async def test_the_console_closes_when_a_later_close_raises(
    tmp_path: Path, builder: FakeBuilder, probe: FakeProbe
) -> None:
    path = str(tmp_path / "sessions.db")
    server = GatewayServer(
        make_cfg(),
        console_options=ConsoleOptions(store_path=path, builder=builder, probe=probe),
    )

    class BadPlayground:
        gate = _QuietGate()

        async def close_sessions(self) -> None:
            raise RuntimeError("playground close failed")

    server.app.state.smoke_playground = BadPlayground()
    with pytest.raises(RuntimeError), TestClient(server.app):
        pass
    # The console's aclose ran in the finally: the owner lock is free.
    again = SessionStore(path, writer="gateway")
    await again.open()
    await again.aclose()
