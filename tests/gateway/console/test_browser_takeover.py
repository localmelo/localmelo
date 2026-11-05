"""Taking over melo's browser over ``/browser/live``: take and give, input only while the person has the page, and nothing typed kept.

The route runs under the loopback guard on a real server (the system picks the
port) and is driven with a WebSocket client, as the panel does. The browser is a
scripted one that streams and records the input it was sent; no Chrome runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from localmelo.melo.agent import Agent
from localmelo.support.browser.live import Frame
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.config import Config
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.tests.browser.test_takeover import Takeable, browser, grace_left
from localmelo.tests.gateway.console import test_browser_live as routes
from localmelo.tests.gateway.console.conftest import (
    FakeAgent,
    FakeProbe,
    open_session,
    quiet_gate,
    seed_turn,
)
from localmelo.tests.gateway.console.test_browser_live import (
    Served,
    browse,
    config,
    messages,
    until,
)

served = routes.served  # the console under the loopback guard on a real server

HOME = "https://example.com/"
SECRET = "hunter2-Secret"
CHINESE = "密码是你好"


class TakingAgent(FakeAgent):
    """A chat's Agent whose browser streams and takes the person's input."""

    def __init__(self, workspace: str) -> None:
        super().__init__(workspace)
        self.driver: Takeable = browser()
        self.provider = BrowserToolProvider(self.driver)  # type: ignore[arg-type]
        self.tool_providers = (self.provider,)

    async def close(self) -> None:
        await self.provider.close()
        await super().close()


class Builder:
    def __init__(self) -> None:
        self.agents: dict[str, TakingAgent] = {}

    async def __call__(
        self, settings: Config, workspace: str, mcp_config: str | None, session_id: str
    ) -> Agent:
        agent = TakingAgent(workspace)
        self.agents[session_id] = agent
        return cast(Agent, agent)


@pytest.fixture
def builder() -> Builder:  # type: ignore[override]
    return Builder()


@pytest.fixture
async def console(tmp_path: Path, builder: Builder) -> AsyncIterator[Console]:  # type: ignore[override]
    found = Console(
        config(),
        port=8401,
        options=ConsoleOptions(
            store_path=str(tmp_path / "sessions.db"),
            builder=builder,
            probe=FakeProbe(),
            gate=quiet_gate(),
        ),
    )
    await found.start()
    try:
        yield found
    finally:
        await found.aclose()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:  # type: ignore[override]
    path = tmp_path / "ws"
    path.mkdir()
    return path


async def chat(console: Console, workspace: Path) -> str:
    turn = await seed_turn(console.store, str(workspace))
    await open_session(console.sessions, turn.session_id)
    return turn.session_id


async def meta_until(ws: Any, test: Any, seconds: float = 3.0) -> dict[str, Any]:
    """The first meta message that passes *test*."""
    async with asyncio.timeout(seconds):
        while True:
            got = await ws.recv()
            if isinstance(got, str):
                meta: dict[str, Any] = json.loads(got)
                if test(meta):
                    return meta


async def opened(
    console: Console, builder: Builder, workspace: Path
) -> tuple[str, TakingAgent]:
    session_id = await chat(console, workspace)
    agent = cast(TakingAgent, await browse(builder, session_id, HOME))  # type: ignore[arg-type]
    return session_id, agent


async def test_the_person_takes_types_and_gives_back_over_the_socket(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id, agent = await opened(console, builder, workspace)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        agent.driver.cast.push("t1", Frame(b"\xff\xd8a", 1280.0, 713.0))
        meta = await meta_until(ws, lambda m: m.get("state") == "live")
        assert meta["owner"] == "melo" and meta["takeover"] is True
        # Input while melo has the page goes nowhere.
        await ws.send(
            json.dumps(
                {
                    "type": "key",
                    "action": "down",
                    "key": "a",
                    "code": "KeyA",
                    "keyCode": 65,
                }
            )
        )
        await ws.send(json.dumps({"type": "take"}))
        meta = await meta_until(ws, lambda m: m.get("owner") == "you")
        assert meta["grace"] == 120
        await ws.send(
            json.dumps(
                {
                    "type": "mouse",
                    "action": "down",
                    "x": 640,
                    "y": 300,
                    "button": "left",
                    "buttons": 1,
                    "clicks": 1,
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "mouse",
                    "action": "up",
                    "x": 640,
                    "y": 300,
                    "button": "left",
                    "buttons": 0,
                    "clicks": 1,
                }
            )
        )
        await ws.send(json.dumps({"type": "compose", "text": "nihao"}))
        await ws.send(json.dumps({"type": "text", "text": CHINESE}))
        await ws.send(json.dumps({"type": "paste", "text": SECRET}))
        await ws.send("not json")
        await ws.send(json.dumps(["a", "list"]))
        await until(lambda: len(agent.driver.inputs) == 5)
        await ws.send(json.dumps({"type": "give"}))
        await meta_until(ws, lambda m: m.get("owner") == "melo")
    methods = [m for _, m, _ in agent.driver.inputs]
    assert methods == [
        "Input.dispatchMouseEvent",
        "Input.dispatchMouseEvent",
        "Input.imeSetComposition",
        "Input.insertText",
        "Input.insertText",
    ]
    assert agent.provider.owners.owner("t1") == "melo"


async def test_a_message_over_the_size_cap_ends_the_stream(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id, agent = await opened(console, builder, workspace)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        await ws.send(json.dumps({"type": "take"}))
        await ws.send(json.dumps({"type": "paste", "text": "x" * 5000}))
        with pytest.raises(ConnectionClosed) as ended:
            async with asyncio.timeout(5):
                while True:
                    await ws.recv()
        assert ended.value.rcvd is not None and ended.value.rcvd.code == 1009
    assert agent.driver.inputs == []


async def test_a_panel_that_leaves_keeps_the_page_for_the_grace_period(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id, agent = await opened(console, builder, workspace)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        await ws.send(json.dumps({"type": "take"}))
        await meta_until(ws, lambda m: m.get("owner") == "you")
    owners = agent.provider.owners
    await until(lambda: grace_left(owners, "t1") is not None)
    assert owners.owner("t1") == "you"
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        meta = await meta_until(ws, lambda m: m.get("type") == "meta")
        assert meta["owner"] == "you", "the person came back within the grace period"
        await until(lambda: grace_left(owners, "t1") is None)


async def test_rotating_the_token_starts_the_grace_period(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id, agent = await opened(console, builder, workspace)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        await ws.send(json.dumps({"type": "take"}))
        await meta_until(ws, lambda m: m.get("owner") == "you")
        served.auth.rotate()
        with pytest.raises(ConnectionClosed):
            async with asyncio.timeout(5):
                while True:
                    await ws.recv()
    owners = agent.provider.owners
    await until(lambda: grace_left(owners, "t1") is not None)
    assert owners.owner("t1") == "you"


async def test_nothing_typed_is_in_the_store_the_logs_or_on_disk(
    console: Console,
    builder: Builder,
    workspace: Path,
    served: Served,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    for name in ("uvicorn.error", "websockets", "websockets.server"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    session_id, agent = await opened(console, builder, workspace)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        agent.driver.cast.push("t1", Frame(b"\xff\xd8a", 1280.0, 713.0))
        await ws.send(json.dumps({"type": "take"}))
        await meta_until(ws, lambda m: m.get("owner") == "you")
        for text in (SECRET, CHINESE):
            await ws.send(
                json.dumps({"type": "compose", "text": text}, ensure_ascii=False)
            )
            await ws.send(
                json.dumps({"type": "text", "text": text}, ensure_ascii=False)
            )
            await ws.send(
                json.dumps({"type": "paste", "text": text}, ensure_ascii=False)
            )
            for char in text:
                await ws.send(
                    json.dumps(
                        {
                            "type": "key",
                            "action": "down",
                            "key": char,
                            "code": "",
                            "keyCode": 0,
                        },
                        ensure_ascii=False,
                    )
                )
        await ws.send(json.dumps({"type": "dialog", "accept": True, "text": SECRET}))
        await until(
            lambda: SECRET in agent.driver.typed() and CHINESE in agent.driver.typed()
        )
        await ws.send(json.dumps({"type": "give"}))
        await meta_until(ws, lambda m: m.get("owner") == "melo")
    events = await console.store.list_events(session_id)
    stored = json.dumps([e.event for e in events], ensure_ascii=False, default=str)
    # The gateway's own logs, at debug level (the test's client logs what it sends: not the gateway's).
    logged = "\n".join(
        record.getMessage()
        for record in caplog.records
        if not record.name.startswith("websockets.client")
    )
    assert any(
        r.name == "uvicorn.error" and r.levelno == logging.DEBUG for r in caplog.records
    ), "the websocket server's debug lines were on"
    for text in (SECRET, CHINESE, SECRET[:7]):
        assert text not in stored
        assert text not in logged
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert text.encode() not in path.read_bytes(), path


@pytest.mark.parametrize("credential", ["bearer", "cookie"])
async def test_a_logger_turned_to_debug_after_startup_logs_no_credential(
    console: Console,
    builder: Builder,
    workspace: Path,
    served: Served,
    caplog: pytest.LogCaptureFixture,
    credential: str,
) -> None:
    # The server started at warning; its loggers are turned to debug after, as
    # a person debugging the gateway might. The credential is the test's own
    # (a token made for this test's gateway), never a real one.
    caplog.set_level(logging.DEBUG)
    for name in ("uvicorn.error", "websockets", "websockets.server"):
        caplog.set_level(logging.DEBUG, logger=name)  # put back after the test
    if credential == "bearer":
        headers, secret = served.bearer, served.token
    else:
        headers = served.cookie()
        secret = headers["Cookie"].split("=", 1)[1]
    session_id, _agent = await opened(console, builder, workspace)
    async with connect(served.url(session_id), additional_headers=headers) as ws:
        await messages(ws, 1)
    logged = [
        record.getMessage()
        for record in caplog.records
        if not record.name.startswith("websockets.client")
    ]
    header = "authorization" if credential == "bearer" else "cookie"
    assert any(
        line.lower() == f"< {header}: [redacted]" for line in logged
    ), "the handshake's header lines were on, with the credential redacted"
    assert len(secret) >= 16
    for line in logged:
        assert secret not in line and secret[:12] not in line
