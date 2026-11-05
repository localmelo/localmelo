"""``/api/sessions/{id}/browser`` and the live stream ``/browser/live``: the chat that owns the tab, the guard, the rotation.

The route runs under the loopback guard on a real server (the system picks the
port) and is read with a WebSocket client. The browser is a scripted one that
streams; no Chrome runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator, MutableMapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.typing import Origin

from localmelo.melo.agent import Agent
from localmelo.support.browser.driver import BrowserError
from localmelo.support.browser.live import Frame
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.config import Config
from localmelo.support.gateway import _ConsoleServer
from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.guard import LoopbackGuard
from localmelo.tests.browser.fakes import FakeDriver, page
from localmelo.tests.browser.test_live import site
from localmelo.tests.browser.test_provider import run

from .conftest import (
    FakeAgent,
    FakeProbe,
    make_cfg,
    open_session,
    quiet_gate,
    seed_turn,
)

HOME = "https://example.com/"
MARKER = b"\xff\xd8secret-picture-bytes"


class BrowsingAgent(FakeAgent):
    """A chat's Agent with a browser that streams (Chrome), or one that cannot (as Safari's windows)."""

    def __init__(
        self, workspace: str, *, browser: bool = True, streams: bool = True
    ) -> None:
        super().__init__(workspace)
        self.driver = site()
        driver = self.driver if streams else FakeDriver(page(HOME, "Welcome."))
        self.provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
        self.tool_providers = (self.provider,) if browser else ()

    async def close(self) -> None:
        await self.provider.close()
        await super().close()


class Builder:
    def __init__(self, *, browser: bool = True) -> None:
        self.browser = browser
        # Whether the next Agents' browser streams: the browser the settings name when it is built.
        self.streams = True
        self.agents: dict[str, BrowsingAgent] = {}

    async def __call__(
        self, settings: Config, workspace: str, mcp_config: str | None, session_id: str
    ) -> Agent:
        agent = BrowsingAgent(workspace, browser=self.browser, streams=self.streams)
        self.agents[session_id] = agent
        return cast(Agent, agent)


def config(driver: str = "chrome", *, live: bool = True) -> Config:
    cfg = make_cfg()
    cfg.browser.driver = driver
    cfg.browser.live = live
    return cfg


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


class Served:
    """The console under the guard on a port the system picked."""

    def __init__(self, port: int, auth: GatewayAuth) -> None:
        self.port = port
        self.auth = auth

    @property
    def token(self) -> str:
        return self.auth.read_token().decode("ascii")

    @property
    def bearer(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def url(self, session_id: str, query: str = "") -> str:
        return (
            f"ws://127.0.0.1:{self.port}/api/sessions/{session_id}/browser/live{query}"
        )

    def cookie(self) -> dict[str, str]:
        code = self.auth.issue_login(self.port)
        cookie = self.auth.consume_login(code, self.port)
        assert cookie
        return {"Cookie": f"melo_session_{self.port}={cookie}"}


@pytest.fixture
async def served(console: Console, tmp_path: Path) -> AsyncIterator[Served]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    auth = GatewayAuth(tmp_path / "auth")
    app = FastAPI()
    console.mount(app)
    server = _ConsoleServer(
        uvicorn.Config(
            LoopbackGuard(app, port=port, auth=auth),
            log_level="warning",
            timeout_graceful_shutdown=3,
        ),
        console=console,
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    for _ in range(500):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    try:
        yield Served(port, auth)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=10)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            sock.close()
            auth.close()


async def chat(console: Console, builder: Builder, workspace: Path) -> str:
    """An open chat with a browser; its session id."""
    turn = await seed_turn(console.store, str(workspace))
    await open_session(console.sessions, turn.session_id)
    return turn.session_id


@pytest.fixture
def workspace(tmp_path: Path) -> Path:  # type: ignore[override]
    path = tmp_path / "ws"
    path.mkdir()
    return path


async def browse(builder: Builder, session_id: str, *urls: str) -> BrowsingAgent:
    """Melo opens the pages, one new tab each."""
    agent = builder.agents[session_id]
    tools = {t.definition.name: t for t in await agent.provider.start()}
    for index, url in enumerate(urls):
        await run(tools, "browser_open", url=url, **({"tab": "new"} if index else {}))
    return agent


async def messages(socket_: Any, count: int) -> list[str | bytes]:
    got: list[str | bytes] = []
    async with asyncio.timeout(5):
        while len(got) < count:
            got.append(await socket_.recv())
    return got


async def until(condition: Any, seconds: float = 3.0) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


# ── The state of a chat's browser ──


async def test_the_state_says_whether_a_page_is_open(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id = await chat(console, builder, workspace)
    url = f"http://127.0.0.1:{served.port}/api/sessions/{session_id}/browser"
    async with httpx.AsyncClient(headers=served.bearer) as client:
        before = (await client.get(url)).json()
        await browse(builder, session_id, HOME)
        after = (await client.get(url)).json()
        unknown = await client.get(url.replace(session_id, "0123456789ab"))
    assert before["available"] is True and before["open"] is False
    assert before["driver"] == "chrome" and before["tab"] is None
    assert (after["open"], after["tab"], after["browsing"]) == (True, "t1", False)
    assert after["title"] == "Home" and after["url"] == HOME and after["viewers"] == 0
    assert unknown.status_code == 404


async def test_the_state_of_a_chat_that_is_not_open_is_calm(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    turn = await seed_turn(console.store, str(workspace))  # in the store, not open
    url = f"http://127.0.0.1:{served.port}/api/sessions/{turn.session_id}/browser"
    async with httpx.AsyncClient(headers=served.bearer) as client:
        body = (await client.get(url)).json()
    assert body["available"] is True and body["open"] is False


async def test_the_view_is_unavailable_with_another_browser_or_when_turned_off(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    turn = await seed_turn(console.store, str(workspace))  # in the store, not open
    url = f"http://127.0.0.1:{served.port}/api/sessions/{turn.session_id}/browser"
    session_id = await chat(console, builder, workspace)
    open_url = f"http://127.0.0.1:{served.port}/api/sessions/{session_id}/browser"
    async with httpx.AsyncClient(headers=served.bearer) as client:
        console.cfg.browser.driver = "safari-automation"
        assert (await client.get(url)).json()["available"] is False
        console.cfg.browser.driver = "chrome"
        console.cfg.browser.live = False
        assert (await client.get(url)).json()["available"] is False
        # Turning the live view off holds at once, also for a chat whose browser streams.
        assert (await client.get(open_url)).json()["available"] is False
        with pytest.raises(InvalidStatus) as refused:
            async with connect(
                served.url(session_id), additional_headers=served.bearer
            ):
                pass
    assert refused.value.response.status_code == 403


async def test_an_open_chat_is_watched_in_the_browser_it_browses_in(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    # A chat whose turn ran when Settings changed the browser keeps the browser it
    # has until its next turn: the state says what that browser can do, not the setting.
    chrome_chat = await chat(console, builder, workspace)
    await browse(builder, chrome_chat, HOME)
    builder.streams = False
    safari_chat = await chat(console, builder, workspace)
    base = f"http://127.0.0.1:{served.port}/api/sessions"
    async with httpx.AsyncClient(headers=served.bearer) as client:
        console.cfg.browser.driver = "safari-automation"
        body = (await client.get(f"{base}/{chrome_chat}/browser")).json()
        assert body["available"] is True and body["open"] is True
        assert body["driver"] == "safari-automation", "the setting, for the next turn"
        async with connect(
            served.url(chrome_chat), additional_headers=served.bearer
        ) as ws:
            assert json.loads(str((await messages(ws, 1))[0]))["tab"] == "t1"
        console.cfg.browser.driver = "chrome"
        body = (await client.get(f"{base}/{safari_chat}/browser")).json()
        assert body["available"] is False and body["open"] is False
        with pytest.raises(InvalidStatus) as refused:
            async with connect(
                served.url(safari_chat), additional_headers=served.bearer
            ):
                pass
    assert refused.value.response.status_code == 403


# ── The stream ──


async def test_a_viewer_gets_the_meta_then_the_pictures_and_the_stream_ends_with_it(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    cast_ = agent.driver.cast
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        (first,) = await messages(ws, 1)
        assert isinstance(first, str)
        import json

        meta = json.loads(first)
        assert meta["type"] == "meta" and meta["state"] == "loading"
        assert (meta["tab"], meta["title"], meta["owner"]) == ("t1", "Home", "melo")
        assert cast_.started == ["t1"]
        cast_.push("t1", Frame(MARKER, width=1280.0, height=713.0))
        live, picture = await messages(ws, 2)
        assert json.loads(str(live))["state"] == "live"
        assert picture == MARKER
        view = agent.provider.live
        assert view is not None and view.viewers == 1
    await until(lambda: view.viewers == 0)
    assert cast_.stopped == ["t1"]
    assert view.streaming == 0


async def test_only_the_newest_picture_reaches_a_viewer_that_is_slow(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        for n in range(1, 41):
            agent.driver.cast.push(
                "t1", Frame(b"\xff\xd8" + str(n).encode(), 1280, 713)
            )
        got = await messages(ws, 2)
        assert got[1] == b"\xff\xd840"
        await asyncio.sleep(0.1)
        view = agent.provider.live
        assert view is not None


async def test_the_viewer_follows_the_tab_melo_switches_to(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    import json

    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        tools = {t.definition.name: t for t in await agent.provider.start()}
        await run(tools, "browser_open", url=HOME + "news", tab="new")
        async with asyncio.timeout(5):
            while True:
                text = await ws.recv()
                if isinstance(text, str) and json.loads(text)["tab"] == "t2":
                    break
    assert agent.driver.cast.started == ["t1", "t2"]


async def test_a_pinned_viewer_may_name_only_a_tab_of_its_own_chat(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    mine = await chat(console, builder, workspace)
    other = await chat(console, builder, workspace)
    await browse(builder, mine, HOME, HOME + "news")  # t1 and t2
    await browse(builder, other, HOME)  # t1 only
    async with connect(served.url(mine, "?tab=t2"), additional_headers=served.bearer):
        pass
    for session_id, query in ((other, "?tab=t2"), (mine, "?tab=../x"), (mine, "?tab=")):
        with pytest.raises(InvalidStatus) as refused:
            async with connect(
                served.url(session_id, query), additional_headers=served.bearer
            ):
                pass
        assert refused.value.response.status_code == 403, query
    # Nothing was started for a refused viewer.
    assert builder.agents[other].driver.cast.started == []


async def test_an_unknown_chat_is_refused(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    with pytest.raises(InvalidStatus) as refused:
        async with connect(
            served.url("0123456789ab"), additional_headers=served.bearer
        ):
            pass
    assert refused.value.response.status_code == 403


async def test_a_chat_with_no_streaming_browser_says_so_and_ends_calmly(
    console: Console, workspace: Path, served: Served, tmp_path: Path
) -> None:
    import json

    turn = await seed_turn(console.store, str(workspace))  # not open: no browser
    async with connect(
        served.url(turn.session_id), additional_headers=served.bearer
    ) as ws:
        meta = json.loads(str(await ws.recv()))
        assert (meta["state"], meta["reason"]) == ("closed", "unavailable")
        with pytest.raises(ConnectionClosed) as ended:
            await ws.recv()
    assert ended.value.rcvd is not None and ended.value.rcvd.code == 1000


async def test_too_many_viewers_of_one_chat_are_refused(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id = await chat(console, builder, workspace)
    await browse(builder, session_id, HOME)
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(4):
            await stack.enter_async_context(
                connect(served.url(session_id), additional_headers=served.bearer)
            )
        with pytest.raises(InvalidStatus):
            async with connect(
                served.url(session_id), additional_headers=served.bearer
            ):
                pass


async def test_a_browser_that_closes_is_told_to_the_viewer_when_the_chat_ends(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    import json

    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        await agent.provider.close()
        told = json.loads(str(await asyncio.wait_for(ws.recv(), 5)))
        assert (told["state"], told["reason"]) == ("closed", "chat")
        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(ws.recv(), 5)


async def test_what_a_viewer_sees_is_not_written_to_the_session_store_or_disk(
    console: Console, builder: Builder, workspace: Path, served: Served, tmp_path: Path
) -> None:
    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        agent.driver.cast.push("t1", Frame(MARKER, 1280, 713))
        assert (await messages(ws, 2))[1] == MARKER
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert MARKER not in path.read_bytes(), path


# ── The guard ──


async def test_the_stream_is_behind_the_guard(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id = await chat(console, builder, workspace)
    await browse(builder, session_id, HOME)
    url = served.url(session_id)
    origin = f"http://127.0.0.1:{served.port}"

    async def refused(**headers: Any) -> int:
        with pytest.raises(InvalidStatus) as caught:
            async with connect(url, **headers):
                pass
        return caught.value.response.status_code

    # No credentials, a wrong token, a foreign Origin (also with a good token)
    # and a cookie from another gateway are all refused.
    assert await refused() == 403
    assert (
        await refused(additional_headers={"Authorization": "Bearer " + "0" * 64}) == 403
    )
    assert (
        await refused(additional_headers=served.bearer, origin="http://evil.example")
        == 403
    )
    stray = {"Cookie": f"melo_session_{served.port + 1}=abc"}
    assert await refused(additional_headers=stray, origin=origin) == 403
    # A bearer needs no Origin; a console cookie needs the page's own.
    async with connect(url, additional_headers=served.bearer) as ws:
        assert isinstance(await ws.recv(), str)
    async with connect(
        url, additional_headers=served.cookie(), origin=Origin(origin)
    ) as ws:
        assert isinstance(await ws.recv(), str)
    assert (
        await refused(additional_headers=served.cookie(), origin="http://localhost:1")
        == 403
    )


async def test_rotating_the_token_ends_an_open_stream(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await messages(ws, 1)
        view = agent.provider.live
        assert view is not None and view.viewers == 1
        served.auth.rotate()
        with pytest.raises(ConnectionClosed) as ended:
            async with asyncio.timeout(5):
                while True:
                    await ws.recv()
        assert ended.value.rcvd is not None and ended.value.rcvd.code == 1008
    await until(lambda: view.viewers == 0)
    assert agent.driver.cast.stopped == ["t1"]
    # The old token opens nothing now.
    with pytest.raises(InvalidStatus):
        async with connect(
            served.url(session_id),
            additional_headers={"Authorization": "Bearer " + "0" * 64},
        ):
            pass


async def test_a_browser_error_while_starting_the_stream_does_not_end_the_view(
    console: Console, builder: Builder, workspace: Path, served: Served
) -> None:
    import json

    session_id = await chat(console, builder, workspace)
    agent = await browse(builder, session_id, HOME)
    real = agent.driver.cast.start_screencast

    async def failing(*args: Any, **kwargs: Any) -> None:
        raise BrowserError("no stream")

    agent.driver.cast.start_screencast = failing  # type: ignore[method-assign,assignment]
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        assert json.loads(str(await ws.recv()))["state"] == "stale"
        agent.driver.cast.start_screencast = real  # type: ignore[method-assign]
        agent.provider.live.poke()  # type: ignore[union-attr]
        await until(lambda: agent.driver.cast.started == ["t1"])


async def test_a_stream_for_another_host_never_reaches_the_route(
    tmp_path: Path,
) -> None:
    reached: list[bool] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        reached.append(True)

    auth = GatewayAuth(tmp_path / "auth")
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(dict(message))

    try:
        for host in (b"evil.example", b"127.0.0.1:1", b"localhost"):
            await LoopbackGuard(app, port=8401, auth=auth)(
                {
                    "type": "websocket",
                    "path": "/api/sessions/a/browser/live",
                    "headers": [
                        (b"host", host),
                        (b"authorization", b"Bearer " + auth.read_token()),
                    ],
                },
                receive,
                send,
            )
        assert not reached
        assert [m["type"] for m in sent] == ["websocket.close"] * 3
        assert {m["code"] for m in sent} == {1008}
    finally:
        auth.close()
