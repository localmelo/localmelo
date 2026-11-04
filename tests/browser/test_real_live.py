"""The live view against the real Google Chrome, headless (opt in).

Skipped unless ``LOCALMELO_CHROME_SMOKE=1`` and Google Chrome is installed.
Two headless Chromes run, neither with a window and each on a profile under
``tmp_path`` (never the person's or melo's own):

* melo's browser, which a chat's :class:`BrowserToolProvider` drives, reading
  pages served by the test on ``127.0.0.1`` (the system picks the port; the
  public-address rule that keeps melo off local addresses is lifted for the
  test alone);
* a second one that plays the person's: it opens the console page, served
  under the real loopback guard, and the Browser panel in it is read and
  pictured.

Every test watches the app in front (``lsappinfo front``) while it runs: one
of the test's own Chromes taking the front would fail it, whatever else the
person does on the computer meanwhile. The rate and the cost of the stream are printed (run with
``-s``) and kept in ``tmp_path / "live-smoke.json"``. Run it by hand:
``LOCALMELO_CHROME_SMOKE=1 uv run pytest tests/browser/test_real_live.py -s``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Response
from websockets.asyncio.client import connect

from localmelo.melo.agent import Agent
from localmelo.support.browser import addresses, chrome
from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.browser.live import DEFAULT_FPS
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.config import Config
from localmelo.support.gateway import _ConsoleServer, webapp
from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway.console import Console, ConsoleOptions
from localmelo.support.gateway.guard import LoopbackGuard
from localmelo.tests.browser.test_provider import run
from localmelo.tests.fixtures.http_stub import HttpStub
from localmelo.tests.gateway.console.conftest import (
    FakeAgent,
    FakeProbe,
    make_cfg,
    open_session,
    quiet_gate,
    seed_turn,
)
from localmelo.tests.gateway.console.test_browser_live import Served

BINARY = os.path.join(chrome.CHROME_APPS[0], chrome.APP_BINARY)
pytestmark = pytest.mark.skipif(
    os.environ.get("LOCALMELO_CHROME_SMOKE") != "1" or not os.access(BINARY, os.X_OK),
    reason="starts the real Chrome, headless: set LOCALMELO_CHROME_SMOKE=1",
)

ANIMATED = """<!doctype html><meta charset="utf-8"><title>Moving page</title>
<body style="margin:0;background:#fff;font:600 44px -apple-system,sans-serif">
<div id="c" style="padding:24px">0</div>
<div style="margin:24px;width:80px;height:80px;border:10px solid #ddd;border-top-color:#183d86;
border-radius:50%;animation:s 1s linear infinite"></div>
<style>@keyframes s{to{transform:rotate(360deg)}}</style>
<script>let n=0;setInterval(()=>c.textContent=++n,50)</script></body>"""
STILL = """<!doctype html><meta charset="utf-8"><title>Still page</title>
<body style="margin:24px;font:16px -apple-system,sans-serif"><h1>Still page</h1>
<p>Nothing on this page moves, so it sends one picture and then none.</p></body>"""


class Pages(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        data = (ANIMATED if self.path.startswith("/moving") else STILL).encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def front() -> tuple[str, int]:
    """The bundle id and process id of the app in front, as ``lsappinfo`` names them (``("", 0)`` when it names none)."""
    asn = subprocess.run(
        ["/usr/bin/lsappinfo", "front"], capture_output=True, text=True, timeout=5
    ).stdout.strip()
    said = subprocess.run(
        ["/usr/bin/lsappinfo", "info", "-only", "bundleid,pid", asn],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    bundle = re.search(r'bundleID="([^"]+)"', said)
    pid = re.search(r'"pid"\s*=\s*(\d+)', said)
    return (bundle.group(1) if bundle else "", int(pid.group(1)) if pid else 0)


def processes_using(marker: str) -> set[int]:
    """The processes whose command line names *marker* (a profile under the test's own folder)."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10
    ).stdout
    return {
        int(line.split(None, 1)[0])
        for line in out.splitlines()
        if marker in line and line.split(None, 1)[0].isdigit()
    }


def cpu_seconds(group: int) -> float:
    """CPU time used so far by the process group *group* (Chrome and its helpers)."""
    out = subprocess.run(
        ["ps", "-axo", "pgid=,cputime="], capture_output=True, text=True, timeout=10
    ).stdout
    total = 0.0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) != group:
            continue
        clock = parts[1].split(":")
        seconds = float(clock[-1]) + 60 * int(clock[-2])
        total += seconds + (3600 * int(clock[-3]) if len(clock) > 2 else 0)
    return total


@pytest.fixture(autouse=True)
def never_the_front(tmp_path: Path) -> Iterator[None]:
    """Sample the app in front every half second: it must never be one of this test's Chromes.

    The person may be using their own Chrome, or anything else, meanwhile; the
    test's Chromes are told apart by their profiles, which sit under ``tmp_path``.
    """
    ours: set[int] = set()
    fronts: set[tuple[str, int]] = set()
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            with contextlib.suppress(Exception):
                fronts.add(front())
                ours.update(processes_using(str(tmp_path)))
            stop.wait(0.5)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join(timeout=10)
    assert ours, "the watch never saw the test's Chromes, so it checked nothing"
    taken = sorted(bundle for bundle, pid in fronts if pid in ours)
    assert not taken, f"a Chrome of this test took the front: {taken}"


@pytest.fixture
def base() -> Iterator[str]:
    with HttpStub(Pages) as stub:
        yield stub.base_url


@pytest.fixture(autouse=True)
def local_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Melo's browser never opens a local address; these pages are the test's own."""

    async def anywhere(url: str) -> None:
        return None

    monkeypatch.setattr(addresses, "require_public", anywhere)


class RealAgent(FakeAgent):
    """A chat's Agent whose browser is the real Chrome, headless, on a profile of the test's."""

    def __init__(self, workspace: str, profile: str) -> None:
        super().__init__(workspace)
        self.driver = ChromeBackgroundDriver(command=[BINARY], profile=profile)
        self.provider = BrowserToolProvider(self.driver)
        self.tool_providers = (self.provider,)

    async def close(self) -> None:
        await self.provider.close()
        await super().close()


class RealBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.agents: dict[str, RealAgent] = {}

    async def __call__(
        self, settings: Config, workspace: str, mcp_config: str | None, session_id: str
    ) -> Agent:
        agent = RealAgent(workspace, str(self.root / f"melo-{len(self.agents)}"))
        self.agents[session_id] = agent
        return cast(Agent, agent)


@pytest.fixture
def builder(tmp_path: Path) -> RealBuilder:
    return RealBuilder(tmp_path)


@pytest.fixture
async def console(tmp_path: Path, builder: RealBuilder) -> AsyncIterator[Console]:
    cfg = make_cfg()
    cfg.browser.driver = "chrome"
    cfg.browser.live = True
    found = Console(
        cfg,
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
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


async def chat(console: Console, workspace: Path) -> str:
    """An open chat, whose Agent the builder makes; its session id."""
    turn = await seed_turn(console.store, str(workspace))
    await open_session(console.sessions, turn.session_id)
    return turn.session_id


@pytest.fixture
async def served(console: Console, tmp_path: Path) -> AsyncIterator[Served]:
    """The console, with its page, on a real server under the loopback guard (the system picks the port)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    auth = GatewayAuth(tmp_path / "auth")
    app = FastAPI()
    webapp.mount(app, console.cfg)

    # The gateway's own route, which the page calls with the code in its address; it comes before the console's.
    @app.post("/api/auth/login")
    async def login(body: dict[str, Any]) -> Response:
        code = body.get("code")
        cookie = auth.consume_login(code, port) if isinstance(code, str) else None
        if cookie is None:
            return Response(status_code=401)
        response = Response(status_code=204, headers={"Cache-Control": "no-store"})
        response.set_cookie(
            f"melo_session_{port}", cookie, httponly=True, samesite="strict", path="/"
        )
        return response

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


async def browse(builder: RealBuilder, session_id: str, url: str) -> RealAgent:
    agent = builder.agents[session_id]
    tools = {t.definition.name: t for t in await agent.provider.start()}
    await run(tools, "browser_open", url=url)
    return agent


@pytest.fixture
async def person(tmp_path: Path) -> AsyncIterator[ChromeBackgroundDriver]:
    """The second headless Chrome, which plays the person's browser."""
    found = ChromeBackgroundDriver(command=[BINARY], profile=str(tmp_path / "person"))
    await found.start()
    try:
        yield found
    finally:
        await found.close()


async def count(socket_: Any, seconds: float) -> tuple[int, list[str]]:
    """How many pictures arrive in *seconds*, and the meta messages that came with them."""
    pictures, metas = 0, []
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        try:
            got = await asyncio.wait_for(socket_.recv(), timeout=left)
        except TimeoutError:
            break
        if isinstance(got, bytes):
            assert got.startswith(b"\xff\xd8")
            pictures += 1
        else:
            metas.append(got)
    return pictures, metas


async def settled(condition: Any, seconds: float = 10.0) -> None:
    async with asyncio.timeout(seconds):
        while not await condition():
            await asyncio.sleep(0.1)


def kept(tmp_path: Path, name: str, value: Any) -> None:
    """Print a measurement, and keep it with the test's files."""
    path = tmp_path / "live-smoke.json"
    found = json.loads(path.read_text()) if path.exists() else {}
    found[name] = value
    path.write_text(json.dumps(found, indent=2))
    print(f"\nlive smoke {name}: {json.dumps(value)}")


# ── The stream ──


async def test_pictures_arrive_at_the_capped_rate_and_stop_when_nobody_watches(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    tmp_path: Path,
) -> None:
    session_id = await chat(console, workspace)
    agent = await browse(builder, session_id, base + "/moving")
    browser = agent.driver._browser
    assert browser is not None
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        # The first pictures come in a burst (about three are in flight when a stream starts): leave them out.
        _, early = await count(ws, 1.5)
        before = cpu_seconds(browser.pid)
        started = time.monotonic()
        pictures, metas = await count(ws, 5.0)
        spent = time.monotonic() - started
        cost = cpu_seconds(browser.pid) - before
        meta = json.loads(next(m for m in reversed([*early, *metas]) if '"live"' in m))
        assert len(browser.casts) == 1, "one watched tab streams"
    fps = pictures / spent
    kept(
        tmp_path,
        "capped",
        {
            "cap_fps": DEFAULT_FPS,
            "fps": round(fps, 1),
            "chrome_cpu_percent_of_a_core": round(100 * cost / spent, 1),
            "view": [meta.get("w"), meta.get("h")],
        },
    )
    assert 5 <= fps <= DEFAULT_FPS * 1.25, fps
    assert meta.get("state") == "live" and meta.get("w") == 1280
    # The viewer went: the stream stops, and Chrome is asked for no more pictures.
    await settled(lambda: _no_streams(browser))
    assert agent.provider.live is not None and agent.provider.live.streaming == 0


async def _no_streams(browser: Any) -> bool:
    return not browser.casts


async def _one_stream(browser: Any) -> bool:
    return len(browser.casts) == 1


async def test_a_page_that_does_not_move_sends_one_picture_and_then_none(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    tmp_path: Path,
) -> None:
    session_id = await chat(console, workspace)
    await browse(builder, session_id, base + "/still")
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        first, _ = await count(ws, 1.5)
        later, _ = await count(ws, 2.0)
    kept(tmp_path, "still", {"first_second": first, "next_two_seconds": later})
    assert first == 1 and later == 0


async def test_the_viewer_follows_melo_to_another_page_and_a_new_viewer_gets_a_picture_at_once(
    console: Console, builder: RealBuilder, workspace: Path, served: Served, base: str
) -> None:
    session_id = await chat(console, workspace)
    agent = await browse(builder, session_id, base + "/still")
    async with connect(served.url(session_id), additional_headers=served.bearer) as ws:
        await count(ws, 1.0)
        tools = {t.definition.name: t for t in await agent.provider.start()}
        await run(tools, "browser_open", url=base + "/moving", tab="new")
        pictures, metas = await count(ws, 2.0)
        assert pictures > 3
        assert json.loads(metas[-1])["tab"] == "t2"
        assert "Moving page" in "".join(metas)
    async with httpx.AsyncClient(headers=served.bearer) as client:
        state = (
            await client.get(
                f"http://127.0.0.1:{served.port}/api/sessions/{session_id}/browser"
            )
        ).json()
    assert state["open"] and state["tab"] == "t2" and state["viewers"] == 0


# ── The panel, in a second headless Chrome ──


async def panel_state(person: ChromeBackgroundDriver) -> dict[str, Any]:
    raw = await person._eval("""JSON.stringify((() => {
          const p = document.getElementById('browser-panel'), c = document.getElementById('bp-canvas');
          const r = p.getBoundingClientRect(), chip = document.getElementById('browser-chip');
          let ink = 0;
          if (!c.hidden && c.width) {
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            for (let i = 0; i < d.length; i += 4 * 97) if (d[i] < 235 || d[i + 1] < 235 || d[i + 2] < 235) ink += 1;
          }
          return {hidden: p.hidden, state: p.dataset.state, browsing: p.dataset.browsing, picture: p.dataset.picture,
            title: document.getElementById('bp-title').textContent, host: document.getElementById('bp-host').textContent,
            status: document.getElementById('bp-status').textContent, note: document.getElementById('bp-note').textContent,
            width: Math.round(r.width), chip: chip.hidden ? '' : chip.textContent, ink,
            melo: getComputedStyle(document.getElementById('panel')).display, view: document.documentElement.dataset.view || ''};
        })())""")
    return cast(dict[str, Any], raw if isinstance(raw, dict) else json.loads(raw))


async def sign_in(person: ChromeBackgroundDriver, served: Served) -> str:
    """Open the console in the person's Chrome, signed in; the address to use after."""
    code = served.auth.issue_login(served.port)
    root = f"http://127.0.0.1:{served.port}/"
    await person.open(f"{root}#login={code}", new_tab=True)
    await settled(lambda: _loaded(person))
    return root


async def _loaded(person: ChromeBackgroundDriver) -> bool:
    return bool(await person._eval("document.documentElement.dataset.ready === '1'"))


async def test_the_panel_alone_shows_the_page_melo_is_on(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    person: ChromeBackgroundDriver,
    tmp_path: Path,
) -> None:
    session_id = await chat(console, workspace)
    agent = await browse(builder, session_id, base + "/moving")
    root = await sign_in(person, served)
    await person.open(f"{root}?pop=1#/live/{session_id}", new_tab=False)
    await settled(lambda: _loaded(person))

    async def shown() -> bool:
        found = await panel_state(person)
        return bool(found["state"] == "live" and found["picture"] == "true")

    await settled(shown)
    found = await panel_state(person)
    assert found["view"] == "live" and found["hidden"] is False
    assert found["title"] == "Moving page" and found["host"].startswith("127.0.0.1:")
    assert (
        found["status"] == "Live" and found["ink"] > 5
    ), "the picture is the page, not a blank"
    shot = await person.screenshot(None)
    (tmp_path / "panel-alone.jpg").write_bytes(base64.b64decode(shot.data))
    # Closing the window ends the stream.
    await person._eval("location.href = 'about:blank'")
    browser = agent.driver._browser
    assert browser is not None
    await settled(lambda: _no_streams(browser))


async def test_the_console_opens_the_panel_beside_the_chat_and_hides_and_shows_it(
    console: Console,
    builder: RealBuilder,
    workspace: Path,
    served: Served,
    base: str,
    person: ChromeBackgroundDriver,
    tmp_path: Path,
) -> None:
    session_id = await chat(console, workspace)
    agent = await browse(builder, session_id, base + "/moving")
    root = await sign_in(person, served)
    await person.open(f"{root}?chat=1#/s/{session_id}", new_tab=False)
    await settled(lambda: _loaded(person))

    async def live() -> bool:
        found = await panel_state(person)
        return bool(found["state"] == "live" and found["picture"] == "true")

    await settled(live)
    found = await panel_state(person)
    # 1280 px is too narrow for Melo's panel as well: the browser takes its place.
    assert (
        found["hidden"] is False
        and 300 <= found["width"] <= 440
        and found["melo"] == "none"
    )
    assert found["chip"].strip() in {"Browser", "melo is browsing"}
    shot = await person.screenshot(None)
    (tmp_path / "console-with-panel.jpg").write_bytes(base64.b64decode(shot.data))
    browser = agent.driver._browser
    assert browser is not None and len(browser.casts) == 1
    # What watching costs, on both sides: melo's Chrome draws and encodes, the person's draws the panel.
    viewer = person._browser
    assert viewer is not None
    mine, theirs, started = (
        cpu_seconds(browser.pid),
        cpu_seconds(viewer.pid),
        time.monotonic(),
    )
    await asyncio.sleep(4.0)
    spent = time.monotonic() - started
    kept(
        tmp_path,
        "panel_open",
        {
            "melo_chrome_cpu_percent_of_a_core": round(
                100 * (cpu_seconds(browser.pid) - mine) / spent, 1
            ),
            "person_chrome_cpu_percent_of_a_core": round(
                100 * (cpu_seconds(viewer.pid) - theirs) / spent, 1
            ),
        },
    )
    # Hide it: nobody watches, so the stream stops; the chip brings it back.
    await person._eval("document.getElementById('bp-hide').click()")
    await settled(lambda: _no_streams(browser))
    found = await panel_state(person)
    assert found["hidden"] is True and found["melo"] != "none" and found["chip"]
    await person._eval("document.getElementById('browser-chip').click()")
    await settled(lambda: _one_stream(browser))
    await settled(live)
