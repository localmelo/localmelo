"""A sign-in helper on this Mac in the real background Chrome (opt in).

Skipped unless ``LOCALMELO_CHROME_SMOKE=1`` and Google Chrome is installed. It
starts Chrome with no window on a profile under ``tmp_path``. Every server is
the test's own, on ``127.0.0.1`` at a port the system picks:

- the pages, under two test names (``site.test`` for any site, ``login.test``
  for the helper's own sign-in page), which Chrome is told are public
  (``--ip-address-space-overrides``), so its Local Network Access check
  treats them as it treats a site on the internet;
- the sign-in helper (``signin.test``, listed for ``login.test`` only), another
  app on this Mac, a WebSocket server, and a stand-in for melo's gateway whose
  port the rule calls melo's own.

The last tests are about a server the person wrote the address of
(``allow_local``, #547): that exact host and port opens, and no other local
address does, a redirect's and a subresource's included.

A page fetches each one and reports what it got; each server records what
reached it, so a test fails if a request it must stop ever arrives.
Run it by hand: ``LOCALMELO_CHROME_SMOKE=1 uv run pytest tests/browser/test_real_local_apps.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, quote, urlsplit

import pytest
import websockets

from localmelo.melo.executor.models import ToolOutput
from localmelo.support import local_ports
from localmelo.support.browser import chrome
from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.browser.driver import BrowserError
from localmelo.support.browser.local_apps import LocalRule, SigninHelper
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.test_acting import asking
from localmelo.tests.web.test_written_local import WroteForBrowser

BINARY = os.path.join(chrome.CHROME_APPS[0], chrome.APP_BINARY)
pytestmark = pytest.mark.skipif(
    os.environ.get("LOCALMELO_CHROME_SMOKE") != "1" or not os.access(BINARY, os.X_OK),
    reason="starts the real Chrome, headless: set LOCALMELO_CHROME_SMOKE=1",
)

REPORT_S = 10.0


def _script(kind: str, target: str) -> str:
    """What a test page does with *target*, then reports to its own server."""
    goal = json.dumps(target)
    report = (
        ".then(r => fetch('/report?r=' + encodeURIComponent(location.host + ' ' + r)))"
    )
    got = ".then(r => r.text()).then(t => 'ok ' + t).catch(e => 'fail ' + e)"
    if kind == "ws":
        return (
            "new Promise(done => { const w = new WebSocket(" + goal + ");"
            " w.onopen = () => done('ok'); w.onerror = () => done('fail');"
            " setTimeout(() => done('timeout'), 4000); })" + report
        )
    if kind == "nav":
        return "setTimeout(() => { location.href = " + goal + "; }, 50)"
    if kind == "post":
        return (
            "fetch(" + goal + ", {method: 'POST', headers: {'content-type': "
            "'application/json'}, body: JSON.stringify({ask: 1})})" + got + report
        )
    return "fetch(" + goal + ")" + got + report


class Pages(BaseHTTPRequestHandler):
    reports: list[str] = []

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, body: str, kind: str = "text/html") -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", f"{kind}; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if parts.path == "/report":
            Pages.reports.append(query.get("r", ""))
            self._send("ok", "text/plain")
        elif parts.path == "/top":
            self._send(
                f'<!doctype html><body>A site<iframe src="{query["src"]}"></iframe>'
            )
        else:
            script = _script(query.get("kind", "fetch"), query.get("target", ""))
            self._send(f"<!doctype html><body>A page<script>{script}</script>")


class App(BaseHTTPRequestHandler):
    """An app on this Mac: answers like a sign-in helper and records every request."""

    seen: dict[int, list[str]] = {}
    other = 0

    def log_message(self, *args: Any) -> None:
        pass

    def _answer(self) -> None:
        port = int(cast(Any, self.server).server_address[1])
        App.seen.setdefault(port, []).append(f"{self.command} {self.path}")
        size = int(self.headers.get("content-length") or 0)
        if size:
            self.rfile.read(size)
        if self.path.startswith("/redir"):
            self.send_response(302)
            self.send_header("location", f"http://127.0.0.1:{App.other}/x")
            self.send_header("access-control-allow-origin", "*")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        data = b"signed-in user"
        self.send_response(200)
        self.send_header(
            "access-control-allow-origin", self.headers.get("origin") or "*"
        )
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-allow-methods", "GET, POST")
        self.send_header("access-control-allow-private-network", "true")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._answer()

    def do_POST(self) -> None:  # noqa: N802
        self._answer()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._answer()


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@dataclass
class World:
    driver: ChromeBackgroundDriver
    pages: int
    helper: int
    other: int
    gateway: int
    socket: int
    sockets: list[str]

    def site(self, kind: str, target: str) -> str:
        return f"http://site.test:{self.pages}/p?kind={kind}&target={quote(target, safe='')}"

    def login_frame(self, kind: str, target: str) -> str:
        """A site that shows the helper's sign-in page in a frame, as a forum shows a sign-in widget."""
        inner = f"http://login.test:{self.pages}/p?kind={kind}&target={quote(target, safe='')}"
        return f"http://site.test:{self.pages}/top?src={quote(inner, safe='')}"

    async def report(self) -> str:
        loop = asyncio.get_running_loop()
        end = loop.time() + REPORT_S
        while not Pages.reports and loop.time() < end:
            await asyncio.sleep(0.05)
        assert Pages.reports, "the page never reported"
        return Pages.reports.pop(0)

    async def ports_told(self) -> list[int]:
        await asyncio.sleep(
            0.3
        )  # a block is told to its page a moment after Chrome pauses it
        return [attempt.port for attempt in self.driver.local_attempts()]


@pytest.fixture
def servers() -> Iterator[dict[str, int]]:
    Pages.reports, App.seen = [], {}
    made = {
        name: _serve(Pages if name == "pages" else App)
        for name in ("pages", "helper", "other", "gateway")
    }
    App.other = made["other"].server_address[1]
    try:
        yield {name: server.server_address[1] for name, server in made.items()}
    finally:
        for server in made.values():
            server.shutdown()
            server.server_close()


@pytest.fixture
async def world(tmp_path: Path, servers: dict[str, int]) -> AsyncIterator[World]:
    sockets: list[str] = []

    async def echo(connection: Any) -> None:
        sockets.append("open")
        await connection.close()

    wss = await websockets.serve(echo, "127.0.0.1", 0)
    socket_port = wss.sockets[0].getsockname()[1]
    pages = servers["pages"]
    helper = SigninHelper(
        "Signin",
        "http",
        "signin.test",
        (servers["helper"], servers["gateway"]),
        "127.0.0.1",
        (f"http://login.test:{pages}",),
    )
    rule = LocalRule((helper,), own=lambda: {servers["gateway"]: local_ports.GATEWAY})
    command = [
        BINARY,
        f"--ip-address-space-overrides=127.0.0.1:{pages}=public",
        "--host-resolver-rules=MAP site.test 127.0.0.1, MAP login.test 127.0.0.1, "
        "MAP signin.test 127.0.0.1, MAP sneaky.test 127.0.0.1",
        "--unsafely-treat-insecure-origin-as-secure="
        f"http://site.test:{pages},http://login.test:{pages}",
    ]
    driver = ChromeBackgroundDriver(
        command=command, profile=str(tmp_path / "p"), local=rule
    )
    await driver.start()
    try:
        yield World(
            driver,
            pages,
            servers["helper"],
            servers["other"],
            servers["gateway"],
            socket_port,
            sockets,
        )
    finally:
        await driver.close()
        wss.close()


async def test_the_listed_helper_answers_its_own_sign_in_page_inside_another_site(
    world: World,
) -> None:
    await world.driver.open(
        world.login_frame("fetch", f"http://signin.test:{world.helper}/api"),
        new_tab=True,
    )
    assert await world.report() == f"login.test:{world.pages} ok signed-in user"
    assert App.seen[world.helper] == ["GET /api"]
    assert await world.ports_told() == []


async def test_a_request_the_browser_must_ask_about_first_is_relayed_too(
    world: World,
) -> None:
    await world.driver.open(
        world.login_frame("post", f"http://signin.test:{world.helper}/authorize"),
        new_tab=True,
    )
    assert await world.report() == f"login.test:{world.pages} ok signed-in user"
    assert App.seen[world.helper] == ["OPTIONS /authorize", "POST /authorize"]


async def test_the_same_helper_is_blocked_for_any_other_page(world: World) -> None:
    await world.driver.open(
        world.site("fetch", f"http://signin.test:{world.helper}/api"), new_tab=True
    )
    assert (await world.report()).startswith(f"site.test:{world.pages} fail")
    assert world.helper not in App.seen
    assert await world.ports_told() == [world.helper]


async def test_another_app_on_this_mac_is_blocked_and_the_step_is_told(
    world: World,
) -> None:
    await world.driver.open(
        world.site("fetch", f"http://127.0.0.1:{world.other}/x"), new_tab=True
    )
    assert (await world.report()).startswith(f"site.test:{world.pages} fail")
    assert world.other not in App.seen
    assert await world.ports_told() == [world.other]


async def test_melos_gateway_port_is_blocked_even_when_listed(world: World) -> None:
    await world.driver.open(
        world.login_frame("fetch", f"http://signin.test:{world.gateway}/v1/agent/run"),
        new_tab=True,
    )
    assert (await world.report()).startswith(f"login.test:{world.pages} fail")
    assert world.gateway not in App.seen
    assert await world.ports_told() == [world.gateway]


async def test_a_redirect_from_the_helper_to_another_app_is_blocked(
    world: World,
) -> None:
    await world.driver.open(
        world.login_frame("fetch", f"http://signin.test:{world.helper}/redir"),
        new_tab=True,
    )
    assert (await world.report()).startswith(f"login.test:{world.pages} fail")
    assert App.seen[world.helper] == ["GET /redir"]
    assert world.other not in App.seen


async def test_a_websocket_to_this_mac_is_blocked(world: World) -> None:
    await world.driver.open(
        world.site("ws", f"ws://127.0.0.1:{world.socket}/"), new_tab=True
    )
    assert await world.report() == f"site.test:{world.pages} fail"
    await world.driver.open(
        world.login_frame("ws", f"ws://signin.test:{world.socket}/"), new_tab=True
    )
    assert await world.report() == f"login.test:{world.pages} fail"
    assert world.sockets == []


async def test_a_name_that_leads_to_this_mac_is_blocked(world: World) -> None:
    await world.driver.open(
        world.site("fetch", f"http://sneaky.test:{world.other}/x"), new_tab=True
    )
    assert (await world.report()).startswith(f"site.test:{world.pages} fail")
    assert world.other not in App.seen


async def test_a_page_cannot_send_its_tab_to_this_mac_with_a_user_part(
    world: World,
) -> None:
    await world.driver.open(
        world.site("nav", f"http://x@127.0.0.1:{world.other}/x"), new_tab=True
    )
    assert await world.ports_told() == [world.other]
    await asyncio.sleep(0.5)
    assert world.other not in App.seen


async def test_a_page_cannot_send_its_tab_to_this_mac(world: World) -> None:
    await world.driver.open(
        world.site("nav", f"http://127.0.0.1:{world.other}/x"), new_tab=True
    )
    assert await world.ports_told() == [world.other]
    await asyncio.sleep(0.5)
    assert world.other not in App.seen


# ── A server on this Mac the person wrote the address of (#547) ──


def local(port: int, path: str = "/hello", host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}{path}"


def asked() -> dict[int, list[str]]:
    """What reached each app, a page's own icon request aside."""
    found = {p: [r for r in rs if "favicon" not in r] for p, rs in App.seen.items()}
    return {port: requests for port, requests in found.items() if requests}


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
async def test_the_server_the_person_wrote_opens(world: World, host: str) -> None:
    world.driver.allow_local(lambda: [("127.0.0.1", world.other)])
    tab = await world.driver.open(local(world.other, host=host), new_tab=True)
    assert tab.url.startswith(f"http://{host}:{world.other}/hello")
    assert asked() == {world.other: ["GET /hello"]}
    assert await world.ports_told() == []


async def test_a_server_the_person_did_not_write_is_refused_and_no_request_is_sent(
    world: World,
) -> None:
    world.driver.allow_local(lambda: [("127.0.0.1", world.other)])
    for port in (world.helper, world.gateway):
        with pytest.raises(BrowserError, match="could not load"):
            await world.driver.open(local(port), new_tab=True)
    assert asked() == {}
    assert sorted(await world.ports_told()) == sorted([world.helper, world.gateway])
    # The one they wrote still opens, so the refusals are not a broken driver.
    await world.driver.open(local(world.other), new_tab=True)
    assert list(asked()) == [world.other]


async def test_a_driver_that_was_given_nothing_reaches_no_local_address(
    world: World,
) -> None:
    with pytest.raises(BrowserError, match="could not load"):
        await world.driver.open(local(world.other), new_tab=True)
    world.driver.allow_local(lambda: [("127.0.0.1", world.other)])
    await world.driver.open(local(world.other), new_tab=True)
    world.driver.allow_local(None)  # the next step's words wrote none
    with pytest.raises(BrowserError, match="could not load"):
        await world.driver.open(local(world.other), new_tab=True)
    assert asked() == {world.other: ["GET /hello"]}


async def test_the_written_server_is_refused_once_the_run_that_wrote_it_is_over(
    world: World,
) -> None:
    """The driver keeps the run's reader: no browser step has to come for the end of the run to count."""
    live = [True]
    world.driver.allow_local(lambda: [("127.0.0.1", world.other)] if live[0] else [])
    tab = await world.driver.open(local(world.other), new_tab=True)
    live[0] = False
    with pytest.raises(BrowserError, match="could not load"):
        await world.driver.open(local(world.other), new_tab=False, tab=tab.id)
    assert asked() == {world.other: ["GET /hello"]}
    assert await world.ports_told() == [world.other]


async def test_melos_own_port_is_refused_even_when_the_person_wrote_it(
    world: World,
) -> None:
    world.driver.allow_local(
        lambda: [("127.0.0.1", world.gateway), ("127.0.0.1", world.other)]
    )
    with pytest.raises(BrowserError, match="could not load"):
        await world.driver.open(local(world.gateway), new_tab=True)
    assert world.gateway not in App.seen
    assert await world.ports_told() == [world.gateway]
    await world.driver.open(local(world.other), new_tab=True)


async def test_a_redirect_from_the_written_server_to_another_local_address_is_refused(
    world: World,
) -> None:
    # The helper's port written: its ``/redir`` sends the tab to ``other``, which is not.
    world.driver.allow_local(lambda: [("127.0.0.1", world.helper)])
    with pytest.raises(BrowserError, match="could not load"):
        await world.driver.open(local(world.helper, "/redir"), new_tab=True)
    assert asked() == {world.helper: ["GET /redir"]}
    assert await world.ports_told() == [world.other]
    # Written too, the redirect's target opens: the exact address, not a pattern.
    world.driver.allow_local(
        lambda: [("127.0.0.1", world.helper), ("127.0.0.1", world.other)]
    )
    tab = await world.driver.open(local(world.helper, "/redir"), new_tab=True)
    assert tab.url.startswith(f"http://127.0.0.1:{world.other}/x")


async def test_a_page_at_the_written_server_reaches_no_other_local_address(
    world: World,
) -> None:
    world.driver.allow_local(lambda: [("127.0.0.1", world.pages)])
    target = f"http://127.0.0.1:{world.other}/x"
    await world.driver.open(
        local(world.pages, f"/p?kind=fetch&target={quote(target, safe='')}"),
        new_tab=True,
    )
    # Its own server answered (the report is a request to the written port) and the other did not.
    assert (await world.report()).startswith(f"127.0.0.1:{world.pages} fail")
    assert world.other not in App.seen
    assert await world.ports_told() == [world.other]


async def test_a_public_page_still_cannot_reach_the_written_server(
    world: World,
) -> None:
    """The written address is let through melo's rule only: Chrome's own check of a public page's request to this Mac still stops it."""
    world.driver.allow_local(lambda: [("127.0.0.1", world.other)])
    await world.driver.open(
        world.site("fetch", f"http://127.0.0.1:{world.other}/x"), new_tab=True
    )
    assert (await world.report()).startswith(f"site.test:{world.pages} fail")
    assert world.other not in App.seen


async def test_the_browser_tools_open_the_server_the_person_wrote_in_the_real_chrome(
    world: World,
) -> None:
    """The whole way: the run's words, the provider's address check, the driver's rule and Chrome."""
    provider = BrowserToolProvider(world.driver)
    tools = {tool.definition.name: tool for tool in await provider.start()}

    async def opened(port: int, path: str = "/hello") -> ToolOutput:
        got = await tools["browser_open"].function(url=local(port, path, "localhost"))
        return got if isinstance(got, ToolOutput) else ToolOutput(text=got)

    try:
        with asking(WroteForBrowser(("127.0.0.1", world.other))):
            ok = await opened(world.other)
            assert not ok.is_error and "signed-in user" in ok.text, ok.text
            assert asked() == {world.other: ["GET /hello"]}
            for port in (world.helper, world.gateway):  # not written
                refused = await opened(port)
                assert refused.is_error, refused.text
        with asking(WroteForBrowser(("127.0.0.1", world.helper))):
            moved = await opened(world.helper, "/redir")  # to a port not written
            assert moved.is_error, moved.text
        assert asked() == {world.other: ["GET /hello"], world.helper: ["GET /redir"]}
        with asking(WroteForBrowser()):  # the next message wrote none
            assert (await opened(world.other)).is_error
        assert asked() == {world.other: ["GET /hello"], world.helper: ["GET /redir"]}
    finally:
        await provider.close()
