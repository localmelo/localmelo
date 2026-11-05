"""The browser driver against a real Safari automation window (opt in).

This is the one test that opens Safari. It is skipped unless
``LOCALMELO_SAFARI=1``, because it needs Safari's remote automation switched
on (Settings > Developer > Allow remote automation) and it opens a window.
Run it by hand: ``LOCALMELO_SAFARI=1 uv run pytest tests/browser/test_real_safari.py -s``.

It drives the driver directly, against a page served on this computer's own
loopback by the test (the system picks the port). The provider is not used,
since melo's tools never open a local address; the driver has no such rule.

It also prints what Safari's helper answers (the shape of its tab list and of
the ``page_interactions`` tool).

The fixture test runs the tools themselves on the pages in ``fixtures/``,
served by two loopback servers (two origins, ports the system picks); for
those two origins alone the test lets the tools open this computer. It checks
the browser's own click (a page that ignores a script's click), the look after
an act (a notice that comes late, a page that redraws), a ref the page redrew
found again, a box from another site and a drawn button clicked at a point.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest

from localmelo.support.browser import addresses, pagescript, sites
from localmelo.support.browser import provider as provider_module
from localmelo.support.browser.driver import Act, Element, RawPage, StaleElementError
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.safari import SafariAutomationDriver
from localmelo.tests.fixtures.http_stub import HttpStub

pytestmark = pytest.mark.skipif(
    os.environ.get("LOCALMELO_SAFARI") != "1",
    reason="opens a real Safari automation window: set LOCALMELO_SAFARI=1",
)

PAGE = """<!doctype html><meta charset="utf-8"><title>melo fixture</title>
<body><main>
<form id="f" action="/sent" method="post">
  <label for="q">Search</label><input id="q" name="q" value="cats">
  <select name="c" aria-label="Country"><option>France</option><option>Spain</option></select>
  <input type="password" name="pw" aria-label="Password">
  <button id="go">Go</button>
</form>
<a href="/news">News</a>
</main></body>"""
SENT = "<!doctype html><title>Sent</title><body><p>Form received.</p></body>"


class Site(BaseHTTPRequestHandler):
    posts: list[dict[str, list[str]]] = []

    def _send(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(PAGE if self.path == "/" else "<title>News</title><p>News.</p>")

    def do_POST(self) -> None:
        size = int(self.headers.get("content-length") or 0)
        Site.posts.append(parse_qs(self.rfile.read(size).decode("utf-8")))
        self._send(SENT)


@pytest.fixture
def base() -> Iterator[str]:
    Site.posts = []
    with HttpStub(Site, path="/") as stub:
        yield stub.base_url


def label(raw: RawPage, text: str) -> Element:
    return next(e for e in raw.elements if e.label == text)


async def test_the_driver_reads_acts_sends_and_names_tabs_in_a_real_safari(
    base: str,
) -> None:
    driver = SafariAutomationDriver()
    await driver.start()
    opened: list[str] = []
    try:
        # What the helper answers, for deciding on a native click and typing later.
        tools = {name: tool.input_schema for name, tool in driver._tools.items()}
        print("\nreal tool names:", sorted(tools))
        print("page_interactions schema:", json.dumps(tools.get("page_interactions")))
        print("list_tabs row shape:", (await driver._call("tabs")).text[:600])

        first = await driver.open(base, new_tab=True)
        opened.append(first.id)
        assert (
            first.url.rstrip("/") == base.rstrip("/") and first.title == "melo fixture"
        )
        raw = await driver.page(first.id)
        assert raw.doc and label(raw, "Go").fp and label(raw, "Search").fp
        assert any(e.sensitive == "password" for e in raw.elements)
        assert label(raw, "Go").submit == base + "sent"

        # Typing and choosing, then reading what the page now holds.
        await driver.act(
            first.id,
            Act("type", raw.url, label(raw, "Search"), text="dogs", doc=raw.doc),
        )
        await driver.act(
            first.id,
            Act("select", raw.url, label(raw, "Country"), text="Spain", doc=raw.doc),
        )
        after = await driver.page(first.id)
        assert label(after, "Search").value == "dogs"
        assert label(after, "Country").value == "Spain"

        # A send nobody approved is stopped and said so; approved, it goes once.
        note = await driver.act(
            first.id, Act("click", after.url, label(after, "Go"), doc=after.doc)
        )
        assert "tried to send a form" in note.note and Site.posts == []
        await driver.act(
            first.id,
            Act("click", after.url, label(after, "Go"), doc=after.doc, send=True),
        )
        assert [p.get("q") for p in Site.posts] == [["dogs"]]
        assert (await driver.page(first.id)).title == "Sent"

        # A tab is the handle Safari gave it: a second tab, two addresses.
        second = await driver.open(base + "news", new_tab=True)
        opened.append(second.id)
        assert second.id != first.id
        assert (await driver.location(first.id)).endswith("/sent")
        assert (await driver.location(second.id)).endswith("/news")
        assert {t.id for t in await driver.tabs()} >= {first.id, second.id}

        # A reloaded page is another document: an act on the old reading is refused.
        again = await driver.open(base, new_tab=False, tab=first.id)
        fresh = await driver.page(again.id)
        assert fresh.doc != raw.doc
        with pytest.raises(StaleElementError):
            await driver.act(
                first.id, Act("click", fresh.url, label(fresh, "News"), doc=raw.doc)
            )
    finally:
        for tab in opened:
            with contextlib.suppress(Exception):
                await driver.close_tab(tab)
        await driver.close()


# ── The fixture pages: native input, the look after, redraws, frames, points ──

FIXTURES = Path(__file__).with_name("fixtures")


class Fixtures(BaseHTTPRequestHandler):
    """The pages in ``fixtures/``; a POST is recorded, and ``/note`` answers late, as a site may."""

    other = ""  # the second origin, for the frame page
    posts: list[str] = []

    def _send(self, body: str, kind: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        name = self.path.lstrip("/").split("?", 1)[0] or "trusted.html"
        file = FIXTURES / name
        if not name.endswith(".html") or not file.is_file():
            self.send_error(404)
            return
        self._send(file.read_text("utf-8").replace("__OTHER__", Fixtures.other))

    def do_POST(self) -> None:
        size = int(self.headers.get("content-length") or 0)
        sent = self.rfile.read(size).decode("utf-8")
        Fixtures.posts.append(self.path + ("?" + sent if sent else ""))
        if self.path.startswith("/note"):
            time.sleep(0.4)  # the site takes a moment to answer
            self._send("Saved: " + parse_qs(sent).get("note", [""])[0], "text/plain")
            return
        self._send("ok", "text/plain")


@pytest.fixture
def two_origins() -> Iterator[tuple[str, str]]:
    """Two servers on this computer's loopback, two origins; the system picks the ports."""
    Fixtures.posts = []
    with HttpStub(Fixtures) as one, HttpStub(Fixtures) as other:
        Fixtures.other = other.base_url
        yield one.base_url, other.base_url


@pytest.fixture
def real_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real page takes real time: the look after an act waits as it does for a person."""
    monkeypatch.setattr(pagescript, "LOOK_STEP_S", 0.1)
    monkeypatch.setattr(pagescript, "QUIET_S", 0.3)
    monkeypatch.setattr(pagescript, "LOOK_CAP_S", 2.0)
    monkeypatch.setattr(provider_module, "SETTLE_S", 0.3)


@pytest.fixture
def local_fixtures(
    monkeypatch: pytest.MonkeyPatch, two_origins: tuple[str, str]
) -> tuple[str, str]:
    """melo's tools never open this computer; for these two fixture servers alone, the test lets them."""
    allowed = {two_origins[0], two_origins[1]}
    real = addresses.require_public

    async def only_the_fixtures(url: str) -> None:
        if sites.origin_of(url) in allowed:
            return
        await real(url)

    monkeypatch.setattr(addresses, "require_public", only_the_fixtures)
    return two_origins


def ref_of(block: str, words: str) -> str:
    """The ref of the element whose line holds *words*, in a page block."""
    for line in block.splitlines():
        if words in line and line.startswith("[e"):
            return line[1:].split(" ", 1)[0]
    raise AssertionError(f"no element {words!r} in:\n{block}")


async def test_the_fixture_pages_in_a_real_safari(
    real_timing: None, local_fixtures: tuple[str, str]
) -> None:
    first, second = local_fixtures
    tools_provider = BrowserToolProvider(SafariAutomationDriver())
    tools = {t.definition.name: t for t in await tools_provider.start()}

    async def call(name: str, **args: Any) -> Any:
        out = await tools[name].function(**args)
        print(f"\n--- {name} {args}\n{getattr(out, 'text', out)}")
        return out

    try:
        # The browser's own click checks in where a script's click does nothing.
        page = await call("browser_open", url=f"{first}/trusted.html")
        clicked = await call("browser_click", ref=ref_of(page, '"Check in"'))
        assert clicked.startswith(f"Clicked {ref_of(page, 'Check in')} (via browser).")
        assert "Checked in: day 12" in clicked and Fixtures.posts == ["/checked-in"]

        # A page that redraws: the look waits it out, and a ref it redrew later is found again.
        page = await call("browser_open", url=f"{first}/redraw.html")
        happy = await call("browser_click", ref=ref_of(page, '"Happy"'))
        assert "Mood: Happy" in happy
        calm = ref_of(happy, '"Calm"')
        await asyncio.sleep(3.0)  # the page redraws itself again, as a late reply would
        again = await call("browser_click", ref=calm)
        assert "Mood: Calm" in again and "is the one element like it now" in again

        # A notice that comes a moment after a send is in what changed.
        page = await call("browser_open", url=f"{first}/notice.html")
        typed = await call("browser_type", ref=ref_of(page, '"Note"'), text="hello")
        assert "(via browser)" in typed
        saved = await call("browser_click", ref=ref_of(typed, '"Save"'))
        assert "Saved: hello" in saved.split("<<end page", 1)[0]

        # A box from another site: its controls are listed and clicked by Safari itself.
        await call("browser_open", url=f"{first}/frame.html")
        await asyncio.sleep(1.0)  # the box loads after the page
        page = await call("browser_read")
        host = second.split("://", 1)[1]
        assert f'"Continue" in a box from {host}]' in page
        boxed = await call("browser_click", ref=ref_of(page, '"Continue"'))
        assert boxed.startswith(f"Clicked {ref_of(page, 'Continue')} (via browser).")
        await asyncio.sleep(0.5)
        assert Fixtures.posts[-1] == "/box-clicked?trusted=true"
        # The same box, at a point of the picture.
        await call("browser_screenshot")
        view = tools_provider._views[tools_provider._tab or "t1"]
        x, y = (40 + 20 + 80) / view.width * 1000, (120 + 20 + 20) / view.height * 1000
        pointed = await call("browser_click_point", x=round(x), y=round(y))
        assert not pointed.is_error and pointed.images
        assert "shows only in the new picture" in pointed.text
        await asyncio.sleep(0.5)
        assert Fixtures.posts.count("/box-clicked?trusted=true") == 2

        # A drawn button: a point, the browser's own click, and a new picture after.
        await call("browser_open", url=f"{first}/canvas.html")
        await call("browser_screenshot")
        view = tools_provider._views[tools_provider._tab or "t1"]
        x, y = (40 + 150) / view.width * 1000, (120 + 75) / view.height * 1000
        started = await call("browser_click_point", x=round(x), y=round(y))
        assert "Started (real click)" in started.text and started.images
    finally:
        await tools_provider.close()


async def test_what_the_real_server_reaches_in_a_box_from_another_site(
    local_fixtures: tuple[str, str],
) -> None:
    """The browser's own reading lists a box's controls; a script cannot run in it."""
    first, second = local_fixtures
    driver = SafariAutomationDriver()
    await driver.start()
    try:
        info = await driver.open(f"{first}/frame.html", new_tab=True)
        await asyncio.sleep(1.0)  # the box loads after the page
        raw = await driver.page(info.id)
        boxed = [e for e in raw.elements if e.site]
        print("\nreached in the box:", [(e.kind, e.label, e.frame) for e in boxed])
        assert [e.label for e in boxed] == ["Continue"] and boxed[0].site == second
        nodes = await driver._nodes("entire_page")
        frame = next(n for n in nodes if n.tag == "iframe")
        inside = await driver._call_with(
            "script", {"expression": "return location.href", "frameId": frame.uid}
        )
        print("a script in the box answers:", inside.text[:200])
        await driver.close_tab(info.id)
    finally:
        await driver.close()
