"""The browser driver against a real Safari automation window (opt in).

This is the one test that opens Safari. It is skipped unless
``LOCALMELO_SAFARI=1``, because it needs Safari's remote automation switched
on (Settings > Developer > Allow remote automation) and it opens a window.
Run it by hand: ``LOCALMELO_SAFARI=1 uv run pytest tests/browser/test_real_safari.py -s``.

It drives the driver directly, against a page served on this computer's own
loopback by the test (the system picks the port). The provider is not used,
since melo's tools never open a local address; the driver has no such rule.

It also prints what Safari's helper answers (the shape of its tab list and of
the ``page_interactions`` tool), so a native click and typing path can be
decided from the real schema.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

import pytest

from localmelo.support.browser.driver import Act, Element, RawPage, StaleElementError
from localmelo.support.browser.safari import SafariAutomationDriver

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

    def log_message(self, *args: Any) -> None:
        pass

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
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)  # the system picks the port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


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
        assert "tried to send a form" in note and Site.posts == []
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
