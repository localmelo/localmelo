"""What only the real Google Chrome can say about the person's page, headless (opt in).

Skipped unless ``LOCALMELO_CHROME_SMOKE=1`` and Google Chrome is installed. One
headless Chrome runs on a profile under ``tmp_path``, no window opens, and the
pages are served by the test on ``127.0.0.1`` (the system picks the port).
Whether a page has a form the person began, in an editor, an open shadow root
and a frame of its own site; and a window a page closed itself against one
closed from outside. Run it by hand:
``LOCALMELO_CHROME_SMOKE=1 uv run pytest tests/browser/test_real_person_page.py -s``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.tests.browser import test_real_live as smoke
from localmelo.tests.fixtures.http_stub import HttpStub

pytestmark = smoke.pytestmark
never_the_front = smoke.never_the_front

PAGES = {
    "/editor": "<div id=box contenteditable style='height:40px'></div>",
    "/shadow": (
        "<div id=host></div><script>"
        "document.getElementById('host').attachShadow({mode:'open'})"
        ".innerHTML='<input id=box>'</script>"
    ),
    "/frame": "<iframe srcdoc='<input id=box>'></iframe>",
    "/opener": "<p>opener</p>",
    "/popup": "<p>popup</p>",
}
FOCUS = {
    "/editor": "document.getElementById('box').focus()",
    "/shadow": "document.getElementById('host').shadowRoot.getElementById('box').focus()",
    "/frame": "document.querySelector('iframe').contentDocument.getElementById('box').focus()",
}


class Pages(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        data = f"<!doctype html><title>t</title><body>{PAGES[self.path]}".encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def base() -> Iterator[str]:
    with HttpStub(Pages) as stub:
        yield stub.base_url


@pytest.fixture
async def driver(tmp_path: Path) -> AsyncIterator[ChromeBackgroundDriver]:
    found = ChromeBackgroundDriver(
        command=[smoke.BINARY], profile=str(tmp_path / "profile")
    )
    await found.start()
    try:
        yield found
    finally:
        await found.close()


async def _in_page(driver: ChromeBackgroundDriver, tab: str, source: str) -> None:
    """Run *source* in the page's own world, as the page would."""
    session = await driver._focus(tab)
    await driver._cmd(
        "Runtime.evaluate", {"expression": source, "userGesture": True}, session
    )


@pytest.mark.parametrize("path", sorted(FOCUS))
async def test_words_typed_in_an_editor_a_shadow_root_or_a_frame_are_a_form_begun(
    driver: ChromeBackgroundDriver, base: str, path: str
) -> None:
    tab = await driver.open(base + path, new_tab=True)
    assert await driver.unsent_input(tab.id) is False, "nothing typed yet"
    await _in_page(driver, tab.id, FOCUS[path])
    session = await driver._focus(tab.id)
    await driver._cmd("Input.insertText", {"text": "half a message"}, session)
    assert await driver.unsent_input(tab.id) is True


async def _until_told(told: list[str]) -> None:
    async with asyncio.timeout(5):
        while not told:
            await asyncio.sleep(0.02)


async def test_a_window_a_page_closed_itself_is_told_apart_from_one_closed_from_outside(
    driver: ChromeBackgroundDriver, base: str
) -> None:
    opener = await driver.open(base + "/opener", new_tab=True)
    await _in_page(
        driver, opener.id, f"window.open({(base + '/popup')!r}, 'p', 'width=400')"
    )
    async with asyncio.timeout(5):
        while len(tabs := await driver.tabs()) < 2:
            await asyncio.sleep(0.02)
    popup = next(t for t in tabs if t.id != opener.id)
    browser = driver._browser
    assert browser is not None
    await driver._cmd("Target.setDiscoverTargets", {"discover": True})
    closes_itself: list[str] = []
    target = driver._resolve(popup.id)
    browser.watch[target] = closes_itself.append
    await driver._hook_close(browser, target)
    await _in_page(driver, popup.id, "setTimeout(function(){window.close()}, 50)")
    await _until_told(closes_itself)
    assert closes_itself == ["gone"]
    closed: list[str] = []
    target = driver._resolve(opener.id)
    browser.watch[target] = closed.append
    await driver._hook_close(browser, target)
    await driver._cmd("Target.closeTarget", {"targetId": target})
    await _until_told(closed)
    assert closed == ["closed"]
