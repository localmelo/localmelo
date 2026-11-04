"""A page asks to open another app, or puts up a box: the person says, melo opens it, the screen is watched.

The browser is a fake (the scripted one, or the Chrome driver over the fake
Chrome peer); the app is named and opened by a fake opener, so nothing here
asks LaunchServices or opens an app.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import never_granted
from localmelo.support.browser import apps, sites
from localmelo.support.browser.chrome import ChromeBackgroundDriver
from localmelo.support.browser.desktop import MELO_APPS, DesktopHelper, DesktopWatch
from localmelo.support.browser.driver import Act, ActResult, Dialog
from localmelo.support.browser.provider import (
    BACKGROUND_NOTE,
    PAGE_NOTE,
    BrowserToolProvider,
)
from localmelo.tests.browser.fakes import FakeDriver, button
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_chrome import FAKE as FAKE_CHROME
from localmelo.tests.browser.test_chrome import Peer
from localmelo.tests.browser.test_other_windows import (
    BUNDLES,
    SIGN_IN,
    Screen,
    run,
)
from localmelo.tests.browser.test_other_windows import FAKE as FAKE_DESKTOP
from localmelo.tests.browser.test_window_cards import settled

CHAT = "chat://login?token=secret-token"


class Opener:
    """Names the app an address is for, and records what melo opened."""

    def __init__(self, app: str = "Chat", then: Callable[[], None] | None = None):
        self.app = app
        self.then = then
        self.named: list[str] = []
        self.opened: list[str] = []

    async def name_for(self, url: str) -> str:
        self.named.append(url)
        return self.app

    async def open(self, url: str) -> None:
        self.opened.append(url)
        if self.then is not None:
            self.then()


def handing_off(
    driver: FakeDriver,
    handoff: dict[str, str] | None = None,
    boxes: dict[str, tuple[Dialog, ...]] | None = None,
) -> FakeDriver:
    """*driver* as Chrome's: an act on a label asks for an address, or puts up boxes."""
    real = driver.act

    async def act(tab: str | None, what: Act) -> ActResult:
        result = await real(tab, what)
        label = what.element.label
        return dataclasses.replace(
            result,
            handoff=(handoff or {}).get(label, ""),
            dialogs=(boxes or {}).get(label, ()),
        )

    driver.act = act  # type: ignore[method-assign,assignment]
    driver.label = "Chrome"  # type: ignore[attr-defined]
    return driver


def chrome_watch(screen: Screen) -> DesktopWatch:
    command = [
        sys.executable,
        "-I",
        FAKE_DESKTOP,
        str(screen.path),
        str(screen.log_path),
    ]

    async def no_handler(url: str) -> frozenset[str]:
        return frozenset()  # LaunchServices is never asked here

    async def bundle(pid: int) -> str:
        return BUNDLES.get(pid, "")

    return DesktopWatch(
        DesktopHelper(command),
        skip=MELO_APPS,
        front=screen.in_front,
        handler=no_handler,
        bundle=bundle,
        clock=screen.clock,
        sleep=screen.clock.sleep,
    )


async def tools(
    driver: FakeDriver, opener: Opener, desk: DesktopWatch | None = None
) -> tuple[BrowserToolProvider, dict[str, Any]]:
    provider = BrowserToolProvider(driver, desktop=desk, opener=opener)  # type: ignore[arg-type]
    return provider, {t.definition.name: t for t in await provider.start()}


async def clicked(
    opener: Opener, asks: Asks | None, desk: DesktopWatch | None = None
) -> Any:
    driver = handing_off(world(button("Open in Chat")), {"Open in Chat": CHAT})
    provider, found = await tools(driver, opener, desk)
    try:
        with asking(asks):
            await run(found, "browser_open", url=HOME)
            return await run(found, "browser_click", ref="e1")
    finally:
        await provider.close()


# ── The hand-off ──


async def test_a_page_that_asks_for_an_app_is_offered_to_the_person_and_opened_on_yes() -> (
    None
):
    opener, asks = Opener(), Asks()
    out = await clicked(opener, asks)
    assert not out.is_error
    name, arguments, preview = asks.asked[-1]
    assert name == "handoff:chat"
    assert arguments == {
        "site": "example.com",
        "access": "handoff",
        "app": "Chat",
        "scheme": "chat",
        "url": CHAT,
        "browser": "chrome",
    }
    assert preview == "open Chat for a page on example.com (chat: address)"
    assert opener.opened == [CHAT]
    assert (
        "The page asked to open Chat (chat:…). You allowed it; melo opened it, so "
        "Chat comes to the front."
    ) in out.text
    assert "secret-token" not in out.text  # the address can hold a sign-in token


async def test_a_refused_app_is_not_opened_and_the_step_says_so() -> None:
    opener = Opener()
    out = await clicked(opener, Asks(deny=("handoff:",)))
    assert opener.opened == []
    assert "The page tried to open Chat (chat:…); melo did not open it." in out.text


async def test_an_address_no_app_opens_asks_nobody() -> None:
    opener, asks = Opener(app=""), Asks()
    out = await clicked(opener, asks)
    assert opener.opened == [] and "handoff:chat" not in asks.names
    assert "no app on this Mac opens it, so nothing opened" in out.text


async def test_with_nobody_to_ask_the_app_is_opened_as_the_click_was() -> None:
    opener = Opener()
    out = await clicked(opener, None)
    assert opener.opened == [CHAT]
    assert "It was allowed; melo opened it" in out.text


async def test_the_apps_window_after_melo_opens_it_is_told_on_that_step(
    tmp_path: Path,
) -> None:
    screen = Screen(tmp_path)
    opener = Opener(then=lambda: screen.clock.at(1.0, lambda: screen.show(SIGN_IN)))
    out = await clicked(opener, Asks(), chrome_watch(screen))
    assert opener.opened == [CHAT]
    # The app melo opened is the one the page asked for: its window is told on its own.
    assert (
        "Chat (the app the page asked for) opened a window 'Sign in to example.com' (w1)"
    ) in out.text
    assert "its buttons: 'Allow', 'Cancel'" in out.text


async def test_after_melo_opens_the_app_another_apps_window_still_waits_for_the_card(
    tmp_path: Path,
) -> None:
    """The app melo opened is the one the page asked for; any other app's window is the person's call."""
    screen = Screen(tmp_path)
    other = {
        **SIGN_IN,
        "window_id": 8,
        "pid": 80,
        "app_name": "Single Sign-On",
        "bundle": "com.example.sso",
        "title": "Approve the sign-in",
    }
    opener = Opener(then=lambda: screen.clock.at(1.0, lambda: screen.show(other)))
    asks = Asks(deny=("window-read:",))
    driver = handing_off(world(button("Open in Chat")), {"Open in Chat": CHAT})
    provider, found = await tools(driver, opener, chrome_watch(screen))
    try:
        with asking(asks):
            await run(found, "browser_open", url=HOME)
            out = await run(found, "browser_click", ref="e1")
            await settled(provider)
    finally:
        await provider.close()
    assert opener.opened == [CHAT]
    assert "Single Sign-On" not in out.text and "Approve the sign-in" not in out.text
    assert "window-read:com.example.sso" in asks.names


def test_an_address_of_another_apps_kind_is_told_from_a_page_and_the_browsers_own() -> (
    None
):
    assert apps.custom_scheme("chat://login?x=1")
    assert apps.custom_scheme("chat:open")
    for own in (
        "https://example.com/",
        "about:blank",
        "chrome://settings",
        "chrome-error://chromewebdata/",
        "javascript:alert(1)",
        "data:text/html,x",
        "",
    ):
        assert not apps.custom_scheme(own), own
    assert apps.shown_scheme(CHAT) == "chat:…"


# ── Boxes a page puts up ──


async def test_the_boxes_a_page_put_up_are_told_in_the_page_block_only() -> None:
    boxes = (
        Dialog("confirm", "Delete the draft?", True),
        Dialog("alert", "Saved", True),
        Dialog("prompt", "Your name?", False),
    )
    driver = handing_off(world(button("Delete")), boxes={"Delete": boxes})
    provider, found = await tools(driver, Opener())
    try:
        with asking(Asks()):
            await run(found, "browser_open", url=HOME)
            out = await run(found, "browser_click", ref="e1")
    finally:
        await provider.close()
    head, block = out.text.split("\n", 1)
    assert "Delete the draft" not in head
    assert "the page showed a question: 'Delete the draft?' (melo answered OK)" in block
    assert "the page showed a message: 'Saved' (melo closed it)" in block
    assert (
        "the page showed a box that asks for text: 'Your name?' (melo answered Cancel)"
        in block
    )


async def test_ok_to_a_confirm_is_the_persons_every_time() -> None:
    driver = handing_off(world(button("Delete")))
    provider = BrowserToolProvider(driver, opener=Opener())  # type: ignore[arg-type]
    asks = Asks()
    with asking(asks):
        assert await provider._confirm("confirm", "Delete it?", HOME + "x") is True
    name, arguments, preview = asks.asked[0]
    assert name == "browser-dialog:https://example.com"
    assert arguments == {
        "site": "example.com",
        "access": "dialog",
        "how": "confirm",
        "text": "Delete it?",
    }
    assert never_granted(name)
    with asking(Asks(deny=("browser-dialog:",))):
        assert await provider._confirm("confirm", "Delete it?", HOME) is False


# ── The browser in the background, in words ──


async def test_questions_tools_and_the_note_name_the_background_browser() -> None:
    driver = handing_off(world(button("Go")))
    provider, found = await tools(driver, Opener())
    asks = Asks()
    try:
        with asking(asks):
            await run(found, "browser_open", url=HOME)
            await run(found, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert asks.asked[0][2] == "open example.com in Chrome and read it"
    assert asks.asked[0][1]["browser"] == "chrome"
    assert asks.asked[1][2] == 'click a button "Go" on example.com in Chrome'
    described = found["browser_open"].definition.description
    assert "Chrome, in the background" in described and "Safari" not in described
    assert provider.system_notes()[0] == BACKGROUND_NOTE != PAGE_NOTE
    assert "Browser panel" in BACKGROUND_NOTE and "cannot type" not in BACKGROUND_NOTE
    assert sites.BROWSER.get() == "Safari"  # only while a call runs


# ── End to end over the Chrome driver and the fake Chrome ──


@pytest.fixture
async def chrome_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    peer = Peer(tmp_path)
    monkeypatch.setenv("FAKE_CHROME_LOG", str(peer.path))
    monkeypatch.delenv("FAKE_CHROME_MODE", raising=False)
    driver = ChromeBackgroundDriver(
        command=[sys.executable, FAKE_CHROME], profile=peer.profile, version="156.0"
    )
    opener = Opener()
    provider = BrowserToolProvider(driver, opener=opener)  # type: ignore[arg-type]
    found = {t.definition.name: t for t in await provider.start()}
    try:
        yield found, opener, peer
    finally:
        await provider.close()


async def test_over_chrome_a_hand_off_and_a_confirm_go_to_the_person(
    chrome_tools: Any,
) -> None:
    found, opener, peer = chrome_tools
    asks = Asks()
    with asking(asks):
        page = await run(found, "browser_open", url="https://example.com/app")
        assert '[e1 button "Open in Chat"]' in page.text
        handed = await run(found, "browser_click", ref="e1")
        assert '[e7 button "Delete"]' in handed.text  # the page read again
        deleted = await run(found, "browser_click", ref="e7")
    assert opener.opened == [CHAT]
    assert "melo opened it, so Chat comes to the front" in handed.text
    assert "Nothing on the page changed" not in handed.text
    assert [n for n in asks.names if n.startswith("browser-dialog:")] == [
        "browser-dialog:https://example.com"
    ]
    assert "the page showed a question: 'Delete the draft?' (melo answered OK)" in (
        deleted.text
    )
    assert "It is gone." in deleted.text
    assert peer.calls("Page.handleJavaScriptDialog")[0]["params"] == {"accept": True}
