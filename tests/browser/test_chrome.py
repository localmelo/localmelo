"""The ``chrome`` driver over a fake Chrome on the pipe: launch, read, act, stale refs, boxes, hand-offs.

The fake (``fake_chrome_peer.py``) is started the way melo starts Chrome and
speaks DevTools on fds 3 and 4; no real Chrome runs here. Each test has a
profile of its own under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser import chrome, pagescript
from localmelo.support.browser.chrome import (
    ChromeBackgroundDriver,
    chrome_installed,
    chrome_version,
    launch_args,
    user_agent,
)
from localmelo.support.browser.driver import (
    Act,
    BrowserError,
    BrowserSetupError,
    CoveredError,
    Dialog,
    Element,
    RawPage,
    StaleElementError,
    WindowBusyError,
)

FAKE = str(Path(__file__).with_name("fake_chrome_peer.py"))
FORM = "https://example.com/form"
APP = "https://example.com/app"
COVER = "https://example.com/cover"


class Peer:
    """Where a fake Chrome writes what it was asked."""

    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "chrome.log"
        self.profile = str(tmp_path / "profile")

    def lines(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def calls(self, method: str) -> list[dict[str, Any]]:
        return [line for line in self.lines() if line.get("method") == method]

    def argv(self) -> list[str]:
        return next(line["argv"] for line in self.lines() if "argv" in line)


@pytest.fixture
def peer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Peer:
    found = Peer(tmp_path)
    monkeypatch.setenv("FAKE_CHROME_LOG", str(found.path))
    monkeypatch.delenv("FAKE_CHROME_MODE", raising=False)
    return found


def make(peer: Peer, **kwargs: Any) -> ChromeBackgroundDriver:
    kwargs.setdefault("version", "156.0.8078.80")
    return ChromeBackgroundDriver(
        command=[sys.executable, FAKE], profile=peer.profile, **kwargs
    )


@pytest.fixture
async def driver(peer: Peer) -> AsyncIterator[ChromeBackgroundDriver]:
    found = make(peer)
    await found.start()
    try:
        yield found
    finally:
        await found.close()


def named(raw: RawPage, label: str) -> Element:
    return next(e for e in raw.elements if e.label == label)


async def opened(driver: ChromeBackgroundDriver, url: str) -> RawPage:
    await driver.open(url, new_tab=True)
    return await driver.page(None)


# ── Launch ──


async def test_chrome_starts_headless_on_the_pipe_with_melos_own_private_profile(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    await driver.open("https://example.com/", new_tab=True)
    argv = peer.argv()
    assert "--headless=new" in argv and "--remote-debugging-pipe" in argv
    assert not [a for a in argv if a.startswith("--remote-debugging-port")]
    assert "--use-mock-keychain" in argv and "--disable-gpu" in argv
    assert "--disable-blink-features=AutomationControlled" in argv
    assert f"--user-data-dir={peer.profile}" in argv
    agent = next(a for a in argv if a.startswith("--user-agent="))
    assert "Chrome/156.0.0.0" in agent and "Headless" not in agent
    assert argv[-1] == "about:blank"
    assert stat.S_IMODE(os.stat(peer.profile).st_mode) == 0o700
    # A page's download is saved by id in melo's own downloads folder (#547).
    assert peer.calls("Browser.setDownloadBehavior")[0]["params"] == {
        "behavior": "allowAndName",
        "downloadPath": str(Path(peer.profile).parent / "downloads"),
        "eventsEnabled": True,
    }
    assert (
        stat.S_IMODE(os.stat(Path(peer.profile).parent / "downloads").st_mode) == 0o700
    )
    created = peer.calls("Target.createTarget")[0]["params"]
    assert created["newWindow"] is True and created["background"] is True
    # The page Chrome started with is nobody's: it goes once melo has its own.
    assert peer.calls("Target.closeTarget")[0]["params"] == {"targetId": "T1"}
    assert [t.id for t in await driver.tabs()] == ["T2"]


def test_the_user_agent_is_the_windowed_chromes_and_the_version_is_the_apps(
    tmp_path: Path,
) -> None:
    assert user_agent("156.0.8078.80") == (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
    )
    assert user_agent("") == "" and user_agent("beta") == ""
    app = tmp_path / "Google Chrome.app" / "Contents"
    (app / "MacOS").mkdir(parents=True)
    binary = app / "MacOS" / "Google Chrome"
    binary.write_text("")
    import plistlib

    (app / "Info.plist").write_bytes(
        plistlib.dumps({"CFBundleShortVersionString": "157.0.1.2"})
    )
    assert chrome_version(str(binary)) == "157.0.1.2"
    assert chrome_version(str(tmp_path / "nowhere")) == ""
    args = launch_args("/p", "", gpu=True)
    assert "--disable-gpu" not in args
    assert not [a for a in args if a.startswith("--user-agent")]


async def test_without_chrome_the_setup_says_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(chrome.CHROME_ENV, raising=False)
    monkeypatch.setattr(chrome, "CHROME_APPS", (str(tmp_path / "No Chrome.app"),))
    assert chrome.chrome_command() is None and not chrome_installed()
    driver = ChromeBackgroundDriver(profile=str(tmp_path / "p"))
    with pytest.raises(BrowserSetupError, match="Google Chrome is not installed"):
        await driver.start()
    monkeypatch.setenv(chrome.CHROME_ENV, f"{sys.executable} {FAKE}")
    assert chrome_installed()


async def test_a_chrome_that_never_answers_or_quits_is_an_error(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_MODE", "silent")
    slow = make(peer, startup_timeout_s=0.5)
    with pytest.raises(BrowserError, match="did not answer within 0.5 s"):
        await slow.start()
    monkeypatch.setenv("FAKE_CHROME_MODE", "exit")
    gone = make(peer)
    with pytest.raises(BrowserError, match="Chrome stopped"):
        await gone.start()


async def test_one_chrome_serves_every_driver_of_a_profile_and_each_sees_its_own_tabs(
    peer: Peer,
) -> None:
    first, second = make(peer), make(peer)
    await first.start()
    await second.start()
    try:
        assert first._browser is second._browser
        a = await first.open("https://example.com/", new_tab=True)
        b = await second.open("https://example.com/news", new_tab=True)
        assert [t.id for t in await first.tabs()] == [a.id]
        assert [t.id for t in await second.tabs()] == [b.id]
        with pytest.raises(BrowserError, match="no longer has that tab"):
            await first.page(b.id)
        assert len([line for line in peer.lines() if "argv" in line]) == 1
        pid = first._browser.pid if first._browser else 0
        await first.close()
        os.kill(pid, 0)  # still running for the other chat
    finally:
        await second.close()
    assert peer.calls("Browser.close")


# ── A window for the person (take over) ──


def starts(peer: Peer) -> list[list[str]]:
    """The command line of every Chrome the test started, in order."""
    return [line["argv"] for line in peer.lines() if "argv" in line]


async def test_every_start_keeps_the_last_session_in_melos_private_profile(
    peer: Peer,
) -> None:
    os.makedirs(peer.profile, mode=0o700)
    with open(os.path.join(peer.profile, "fake-session.json"), "w") as handle:
        json.dump({"urls": ["https://example.com/news"], "n": 7}, handle)
    driver = make(peer)
    await driver.start()
    try:
        assert "--restore-last-session" in peer.argv()
        prefs = Path(peer.profile) / "Default" / "Preferences"
        saved = json.loads(prefs.read_text())
        assert saved["session"]["restore_on_startup"] == 1
        assert saved["profile"]["exit_type"] == "Normal"
        assert stat.S_IMODE(prefs.stat().st_mode) == 0o600
        # The page the session brought back is nobody's: it goes at once.
        await driver.open("https://example.com/", new_tab=True)
        closed = [c["params"]["targetId"] for c in peer.calls("Target.closeTarget")]
        assert "T8" in closed
    finally:
        await driver.close()


async def test_taking_over_opens_a_window_on_the_same_pages_and_done_goes_back(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    mine = await driver.open("https://example.com/", new_tab=True)
    other = await driver.open("https://example.com/news", new_tab=True)
    closed: list[str] = []
    assert await driver.show_window(mine.id, closed.append) is True
    window = starts(peer)[-1]
    assert "--headless=new" not in window and "about:blank" not in window
    assert {(t.id, t.url) for t in await driver.tabs()} == {
        (mine.id, "https://example.com/"),
        (other.id, "https://example.com/news"),
    }, "the same ids on the restored pages"
    bounds = [c["params"]["bounds"] for c in peer.calls("Browser.setWindowBounds")]
    assert {"windowState": "minimized"} in bounds and {
        "windowState": "normal"
    } in bounds
    assert {"left": 80, "top": 60, "width": 1280, "height": 800} in bounds
    assert peer.calls("Page.bringToFront")
    await driver.hide_window()
    assert "--headless=new" in starts(peer)[-1] and len(starts(peer)) == 3
    assert {t.id for t in await driver.tabs()} == {mine.id, other.id}
    assert (await driver.page(mine.id)).title == "Example Home"
    assert closed == [], "melo's own restart is no Done"


async def test_closing_the_window_is_done_and_the_page_comes_back_in_the_background(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_PERSON_CLOSES", "0.2")
    driver = make(peer)
    await driver.start()
    try:
        tab = await driver.open("https://example.com/news", new_tab=True)
        done: list[str] = []
        assert await driver.show_window(tab.id, done.append) is True
        await until(lambda: done == ["closed"])
        await driver.hide_window()
        assert [(t.id, t.url) for t in await driver.tabs()] == [
            (tab.id, "https://example.com/news")
        ], "opened again at its last address, under the same id"
    finally:
        await driver.close()


async def test_a_window_that_cannot_open_leaves_melo_in_the_background(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_NO_WINDOW", "1")
    driver = make(peer)
    await driver.start()
    try:
        tab = await driver.open("https://example.com/", new_tab=True)
        assert await driver.show_window(tab.id, lambda how: None) is False
        assert "--headless=new" in starts(peer)[-1]
        assert [t.id for t in await driver.tabs()] == [tab.id]
    finally:
        await driver.close()


async def test_while_the_window_is_open_another_chat_waits_and_keeps_its_tab(
    peer: Peer,
) -> None:
    first, second = make(peer), make(peer)
    await first.start()
    await second.start()
    try:
        a = await first.open("https://example.com/", new_tab=True)
        b = await second.open("https://example.com/news", new_tab=True)
        assert await first.show_window(a.id, lambda how: None) is True
        with pytest.raises(WindowBusyError, match="open in a window for the person"):
            await second.tabs()
        await first.hide_window()
        assert [t.id for t in await second.tabs()] == [b.id]
    finally:
        await first.close()
        await second.close()


async def until(condition: Any, seconds: float = 5.0) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.02)


async def test_a_call_during_the_restart_waits_for_it_and_keeps_its_tabs(
    peer: Peer,
) -> None:
    first, second = make(peer), make(peer)
    await first.start()
    await second.start()
    try:
        a = await first.open("https://example.com/", new_tab=True)
        b = await second.open("https://example.com/news", new_tab=True)
        old = first._browser
        assert old is not None
        showing = asyncio.ensure_future(first.show_window(a.id, lambda how: None))
        await until(lambda: old.closed)
        assert old.successor is None, "inside the restart"
        with pytest.raises(WindowBusyError):  # it waited, then met the person's window
            await second.tabs()
        assert await showing is True
        window = first._browser
        assert window is not None and window.windowed and window.owner is first
        assert a.id in [t.id for t in await first.tabs()]
        await first.hide_window()
        assert [t.id for t in await second.tabs()] == [b.id]
    finally:
        await first.close()
        await second.close()


async def test_a_step_during_the_restart_reaches_its_tab_on_the_new_chrome(
    peer: Peer,
) -> None:
    """What the old Chrome called the tab and its session does not outlive it."""
    driver = make(peer)
    await driver.start()
    try:
        a = await driver.open("https://example.com/news", new_tab=True)
        b = await driver.open("https://example.com/news", new_tab=True)
        old = driver._browser
        assert old is not None
        stale = driver._resolve(b.id)
        del driver._sessions[stale]  # a page melo has not spoken to in this Chrome
        showing = asyncio.ensure_future(driver.show_window(a.id, lambda how: None))
        await until(lambda: old.closed)
        assert old.successor is None, "inside the restart"
        back, attached = await asyncio.gather(
            driver.back(a.id), driver._attach(stale), return_exceptions=True
        )
        assert "no earlier page" in str(back), "the new Chrome answered, not an error"
        asked = peer.calls("Page.getNavigationHistory")[-1]
        assert asked["session"] == driver._sessions[driver._resolve(a.id)]
        assert attached == driver._sessions[driver._resolve(b.id)]
        await showing
    finally:
        await driver.hide_window()
        await driver.close()


async def test_another_chat_ending_leaves_the_persons_window(peer: Peer) -> None:
    from localmelo.support.browser.provider import BrowserToolProvider

    first, second = make(peer), make(peer)
    await first.start()
    await second.start()
    other = BrowserToolProvider(second)  # type: ignore[arg-type]
    try:
        a = await first.open("https://example.com/", new_tab=True)
        await second.open("https://example.com/news", new_tab=True)
        told: list[str] = []
        assert await first.show_window(a.id, told.append) is True
        await other.close()  # chat B ends: deleted, evicted to make room, rebuilt
        first._follow()
        window = first._browser
        assert window is not None and window.windowed and not window.closed
        assert told == [] and "--headless=new" not in starts(peer)[-1]
        await first.hide_window()
    finally:
        await first.close()
        await second.close()


async def test_a_page_that_closes_itself_is_no_done_and_does_not_come_back(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_PAGE_CLOSES", "0.2")
    driver = make(peer)
    await driver.start()
    try:
        tab = await driver.open("https://example.com/news", new_tab=True)
        told: list[str] = []
        assert await driver.show_window(tab.id, told.append) is True
        assert peer.calls("Runtime.addBinding"), "the page says it closes itself"
        await until(lambda: told == ["gone"])
        await driver.hide_window()
        assert tab.id not in [t.id for t in await driver.tabs()]
    finally:
        await driver.close()


async def test_a_sign_in_window_that_closes_itself_ends_the_step_with_how_it_ended(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.melo.agent._pause import Hold
    from localmelo.melo.contracts.pause import PAUSE
    from localmelo.support.browser.provider import BrowserToolProvider
    from localmelo.tests.browser.test_acting import Asks, asking
    from localmelo.tests.browser.test_provider import run

    monkeypatch.setenv("FAKE_CHROME_PAGE_CLOSES", "0.2")
    provider = BrowserToolProvider(make(peer))  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    said: list[Any] = []
    token = PAUSE.set(Hold(lambda w, d: None, lambda w, h, s: said.append(h)))
    try:
        with asking(Asks()):  # the page needs the person: the card, then the window
            out = await asyncio.wait_for(
                run(tools, "browser_open", url="https://example.com/"), 10
            )
        assert said == ["gone"] and "closed its own window" in out.text
    finally:
        PAUSE.reset(token)
        await provider.close()


@pytest.mark.parametrize(("code", "how"), [(0, "closed"), (1, "lost")])
async def test_a_chrome_that_ends_is_the_persons_quit_only_when_it_quit_cleanly(
    peer: Peer, monkeypatch: pytest.MonkeyPatch, code: int, how: str
) -> None:
    monkeypatch.setenv("FAKE_CHROME_QUITS", f"1.0:{code}")
    driver = make(peer)
    await driver.start()
    try:
        tab = await driver.open("https://example.com/news", new_tab=True)
        told: list[str] = []
        assert await driver.show_window(tab.id, told.append) is True
        await until(lambda: told == [how])
        monkeypatch.delenv("FAKE_CHROME_QUITS")
        await driver.hide_window()
        assert [t.id for t in await driver.tabs()] == [tab.id]
    finally:
        await driver.close()


async def test_a_second_chats_take_while_a_window_is_open_is_its_page_in_the_app(
    peer: Peer,
) -> None:
    from localmelo.melo.agent._pause import Hold
    from localmelo.melo.contracts.pause import PAUSE
    from localmelo.support.browser.provider import BrowserToolProvider
    from localmelo.tests.browser.test_acting import Asks, asking
    from localmelo.tests.browser.test_provider import run

    first, second = make(peer), make(peer)
    other = BrowserToolProvider(second)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await other.start()}
    said: list[Any] = []
    hold = Hold(lambda w, d: said.append(d), lambda w, h, s: said.append(h))
    token = PAUSE.set(hold)
    try:
        await first.start()
        with asking(Asks()):
            await run(tools, "browser_open", url="https://example.com/news")
        a = await first.open("https://example.com/", new_tab=True)
        assert await first.show_window(a.id, lambda how: None) is True
        other.owners.take("t1")
        await until(lambda: bool(said))
        await other._switched()
        where = other.owners.where("t1")
        assert where is not None and (where.place, where.failed, where.busy) == (
            "app",
            True,
            True,
        )
        assert said == [
            {
                "tab": "t1",
                "place": "app",
                "site": "example.com",
                "failed": True,
                "busy": True,
            }
        ]
    finally:
        PAUSE.reset(token)
        other.hand_back()
        await first.hide_window()
        await other.close()
        await first.close()


async def test_a_take_with_the_panel_open_keeps_the_window_and_every_tab(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel's viewer looks at the tabs while Chrome starts again for the window."""
    from localmelo.melo.agent._pause import Hold
    from localmelo.melo.contracts.pause import PAUSE
    from localmelo.support.browser.provider import BrowserToolProvider
    from localmelo.tests.browser.test_acting import Asks, asking
    from localmelo.tests.browser.test_provider import run

    monkeypatch.setenv("FAKE_CHROME_FRAMES", "100000")
    driver = make(peer)
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    said: list[Any] = []
    hold = Hold(lambda w, d: said.append(d), lambda w, h, s: said.append(h))
    token = PAUSE.set(hold)
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url="https://example.com/news")
        live = provider.live
        assert live is not None
        viewer = await live.watch()
        await until(lambda: viewer.meta.get("state") == "live")
        provider.owners.take("t1")
        await until(lambda: bool(said))
        await provider._switched()
        where = provider.owners.where("t1")
        assert where is not None and (where.place, where.failed) == ("window", False)
        window = driver._browser
        assert window is not None and window.windowed and window.owner is driver
        raw = provider.live_raw("t1")
        assert raw in [t.id for t in await driver.tabs()]
        assert viewer.meta.get("state") != "closed"
        provider.hand_back()
        await provider._switched()
        assert driver._browser is not None and not driver._browser.windowed
    finally:
        PAUSE.reset(token)
        await provider.close()


async def test_a_page_taken_beside_another_chats_window_lives_in_the_panel(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app is really where the page is: pictures come, input goes, the window stays."""
    from localmelo.melo.agent._pause import Hold
    from localmelo.melo.contracts.pause import PAUSE
    from localmelo.support.browser.provider import BrowserToolProvider
    from localmelo.tests.browser.test_acting import Asks, asking
    from localmelo.tests.browser.test_provider import run

    monkeypatch.setenv("FAKE_CHROME_FRAMES", "100000")
    first, second = make(peer), make(peer)
    other = BrowserToolProvider(second)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await other.start()}
    said: list[Any] = []
    hold = Hold(lambda w, d: said.append(d), lambda w, h, s: said.append(h))
    token = PAUSE.set(hold)
    try:
        await first.start()
        with asking(Asks()):
            await run(tools, "browser_open", url="https://example.com/news")
        live = other.live
        assert live is not None
        viewer = await live.watch()
        await until(lambda: viewer.meta.get("state") == "live")
        a = await first.open("https://example.com/", new_tab=True)
        assert await first.show_window(a.id, lambda how: None) is True
        window = first._browser
        assert await live.take(viewer) is True
        await until(lambda: bool(said))
        await other._switched()
        await until(lambda: viewer.meta.get("state") == "live")
        assert viewer.meta["owner"] == "you" and viewer.meta["place"] == "app"
        await until(lambda: viewer.meta.get("w") is not None)
        click = {
            "type": "mouse",
            "action": "down",
            "x": 10,
            "y": 10,
            "button": "left",
            "buttons": 1,
        }
        assert live.send_input(viewer, click) is True
        await until(lambda: bool(peer.calls("Input.dispatchMouseEvent")))
        assert first._browser is window and window is not None
        assert window.windowed and window.owner is first
    finally:
        PAUSE.reset(token)
        other.hand_back()
        await first.hide_window()
        await other.close()
        await first.close()


async def test_a_window_clears_no_form_on_another_page_unless_the_person_chooses(
    peer: Peer,
) -> None:
    """Chrome starts again for a window on every page: taking a plain one waits for a choice."""
    from localmelo.melo.agent._pause import Hold
    from localmelo.melo.contracts.pause import PAUSE
    from localmelo.support.browser.provider import BrowserToolProvider
    from localmelo.tests.browser.test_acting import Asks, asking
    from localmelo.tests.browser.test_provider import run

    driver = make(peer)
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    hold = Hold(lambda w, d: None, lambda w, h, s: None)
    token = PAUSE.set(hold)
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url="https://example.com/filled")
            await run(tools, "browser_open", url="https://example.com/news", tab="new")
        before = driver._browser
        assert before is not None
        provider.owners.take("t1")
        await until(lambda: hold.held)
        await provider._switched()
        provider.owners.take("t2")
        await provider._switched()
        where = provider.owners.where("t2")
        assert where is not None and (where.place, where.form) == ("app", True)
        assert driver._browser is before and not before.closing, "nothing reloaded"
        assert provider.show_to_person() is True, "the person's choice covers it"
        await provider._switched()
        assert driver._browser is not before and driver._browser.windowed
    finally:
        PAUSE.reset(token)
        await provider.close()


def test_preferences_save_no_password_and_never_write_through_a_planted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "chrome"
    folder = profile / "Default"
    folder.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    os.link(outside, folder / f"Preferences.melo-{os.getpid()}")  # the old fixed name
    chrome.keep_sessions(str(profile))
    assert outside.read_text() == "keep me"
    saved = json.loads((folder / "Preferences").read_text())
    assert saved["credentials_enable_service"] is False
    assert saved["profile"] == {
        "exit_type": "Normal",
        "password_manager_enabled": False,
    }
    assert saved["autofill"] == {"profile_enabled": False, "credit_card_enabled": False}
    (folder / "Preferences").unlink()

    def full(*args: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(chrome.os, "replace", full)
    with pytest.raises(OSError):
        chrome.keep_sessions(str(profile))
    left = sorted(p.name for p in folder.iterdir())
    assert left == [f"Preferences.melo-{os.getpid()}"], "no temporary file is left"


async def test_a_form_on_any_page_is_told_and_no_chats_current_tab_moves(
    peer: Peer,
) -> None:
    first, second = make(peer), make(peer)
    await first.start()
    await second.start()
    try:
        home = await first.open("https://example.com/", new_tab=True)
        assert await first.unsent_input() is False
        await second.open("https://example.com/filled", new_tab=True)
        here = first._current
        assert await first.unsent_input() is True, "another chat's page reloads too"
        assert first._current == here == first._resolve(home.id)
    finally:
        await first.close()
        await second.close()


# ── Reading ──


async def test_a_page_is_read_in_melos_isolated_world_and_acts_in_the_page(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, FORM)
    assert raw.url == FORM and raw.title == "Form" and raw.doc == "doc-1"
    assert [e.label for e in raw.elements] == [
        "Check in",
        "Nickname",
        "Country",
        "Hidden away",
    ]
    reads = [
        c
        for c in peer.calls("Runtime.evaluate")
        if c["script"].startswith("(function () {")
    ]
    assert reads and all(c["params"].get("contextId") == 42 for c in reads)
    await driver.act(None, Act("click", raw.url, named(raw, "Check in"), doc=raw.doc))
    acts = [
        c
        for c in peer.calls("Runtime.evaluate")
        if c["script"].startswith("(function (spec)")
    ]
    assert acts and all("contextId" not in c["params"] for c in acts)


async def test_tabs_back_wait_blank_and_a_picture(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    tab = await driver.open("https://example.com/news", new_tab=True)
    again = await driver.open("https://example.com/", new_tab=False, tab=tab.id)
    assert again.id == tab.id and again.title == "Example Home"
    back = await driver.back(tab.id)
    assert back.url == "https://example.com/news"
    with pytest.raises(BrowserError, match="no earlier page"):
        await driver.back(tab.id)
    assert await driver.wait(tab.id, "text", "news", 1) is True
    assert await driver.wait(tab.id, "load", "", 1) is True
    # One look at the same condition, with no waiting.
    assert await driver.holds(tab.id, "load", "") is True
    assert await driver.holds(tab.id, "text", "news") is True
    assert await driver.holds(tab.id, "text", "nowhere") is False
    assert await driver.holds(tab.id, "gone", "nowhere") is True
    assert await driver.holds(tab.id, "gone", "news") is False
    shot = await driver.screenshot(tab.id)
    assert shot.mime_type == "image/jpeg"
    assert base64.b64decode(shot.data).startswith(b"\xff\xd8")
    assert shot.view is not None and (shot.view.width, shot.view.ratio) == (1280, 1)
    capture = peer.calls("Page.captureScreenshot")[0]["params"]
    assert capture == {"format": "jpeg", "quality": 70}
    await driver.blank(tab.id)
    assert await driver.location(tab.id) == "about:blank"
    await driver.close_tab(tab.id)
    assert await driver.tabs() == []
    assert await driver.location(tab.id) == ""


async def test_a_wait_goes_on_while_the_page_does_not_answer_but_not_once_chrome_is_gone(
    driver: ChromeBackgroundDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = await driver.open(FORM, new_tab=True)
    monkeypatch.setattr(pagescript, "POLL_S", 0.0)
    replies: list[object] = [BrowserError("between two documents"), True]

    async def read(source: str) -> object:
        reply = replies.pop(0)
        if isinstance(reply, BrowserError):
            raise reply
        return reply

    monkeypatch.setattr(driver, "_read", read)
    assert await driver.wait(tab.id, "load", "", 5) is True
    # One look gives a page that does not answer as not holding.
    replies[:] = [BrowserError("between two documents")]
    assert await driver.holds(tab.id, "load", "") is False
    assert driver._browser is not None
    monkeypatch.setattr(driver._browser, "closed", True)
    replies[:] = [BrowserError("gone")]
    with pytest.raises(BrowserError, match="gone"):
        await driver.wait(tab.id, "load", "", 5)


async def test_an_act_in_a_box_from_another_site_is_refused_without_naming_safari(
    driver: ChromeBackgroundDriver,
) -> None:
    await driver.open(FORM, new_tab=True)
    box = Element("button", "Allow", frame="x7")
    with pytest.raises(BrowserError, match="cannot click there") as refused:
        await driver.act(None, Act("click", FORM, box))
    assert "Safari" not in str(refused.value)


async def test_a_page_that_does_not_load_says_why(
    driver: ChromeBackgroundDriver, monkeypatch: pytest.MonkeyPatch, peer: Peer
) -> None:
    await driver.close()
    monkeypatch.setenv("FAKE_CHROME_MODE", "nav_error")
    failing = make(peer)
    await failing.start()
    try:
        with pytest.raises(BrowserError, match="ERR_NAME_NOT_RESOLVED"):
            await failing.open("https://nowhere.example/", new_tab=True)
    finally:
        await failing.close()


async def test_without_an_isolated_world_nothing_is_read(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_MODE", "no_world")
    driver = make(peer)
    await driver.start()
    try:
        await driver.open(FORM, new_tab=True)
        with pytest.raises(BrowserError, match="Cannot create isolated world"):
            await driver.page(None)
    finally:
        await driver.close()


# ── Acting ──


async def test_a_click_is_chromes_own_trusted_click_at_the_elements_middle(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, FORM)
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Check in"), doc=raw.doc)
    )
    assert result.via == "browser" and result.why == ""
    mouse = [c["params"] for c in peer.calls("Input.dispatchMouseEvent")]
    assert [m["type"] for m in mouse] == ["mouseMoved", "mousePressed", "mouseReleased"]
    assert all((m["x"], m["y"]) == (100, 50) for m in mouse)
    assert await driver.location(None) == "https://example.com/done"


async def test_typing_clicks_the_box_selects_all_then_puts_the_text_in(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, FORM)
    box = named(raw, "Nickname")
    result = await driver.act(
        None, Act("type", raw.url, box, text="Alex 你好", enter=True, doc=raw.doc)
    )
    assert result.via == "browser"
    keys = [c["params"] for c in peer.calls("Input.dispatchKeyEvent")]
    assert keys[0]["commands"] == ["selectAll"] and keys[0]["type"] == "rawKeyDown"
    assert [c["params"]["text"] for c in peer.calls("Input.insertText")] == [
        "Alex 你好"
    ]
    assert any(k.get("key") == "Enter" and k["type"] == "keyDown" for k in keys)
    assert peer.lines()[-1]["typed"]["Nickname"] == "Alex 你好\n"


async def test_a_choice_and_a_covered_element_are_the_page_scripts(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, FORM)
    chose = await driver.act(
        None, Act("select", raw.url, named(raw, "Country"), text="Spain", doc=raw.doc)
    )
    assert (chose.via, chose.why) == ("page-script", "")
    covered = await driver.act(
        None, Act("click", raw.url, named(raw, "Hidden away"), doc=raw.doc)
    )
    assert covered.via == "page-script" and "covers it" in covered.why
    assert not peer.calls("Input.dispatchMouseEvent")


def clicks(peer: Peer) -> list[tuple[float, float]]:
    """Where the trusted mouse went, once per press."""
    return [
        (c["params"]["x"], c["params"]["y"])
        for c in peer.calls("Input.dispatchMouseEvent")
        if c["params"]["type"] == "mousePressed"
    ]


def scrolls(peer: Peer) -> int:
    return len([line for line in peer.lines() if line.get("act_mode") == "scroll"])


async def test_a_target_whose_middle_is_covered_is_clicked_at_a_part_that_is_clear(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, COVER)
    done = await driver.act(
        None, Act("click", raw.url, named(raw, "Half covered"), doc=raw.doc)
    )
    assert (done.via, done.why) == ("browser", "")
    assert clicks(peer) == [(130, 400)]
    assert await driver.location(None) == "https://example.com/done"


async def test_a_covered_target_is_scrolled_to_the_middle_once_more_before_it_is_given_up(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, COVER)
    done = await driver.act(
        None, Act("click", raw.url, named(raw, "Clears later"), doc=raw.doc)
    )
    assert (done.via, done.why) == ("browser", "")
    assert scrolls(peer) == 2 and len(clicks(peer)) == 1
    assert await driver.location(None) == "https://example.com/done"


async def test_a_target_that_stays_covered_goes_to_the_page_script_and_names_what_covers_it(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, COVER)
    done = await driver.act(
        None, Act("click", raw.url, named(raw, "Under the header"), doc=raw.doc)
    )
    assert done.via == "page-script"
    assert "a sticky header (h1) covers it" in done.why
    assert scrolls(peer) == 2 and not clicks(peer)


async def test_a_click_that_lands_on_another_element_is_tried_once_more(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, COVER)
    done = await driver.act(
        None, Act("click", raw.url, named(raw, "Slips away"), doc=raw.doc)
    )
    assert (done.via, done.why) == ("browser", "")
    assert len(clicks(peer)) == 2 and scrolls(peer) == 2
    assert await driver.location(None) == "https://example.com/done"


async def test_a_click_that_lands_on_another_element_twice_is_stopped_and_names_it(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, COVER)
    with pytest.raises(CoveredError) as stopped:
        await driver.act(
            None, Act("click", raw.url, named(raw, "Always slips"), doc=raw.doc)
        )
    said = str(stopped.value)
    assert said.startswith("An ad (a frame from another site) covers that button")
    assert "Nothing was clicked or typed" in said
    assert len(clicks(peer)) == 2
    assert await driver.location(None) == COVER


async def test_input_that_reaches_nothing_falls_to_the_page_script_and_says_so(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CHROME_MODE", "native_missed")
    driver = make(peer)
    await driver.start()
    try:
        raw = await opened(driver, FORM)
        result = await driver.act(
            None, Act("click", raw.url, named(raw, "Check in"), doc=raw.doc)
        )
        assert result.via == "page-script"
        assert "did not reach it" in result.why
        assert await driver.location(None) == "https://example.com/done"
    finally:
        await driver.close()


async def test_a_ref_of_another_page_or_document_or_a_moved_element_is_stale(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, FORM)
    item = named(raw, "Check in")
    with pytest.raises(StaleElementError) as page:
        await driver.act(None, Act("click", "https://example.com/", item, doc=raw.doc))
    assert page.value.why == "page"
    with pytest.raises(StaleElementError) as doc:
        await driver.act(None, Act("click", raw.url, item, doc="doc-0"))
    assert doc.value.why == "doc"
    moved = Element("button", "Check in", at=item.at, fp="button|elsewhere")
    with pytest.raises(StaleElementError) as gone:
        await driver.act(None, Act("click", raw.url, moved, doc=raw.doc))
    assert gone.value.why == "gone"
    assert not peer.calls("Input.dispatchMouseEvent")
    assert await driver.location(None) == FORM


async def test_a_point_is_clicked_where_the_picture_says(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, FORM)
    shot = await driver.screenshot(None)
    assert shot.view is not None
    await driver.act(
        None,
        Act(
            "click",
            raw.url,
            named(raw, "Check in"),
            doc=raw.doc,
            point=(100, 50),
            view=shot.view,
        ),
    )
    pressed = peer.calls("Input.dispatchMouseEvent")[1]["params"]
    assert (pressed["x"], pressed["y"]) == (100, 50)


async def test_a_trusted_only_button_takes_chromes_click(
    driver: ChromeBackgroundDriver,
) -> None:
    raw = await opened(driver, APP)
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Daily"), doc=raw.doc)
    )
    assert result.via == "browser"
    assert result.changes is not None and result.changes.texts == (
        "Checked in: day 12",
    )


# ── Another app's address, boxes, new tabs ──


async def test_an_address_of_another_apps_kind_is_put_on_the_result_and_not_followed(
    driver: ChromeBackgroundDriver,
) -> None:
    raw = await opened(driver, APP)
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Open in Chat"), doc=raw.doc)
    )
    assert result.handoff == "chat://login?token=secret-token"
    assert await driver.location(None) == APP
    again = await driver.act(
        None, Act("click", raw.url, named(raw, "Daily"), doc=raw.doc)
    )
    assert again.handoff == ""  # taken once, by the act it came with


async def test_an_address_asked_for_before_an_act_began_is_told_late_never_as_its_own(
    driver: ChromeBackgroundDriver,
) -> None:
    raw = await opened(driver, APP)
    session = await driver._session()
    (await driver._live()).handoffs[session] = ["chat://asked-while-no-act-ran"]
    # An act that fails leaves it for the next one.
    with pytest.raises(StaleElementError):
        await driver.act(None, Act("click", raw.url, named(raw, "Daily"), doc="other"))
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Open in Chat"), doc=raw.doc)
    )
    assert result.late == ("chat://asked-while-no-act-ran",)
    assert result.handoff == "chat://login?token=secret-token", "its own is its own"
    again = await driver.act(
        None, Act("click", raw.url, named(raw, "Daily"), doc=raw.doc)
    )
    assert (again.late, again.handoff) == ((), ""), "told once"


@pytest.mark.parametrize("answer", [True, False])
async def test_a_confirm_during_an_act_is_put_to_the_person(
    driver: ChromeBackgroundDriver, peer: Peer, answer: bool
) -> None:
    asked: list[tuple[str, str, str]] = []

    async def person(kind: str, message: str, url: str) -> bool:
        asked.append((kind, message, url))
        return answer

    driver.on_dialog = person
    raw = await opened(driver, APP)
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Delete"), doc=raw.doc)
    )
    assert asked == [("confirm", "Delete the draft?", APP)]
    assert result.dialogs == (Dialog("confirm", "Delete the draft?", answer),)
    assert result.dialogs[0].url == APP, "the page that put it up, as the browser says"
    answered = peer.calls("Page.handleJavaScriptDialog")[0]["params"]
    assert answered == {"accept": answer}
    landed = await driver.location(None)
    assert landed == ("https://example.com/deleted" if answer else APP)


async def test_an_alert_is_closed_with_ok_without_asking(
    driver: ChromeBackgroundDriver,
) -> None:
    asked: list[str] = []

    async def person(kind: str, message: str, url: str) -> bool:
        asked.append(kind)
        return False

    driver.on_dialog = person
    raw = await opened(driver, APP)
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Hello"), doc=raw.doc)
    )
    assert asked == [] and result.dialogs == (Dialog("alert", "Hi there", True),)


async def test_a_confirm_with_nobody_to_ask_is_answered_cancel(
    driver: ChromeBackgroundDriver,
) -> None:
    raw = await opened(driver, APP)
    result = await driver.act(
        None, Act("click", raw.url, named(raw, "Delete"), doc=raw.doc)
    )
    assert result.dialogs == (Dialog("confirm", "Delete the draft?", False),)
    assert await driver.location(None) == APP


async def test_a_tab_a_page_opens_is_melos_and_another_chats_never_is(
    driver: ChromeBackgroundDriver, peer: Peer
) -> None:
    raw = await opened(driver, APP)
    before = {t.id for t in await driver.tabs()}
    await driver.act(None, Act("click", raw.url, named(raw, "More"), doc=raw.doc))
    after = {t.id: t for t in await driver.tabs()}
    new = [t for t in after if t not in before]
    assert len(new) == 1 and after[new[0]].url == "https://example.com/news"
    read = await driver.page(new[0])
    assert read.title == "News"


class Observer:
    """What ``_look`` asks of a driver: the page script's report of the changes it saw, by the time."""

    def __init__(self, changes_at: float | None, quiet_after: float | None) -> None:
        self.began = time.monotonic()
        self.changes_at, self.quiet_after = changes_at, quiet_after

    async def _eval(self, source: str) -> dict[str, Any]:
        spent = time.monotonic() - self.began
        if self.changes_at is None or spent < self.changes_at:
            return {"count": 0, "quiet": spent * 1000}
        if self.quiet_after is None:  # steady background traffic: never quiet
            return {"count": 5, "quiet": 0}
        return {"count": 1, "quiet": max(spent - self.changes_at, 0) * 1000}

    async def _finish(self, key: str, doc: str) -> dict[str, Any]:
        return {"count": 1}


async def test_a_page_that_never_goes_quiet_is_left_soon_after_its_first_change_but_a_late_one_still_shows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in (
        ("LOOK_STEP_S", 0.01),
        ("QUIET_S", 0.05),
        ("LOOK_CAP_S", 0.5),
        ("LOOK_BUSY_S", 0.05),
    ):
        monkeypatch.setattr(pagescript, name, value)
    act = Act("click", FORM, Element("button", "Go"))

    async def looked(changes_at: float | None, quiet_after: float | None) -> Any:
        found = Observer(changes_at, quiet_after)
        began = time.monotonic()
        report, quiet = await pagescript.PageScriptActs._look(found, "k", "d", act)  # type: ignore[arg-type]
        return quiet, time.monotonic() - began

    # Steady traffic: it ends a moment after the first change, not at the cap.
    quiet, spent = await looked(0.0, None)
    assert quiet is False and spent < 0.25
    # A change that comes late, then quiet: waited for, and reported quiet.
    quiet, spent = await looked(0.2, 0.05)
    assert quiet is True and 0.2 <= spent < 0.45
    # Nothing at all: the page is left at the cap.
    quiet, spent = await looked(None, None)
    assert quiet is True and spent >= 0.5
