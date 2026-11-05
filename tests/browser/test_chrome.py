"""The ``chrome`` driver over a fake Chrome on the pipe: launch, read, act, stale refs, boxes, hand-offs.

The fake (``fake_chrome_peer.py``) is started the way melo starts Chrome and
speaks DevTools on fds 3 and 4; no real Chrome runs here. Each test has a
profile of its own under ``tmp_path``.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
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
    assert peer.calls("Browser.setDownloadBehavior")[0]["params"] == {
        "behavior": "deny"
    }
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
    assert said.startswith("An ad (iframe from ads.example.net) covers that button")
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
        assert result.via == "page-script" and result.native_failed is True
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
