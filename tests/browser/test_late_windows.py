"""A window another app opens a moment after melo's act: told by the next step, or waited for.

The desktop control helper is the fake MCP server over a JSON screen file, the
app in front is scripted, and the clock is a fake one that moves only when the
watch sleeps (``Screen.clock``), so no test waits in real time and none touches
the real screen.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser import desktop as desktop_mod
from localmelo.support.browser.desktop import (
    LATE_POLL_S,
    LATE_S,
    MARK_TTL_S,
    DesktopHelper,
    DesktopWatch,
    Front,
    HelperStatus,
    helper_status,
)
from localmelo.support.browser.driver import Changes, Element, PointHit
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDriver, button
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_other_windows import (
    FAKE,
    SIGN_IN,
    Screen,
    run,
    tools_with,
    watch,
)

CHAT = Front("Chat", "com.example.chat", 70)


@pytest.fixture
def screen(tmp_path: Path) -> Screen:
    return Screen(tmp_path)


def ref_of(out: Any) -> str:
    """The ref of the page's one button, as a reading names it."""
    found = re.search(r"\[(e\d+) button", out.text)
    assert found is not None
    return found.group(1)


async def opened(
    driver: FakeDriver, desk: DesktopWatch | None
) -> tuple[BrowserToolProvider, dict[str, Any]]:
    provider, tools = await tools_with(driver, desk)
    with asking(Asks()):
        await run(tools, "browser_open", url=HOME)
    return provider, tools


# ── A window that comes after the act's own look ──


async def test_a_window_that_comes_after_the_acts_look_is_told_by_the_next_step(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            click = await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN)  # two seconds after the click's look is over
            screen.clock.now += 2
            read = await run(tools, "browser_read")
            again = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "Another app" not in click.text and "opened a window" not in click.text
    assert "Another app put a window on the screen since your last step" in read.text
    assert (
        "- Chat opened a window 'Sign in to example.com' (w1); its buttons: "
        "'Allow', 'Cancel'. window_read w1 shows it; window_press presses a button."
    ) in read.text
    assert "Groceries" not in read.text  # the person's own windows are never named
    assert "opened a window" not in again.text  # told once
    assert not window.is_error and '[button "Allow"]' in window.text


@pytest.mark.parametrize(
    "tool, args",
    [
        ("browser_screenshot", {}),
        ("browser_find", {"query": "Quick"}),
        ("browser_tabs", {"action": "list"}),
        ("browser_wait", {"until": "load", "seconds": 1}),
    ],
)
async def test_every_browser_step_tells_a_window_that_came_since_the_last_look(
    screen: Screen, tool: str, args: dict[str, Any]
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN)
            out = await run(tools, tool, **args)
    finally:
        await provider.close()
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text


async def test_a_window_told_once_is_not_told_again_by_any_later_step(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            screen.clock.at(1.0, lambda: screen.show(SIGN_IN))
            click = await run(tools, "browser_click", ref="e1")
            texts = [
                (await run(tools, "browser_read")).text,
                (await run(tools, "browser_tabs", action="list")).text,
                (await run(tools, "browser_wait", until="load", seconds=1)).text,
            ]
    finally:
        await provider.close()
    assert click.text.count("opened a window") == 1
    assert all("opened a window" not in text for text in texts)
    assert all("(w1)" not in text for text in texts)


# ── A window that comes during the late wait ──


async def test_a_window_that_comes_within_the_late_wait_is_told_by_the_click_itself(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            screen.clock.at(1.2, lambda: screen.show(SIGN_IN))
            out = await run(tools, "browser_click", ref="e1")
            after = await run(tools, "browser_read")
    finally:
        await provider.close()
    first = out.text.splitlines()[0]
    assert "Another app put a window on the screen: see what changed." in first
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text
    assert "opened a window" not in after.text
    # It stopped waiting when the window came (polls of half a second).
    assert sum(screen.clock.slept) == pytest.approx(1.5)
    assert max(screen.clock.slept) <= LATE_POLL_S


async def test_the_late_wait_ends_at_its_limit_when_nothing_comes(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert "Another app" not in out.text
    assert sum(screen.clock.slept) == pytest.approx(LATE_S)
    assert all(0 < pause <= LATE_POLL_S for pause in screen.clock.slept)
    assert len(screen.clock.slept) == round(LATE_S / LATE_POLL_S)


async def test_a_point_click_and_a_send_wait_for_a_late_window_too(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            screen.clock.at(2.0, lambda: screen.show(SIGN_IN))
            await run(tools, "browser_screenshot")
            driver.hits[(500, 250)] = PointHit(
                "control", Element("button", "Quick sign-in")
            )
            out = await run(tools, "browser_click_point", x=500, y=500)
    finally:
        await provider.close()
    assert not out.is_error
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text
    assert 1.5 <= sum(screen.clock.slept) <= 2.5


async def test_acts_that_cannot_hand_off_do_not_wait(screen: Screen) -> None:
    driver = world(
        Element("textbox", "Name"),
        Element("select", "Size"),
        button("Quick sign-in"),
    )
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            typed = await run(tools, "browser_type", ref="e1", text="Alex")
            chosen = await run(tools, "browser_select", ref="e5", option="Large")
    finally:
        await provider.close()
    assert not typed.is_error and not chosen.is_error
    assert screen.clock.slept == []


async def test_a_click_that_changed_the_page_does_not_wait(screen: Screen) -> None:
    driver = world(button("Quick sign-in"))
    driver.changes["Quick sign-in"] = Changes(texts=("Opening...",), changed=True)
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            out = await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN)  # it comes after the click: the next step tells it
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert not out.is_error and screen.clock.slept == []
    assert "Chat opened a window" in read.text


async def test_the_wait_stops_when_the_page_changes_by_itself(screen: Screen) -> None:
    driver = world(button("Quick sign-in"))
    provider, tools = await opened(driver, watch(screen))
    real = driver.location
    moved = ""

    async def location(tab: str | None) -> str:
        # The page sends the browser on one second after the click.
        if screen.clock.now >= 1001.0 and moved:
            return moved
        return await real(tab)

    driver.location = location  # type: ignore[method-assign]
    try:
        with asking(Asks()):
            moved = HOME + "elsewhere"
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert not out.is_error
    # One poll (half a second) passed before the page was seen to move on.
    assert sum(screen.clock.slept) <= 1.0


async def test_the_address_of_an_app_hands_the_act_off_and_the_window_is_waited_for(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    driver.lands["Quick sign-in"] = "chat://authorize?site=example.com"
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            screen.clock.at(1.0, lambda: screen.show(SIGN_IN))
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert out.is_error and "not an http or https address" in out.text
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text
    assert sum(screen.clock.slept) == pytest.approx(1.0)


# ── The wait step ──


async def test_a_window_that_came_before_the_wait_is_told_at_once(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    waits = 0
    real = driver.wait

    async def wait(*args: Any, **kwargs: Any) -> bool:
        nonlocal waits
        waits += 1
        return await real(*args, **kwargs)

    driver.wait = wait  # type: ignore[method-assign]
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN)
            out = await run(
                tools, "browser_wait", until="text", text="never", seconds=30
            )
    finally:
        await provider.close()
    assert out.text.startswith("Stopped waiting: another app put a window")
    assert "Chat opened a window" in out.text
    assert waits == 0  # it did not wait at all


async def test_a_wait_that_ends_in_success_still_tells_a_window_that_came_meanwhile(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    real = driver.wait

    async def wait(*args: Any, **kwargs: Any) -> bool:
        screen.show(SIGN_IN)
        return await real(*args, **kwargs)

    driver.wait = wait  # type: ignore[method-assign]
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            out = await run(tools, "browser_wait", until="load", seconds=5)
    finally:
        await provider.close()
    assert out.text.startswith("Reached: the page loaded. Another app put a window")
    assert "Chat opened a window" in out.text


# ── An old mark is not trusted ──


async def test_a_window_long_after_the_last_look_is_not_taken_for_the_acts(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN, front=False)  # the person opens it themselves
            screen.clock.now += MARK_TTL_S + 1
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "opened a window" not in read.text
    assert window.is_error and "No window of another app appeared" in window.text


# ── Without the helper: the app in front ──


async def test_without_the_helper_an_app_that_comes_to_the_front_late_is_named(
    screen: Screen,
) -> None:
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, helper=False)
    )
    try:
        with asking(Asks()):
            click = await run(tools, "browser_click", ref="e1")
            screen.front = CHAT
            read = await run(tools, "browser_read")
            again = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert "came to the front" not in click.text
    assert (
        "- Chat came to the front: it may show a window melo cannot see. Tell the "
        "person to answer it in Chat."
    ) in read.text
    assert "came to the front" not in again.text


async def test_without_the_helper_the_late_wait_names_the_app_that_comes_forward(
    screen: Screen,
) -> None:
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, helper=False)
    )
    try:
        with asking(Asks()):
            screen.clock.at(1.0, lambda: setattr(screen, "front", CHAT))
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert "- Chat came to the front" in out.text
    assert sum(screen.clock.slept) == pytest.approx(1.0)


# ── A helper that cannot run is said once per chat ──

NO_ACCESS = (
    "melo can't read other apps' windows: the desktop control helper needs "
    "Accessibility (System Settings → Privacy & Security → Accessibility)."
)


async def test_a_helper_without_accessibility_is_said_once_per_chat_and_names_its_app(
    screen: Screen,
) -> None:
    screen.set(grants={"accessibility": False, "screen_recording": True})
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            first = await run(tools, "browser_click", ref="e1")
            read = await run(tools, "browser_read")
            provider.forget()  # a new chat
            screen.set(grants={"accessibility": True, "screen_recording": True})
            await run(tools, "browser_open", url=HOME)
            fixed = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert f"- {NO_ACCESS} Turn on Cua Driver in that list." in first.text
    assert first.text.count("can't read other apps' windows") == 1
    assert "can't read other apps' windows" not in read.text
    # The next chat looks at the grant again: it was turned on meanwhile.
    assert "can't read other apps' windows" not in fixed.text


async def test_the_line_comes_again_in_the_next_chat_while_the_grant_is_missing(
    screen: Screen,
) -> None:
    screen.set(grants={"accessibility": False})
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            first = await run(tools, "browser_click", ref="e1")
            provider.forget()
            again = await run(tools, "browser_open", url=HOME)
            second = await run(tools, "browser_click", ref=ref_of(again))
    finally:
        await provider.close()
    assert NO_ACCESS in first.text and NO_ACCESS in second.text


async def test_a_helper_that_is_not_installed_says_nothing_extra(
    screen: Screen,
) -> None:
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, helper=False)
    )
    try:
        with asking(Asks()):
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert "can't read other apps' windows" not in out.text


async def test_a_helper_that_does_not_start_is_said_with_its_reason(
    screen: Screen,
) -> None:
    bad = DesktopHelper(["/nonexistent/helper-program", "mcp"])
    desk = DesktopWatch(
        bad, front=screen.in_front, clock=screen.clock, sleep=screen.clock.sleep
    )
    provider, tools = await opened(world(button("Quick sign-in")), desk)
    try:
        with asking(Asks()):
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert (
        "- melo can't read other apps' windows: the desktop control helper did not "
        "start ("
    ) in out.text


async def test_the_app_to_give_the_grant_is_the_one_the_helper_names(
    screen: Screen,
) -> None:
    daemon = {"attribution": "driver-daemon", "bundle_id": "com.trycua.driver"}
    host = {
        "attribution": "host",
        "executable": "/Applications/Host.app/Contents/MacOS/Host",
    }
    names = {}
    for label, source in (("none", None), ("daemon", daemon), ("host", host)):
        grants: dict[str, Any] = {"accessibility": False}
        if source is not None:
            grants["source"] = source
        screen.set(grants=grants)
        helper = DesktopHelper(
            [sys.executable, "-I", FAKE, str(screen.path), str(screen.log_path)]
        )
        assert await helper.ready() is False
        names[label] = helper.app
        await helper.close()
    # `cua-driver mcp` runs the helper's own daemon: its grants are the daemon's,
    # whatever app started the proxy, so the daemon is the default.
    assert names == {"none": "Cua Driver", "daemon": "Cua Driver", "host": "Host"}


# ── The helper's status, for the settings page ──


async def test_the_helper_status_says_missing_installed_granted_or_what_is_wrong(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop_mod.forget_status()
    monkeypatch.setenv(desktop_mod.HELPER_ENV, "")
    assert (await helper_status(check=True)).state == "missing"
    command = f"{sys.executable} -I {FAKE} {screen.path} {screen.log_path}"
    monkeypatch.setenv(desktop_mod.HELPER_ENV, command)
    assert await helper_status() == HelperStatus("installed")  # not looked at yet
    granted = await helper_status(check=True)
    assert granted == HelperStatus("granted", "Cua Driver", True)
    desktop_mod.forget_status()
    screen.set(grants={"accessibility": True, "screen_recording": False})
    assert (await helper_status(check=True)).pictures is False
    desktop_mod.forget_status()
    screen.set(grants={"accessibility": False, "screen_recording": True})
    refused = await helper_status(check=True)
    assert refused.state == "no_grant" and refused.app == "Cua Driver"
    assert "no Accessibility grant" in refused.problem
    # Not looked at again within a few seconds, whatever the person did meanwhile.
    screen.set(grants={"accessibility": True, "screen_recording": True})
    assert (await helper_status(check=True)).state == "no_grant"
    desktop_mod.forget_status()
    monkeypatch.setenv(desktop_mod.HELPER_ENV, "/nonexistent/helper mcp")
    broken = await helper_status(check=True)
    assert broken.state == "failed" and "did not start" in broken.problem
    desktop_mod.forget_status()


async def test_a_chats_first_use_of_the_helper_is_the_newest_status(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop_mod.forget_status()
    monkeypatch.setenv(
        desktop_mod.HELPER_ENV,
        f"{sys.executable} -I {FAKE} {screen.path} {screen.log_path}",
    )
    screen.set(grants={"accessibility": False})
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    # Not asked for a check, yet it knows what the chat found.
    assert (await helper_status()).state == "no_grant"
    desktop_mod.forget_status()
