"""A window another app opens a moment after melo's act: told by the next step, or waited for.

The desktop control helper is the fake MCP server over a JSON screen file, the
app in front is scripted, and the clock is a fake one that moves only when the
watch sleeps (``Screen.clock``), so no test waits in real time and none touches
the real screen.
"""

from __future__ import annotations

import asyncio
import plistlib
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.browser import desktop as desktop_mod
from localmelo.support.browser import provider as provider_mod
from localmelo.support.browser.desktop import (
    ADMIT_S,
    LATE_POLL_S,
    LATE_S,
    DesktopHelper,
    DesktopWatch,
    Front,
    HelperStatus,
    helper_status,
)
from localmelo.support.browser.driver import Changes, Element, PointHit
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDriver, button, forget_helper_status
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_other_windows import (
    FAKE,
    SIGN_IN,
    Screen,
    on_act,
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
            await run(tools, "browser_click", ref="e1")  # opens the deadline
            out = await run(tools, "browser_wait", until="load", seconds=5)
    finally:
        await provider.close()
    assert out.text.startswith("Reached: the page loaded. Another app put a window")
    assert "Chat opened a window" in out.text


# ── Only an act opens admission, and only for a while ──


def window_calls(screen: Screen) -> list[str]:
    return [name for name, _ in screen.calls() if name == "get_window_state"]


async def test_read_only_looks_never_keep_admission_open_for_a_window_of_the_person(
    screen: Screen,
) -> None:
    """The review's case: reads 300 s apart, then a window the person opens at +900 s."""
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")  # hands nothing off
            reads = []
            for _ in range(3):
                screen.clock.now += 300
                reads.append(await run(tools, "browser_read"))
            screen.clock.now += 0  # now 900 s after the act
            screen.show(SIGN_IN, front=False)  # the person opens it themselves
            read = await run(tools, "browser_read")
            shot = await run(tools, "browser_screenshot")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert all("opened a window" not in out.text for out in reads)
    for out in (read, shot):
        assert "opened a window" not in out.text and "Sign in" not in out.text
    assert "Allow" not in read.text  # no button was read out
    assert window.is_error and "No window of another app appeared" in window.text
    assert window_calls(screen) == []  # its state was never read


async def test_a_window_just_after_the_deadline_is_not_taken_for_the_act(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S + 1
            screen.show(SIGN_IN)  # in front, too: the front app is not admitted either
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "Chat" not in read.text and "came to the front" not in read.text
    assert window.is_error and window_calls(screen) == []


async def test_a_window_inside_the_deadline_is_told_by_a_later_step(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S - 4  # a slow model's next step
            screen.show(SIGN_IN, front=False)
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "Chat opened a window 'Sign in to example.com' (w1)" in read.text
    assert not window.is_error and '[button "Allow"]' in window.text


async def test_a_window_seen_late_is_lost_not_told_by_a_later_look(
    screen: Screen,
) -> None:
    """A look after the deadline only renews the baseline: the window is not kept for a later one."""
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S + 1
            screen.show(SIGN_IN, front=False)
            again = await run(tools, "browser_read")
            second = await run(tools, "browser_click", ref=ref_of(again))
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    # The window of the person came after the first deadline: a new act does not
    # bring it back, it was only part of the baseline when that act looked.
    assert not second.is_error
    assert "opened a window" not in read.text and window_calls(screen) == []


async def test_a_new_act_opens_a_new_deadline(screen: Screen) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S * 4
            again = await run(tools, "browser_read")
            second = await run(tools, "browser_click", ref=ref_of(again))
            screen.show(SIGN_IN, front=False)
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert not second.is_error
    assert "Chat opened a window" in read.text


async def test_acts_that_hand_nothing_off_open_no_deadline(screen: Screen) -> None:
    driver = world(Element("textbox", "Name"), Element("select", "Size"))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_type", ref="e1", text="Alex")
            await run(tools, "browser_select", ref="e5", option="Large")
            screen.show(SIGN_IN)
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "opened a window" not in read.text
    assert window.is_error and window_calls(screen) == []


async def test_reads_waits_screenshots_and_finds_do_not_open_a_deadline(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_read")
            await run(tools, "browser_find", query="Quick")
            await run(tools, "browser_screenshot")
            await run(tools, "browser_wait", until="load", seconds=1)
            screen.show(SIGN_IN)
            read = await run(tools, "browser_read")
            waited = await run(tools, "browser_wait", until="load", seconds=1)
    finally:
        await provider.close()
    assert "opened a window" not in read.text + waited.text
    assert window_calls(screen) == []


async def test_a_wait_tells_a_window_inside_the_deadline_and_not_after(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S + 1
            screen.show(SIGN_IN, front=False)
            late = await run(
                tools, "browser_wait", until="text", text="never", seconds=1
            )
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert late.text.startswith("Timed out after 1 s")
    assert "opened a window" not in late.text
    assert window.is_error and window_calls(screen) == []


async def test_a_wait_stops_polling_the_screen_once_the_deadline_has_closed(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_mod, "WATCH_S", 0.01)
    driver = world(button("Quick sign-in"))
    waits = 0
    real = driver.wait

    async def wait(*args: Any, **kwargs: Any) -> bool:
        nonlocal waits
        waits += 1
        screen.clock.now += 10  # each slice of the wait takes ten seconds
        return await real(*args, **kwargs)

    driver.wait = wait  # type: ignore[method-assign]
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            before = len(screen.calls())
            out = await run(
                tools, "browser_wait", until="text", text="never", seconds=0.1
            )
    finally:
        await provider.close()
    assert out.text.startswith("Timed out")
    looks = [
        n for n, a in screen.calls()[before:] if n == "list_windows" and "pid" not in a
    ]
    assert len(looks) <= 2  # the first look and the one after the first slice


# ── A window that comes while the model plans its next step ──


def sampling(provider: BrowserToolProvider) -> bool:
    task = provider._sampler
    return task is not None and not task.done()


async def test_a_window_that_comes_during_the_deadline_is_told_by_a_step_after_it(
    screen: Screen,
) -> None:
    """A slow model: the window comes 5 s after the click, the next step runs at 20 s."""
    desk = watch(screen)
    provider, tools = await opened(world(button("Quick sign-in")), desk)
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            assert sampling(provider)
            screen.clock.now += 5
            screen.show(SIGN_IN, front=False)
            await screen.pacer.tick(provider)  # a sample sees it at 5 s
            screen.clock.now += 15  # the next step runs 20 s after the click
            assert not desk.admitting(provider._desk_mark)
            read = await run(tools, "browser_read")
            again = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "Another app put a window on the screen since your last step" in read.text
    assert "Chat opened a window 'Sign in to example.com' (w1)" in read.text
    assert read.text.count("opened a window") == 1
    assert "opened a window" not in again.text  # told once
    # It was admitted when it was first seen: it is still there to read.
    assert not window.is_error and '[button "Allow"]' in window.text
    assert len(window_calls(screen)) == 2  # the buttons when seen, then the read


async def test_a_window_that_comes_after_the_deadline_is_not_sampled(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S + 1  # 16 s after the click
            screen.show(SIGN_IN, front=False)
            await screen.pacer.tick(provider)
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read")
            ended = not sampling(provider)
    finally:
        await provider.close()
    assert ended
    assert "opened a window" not in read.text
    assert window.is_error and window_calls(screen) == []


async def test_a_wait_after_the_deadline_tells_what_a_sample_admitted_inside_it(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += 3
            screen.show(SIGN_IN, front=False)
            await screen.pacer.tick(provider)
            screen.clock.now += ADMIT_S  # the wait starts after the deadline
            out = await run(
                tools, "browser_wait", until="text", text="never", seconds=30
            )
    finally:
        await provider.close()
    assert out.text.startswith("Stopped waiting: another app put a window")
    assert "Chat opened a window" in out.text


async def test_a_step_inside_the_deadline_tells_a_sampled_window_once(
    screen: Screen,
) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += 2
            screen.show(SIGN_IN, front=False)
            await screen.pacer.tick(provider)
            first = await run(tools, "browser_read")
            second = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert first.text.count("(w1)") == 1 and "opened a window" not in second.text


async def test_the_sampler_stops_when_the_deadline_closes(screen: Screen) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += 6
            await screen.pacer.tick(provider)
            still = sampling(provider)  # a sample at 6 s: it goes on
            screen.clock.now += ADMIT_S
            await screen.pacer.tick(provider)  # the next one finds the deadline closed
            ended = not sampling(provider)
    finally:
        await provider.close()
    assert still and ended


async def test_the_sampler_stops_when_the_browser_closes(screen: Screen) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    with asking(Asks()):
        await run(tools, "browser_click", ref="e1")
    task = provider._sampler
    assert task is not None and not task.done()
    await provider.close()
    assert task.done() and provider._sampler is None


async def test_the_sampler_stops_when_a_new_chat_starts(screen: Screen) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += 2
            screen.show(SIGN_IN, front=False)
            await screen.pacer.tick(provider)  # queued, not yet told
            task = provider._sampler
            provider.forget()
            await asyncio.sleep(0)
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert task is not None and task.done()
    assert "opened a window" not in read.text  # the old chat's window is not told


async def test_a_new_act_replaces_the_sampler_with_its_own(screen: Screen) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            first = provider._sampler
            reading = await run(tools, "browser_read")
            await run(tools, "browser_click", ref=ref_of(reading))
            second = provider._sampler
            await asyncio.sleep(0.05)
            live = [t for t in (first, second) if t is not None and not t.done()]
    finally:
        await provider.close()
    assert first is not None and second is not None and first is not second
    assert live == [second]  # one sampler at a time


async def test_there_is_no_sampler_without_an_act_that_can_hand_off(
    screen: Screen,
) -> None:
    driver = world(Element("textbox", "Name"), Element("select", "Size"))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_read")
            await run(tools, "browser_find", query="Name")
            await run(tools, "browser_screenshot")
            await run(tools, "browser_wait", until="load", seconds=1)
            await run(tools, "browser_type", ref="e1", text="Alex")
            await run(tools, "browser_select", ref="e5", option="Large")
            seen = provider._sampler
    finally:
        await provider.close()
    assert seen is None and screen.pacer.calls == 0


async def test_a_sample_is_cut_at_the_late_waits_limit(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_mod, "LATE_S", 0.3)
    slow = False

    async def front() -> Front | None:
        if slow:
            await asyncio.sleep(30)
        return screen.front

    desk = watch(screen)
    desk._front = front
    provider, tools = await opened(world(button("Quick sign-in")), desk)
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            slow = True
            began = time.monotonic()
            await screen.pacer.tick(provider)  # the sample hangs until it is cut
            took = time.monotonic() - began
            held = list(provider._held)
    finally:
        await provider.close()
    assert took < 3.0 and held == []


async def test_a_sample_is_not_a_steps_own_look(screen: Screen) -> None:
    provider, tools = await opened(world(button("Quick sign-in")), watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            provider._looked = False
            await provider._look(sampling=True)
            sampled = provider._looked
            await provider._look()
            looked = provider._looked
    finally:
        await provider.close()
    assert sampled is False and looked is True


async def test_the_sampler_runs_on_its_own_beside_the_steps_in_real_time(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No hand-paced clock: a short deadline, a window that comes in it, a step after it."""
    monkeypatch.setattr(desktop_mod, "ADMIT_S", 1.0)
    monkeypatch.setattr(provider_mod, "SAMPLE_S", 0.05)
    monkeypatch.setattr(provider_mod, "LATE_S", 0.2)
    command = [sys.executable, "-I", FAKE, str(screen.path), str(screen.log_path)]
    desk = DesktopWatch(DesktopHelper(command), front=screen.in_front)
    provider, tools = await opened(world(button("Quick sign-in")), desk)
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            await asyncio.sleep(0.3)
            screen.show(SIGN_IN, front=False)
            await asyncio.sleep(1.5)  # the model plans; the deadline closes meanwhile
            ended = not sampling(provider)
            read = await run(tools, "browser_read")
            again = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert ended
    assert read.text.count("Chat opened a window 'Sign in to example.com' (w1)") == 1
    assert "opened a window" not in again.text


async def test_without_the_helper_the_sampler_watches_the_front_app(
    screen: Screen,
) -> None:
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, helper=False)
    )
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += 5
            screen.front = CHAT
            await screen.pacer.tick(provider)
            screen.clock.now += 15
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert "- Chat came to the front" in read.text


# ── The app the page asked for is a hint, not a filter ──


SSO = {
    **SIGN_IN,
    "window_id": 8,
    "pid": 80,
    "app_name": "SSO Agent",
    "bundle": "com.example.sso",
    "title": "Approve sign-in",
    "z_index": 12,  # above the chat app's window
}


async def test_a_window_of_another_app_is_admitted_after_the_one_the_page_asked_for(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    driver.lands["Quick sign-in"] = "chat://authorize?site=example.com"
    desk = watch(screen, asked=("chat", "com.example.chat"))
    provider, tools = await opened(driver, desk)
    try:
        with asking(Asks()):

            def both() -> None:
                screen.show(SSO)
                screen.show(SIGN_IN)

            screen.clock.at(1.0, both)
            out = await run(tools, "browser_click", ref="e1")
            other = await run(tools, "window_read", window="w2")
    finally:
        await provider.close()
    lines = [line for line in out.text.splitlines() if "opened a window" in line]
    assert len(lines) == 2
    # The app the page asked for first and named so, though the other window is on top.
    assert lines[0].startswith(
        "- Chat (the app the page asked for) opened a window 'Sign in to example.com' (w1)"
    )
    assert lines[1].startswith("- SSO Agent opened a window 'Approve sign-in' (w2)")
    assert "asked for" not in lines[1]
    assert not other.is_error  # admitted: its state is read and it has an id


async def test_the_asked_for_app_is_only_a_hint_when_it_is_not_known(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    driver.lands["Quick sign-in"] = "chat://authorize?site=example.com"
    provider, tools = await opened(driver, watch(screen))  # no app known
    try:
        with asking(Asks()):
            screen.clock.at(1.0, lambda: screen.show(SSO))
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert "SSO Agent opened a window 'Approve sign-in' (w1)" in out.text
    assert "asked for" not in out.text


async def test_a_scheme_handler_that_fails_is_just_not_known(screen: Screen) -> None:
    async def broken(url: str) -> frozenset[str]:
        raise OSError("no")

    desk = watch(screen)
    desk._handler = broken  # type: ignore[assignment]
    assert await desk.scheme_app("chat://x") == frozenset()


# ── The late wait is a wall-clock cap ──


async def test_the_late_wait_is_cut_at_its_limit_when_a_call_is_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A front-app lookup that hangs does not hold the act past the late wait."""
    slow = False

    async def front() -> Front | None:
        if slow:
            await asyncio.sleep(30)
        return Front("Safari", "com.apple.Safari", 10)

    desk = DesktopWatch(None, front=front)  # the real clock and sleep
    mark = desk.admit(await desk.mark())
    slow = True
    began = time.monotonic()
    seen, fresh = await desk.look(mark, wait_s=0.3)
    took = time.monotonic() - began
    assert seen == [] and fresh is mark  # nothing learned, the mark as it was
    assert 0.2 <= took < 2.0


async def test_the_late_wait_counts_the_calls_it_makes(screen: Screen) -> None:
    """Time a call takes comes off the wait: three seconds of calls and sleeps in all."""
    spent = 0.0

    async def front() -> Front | None:
        nonlocal spent
        screen.clock.now += 1.0  # each look at the front app takes a second
        spent += 1.0
        return Front("Safari", "com.apple.Safari", 10)

    desk = DesktopWatch(None, front=front, clock=screen.clock, sleep=screen.clock.sleep)
    mark = desk.admit(await desk.mark())
    start = screen.clock.now
    await desk.look(mark, wait_s=LATE_S)
    # The first look takes a second, then half a second of sleep and a look, and so on:
    # the wait ends once three seconds are gone, within one call of the limit.
    assert 3.0 <= screen.clock.now - start <= 3.0 + 1.0


async def test_a_slow_helper_does_not_hold_the_act_past_the_late_wait(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_mod, "LATE_S", 0.4)
    slow = False

    async def front() -> Front | None:
        if slow:
            await asyncio.sleep(30)
        return screen.front

    desk = watch(screen)
    desk._front = front
    driver = world(button("Quick sign-in"))

    def hang() -> None:
        nonlocal slow
        slow = True

    on_act(driver, hang)
    provider, tools = await opened(driver, desk)
    try:
        with asking(Asks()):
            began = time.monotonic()
            out = await run(tools, "browser_click", ref="e1")
            took = time.monotonic() - began
    finally:
        await provider.close()
    assert not out.is_error and took < 5.0


# ── Enter and submit hand off too ──


async def test_a_send_with_enter_waits_for_a_late_window(screen: Screen) -> None:
    driver = world(Element("textbox", "Name", submit=HOME))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            screen.clock.at(1.0, lambda: screen.show(SIGN_IN))
            out = await run(tools, "browser_type", ref="e1", text="Alex", submit=True)
    finally:
        await provider.close()
    assert not out.is_error
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text
    assert sum(screen.clock.slept) == pytest.approx(1.0)


async def test_a_form_button_waits_for_a_late_window(screen: Screen) -> None:
    driver = world(Element("button", "Send", submit=HOME))
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            screen.clock.at(1.5, lambda: screen.show(SIGN_IN))
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert not out.is_error
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text
    assert sum(screen.clock.slept) == pytest.approx(1.5)


# ── The app the address opens ──


def test_an_apps_names_are_its_file_name_and_its_plist(tmp_path: Path) -> None:
    app = tmp_path / "Chat Pro.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleName": "Chat",
                "CFBundleDisplayName": "Chat Pro",
                "CFBundleExecutable": "chatd",
                "CFBundleIdentifier": "com.example.Chat",
                "Other": 3,
            }
        )
    )
    assert desktop_mod.app_names(str(app)) == {
        "chat pro",
        "chat",
        "chatd",
        "com.example.chat",
    }
    # An unreadable plist leaves the file name; a path that is no app, nothing.
    broken = tmp_path / "Broken.app"
    broken.mkdir()
    assert desktop_mod.app_names(str(broken)) == {"broken"}
    assert desktop_mod.app_names("/usr/bin/true") == frozenset()
    assert desktop_mod.app_names("") == frozenset()


async def test_the_app_that_opens_a_scheme_is_asked_of_the_system_by_scheme_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "Chat.app"
    app.mkdir()
    log = tmp_path / "args.txt"
    script = tmp_path / "osascript"
    script.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > {log}\nprintf "%s\\n" "{app}"\n'
    )
    script.chmod(0o755)
    monkeypatch.setattr(desktop_mod, "OSASCRIPT", str(script))
    names = await desktop_mod.scheme_apps("chat://authorize?token=secret-value")
    assert names == {"chat"}
    asked = log.read_text()
    assert "secret-value" not in asked and "authorize" not in asked
    assert asked.rstrip().endswith("chat")  # the scheme is the argument


async def test_a_scheme_that_cannot_be_asked_or_answered_is_not_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "osascript"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    monkeypatch.setattr(desktop_mod, "OSASCRIPT", str(script))
    monkeypatch.setattr(desktop_mod, "SCHEME_S", 0.2)
    assert await desktop_mod.scheme_apps("chat://x") == frozenset()  # timed out
    monkeypatch.setattr(desktop_mod, "OSASCRIPT", str(tmp_path / "missing"))
    assert await desktop_mod.scheme_apps("chat://x") == frozenset()  # cannot start
    for url in ("", "not a url", "chat-app://x", "http s://x", "9chat://x"):
        assert await desktop_mod.scheme_apps(url) == frozenset()


def test_without_a_mark_nothing_is_admitted_yet() -> None:
    desk = DesktopWatch(None)
    assert desk.admit(None) is None and desk.admitting(None) is False


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


async def test_without_the_helper_an_app_in_front_after_the_deadline_is_not_named(
    screen: Screen,
) -> None:
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, helper=False)
    )
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S + 1
            screen.front = CHAT  # the person switched apps themselves
            read = await run(tools, "browser_read")
            again = await run(tools, "browser_read")
            await run(tools, "browser_click", ref=ref_of(again))
            after_act = await run(tools, "browser_read")
    finally:
        await provider.close()
    # The switch is absorbed into the baseline: not told now, nor by a later act.
    assert "came to the front" not in read.text + again.text + after_act.text


async def test_without_the_helper_reads_never_keep_the_front_app_admitted(
    screen: Screen,
) -> None:
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, helper=False)
    )
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            for _ in range(3):
                screen.clock.now += 300
                await run(tools, "browser_read")
            screen.front = CHAT
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert "came to the front" not in read.text


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
        bad,
        front=screen.in_front,
        clock=screen.clock,
        sleep=screen.clock.sleep,
        pace=screen.pacer,
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
    forget_helper_status()
    monkeypatch.setenv(desktop_mod.HELPER_ENV, "")
    assert (await helper_status(check=True)).state == "missing"
    command = f"{sys.executable} -I {FAKE} {screen.path} {screen.log_path}"
    monkeypatch.setenv(desktop_mod.HELPER_ENV, command)
    assert await helper_status() == HelperStatus("installed")  # not looked at yet
    granted = await helper_status(check=True)
    assert granted == HelperStatus("granted", "Cua Driver", True)
    forget_helper_status()
    screen.set(grants={"accessibility": True, "screen_recording": False})
    assert (await helper_status(check=True)).pictures is False
    forget_helper_status()
    screen.set(grants={"accessibility": False, "screen_recording": True})
    refused = await helper_status(check=True)
    assert refused.state == "no_grant" and refused.app == "Cua Driver"
    assert "no Accessibility grant" in refused.problem
    # Not looked at again within a few seconds, whatever the person did meanwhile.
    screen.set(grants={"accessibility": True, "screen_recording": True})
    assert (await helper_status(check=True)).state == "no_grant"
    forget_helper_status()
    monkeypatch.setenv(desktop_mod.HELPER_ENV, "/nonexistent/helper mcp")
    broken = await helper_status(check=True)
    assert broken.state == "failed" and "did not start" in broken.problem
    forget_helper_status()


async def test_a_chats_first_use_of_the_helper_is_the_newest_status(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    forget_helper_status()
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
    forget_helper_status()
