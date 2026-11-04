"""Another app's panel on any layer, the person's own windows on another Space, and buttons an app draws itself.

A sign-in app can answer melo's click with a panel above every other window
(layer 23 here) without coming to the front, and with a hidden main-layer
window beside it. The person's own windows show up when they switch Spaces or
un-hide an app. The fake desktop helper lists windows as the real one does:
every Space, on the screen or not, layer 0 of every app and every layer of one
process. Nothing here touches the real screen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.support.browser.desktop import ADMIT_S, Front
from localmelo.tests.browser.fakes import button
from localmelo.tests.browser.test_acting import Asks, asking, world
from localmelo.tests.browser.test_late_windows import opened, window_calls
from localmelo.tests.browser.test_other_windows import Screen, on_act, run, watch
from localmelo.tests.browser.test_window_cards import (
    ALLOW_ONCE,
    DENY,
    Person,
    settled,
    told,
)

CHAT = "window-read:com.example.chat"
# The chat app's main window, open on every Space.
CHAT_MAIN = {
    "window_id": 4,
    "pid": 70,
    "app_name": "Chat",
    "bundle": "com.example.chat",
    "title": "Chat",
    "z_index": 3,
}
# Its authorization panel: above every other window, on every Space, and drawn by the
# app itself (its accessibility tree is the window alone).
PANEL = {
    "window_id": 41,
    "pid": 70,
    "app_name": "Chat",
    "bundle": "com.example.chat",
    "title": "Allow example.com?",
    "z_index": 30,
    "layer": 23,
    "elements": [{"element_index": 0, "role": "AXWindow"}],
}
# The main-layer window the app makes with it, never shown.
COMPANION = {
    "window_id": 40,
    "pid": 70,
    "app_name": "Chat",
    "bundle": "com.example.chat",
    "title": "",
    "z_index": 0,
    "on_screen": False,
    "spaces": [],
}
# The person's own windows on Space 2: a terminal and its floating inspector, which
# the window server numbered after every window listed on Space 1.
TERMINAL = {
    "window_id": 3,
    "pid": 30,
    "app_name": "Terminal",
    "title": "build",
    "z_index": 4,
    "spaces": [2],
}
INSPECTOR = {
    "window_id": 60,
    "pid": 30,
    "app_name": "Terminal",
    "title": "Inspector",
    "z_index": 5,
    "layer": 3,
    "spaces": [2],
}
POINT = {"x": 700, "y": 850}


@pytest.fixture
def screen(tmp_path: Path) -> Screen:
    return Screen(tmp_path)


def change(screen: Screen, window_id: int, **values: Any) -> None:
    """Change one window of the screen (shown, un-hidden ...)."""
    screen.state = screen.load()
    for window in screen.state["windows"]:
        if window["window_id"] == window_id:
            window.update(values)
    screen.save()


def showing(screen: Screen, *windows: dict[str, Any]) -> Callable[[], None]:
    """What another app does right after the act: it shows *windows*, none in front."""

    def show() -> None:
        for window in windows:
            screen.show(window, front=False)

    return show


def cards_for(person: Person) -> list[str]:
    """The titles of the windows the person was asked about."""
    return [str(card.arguments.get("window")) for card in person.cards]


def clicks(screen: Screen) -> list[dict[str, Any]]:
    """The clicks at a point the helper was asked for, in order."""
    return [args for name, args in screen.calls() if name == "click" and "x" in args]


# ── A panel above the main layer, of an app that does not come to the front ──


@pytest.mark.parametrize("how", ["with a hidden window", "beside its open window"])
async def test_a_layer_23_panel_of_an_app_not_in_front_asks_the_person(
    screen: Screen, how: str
) -> None:
    if how == "beside its open window":
        screen.show(CHAT_MAIN, front=False)
        made = [PANEL]
    else:
        made = [COMPANION, PANEL]
    person = Person(ALLOW_ONCE)
    driver = world(button("Quick sign-in"))
    on_act(driver, showing(screen, *made))
    provider, tools = await opened(driver, watch(screen, strict=True))
    try:
        with asking(person):
            click = await run(tools, "browser_click", ref="e1")
            await settled(provider)
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read", window="w1")
    finally:
        await provider.close()
    assert screen.front is not None and screen.front.name == "Safari"  # never in front
    assert [card.tool_name for card in person.cards] == [CHAT]
    [preview] = [p for name, _, p in person.asked if name == CHAT]
    assert preview == (
        'Chat opened a window "Allow example.com?" right after melo\'s click: '
        "let melo see it?"
    )
    assert (
        "Chat opened a window 'Allow example.com?' (w1); it names no buttons. "
        "window_read w1 shows its picture; window_press with x and y presses a "
        "point of it."
    ) in told(click, read)
    assert not window.is_error and window.images
    assert "This window names no buttons: it draws its own." in window.text


async def test_nothing_about_the_panel_reaches_the_model_before_an_allow(
    screen: Screen,
) -> None:
    person = Person(DENY)
    driver = world(button("Quick sign-in"))
    on_act(driver, showing(screen, COMPANION, PANEL))
    provider, tools = await opened(driver, watch(screen, strict=True))
    try:
        with asking(person):
            click = await run(tools, "browser_click", ref="e1")
            await settled(provider)
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read")
    finally:
        await provider.close()
    assert [card.tool_name for card in person.cards] == [CHAT]
    text = told(click, read)
    assert "Allow example.com?" not in text and "Chat" not in text
    assert "opened a window" not in text and "Another app" not in text
    assert window.is_error and window_calls(screen) == []
    assert provider._held == []


async def test_only_layers_a_window_to_answer_sits_on_are_candidates(
    screen: Screen,
) -> None:
    """The Dock's level, the menu bar, its status items, pop-up menus and the screen saver's are never another app's window."""
    screen.show(CHAT_MAIN, front=False)
    others = [
        {**PANEL, "window_id": 50 + n, "title": f"level {layer}", "layer": layer}
        for n, layer in enumerate((20, 24, 25, 101, 1000))
    ]
    person = Person(ALLOW_ONCE, ALLOW_ONCE, ALLOW_ONCE)
    driver = world(button("Quick sign-in"))
    on_act(driver, showing(screen, *others, PANEL))
    provider, tools = await opened(driver, watch(screen, strict=True))
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            await settled(provider)
    finally:
        await provider.close()
    assert cards_for(person) == ["Allow example.com?"]


# ── The person's own windows becoming visible are not new ──


async def test_a_space_switch_that_shows_old_windows_raises_no_card(
    screen: Screen,
) -> None:
    screen.show(TERMINAL, front=False)
    screen.show(INSPECTOR, front=False)
    person = Person()
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            click = await run(tools, "browser_click", ref="e1")
            # The person switches to Space 2: its windows come on the screen, and its
            # app to the front.
            screen.set(space=2)
            screen.front = Front("Terminal", "com.apple.Terminal", 30)
            screen.clock.now += 2
            await screen.pacer.tick(provider)
            read = await run(tools, "browser_read")
            await settled(provider)
    finally:
        await provider.close()
    assert person.cards == [] and person.windows == []
    text = told(click, read)
    assert "Terminal" not in text and "came to the front" not in text
    assert "opened a window" not in text and window_calls(screen) == []


async def test_an_unhidden_old_window_raises_no_card(screen: Screen) -> None:
    hidden_main = {**CHAT_MAIN, "hidden": True}
    hidden_panel = {**PANEL, "window_id": 3, "title": "Old notice", "hidden": True}
    screen.show(hidden_main, front=False)
    screen.show(hidden_panel, front=False)
    person = Person()
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            click = await run(tools, "browser_click", ref="e1")
            change(screen, 4, hidden=False)
            change(screen, 3, hidden=False)
            screen.clock.now += 2
            await screen.pacer.tick(provider)
            read = await run(tools, "browser_read")
            await settled(provider)
    finally:
        await provider.close()
    assert person.cards == [] and person.windows == []
    assert "opened a window" not in told(click, read) and window_calls(screen) == []


async def test_an_app_that_shows_an_old_panel_with_its_new_one_asks_only_about_the_new(
    screen: Screen,
) -> None:
    """Its other layers were not listed before: only a window numbered after every window listed then is new."""
    screen.show({**CHAT_MAIN, "window_id": 8, "hidden": True}, front=False)
    screen.show(
        {**PANEL, "window_id": 6, "title": "Old notice", "hidden": True}, front=False
    )
    person = Person(ALLOW_ONCE, ALLOW_ONCE)

    def answer() -> None:
        change(screen, 8, hidden=False)
        change(screen, 6, hidden=False)
        screen.show(COMPANION, front=False)
        screen.show(PANEL, front=False)

    driver = world(button("Quick sign-in"))
    on_act(driver, answer)
    provider, tools = await opened(driver, watch(screen, strict=True))
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            await settled(provider)
    finally:
        await provider.close()
    assert cards_for(person) == ["Allow example.com?"]


async def test_a_window_made_hidden_after_the_act_is_new_when_it_shows(
    screen: Screen,
) -> None:
    later = {**CHAT_MAIN, "window_id": 42, "title": "Sign in", "on_screen": False}
    person = Person(ALLOW_ONCE)
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(later, front=False))
    provider, tools = await opened(driver, watch(screen, strict=True))
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            assert person.cards == []  # made, not shown yet
            screen.clock.now += 2
            change(screen, 42, on_screen=True)
            await screen.pacer.tick(provider)
            await settled(provider)
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert cards_for(person) == ["Sign in"]
    assert "Chat opened a window 'Sign in' (w1)" in read.text


async def test_a_window_made_hidden_and_shown_after_the_deadline_is_not_taken(
    screen: Screen,
) -> None:
    later = {**CHAT_MAIN, "window_id": 42, "title": "Sign in", "on_screen": False}
    person = Person()
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(later, front=False))
    provider, tools = await opened(driver, watch(screen, strict=True))
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += ADMIT_S + 1
            change(screen, 42, on_screen=True)
            read = await run(tools, "browser_read")
            await settled(provider)
    finally:
        await provider.close()
    assert person.cards == [] and "opened a window" not in read.text


# ── Buttons the app draws itself: a point of its picture ──


async def panel_pressed(
    screen: Screen, clicks_do: dict[str, str], asks: Asks
) -> tuple[Any, Any, Any]:
    """Click, read the panel and press a point of it; the panel's clicks do *clicks_do*."""
    driver = world(button("Quick sign-in"))
    on_act(driver, showing(screen, COMPANION, {**PANEL, "clicks": clicks_do}))
    provider, tools = await opened(driver, watch(screen))  # read on its own
    try:
        with asking(asks):
            click = await run(tools, "browser_click", ref="e1")
            read = await run(tools, "window_read", window="w1")
            pressed = await run(tools, "window_press", window="w1", **POINT)
    finally:
        await provider.close()
    return click, read, pressed


@pytest.mark.parametrize(
    ("closes", "shown"),
    [
        ("close", " The picture shows where."),
        # The click closed the window before the helper took the picture that marks the
        # point, so that picture failed: the click was delivered, and it is no refusal.
        ("close_then_fail", ""),
    ],
)
async def test_a_window_that_names_no_buttons_is_pressed_at_a_point_of_its_picture(
    screen: Screen, closes: str, shown: str
) -> None:
    asks = Asks()
    click, read, pressed = await panel_pressed(screen, {"background": closes}, asks)
    assert "it names no buttons" in click.text
    assert "(no readable text)" in read.text
    assert (
        "[This window names no buttons: it draws its own. window_press with x and "
        "y, 0-1000 across and down this picture, presses a point of it. melo does "
        "not type into other apps.]"
    ) in read.text
    assert not pressed.is_error
    assert pressed.text == (
        f"Clicked (700, 850) on the picture of Chat's window (via point).{shown} "
        "Window w1 closed."
    )
    assert bool(pressed.images) is bool(shown)
    [only] = clicks(screen)
    assert only.get("delivery_mode", "background") == "background"
    assert [name for name in asks.names if name.startswith("desktop-act:")] == [
        "desktop-act:Chat"
    ]


async def test_a_point_click_that_changes_nothing_is_tried_once_in_the_foreground(
    screen: Screen,
) -> None:
    asks = Asks()
    _, _, pressed = await panel_pressed(
        screen, {"background": "nothing", "foreground": "close"}, asks
    )
    assert not pressed.is_error
    assert pressed.text == (
        "Clicked (700, 850) on the picture of Chat's window (via point). The "
        "picture shows where. The click in the background changed nothing, so melo "
        "clicked there again in the foreground: the pointer moved and Chat came to "
        "the front for a moment. Window w1 closed."
    )
    background, foreground = clicks(screen)
    assert "delivery_mode" not in background
    assert foreground["delivery_mode"] == "foreground"
    assert (foreground["x"], foreground["y"]) == (560.0, 510.0)  # on 800x600
    # The window was read again right before the foreground click, in the same session.
    calls = screen.calls()
    at = calls.index(("click", foreground))
    assert calls[at - 1][0] == "get_window_state"
    assert calls[at - 1][1]["include_screenshot"] is True
    # One question covers both clicks, and says what the second does.
    [(_, arguments, preview)] = [
        item for item in asks.asked if item[0] == "desktop-act:Chat"
    ]
    assert arguments["delivery"] == "foreground_retry"
    assert preview == (
        "press a point (700, 850) of its picture in Chat's window \"Allow "
        'example.com?"; if a click in the background does not reach it, melo '
        "clicks once more in the foreground: the pointer moves there and Chat comes "
        "to the front for a moment"
    )
    assert sum(screen.clock.slept) >= 1.0  # it gave the click a second first


class ClosesLate(Asks):
    """The panel closes a moment after the foreground click, not at once."""

    def __init__(self, screen: Screen) -> None:
        super().__init__()
        self.screen = screen

    async def __call__(
        self, name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        if name.startswith("desktop-act:"):
            # The background click and its second of waiting come first, so this is
            # half a second after the foreground click.
            self.screen.clock.at(1.5, lambda: remove(self.screen, 41))
        return await super().__call__(name, arguments, preview)


def remove(screen: Screen, window_id: int) -> None:
    """The window (the panel is 41) is gone from the screen."""
    screen.state = screen.load()
    screen.state["windows"] = [
        w for w in screen.state["windows"] if w["window_id"] != window_id
    ]
    screen.save()


async def test_the_foreground_click_is_given_a_moment_before_the_window_is_reported(
    screen: Screen,
) -> None:
    _, _, pressed = await panel_pressed(
        screen, {"background": "nothing", "foreground": "nothing"}, ClosesLate(screen)
    )
    assert len(clicks(screen)) == 2
    assert pressed.text.endswith(
        "in the foreground: the pointer moved and Chat came to the front for a moment. Window w1 closed."
    )
    assert "still open" not in pressed.text


async def test_a_point_click_that_changed_the_window_is_not_repeated(
    screen: Screen,
) -> None:
    asks = Asks()
    _, _, pressed = await panel_pressed(
        screen, {"background": "change", "foreground": "close"}, asks
    )
    assert len(clicks(screen)) == 1
    assert "The window changed: window_read w1 shows it now." in pressed.text
    assert [name for name in asks.names if name.startswith("desktop-act:")] == [
        "desktop-act:Chat"
    ]


async def test_a_click_the_helper_refused_stays_an_error_even_when_the_window_is_gone(
    screen: Screen,
) -> None:
    # The window went away just before the click, and the helper refused it: nothing was clicked.
    _, _, pressed = await panel_pressed(screen, {"background": "gone"}, Asks())
    assert pressed.is_error and "window_id_not_found" in pressed.text
    assert "Clicked" not in pressed.text


async def test_a_second_point_needs_a_new_read(screen: Screen) -> None:
    driver = world(button("Quick sign-in"))
    on_act(
        driver,
        showing(screen, COMPANION, {**PANEL, "clicks": {"background": "change"}}),
    )
    provider, tools = await opened(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_click", ref="e1")
            await run(tools, "window_read", window="w1")
            first = await run(tools, "window_press", window="w1", **POINT)
            again = await run(tools, "window_press", window="w1", **POINT)
            before = len(clicks(screen))
            await run(tools, "window_read", window="w1")
            third = await run(tools, "window_press", window="w1", **POINT)
    finally:
        await provider.close()
    assert "The window changed: window_read w1 shows it now." in first.text
    assert again.is_error and "Use window_read w1 first" in again.text
    assert before == 1  # the second point clicked nothing
    assert not third.is_error and len(clicks(screen)) > before
