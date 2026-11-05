"""A window another app opens after melo acts: noticed, read and answered, or named when it cannot be read.

The desktop control helper is a fake MCP server over a JSON screen file
(``fake_desktop_server.py``); the app in front is scripted. Nothing here
touches the real screen.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import NEVER_GRANTED, SELF_ASKING_TOOLS
from localmelo.melo.executor.models import ToolOutput
from localmelo.support.browser import desktop as desktop_mod
from localmelo.support.browser import provider as provider_mod
from localmelo.support.browser.desktop import (
    SHOT_MAX_BYTES,
    AppWindow,
    DesktopHelper,
    DesktopWatch,
    Front,
    Mark,
    parse_front,
)
from localmelo.support.browser.driver import Act, ActResult
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDriver, button
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world

FAKE = str(Path(__file__).with_name("fake_desktop_server.py"))
BROWSER = {
    "window_id": 1,
    "pid": 10,
    "app_name": "Safari",
    "title": "Forum",
    "z_index": 1,
}
MINE = {
    "window_id": 2,
    "pid": 20,
    "app_name": "Notes",
    "title": "Groceries",
    "z_index": 2,
}
SIGN_IN = {
    "window_id": 7,
    "pid": 70,
    "app_name": "Chat",
    "bundle": "com.example.chat",
    "title": "Sign in to example.com",
    "z_index": 9,
    "layer": 0,
    "elements": [
        {"element_index": 0, "role": "AXStaticText", "label": "Allow example.com to sign in with Chat?"},
        {"element_index": 1, "role": "AXButton", "label": "Allow", "element_token": "s0000000a:1"},
        {"element_index": 2, "role": "AXButton", "label": "Cancel", "element_token": "s0000000a:2"},
    ],
    "press": {"s0000000a:1": {"close": True}, "s0000000a:2": {"close": True}},
}  # fmt: skip


class Clock:
    """The watch's clock and sleep: time moves only when the watch sleeps, so a late wait costs no real time.

    ``at(seconds, then)`` runs *then* when the clock first passes that many
    seconds from now: what another app does a moment after an act.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []
        self._due: list[tuple[float, Callable[[], None]]] = []

    def __call__(self) -> float:
        return self.now

    def at(self, seconds: float, then: Callable[[], None]) -> None:
        self._due.append((self.now + seconds, then))

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        for due, then in [d for d in self._due if d[0] <= self.now]:
            self._due.remove((due, then))
            then()


class Pacer:
    """The pause between two background samples: none passes until a test ticks.

    The sampler parks here; ``tick`` lets it take one sample and returns when it
    is parked again (or has ended), so a test decides what the screen shows at
    each sample.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.waiting = asyncio.Event()
        self._go: asyncio.Queue[None] = asyncio.Queue()

    async def __call__(self, seconds: float) -> None:
        self.calls += 1
        self.waiting.set()
        await self._go.get()
        self.waiting.clear()

    async def tick(self, provider: BrowserToolProvider) -> None:
        end = time.monotonic() + 5.0

        def over() -> bool:
            task = provider._sampler
            return task is None or task.done() or time.monotonic() > end

        while not self.waiting.is_set():
            if over():
                return
            await asyncio.sleep(0.01)
        calls = self.calls
        self._go.put_nowait(None)
        while self.calls == calls and not over():
            await asyncio.sleep(0.01)


class Screen:
    """The fake helper's screen file, and the app the test puts in front."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / "screen.json"
        self.log_path = folder / "calls.jsonl"
        self.log_path.write_text("")
        self.state: dict[str, Any] = {"windows": [BROWSER, MINE]}
        self.front: Front | None = Front("Safari", "com.apple.Safari", 10)
        self.clock = Clock()
        self.pacer = Pacer()
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state))

    def load(self) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.path.read_text())
        return loaded

    def show(self, window: dict[str, Any], *, front: bool = True) -> None:
        self.state = self.load()
        self.state["windows"] = [*self.state["windows"], window]
        self.save()
        if front:
            self.front = Front(
                window["app_name"], window.get("bundle", ""), window["pid"]
            )

    def set(self, **values: Any) -> None:
        self.state = self.load()
        self.state.update(values)
        self.save()

    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (name, args)
            for name, args in map(json.loads, self.log_path.read_text().splitlines())
        ]

    async def in_front(self) -> Front | None:
        return self.front


@pytest.fixture
def screen(tmp_path: Path) -> Screen:
    return Screen(tmp_path)


class Trusting(DesktopWatch):
    """A watch that reads every window that appears inside the deadline on its own.

    The tests of the timing (deadline, late wait, sampler) are about when a window is
    noticed, not about who may see it, so they use this; the real rule (only the app
    the page asked for is read on its own, any other waits for the person) is tested
    with ``watch(..., strict=True)``.
    """

    def _alone(self, mark: Mark, window: AppWindow) -> bool:
        return True


# The apps of the fake screen's windows, by process id (what lsappinfo would say).
BUNDLES = {
    10: "com.apple.Safari",
    20: "com.apple.Notes",
    70: "com.example.chat",
    80: "com.example.sso",
}


def watch(
    screen: Screen,
    *,
    helper: bool = True,
    asked: tuple[str, ...] = (),
    strict: bool = False,
) -> DesktopWatch:
    """A watch over *screen*.

    *asked* are the names of the app an address of another kind opens (none: not
    known). Unless *strict*, every window inside the deadline is read on its own.
    """
    command = [sys.executable, "-I", FAKE, str(screen.path), str(screen.log_path)]

    async def handler(url: str) -> frozenset[str]:
        return frozenset(asked)

    async def bundle(pid: int) -> str:
        return BUNDLES.get(pid, "")

    kind = DesktopWatch if strict else Trusting
    return kind(
        DesktopHelper(command) if helper else None,
        front=screen.in_front,
        handler=handler,
        bundle=bundle,
        clock=screen.clock,
        sleep=screen.clock.sleep,
        pace=screen.pacer,
    )


async def tools_with(
    driver: FakeDriver, desk: DesktopWatch | None
) -> tuple[BrowserToolProvider, dict[str, Any]]:
    provider = BrowserToolProvider(driver, desktop=desk)
    return provider, {tool.definition.name: tool for tool in await provider.start()}


async def run(tools: dict[str, Any], name: str, **args: Any) -> ToolOutput:
    result = await tools[name].function(**args)
    return result if isinstance(result, ToolOutput) else ToolOutput(text=result)


def on_act(driver: FakeDriver, then: Callable[[], None]) -> None:
    """*then* runs right after the driver's act: what the click made another app do."""
    real = driver.act

    async def act(tab: str | None, what: Act) -> ActResult:
        result = await real(tab, what)
        then()
        return result

    driver.act = act  # type: ignore[method-assign,assignment]


async def closing(provider: BrowserToolProvider, body: Awaitable[None]) -> None:
    try:
        await body
    finally:
        await provider.close()


# ── Seeing it ──


async def test_a_window_another_app_opens_after_a_click_is_named_with_its_buttons(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(SIGN_IN))
    provider, tools = await tools_with(driver, watch(screen))
    try:
        assert {"window_read", "window_press"} <= set(tools)
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert not out.is_error
    first = out.text.splitlines()[0]
    assert "Another app put a window on the screen: see what changed." in first
    assert (
        "- Chat opened a window 'Sign in to example.com' (w1); its buttons: "
        "'Allow', 'Cancel'. window_read w1 shows it; window_press presses a button."
    ) in out.text
    # Only windows that appeared are named; the person's own windows never are.
    assert "Groceries" not in out.text and "Notes" not in out.text
    # The cheapest call watches: each look lists the main layer of every app on every
    # Space, then every layer of the apps with a window on the screen (Notes here; the
    # browser is not another app).
    calls = screen.calls()
    assert [name for name, _ in calls][:1] == ["check_permissions"]
    assert calls[1:3] == [("list_windows", {}), ("list_windows", {"pid": 20})]
    assert all(
        args.get("include_screenshot") is False
        for name, args in screen.calls()
        if name == "get_window_state"
    )


async def test_the_window_is_read_with_its_text_buttons_and_a_picture_in_the_limits(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(SIGN_IN))
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            await run(tools, "browser_click", ref="e1")
            out = await run(tools, "window_read")
    finally:
        await provider.close()
    assert not out.is_error
    assert out.text.startswith("<<page id=w1 nonce=") and 'app="Chat"' in out.text
    assert "Allow example.com to sign in with Chat?" in out.text
    assert '[button "Allow"]' in out.text and '[button "Cancel"]' in out.text
    assert "melo does not type into other apps" in out.text
    [picture] = out.images
    assert picture.mime_type == "image/jpeg"
    assert len(base64.b64decode(picture.data)) <= SHOT_MAX_BYTES


async def test_a_button_is_pressed_by_name_through_accessibility_after_a_question(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(SIGN_IN))
    asks = Asks()
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(asks):
            await run(tools, "browser_open", url=HOME)
            await run(tools, "browser_click", ref="e1")
            out = await run(tools, "window_press", button="allow")
    finally:
        await provider.close()
    assert out.text == "Pressed 'Allow' in Chat (via accessibility). Window w1 closed."
    name, arguments, preview = asks.asked[-1]
    assert name == "desktop-act:Chat"
    assert arguments["access"] == "desktop-act" and arguments["button"] == "Allow"
    assert preview == 'press "Allow" in Chat\'s window "Sign in to example.com"'
    assert ("click", {"element_token": "s0000000a:1"}) in screen.calls()


async def test_a_refused_press_presses_nothing(screen: Screen) -> None:
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(SIGN_IN))
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks(deny=("desktop-act:",))):
            await run(tools, "browser_open", url=HOME)
            await run(tools, "browser_click", ref="e1")
            out = await run(tools, "window_press", button="Allow")
    finally:
        await provider.close()
    assert out.is_error and out.text.startswith("melo did not press it in Chat.")
    assert not [args for name, args in screen.calls() if name == "click"]


async def test_when_accessibility_fails_a_point_on_the_picture_is_clicked_and_marked(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    # The click reaches the window in the background: it changes, so it is not repeated.
    on_act(driver, lambda: screen.show({**SIGN_IN, "clicks": {"background": "change"}}))
    screen.set(press_fails=True)
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            await run(tools, "browser_click", ref="e1")
            failed = await run(tools, "window_press", button="Allow")
            early = await run(tools, "window_press", x=500, y=800)
            await run(tools, "window_read", window="w1")
            out = await run(tools, "window_press", window="w1", x=500, y=800)
    finally:
        await provider.close()
    assert failed.is_error and "window_press with x and y" in failed.text
    assert early.is_error and "Use window_read w1 first" in early.text
    assert not out.is_error and "(via point). The picture shows where." in out.text
    [(_, args)] = [c for c in screen.calls() if c[0] == "click" and "x" in c[1]]
    assert (args["x"], args["y"]) == (400.0, 480.0)  # 0-1000 on an 800x600 picture
    assert args["debug_image_out"].endswith("point.png")  # the point marked
    assert out.images and out.images[0].mime_type == "image/jpeg"


async def test_browser_wait_stops_early_when_another_apps_window_appears(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_mod, "WATCH_S", 0.01)
    driver = world(button("Quick sign-in"))
    real = driver.wait
    looks = 0

    async def wait(*args: Any, **kwargs: Any) -> bool:
        nonlocal looks
        looks += 1
        if looks == 2:
            screen.show(SIGN_IN)
        return await real(*args, **kwargs)

    driver.wait = wait  # type: ignore[method-assign]
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            # A wait tells a window only inside the deadline of an act that can hand off.
            await run(tools, "browser_click", ref="e1")
            out = await run(
                tools, "browser_wait", until="text", text="never", seconds=30
            )
    finally:
        await provider.close()
    assert out.text.startswith("Stopped waiting: another app put a window")
    assert "Chat opened a window" in out.text and "(w1)" in out.text
    assert looks == 2  # it did not wait out the 30 seconds


async def test_a_panel_of_an_app_that_came_to_the_front_is_new_only_against_its_own_listing(
    screen: Screen,
) -> None:
    """A panel is not in the main layer, so melo lists the app's own windows on every layer.

    The first time the app comes to the front there is no such listing from
    before the act: its panel is named, never read. After that, the app's windows
    are listed before each act, and a panel that appears is melo's to read.
    """
    menu_item = {**SIGN_IN, "window_id": 8, "title": "", "layer": 25, "elements": []}
    first = {**SIGN_IN, "layer": 3}
    second = {**SIGN_IN, "window_id": 9, "title": "Confirm sign-in", "layer": 3}
    screen.show(menu_item, front=False)  # the app's menu bar item, there before
    driver = world(button("Quick sign-in"), button("Again"))
    shown = [first, second]
    on_act(driver, lambda: screen.show(shown.pop(0)))
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            once = await run(tools, "browser_click", ref="e1")
            unread = await run(tools, "window_read")
            screen.front = Front("Safari", "com.apple.Safari", 10)
            await run(tools, "browser_open", url=HOME)
            again = await run(tools, "browser_click", ref="e5")
    finally:
        await provider.close()
    assert "(w1)" not in once.text
    assert (
        "- Chat came to the front: it may show a window melo cannot see. Tell the "
        "person to answer it in Chat."
    ) in once.text
    assert (
        "Sign in to example.com" not in once.text
    )  # a window it cannot place: no title
    assert unread.is_error and "No window of another app appeared" in unread.text
    assert "Chat opened a window 'Confirm sign-in' (w1)" in again.text
    assert "Sign in to example.com" not in again.text  # there before: not new
    assert "(w2)" not in again.text  # the menu bar item is never taken for it
    assert ("list_windows", {"pid": 70}) in screen.calls()


async def test_a_panel_open_before_the_act_is_never_read_when_its_app_comes_forward(
    screen: Screen,
) -> None:
    """Only the app in front changes: the panel was there all along, so it is named, not read."""
    private = {**SIGN_IN, "window_id": 5, "title": "Private notes", "layer": 3}
    screen.show({**SIGN_IN, "window_id": 4, "title": "Chat"}, front=False)
    screen.show(private, front=False)
    driver = world(button("Quick sign-in"))
    on_act(
        driver, lambda: setattr(screen, "front", Front("Chat", "com.example.chat", 70))
    )
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e1")
            read = await run(tools, "window_read")
    finally:
        await provider.close()
    assert "(w1)" not in out.text and "its buttons" not in out.text
    assert "- Chat came to the front: it may show a window melo cannot see." in out.text
    assert "Private notes" not in out.text  # no title of a window melo cannot place
    assert read.is_error
    assert not [name for name, _ in screen.calls() if name == "get_window_state"]


async def test_with_accessibility_alone_melo_reads_and_presses_by_name_without_a_picture(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(SIGN_IN))
    screen.set(grants={"accessibility": True, "screen_recording": False})
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e1")
            read = await run(tools, "window_read")
            point = await run(tools, "window_press", x=500, y=800)
            pressed = await run(tools, "window_press", button="Allow")
    finally:
        await provider.close()
    assert "Chat opened a window 'Sign in to example.com' (w1)" in out.text
    assert not read.is_error and not read.images
    assert '[button "Allow"]' in read.text
    assert "no Screen Recording permission" in read.text
    assert point.is_error and "no picture to point at" in point.text
    assert (
        pressed.text == "Pressed 'Allow' in Chat (via accessibility). Window w1 closed."
    )
    assert all(
        args.get("include_screenshot") is False
        for name, args in screen.calls()
        if name == "get_window_state"
    )
    assert not [
        args for name, args in screen.calls() if name == "click" and "x" in args
    ]


async def test_a_failed_press_without_screen_recording_asks_the_person(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    on_act(driver, lambda: screen.show(SIGN_IN))
    screen.set(grants={"accessibility": True}, press_fails=True)
    provider, tools = await tools_with(driver, watch(screen))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            await run(tools, "browser_click", ref="e1")
            failed = await run(tools, "window_press", button="Allow")
    finally:
        await provider.close()
    assert failed.is_error
    assert "no picture to click a point of: ask the person to press it in Chat" in (
        failed.text
    )


# ── Without the helper ──


@pytest.mark.parametrize("helper", ["missing", "no grant"])
async def test_without_the_helper_melo_names_the_app_in_front_in_one_line(
    screen: Screen, helper: str
) -> None:
    driver = world(button("Quick sign-in"))
    on_act(
        driver, lambda: setattr(screen, "front", Front("Chat", "com.example.chat", 70))
    )
    if helper == "no grant":
        screen.set(grants={"accessibility": False, "screen_recording": True})
    desk = watch(screen, helper=helper != "missing")
    provider, tools = await tools_with(driver, desk)
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e1")
            read = await run(tools, "window_read") if "window_read" in tools else None
    finally:
        await provider.close()
    assert (
        "- Chat came to the front: it may show a window melo cannot see. Tell the "
        "person to answer it in Chat."
    ) in out.text
    if helper == "missing":
        assert "window_read" not in tools  # nothing to read with
    else:
        assert read is not None and read.is_error
        assert "no Accessibility grant" in read.text


async def test_the_browser_coming_to_the_front_is_not_another_app(
    screen: Screen,
) -> None:
    driver = world(button("Quick sign-in"))
    screen.front = Front("Terminal", "com.apple.Terminal", 5)
    on_act(
        driver,
        lambda: setattr(screen, "front", Front("Safari", "com.apple.Safari", 10)),
    )
    provider, tools = await tools_with(driver, watch(screen, helper=False))
    try:
        with asking(Asks()):
            await run(tools, "browser_open", url=HOME)
            out = await run(tools, "browser_click", ref="e1")
    finally:
        await provider.close()
    assert "came to the front" not in out.text and "Another app" not in out.text


# ── Scope ──


async def test_no_window_is_read_unless_one_appeared_after_melos_own_act(
    screen: Screen,
) -> None:
    provider, tools = await tools_with(world(), watch(screen))
    try:
        out = await run(tools, "window_read")
        named = await run(tools, "window_read", window="w2")
    finally:
        await provider.close()
    assert out.is_error and "it does not look around the desktop" in out.text
    assert named.is_error and "is not a window that appeared" in named.text
    assert not [name for name, _ in screen.calls() if name == "get_window_state"]


async def test_a_press_is_its_own_question_every_time_and_the_tool_asks_it_itself() -> (
    None
):
    assert any(prefix == "desktop-act:" for prefix in NEVER_GRANTED)
    assert "window_press" in SELF_ASKING_TOOLS


async def test_no_watch_no_window_tools_and_no_note() -> None:
    provider, tools = await tools_with(world(), None)
    assert "window_read" not in tools
    assert len(provider.system_notes()) == 1


def test_the_front_app_is_read_from_lsappinfo_words() -> None:
    text = (
        '"Chat" ASN:0x0-0x1c01c: (in front) \n    bundleID="com.example.chat"\n'
        "    bundle path=[ NULL ] \n    pid = 70 !cgsConnection !signalled\n"
    )
    assert parse_front(text) == Front("Chat", "com.example.chat", 70)
    assert parse_front("") is None


def test_the_helper_is_found_by_its_command_or_not_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(desktop_mod.HELPER_ENV, "/opt/helper --flag")
    assert desktop_mod.helper_command() == ["/opt/helper", "--flag"]
    monkeypatch.setenv(desktop_mod.HELPER_ENV, "")
    assert desktop_mod.helper_command() is None
    assert desktop_mod.watch_for(False) is None
