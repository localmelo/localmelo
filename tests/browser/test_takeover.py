"""The person takes over melo's browser: who has a tab, melo's steps that wait, the person's input, the card.

No real browser runs. The owners and the input mapping are tested alone; the
relay and the browser tools over a scripted browser that streams and takes
input (it records what it was sent). Every test that types checks that the
text went to the browser and nowhere else: not to a log, a step result, a
question or the live view's messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

from localmelo.support.browser import provider as provider_mod
from localmelo.support.browser import takeover
from localmelo.support.browser.driver import Element
from localmelo.support.browser.live import TAKEN_FPS, Frame, LiveView
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.browser.takeover import (
    META,
    SHIFT,
    Bucket,
    InputRefusedError,
    Owners,
    QuietFrames,
    cdp_events,
    held_words,
    needs_person,
    quiet_frames,
)
from localmelo.tests.browser.fakes import button, link, page
from localmelo.tests.browser.test_acting import Asks, asking
from localmelo.tests.browser.test_live import Stream, Streaming
from localmelo.tests.browser.test_provider import run

HOME = "https://example.com/"
LOGIN = "https://example.com/login"
SECRET = "hunter2-Secret"
CHINESE = "密码是你好"
VIEW = (1280.0, 713.0)


def grace_left(owners: Owners, tab: str) -> float | None:
    """Seconds until a held tab nobody watches comes back; None while watched or not held."""
    held = owners._held.get(tab)
    if held is None or held.away is None:
        return None
    return max(0.0, held.away + owners.grace_s - owners._clock())


class Later:
    """A test's timers: ``later(seconds, fn)`` keeps them, ``fire()`` runs the due ones."""

    def __init__(self, clock: list[float]) -> None:
        self.clock = clock
        self.timers: list[list[Any]] = []

    def __call__(self, seconds: float, fn: Callable[[], None]) -> Any:
        timer = [self.clock[0] + seconds, fn, False]

        class Handle:
            def cancel(self_inner) -> None:  # noqa: N805
                timer[2] = True

        self.timers.append(timer)
        return Handle()

    def advance(self, seconds: float) -> None:
        self.clock[0] += seconds
        for timer in list(self.timers):
            if not timer[2] and timer[0] <= self.clock[0]:
                timer[2] = True
                timer[1]()


def owners(changes: list[int] | None = None) -> tuple[Owners, Later]:
    clock = [100.0]
    later = Later(clock)
    found = Owners(
        on_change=(lambda: changes.append(1)) if changes is not None else None,
        clock=lambda: clock[0],
        later=later,
    )
    return found, later


# ── Who has a tab ──


def test_a_tab_is_melos_until_the_person_takes_it_and_back_when_they_give_it() -> None:
    changes: list[int] = []
    found, later = owners(changes)
    assert found.owner("t1") == "melo" and found.owner(None) == "melo"
    assert found.take("t1") is True and found.owner("t1") == "you"
    assert found.take("t1") is False, "taking a tab one has changes nothing"
    later.advance(42)
    back = found.give("t1")
    assert back is not None and (back.tab, round(back.seconds), back.how) == (
        "t1",
        42,
        "gave",
    )
    assert found.owner("t1") == "melo" and found.give("t1") is None
    assert [r.tab for r in found.returned()] == ["t1"] and found.returned() == []
    assert len(changes) == 2


def test_a_tab_nobody_watches_comes_back_after_the_grace_period() -> None:
    found, later = owners()
    found.take("t1")
    found.watching("t1", 1)
    assert grace_left(found, "t1") is None
    found.watching(
        "t1", 0
    )  # the panel closed, the page was hidden or the token rotated
    assert grace_left(found, "t1") == takeover.GRACE_S
    later.advance(takeover.GRACE_S - 1)
    assert found.owner("t1") == "you"
    later.advance(2)
    assert found.owner("t1") == "melo"
    [back] = found.returned()
    assert back.how == "left"


def test_a_panel_that_comes_back_within_the_grace_period_keeps_the_tab() -> None:
    found, later = owners()
    found.take("t1")
    found.watching("t1", 0)
    later.advance(60)
    found.watching("t1", 2)
    assert grace_left(found, "t1") is None
    later.advance(takeover.GRACE_S * 2)
    assert found.owner("t1") == "you", "the cancelled timer did not give it back"


def test_a_tab_taken_from_the_card_with_no_panel_open_starts_its_grace_at_once() -> (
    None
):
    found, later = owners()
    found.take("t1", watched=False)
    assert grace_left(found, "t1") == takeover.GRACE_S
    later.advance(takeover.GRACE_S + 1)
    assert found.owner("t1") == "melo"


def test_a_closed_tab_and_the_end_of_the_chat_leave_nothing_held() -> None:
    found, later = owners()
    found.take("t1")
    found.take("t2")
    found.give("t2")
    found.forget("t2")
    assert found.returned() == [], "nothing is told about a tab that is gone"
    found.forget("t1")
    assert found.held == frozenset()
    found.take("t3")
    found.watching("t3", 0)
    found.clear()
    later.advance(takeover.GRACE_S + 1)
    assert found.held == frozenset() and found.returned() == []


async def test_a_step_waits_until_the_person_gives_the_tab_back() -> None:
    found = Owners()
    found.take("t1")
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, found.give, "t1")
    assert await found.wait_for_melo("t1", 2.0) is True
    found.take("t1")
    assert await found.wait_for_melo("t1", 0.05) is False
    assert await found.wait_for_melo("t2", 0.0) is True


def test_how_long_the_person_had_it_reads_plainly() -> None:
    assert [held_words(s) for s in (0.2, 42, 60, 125)] == [
        "1 s",
        "42 s",
        "1 min",
        "2 min 5 s",
    ]


# ── The person's input ──


def test_the_rate_is_held_with_a_burst() -> None:
    clock = [0.0]
    bucket = Bucket(rate=10, burst=5, clock=lambda: clock[0])
    assert [bucket.allow() for _ in range(7)] == [True] * 5 + [False] * 2
    clock[0] += 0.3  # three more
    assert [bucket.allow() for _ in range(4)] == [True] * 3 + [False]
    assert bucket.dropped == 3


def test_a_click_is_the_browsers_own_mouse_input_kept_on_the_page() -> None:
    [(method, params)] = cdp_events(
        {
            "type": "mouse",
            "action": "down",
            "x": 100.5,
            "y": 20,
            "button": "left",
            "buttons": 1,
            "clicks": 2,
        },
        VIEW,
    )
    assert method == "Input.dispatchMouseEvent"
    assert params == {
        "type": "mousePressed", "x": 100.5, "y": 20.0, "modifiers": 0,
        "button": "left", "buttons": 1, "clickCount": 2,
    }  # fmt: skip
    [(_, moved)] = cdp_events(
        {"type": "mouse", "action": "move", "x": 5000, "y": -3}, VIEW
    )
    assert (moved["type"], moved["x"], moved["y"], moved["button"]) == (
        "mouseMoved",
        1280.0,
        0.0,
        "none",
    )
    assert "clickCount" not in moved
    [(_, wheel)] = cdp_events(
        {"type": "wheel", "x": 1, "y": 1, "dx": 0, "dy": 99999}, VIEW
    )
    assert wheel["type"] == "mouseWheel" and wheel["deltaY"] == takeover.MAX_WHEEL


@pytest.mark.parametrize(
    "message",
    [
        {"type": "mouse", "action": "drag", "x": 1, "y": 1},
        {"type": "mouse", "action": "down", "x": "1", "y": 1},
        {"type": "mouse", "action": "down", "x": float("nan"), "y": 1},
        {"type": "mouse", "action": "down", "x": 1, "y": 1, "button": "laser"},
        {"type": "mouse", "action": "down", "x": 1, "y": 1, "clicks": 9},
        {"type": "mouse", "action": "down", "x": 1, "y": 1, "mods": True},
        {"type": "key", "action": "down", "key": "<script>"},
        {"type": "key", "action": "down", "key": "a", "code": "Key A!"},
        {"type": "key", "action": "press", "key": "a"},
        {"type": "text", "text": 5},
        {"type": "text", "text": "x" * (takeover.MAX_TEXT + 1)},
        {"type": "compose", "text": "x" * (takeover.MAX_COMPOSE + 1)},
        {"type": "compose", "text": "ab", "start": 3},
        {"type": "eval", "text": "1"},
    ],
)
def test_a_message_of_another_shape_is_refused_without_its_content(
    message: dict[str, Any],
) -> None:
    with pytest.raises(InputRefusedError) as refused:
        cdp_events(message, VIEW)
    said = str(refused.value)
    assert "<script>" not in said and "xxx" not in said and "Key A!" not in said


def test_no_point_is_taken_before_the_page_has_a_size() -> None:
    with pytest.raises(InputRefusedError):
        cdp_events({"type": "mouse", "action": "down", "x": 1, "y": 1}, (0.0, 0.0))


def test_a_printable_key_types_it_and_a_named_key_does_not() -> None:
    [(method, down)] = cdp_events(
        {"type": "key", "action": "down", "key": "a", "code": "KeyA", "keyCode": 65},
        VIEW,
    )
    assert method == "Input.dispatchKeyEvent"
    assert (down["type"], down["text"], down["windowsVirtualKeyCode"]) == (
        "keyDown",
        "a",
        65,
    )
    [(_, enter)] = cdp_events(
        {"type": "key", "action": "down", "key": "Enter", "code": "Enter"}, VIEW
    )
    assert (enter["type"], enter["text"], enter["windowsVirtualKeyCode"]) == (
        "keyDown",
        "\r",
        13,
    )
    [(_, back)] = cdp_events(
        {"type": "key", "action": "down", "key": "Backspace", "code": "Backspace"}, VIEW
    )
    assert back["type"] == "rawKeyDown" and "text" not in back
    assert back["commands"] == ["deleteBackward"]
    [(_, tab)] = cdp_events(
        {"type": "key", "action": "down", "key": "Tab", "code": "Tab"}, VIEW
    )
    assert (
        tab["type"] == "rawKeyDown"
        and "text" not in tab
        and tab["windowsVirtualKeyCode"] == 9
    )
    [(_, up)] = cdp_events(
        {"type": "key", "action": "up", "key": "a", "code": "KeyA", "keyCode": 65}, VIEW
    )
    assert up["type"] == "keyUp" and "text" not in up


def test_a_shortcut_types_nothing_and_says_its_editing_command() -> None:
    [(_, select)] = cdp_events(
        {
            "type": "key",
            "action": "down",
            "key": "a",
            "code": "KeyA",
            "keyCode": 65,
            "mods": META,
        },
        VIEW,
    )
    assert select["type"] == "rawKeyDown" and "text" not in select
    assert select["commands"] == ["selectAll"] and select["modifiers"] == META
    [(_, redo)] = cdp_events(
        {
            "type": "key",
            "action": "down",
            "key": "Z",
            "code": "KeyZ",
            "keyCode": 90,
            "mods": META | SHIFT,
        },
        VIEW,
    )
    assert redo["commands"] == ["redo"]
    [(_, shifted)] = cdp_events(
        {
            "type": "key",
            "action": "down",
            "key": "A",
            "code": "KeyA",
            "keyCode": 65,
            "mods": SHIFT,
        },
        VIEW,
    )
    assert shifted["text"] == "A" and "commands" not in shifted


def test_an_input_methods_composition_and_commit_and_a_paste_are_text() -> None:
    [(method, composing)] = cdp_events(
        {"type": "compose", "text": "nihao", "start": 5, "end": 5}, VIEW
    )
    assert method == "Input.imeSetComposition"
    assert composing == {"text": "nihao", "selectionStart": 5, "selectionEnd": 5}
    assert cdp_events({"type": "text", "text": "你好"}, VIEW) == [
        ("Input.insertText", {"text": "你好"})
    ]
    assert cdp_events({"type": "paste", "text": "a\tb\nc\x07\x1b"}, VIEW) == [
        ("Input.insertText", {"text": "a\tb\nc"})
    ]
    assert cdp_events({"type": "text", "text": ""}, VIEW) == []
    [(_, cancel)] = cdp_events({"type": "compose", "text": ""}, VIEW)
    assert cancel == {"text": "", "selectionStart": 0, "selectionEnd": 0}


def test_the_websocket_frame_lines_never_reach_a_log() -> None:
    record = logging.LogRecord(
        "uvicorn.error", logging.DEBUG, "", 0, "< %s", ('TEXT \'{"text":"x"}\'',), None
    )
    assert QuietFrames().filter(record) is False
    other = logging.LogRecord(
        "uvicorn.error", logging.DEBUG, "", 0, "connection open", (), None
    )
    assert QuietFrames().filter(other) is True
    quiet_frames(["test.frames"])
    quiet_frames(["test.frames"])
    assert (
        sum(
            isinstance(f, QuietFrames) for f in logging.getLogger("test.frames").filters
        )
        == 1
    )


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING])
def test_a_handshake_header_that_carries_a_credential_is_logged_redacted(
    level: int,
) -> None:
    token = "synthetic-bearer-7f3a"  # noqa: S105 - not a credential
    for line, name, value in [
        ("< %s: %s", "Authorization", f"Bearer {token}"),
        ("< %s: %s", "cookie", f"melo_session_1={token}"),
        ("> %s: %s", "Set-Cookie", f"melo_session_1={token}; HttpOnly"),
        ("< %s: %s", "X-Api-Token", token),
    ]:
        record = logging.LogRecord(
            "uvicorn.error", level, "", 0, line, (name, value), None
        )
        assert QuietFrames().filter(record) is True, "the line stays, redacted"
        said = record.getMessage()
        assert token not in said and said.endswith(f"{name}: {takeover.REDACTED}")
    host = logging.LogRecord(
        "uvicorn.error", level, "", 0, "< %s: %s", ("Host", "127.0.0.1:9"), None
    )
    assert QuietFrames().filter(host) is True
    assert host.getMessage() == "< Host: 127.0.0.1:9"


# ── When melo needs the person ──


@pytest.mark.parametrize(
    ("kwargs", "need"),
    [
        (
            {
                "url": LOGIN,
                "title": "Example",
                "text": "",
                "sensitive": ["password"],
                "elements": 80,
            },
            "sign-in",
        ),
        (
            {
                "url": HOME,
                "title": "Sign in · Example",
                "text": "",
                "sensitive": ["password"],
                "elements": 80,
            },
            "sign-in",
        ),
        (
            {
                "url": HOME,
                "title": "Example",
                "text": "",
                "sensitive": ["password"],
                "elements": 6,
            },
            "sign-in",
        ),
        # A small sign-in form at the side of a long page is not a sign-in page.
        (
            {
                "url": HOME,
                "title": "Example News",
                "text": "Log in",
                "sensitive": ["password"],
                "elements": 120,
            },
            "",
        ),
        (
            {
                "url": HOME + "verify",
                "title": "Check",
                "text": "",
                "sensitive": ["code"],
            },
            "code",
        ),
        (
            {
                "url": HOME,
                "title": "Shop",
                "text": "Add to cart",
                "sensitive": ["code"],
            },
            "",
        ),
        (
            {
                "url": HOME,
                "title": "登录",
                "text": "请使用微信扫码登录",
                "sensitive": [],
            },
            "qr",
        ),
        (
            {
                "url": LOGIN,
                "title": "Sign in",
                "text": "Scan the QR code with the app",
                "sensitive": [],
            },
            "qr",
        ),
        (
            {"url": HOME, "title": "Just a moment...", "text": "", "sensitive": []},
            "captcha",
        ),
        (
            {
                "url": HOME,
                "title": "Example",
                "text": "",
                "sensitive": [],
                "frames": ["https://newassets.hcaptcha.com"],
            },
            "captcha",
        ),
        (
            {
                "url": HOME,
                "title": "Example",
                "text": "Please verify you are human",
                "sensitive": [],
            },
            "captcha",
        ),
        (
            {
                "url": HOME,
                "title": "Example",
                "text": "Protected by reCAPTCHA. Privacy - Terms",
                "sensitive": [],
            },
            "",
        ),
        ({"url": HOME, "title": "Example", "text": "Hello.", "sensitive": []}, ""),
    ],
)
def test_what_on_a_page_needs_the_person(kwargs: dict[str, Any], need: str) -> None:
    assert needs_person(**kwargs) == need


# ── The relay ──


class TakeStream(Stream):
    """A stream that also says the rate it was asked for and keeps the box function."""

    def __init__(self) -> None:
        super().__init__()
        self.rates: list[float] = []
        self.boxes: dict[str, Callable[[], None]] = {}

    async def start_screencast(
        self,
        tab: str,
        on_frame: Callable[[Frame], None],
        on_end: Callable[[], None],
        *,
        fps: float = 15.0,
        quality: int = 70,
        on_dialog: Callable[[], None] | None = None,
    ) -> None:
        await super().start_screencast(tab, on_frame, on_end, fps=fps, quality=quality)
        self.rates.append(fps)
        if on_dialog is not None:
            self.boxes[tab] = on_dialog


class Takeable(Streaming):
    """A scripted browser that streams and takes the person's input, which it records."""

    def __init__(self, *pages: Any) -> None:
        super().__init__(*pages)
        self.cast: TakeStream = TakeStream()
        self.inputs: list[tuple[str, str, dict[str, Any]]] = []
        self.held: dict[str, bool] = {}
        self.box: dict[str, str] | None = None
        self.answers: list[tuple[str, bool, str]] = []

    def person_input(self, tab: str, method: str, params: dict[str, Any]) -> bool:
        self.inputs.append((tab, method, params))
        return True

    def hold_dialogs(self, tab: str, held: bool) -> None:
        self.held[tab] = held

    def dialog_of(self, tab: str) -> dict[str, str] | None:
        return self.box

    async def answer_dialog(self, tab: str, accept: bool, text: str = "") -> bool:
        self.answers.append((tab, accept, text))
        self.box = None
        return True

    def typed(self) -> str:
        """Everything the person's input carried, as one text."""
        return json.dumps([params for _, _, params in self.inputs], ensure_ascii=False)


class HeldStart(Takeable):
    """A browser whose start waits for the test once ``hold()`` is called."""

    def __init__(self, *pages: Any) -> None:
        super().__init__(*pages)
        self.starting = asyncio.Event()
        self._go = asyncio.Event()
        self._go.set()

    def hold(self) -> None:
        """The next start stops, and says so with ``starting``, until ``release()``."""
        self.starting.clear()
        self._go.clear()

    def release(self) -> None:
        self._go.set()

    async def start(self) -> None:
        self.starting.set()
        await self._go.wait()
        await super().start()


def textbox(label: str, **more: Any) -> Element:
    return Element("textbox", label, **more)


def browser() -> Takeable:
    return Takeable(
        page(HOME, "Welcome.", link("Sign in", LOGIN), button("Go"), title="Home"),
        page(
            LOGIN,
            "Sign in to Example.",
            textbox("Email"),
            textbox("Password", sensitive="password"),
            button("Sign in"),
            title="Sign in",
        ),
    )


async def started(
    driver: Takeable,
) -> tuple[BrowserToolProvider, dict[str, Any], LiveView]:
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    live = provider.live
    assert live is not None and live.can_take()
    return provider, tools, live


async def until(condition: Callable[[], bool], seconds: float = 2.0) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


def framed(driver: Takeable, tab: str = "t1") -> None:
    """The tab sends its first picture, so its size is known."""
    driver.cast.push(tab, Frame(b"\xff\xd8x", width=1280.0, height=713.0))


async def test_the_person_takes_the_tab_from_the_panel_and_gives_it_back() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await until(lambda: viewer.meta.get("state") == "live")
    assert viewer.meta["owner"] == "melo" and viewer.meta["takeover"] is True
    assert await live.take(viewer) is True
    await until(lambda: viewer.meta["owner"] == "you")
    assert viewer.meta["grace"] == int(takeover.GRACE_S)
    assert (
        driver.cast.rates[-1] == TAKEN_FPS
    ), "the person sees what they do at full rate"
    assert driver.held == {"t1": True}, "the page's boxes are theirs now"
    assert await live.give(viewer) is True
    await until(lambda: viewer.meta["owner"] == "melo")
    assert driver.cast.rates[-1] == live.fps and driver.held == {"t1": False}
    await provider.close()


async def test_input_goes_to_the_page_only_while_the_person_has_it_and_melo_is_not_mid_step() -> (
    None
):
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await until(lambda: viewer.meta.get("w") == 1280.0)
    click = {
        "type": "mouse",
        "action": "down",
        "x": 10,
        "y": 10,
        "button": "left",
        "buttons": 1,
    }
    assert live.send_input(viewer, click) is False, "melo has the tab"
    await live.take(viewer)
    assert live.send_input(viewer, click) is True
    provider._busy = True
    assert live.send_input(viewer, click) is False, "melo is finishing a step"
    provider._busy = False
    assert live.send_input(viewer, {"type": "eval", "text": "x"}) is False
    assert [m for _, m, _ in driver.inputs] == ["Input.dispatchMouseEvent"]
    await provider.close()


async def test_the_persons_input_is_held_to_a_rate() -> None:
    driver = browser()
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    clock = [0.0]
    live = LiveView(
        driver,
        provider,
        owners=provider.owners,
        bucket=lambda: Bucket(rate=10, burst=3, clock=lambda: clock[0]),
    )
    provider._live = live
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await until(lambda: viewer.meta.get("w") == 1280.0)
    await live.take(viewer)
    move = {"type": "mouse", "action": "move", "x": 1, "y": 1}
    assert [live.send_input(viewer, move) for _ in range(5)] == [True] * 3 + [False] * 2
    assert len(driver.inputs) == 3 and viewer.bucket.dropped == 2
    await provider.close()


async def test_a_box_the_page_puts_up_goes_to_the_panel_and_its_answer_back() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)
    driver.box = {"kind": "confirm", "message": "Leave the form?", "default": ""}
    driver.cast.boxes["t1"]()
    await until(lambda: "dialog" in viewer.meta)
    assert viewer.meta["dialog"]["message"] == "Leave the form?"
    assert await live.answer_dialog(viewer, True, "") is True
    assert driver.answers == [("t1", True, "")]
    await until(lambda: "dialog" not in viewer.meta)
    await live.give(viewer)
    driver.box = {"kind": "alert", "message": "Hi", "default": ""}
    assert (
        await live.answer_dialog(viewer, True) is False
    ), "melo's own boxes are melo's again"
    await provider.close()


async def test_the_view_stays_on_the_persons_tab_while_melo_opens_another() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)
    await run(tools, "browser_open", url=LOGIN, tab="new")
    await live._sync()
    assert viewer.target[0] == "t1" and provider.live_current() == "t2"
    await live.give(viewer)
    await live._sync()
    assert viewer.target[0] == "t2", "given back, the view follows melo again"
    await provider.close()


async def test_a_panel_that_goes_away_starts_the_grace_period() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)
    assert grace_left(provider.owners, "t1") is None
    await live.release(viewer)
    left = grace_left(provider.owners, "t1")
    assert left is not None and left > takeover.GRACE_S - 5
    again = await live.watch()
    assert grace_left(provider.owners, "t1") is None and again.target[0] == "t1"
    await provider.close()


# ── melo's steps while the person has the page ──


async def test_melos_step_waits_for_the_person_and_reads_the_page_again_after() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)
    asyncio.get_running_loop().call_later(0.05, provider.owners.give, "t1")
    out = await run(tools, "browser_click", ref="e2")
    assert not out.is_error
    assert out.text.startswith("The user had tab t1 for 1 s and gave it back to melo.")
    assert "melo did not do this step" in out.text and "<<page " in out.text
    assert [
        a.element.label for a in driver.acts
    ] == [], "the click on the old reading was not done"
    await provider.close()


async def test_a_step_that_waits_too_long_says_the_person_is_using_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_mod, "STEP_WAIT_S", 0.05)
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    await live.take(viewer)
    out = await run(tools, "browser_read")
    assert not out.is_error, "waiting for the person is no failure"
    assert "The user is using the page in tab t1" in out.text
    # A new tab is not theirs: melo may work there meanwhile.
    other = await run(tools, "browser_open", url=LOGIN, tab="new")
    assert "<<page " in other.text
    await provider.close()


async def test_melo_never_tidies_away_the_persons_tab() -> None:
    driver = browser()
    provider = BrowserToolProvider(driver, idle_close_s=0.2)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    live = provider.live
    assert live is not None
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    await live.take(viewer)
    provider._arm_idle()
    await asyncio.sleep(0.3)
    assert ("close", "t1") not in driver.calls and provider.live_raw("t1") is not None
    await provider.close()


async def test_a_step_naming_another_tab_never_acts_on_the_persons_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The person took the current tab over from the panel; its reading is kept,
    # so a ref of it is still real. A step naming another tab, a tab of no chat
    # or the person's tab spelt differently is refused or waits: the tab whose
    # owner was checked is the tab the step works on.
    monkeypatch.setattr(provider_mod, "STEP_WAIT_S", 0.05)
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)  # t1: e1 is the "Sign in" link
    await run(tools, "browser_open", url=LOGIN, tab="new")  # t2: e3 on
    await run(tools, "browser_tabs", action="switch", tab="t1")
    viewer = await live.watch()
    framed(driver)
    assert await live.take(viewer) is True and provider.owners.owner("t1") == "you"
    calls = len(driver.calls)

    other = await run(tools, "browser_follow", ref="e1", tab="t2")
    assert other.is_error and "e1" in other.text, "t2's reading has no e1"
    unknown = await run(tools, "browser_follow", ref="e1", tab="t9")
    assert unknown.is_error and '"t9" is not a tab of this chat' in unknown.text
    spelt = await run(tools, "browser_follow", ref="e1", tab=" T1 ")
    assert not spelt.is_error
    assert spelt.text.startswith("The user is using the page in tab t1")

    assert driver.calls[calls:] == [], "no page was opened, read or touched"
    assert driver.open_in == [], "nothing was loaded into a tab of before"
    assert provider.owners.owner("t1") == "you"
    await provider.close()


async def test_a_step_keeps_its_tab_when_the_current_tab_moves_in_the_start() -> None:
    # A step finds its tab once, before the browser starts, and Chrome may take a
    # moment to start. If the chat's current tab moves in the meantime, the step
    # still works on the tab it found: it never lands on the new current tab.
    driver = HeldStart(*browser().pages.values())
    provider, tools, _ = await started(driver)
    await run(tools, "browser_open", url=HOME)  # t1: e1 is "Sign in", e2 is "Go"
    await run(tools, "browser_open", url=LOGIN, tab="new")  # t2: e3 on
    await run(tools, "browser_tabs", action="switch", tab="t1")
    assert provider._tab == "t1"

    driver.hold()
    step = asyncio.ensure_future(run(tools, "browser_click", ref="e2"))
    await asyncio.wait_for(driver.starting.wait(), 2)
    # Another tool call waits for the step's lock, so the current tab is moved
    # directly: whatever moves it while the browser starts.
    provider._tab = "t2"
    driver.release()

    out = await asyncio.wait_for(step, 2)
    assert not out.is_error, out.text
    assert [(act.element.label, act.url) for act in driver.acts] == [("Go", HOME)]
    await provider.close()


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("browser_open", {"url": LOGIN}),
        ("browser_read", {}),
        ("browser_find", {"query": "Sign"}),
        ("browser_follow", {"ref": "e1"}),
        ("browser_click", {"ref": "e2"}),
        ("browser_type", {"ref": "e1", "text": "x"}),
        ("browser_select", {"ref": "e1", "option": "x"}),
        ("browser_back", {}),
        ("browser_tabs", {"action": "switch"}),
        ("browser_tabs", {"action": "close"}),
        ("browser_wait", {"until": "load"}),
        ("browser_screenshot", {}),
        ("browser_click_point", {"x": 10, "y": 10}),
    ],
)
async def test_every_step_naming_a_tab_of_no_chat_is_refused_before_it_waits_or_acts(
    tool: str, args: dict[str, Any]
) -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)  # a wait would show: the current tab is the person's
    calls = len(driver.calls)
    out = await run(tools, tool, **args, tab="t9")
    assert out.is_error and '"t9" is not a tab of this chat' in out.text
    assert driver.calls[calls:] == [] and driver.acts == []
    await provider.close()


# ── The card ──


def shop() -> Takeable:
    """A page with a code field that does not say it verifies anything: no card when it is read."""
    return Takeable(
        page(
            HOME,
            "Add to cart.",
            textbox("Gift code", sensitive="code"),
            button("Buy"),
            title="Shop",
        ),
    )


def the_person(
    provider: BrowserToolProvider, live: LiveView, viewer: Any, *texts: str
) -> None:
    """When the card hands them the tab, the person types *texts* in the panel and gives the page back."""
    loop = asyncio.get_running_loop()
    original = provider.owners.take

    def person() -> None:
        for text in texts:
            live.send_input(viewer, {"type": "text", "text": text})
        provider.owners.give("t1")

    def taken(tab: str, *, watched: bool = True) -> bool:
        changed = original(tab, watched=watched)
        loop.call_later(0.05, person)
        return changed

    provider.owners.take = taken  # type: ignore[method-assign]


async def test_a_secret_field_asks_the_person_to_take_over_and_melo_carries_on() -> (
    None
):
    driver = shop()
    provider, tools, live = await started(driver)
    asks = Asks()
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        assert not [
            n for n in asks.names if n.startswith("takeover:")
        ], "reading the page asked nothing"
        viewer = await live.watch()
        framed(driver)
        the_person(provider, live, viewer, SECRET, CHINESE)
        out = await run(tools, "browser_type", ref="e1", text="anything")
    name, arguments, preview = next(
        a for a in asks.asked if a[0].startswith("takeover:")
    )
    assert name == "takeover:https://example.com"
    assert arguments == {
        "site": "example.com",
        "access": "takeover",
        "need": "code",
        "tab": "t1",
        "browser": "chrome",
    }
    assert "in the Browser panel" in preview
    assert not out.is_error
    assert out.text.startswith(
        "That is a one-time or verification code field: melo never types into"
    )
    assert "The user had tab t1 for 1 s and gave it back to melo." in out.text
    assert "The page as it is now:" in out.text and "<<page " in out.text
    assert [a for a in driver.acts] == [], "melo typed nothing"
    assert (
        SECRET in driver.typed() and CHINESE in driver.typed()
    ), "the person's text reached the page"
    assert SECRET not in out.text and CHINESE not in out.text
    assert not any(SECRET in json.dumps(a, ensure_ascii=False) for a in asks.asked)
    await provider.close()


async def test_a_sign_in_page_asks_the_person_and_melo_reads_it_again_after() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    asks = Asks()
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        viewer = await live.watch()
        framed(driver)
        the_person(provider, live, viewer, SECRET)
        out = await run(tools, "browser_follow", ref="e1")
    assert [n for n in asks.names if n.startswith("takeover:")] == [
        "takeover:https://example.com"
    ]
    assert (
        next(a for a in asks.asked if a[0].startswith("takeover:"))[1]["need"]
        == "sign-in"
    )
    assert "The user had tab t1 for 1 s and gave it back to melo." in out.text
    assert out.text.count("<<page ") == 1, "the page as the person left it, once"
    assert SECRET not in out.text
    await provider.close()


async def test_a_sign_in_page_asks_once_and_not_again_after_not_now() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    asks = Asks(deny=("takeover:",))
    with asking(asks):
        first = await run(tools, "browser_open", url=LOGIN)
        again = await run(tools, "browser_read")
        typed = await run(tools, "browser_type", ref="e5", text="x")
    takeovers = [n for n in asks.names if n.startswith("takeover:")]
    assert takeovers == ["takeover:https://example.com", "takeover:https://example.com"]
    assert "they did not take it over" in first.text and "<<page " in first.text
    assert "did not take" not in again.text, "not asked again on the site"
    assert typed.is_error and "melo never types into passwords" in typed.text
    await provider.close()


async def test_with_no_panel_the_password_field_is_refused_as_before() -> None:
    driver = browser()
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {
        t.definition.name: t for t in await provider.start()
    }  # no console asked for the live view
    asks = Asks()
    with asking(asks):
        await run(tools, "browser_open", url=LOGIN)
        out = await run(tools, "browser_type", ref="e2", text=SECRET)
    assert out.is_error and "Ask the person to fill it in themselves" in out.text
    assert not [n for n in asks.names if n.startswith("takeover:")]
    assert driver.inputs == []
    await provider.close()


async def test_nothing_the_person_types_reaches_a_log_a_result_or_a_meta(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)
    metas: list[str] = []
    for text in (SECRET, CHINESE):
        live.send_input(viewer, {"type": "compose", "text": text})
        live.send_input(viewer, {"type": "text", "text": text})
        live.send_input(viewer, {"type": "paste", "text": text})
        for char in text:
            live.send_input(
                viewer,
                {
                    "type": "key",
                    "action": "down",
                    "key": char,
                    "code": "",
                    "keyCode": 0,
                },
            )
    live.send_input(
        viewer, {"type": "text", "text": "x" * 5000 + SECRET}
    )  # refused for its size
    await live.give(viewer)
    while True:
        try:
            async with asyncio.timeout(0.05):
                item = await viewer.next()
        except TimeoutError:
            break
        if item and item[0] == "meta":
            metas.append(json.dumps(item[1], ensure_ascii=False))
    out = await run(tools, "browser_read")
    assert SECRET in driver.typed() and CHINESE in driver.typed()
    for text in (SECRET, CHINESE, SECRET[:6]):
        assert text not in caplog.text
        assert text not in out.text
        assert not any(text in m for m in metas)
    await provider.close()
