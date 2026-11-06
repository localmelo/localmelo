"""The person takes over melo's browser: who has a tab, melo's steps that wait, the person's input, the card.

No real browser runs. The owners and the input mapping are tested alone; the
relay and the browser tools over a scripted browser that streams and takes
input (it records what it was sent). Every test that types checks that the
text went to the browser and nowhere else: not to a log, a step result, a
question or the live view's messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from localmelo.melo.agent._pause import Hold
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.pause import PAUSE
from localmelo.support.browser import takeover
from localmelo.support.browser.driver import (
    Act,
    ActResult,
    Element,
    PointHit,
    RawPage,
    View,
)
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
    is_human_check,
    names_a_check,
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


def owners(
    changes: list[int] | None = None, heard: list[str] | None = None
) -> tuple[Owners, list[float]]:
    clock = [100.0]
    found = Owners(
        on_change=(lambda: changes.append(1)) if changes is not None else None,
        on_take=(
            (lambda tab: heard.append(f"take {tab}")) if heard is not None else None
        ),
        on_give=(
            (lambda back: heard.append(f"give {back.tab} {back.how}"))
            if heard is not None
            else None
        ),
        clock=lambda: clock[0],
    )
    return found, clock


# ── Who has a tab ──


def test_a_tab_is_melos_until_the_person_takes_it_and_back_when_they_give_it() -> None:
    changes: list[int] = []
    heard: list[str] = []
    found, clock = owners(changes, heard)
    assert found.owner("t1") == "melo" and found.owner(None) == "melo"
    assert found.take("t1") is True and found.owner("t1") == "you"
    assert found.take("t1") is False, "taking a tab one has changes nothing"
    clock[0] += 42
    back = found.give("t1", "closed")
    assert back is not None and (back.tab, round(back.seconds), back.how) == (
        "t1",
        42,
        "closed",
    )
    assert found.owner("t1") == "melo" and found.give("t1") is None
    assert [r.tab for r in found.returned()] == ["t1"] and found.returned() == []
    assert len(changes) == 2 and heard == ["take t1", "give t1 closed"]


def test_where_the_person_has_a_tab_and_no_clock_ever_gives_it_back() -> None:
    found, clock = owners()
    found.take("t1", "app", form=True)
    where = found.where("t1")
    assert where is not None and (where.place, where.form, where.failed) == (
        "app",
        True,
        False,
    )
    found.move("t1", "window")
    assert found.where("t1") == dataclasses.replace(where, place="window")
    clock[0] += 86400
    assert found.owner("t1") == "you", "nothing counts down"
    assert found.where("t2") is None


def test_a_closed_tab_and_the_end_of_the_chat_leave_nothing_held() -> None:
    heard: list[str] = []
    found, _ = owners(heard=heard)
    found.take("t1")
    found.take("t2")
    found.give("t2")
    found.forget("t2")
    assert found.returned() == [], "nothing is told about a tab that is gone"
    found.forget("t1")
    found.take("t3")
    found.clear()
    assert found.held == frozenset() and found.returned() == []
    assert heard == ["take t1", "take t2", "give t2 done", "take t3"]


async def test_a_step_waits_until_the_person_gives_the_tab_back() -> None:
    found = Owners()
    found.take("t1")
    asyncio.get_running_loop().call_later(0.05, found.give, "t1")
    async with asyncio.timeout(2):
        await found.wait_for_melo("t1")
        await found.wait_for_melo("t2")
    assert found.owner("t1") == "melo"


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
        # A human check is melo's to press: the page alone raises no card.
        (
            {"url": HOME, "title": "Just a moment...", "text": "", "sensitive": []},
            "",
        ),
        (
            {
                "url": HOME,
                "title": "Example",
                "text": "Please verify you are human",
                "sensitive": [],
            },
            "",
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
    assert viewer.meta["place"] == "app", "no window here: the panel is the place"
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


async def test_a_panel_that_goes_away_leaves_the_page_the_persons() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    framed(driver)
    await live.take(viewer)
    await live.release(viewer)
    await asyncio.sleep(0.1)
    assert provider.owners.owner("t1") == "you", "only Done gives it back"
    again = await live.watch()
    assert again.target[0] == "t1"
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


async def test_a_step_on_the_persons_page_waits_as_long_as_they_need() -> None:
    driver = browser()
    provider, tools, live = await started(driver)
    await run(tools, "browser_open", url=HOME)
    viewer = await live.watch()
    await live.take(viewer)
    step = asyncio.ensure_future(run(tools, "browser_read"))
    await asyncio.sleep(0.3)
    assert not step.done(), "no time limit hands the page back"
    assert provider.hand_back() is True
    out = await asyncio.wait_for(step, 2)
    assert out.text.startswith("The user had tab t1 for")
    assert "gave it back to melo" in out.text and "<<page " in out.text
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


async def test_a_step_naming_another_tab_never_acts_on_the_persons_tab() -> None:
    # The person took the current tab over from the panel; its reading is kept,
    # so a ref of it is still real. A step naming another tab, a tab of no chat
    # or the person's tab spelt differently is refused or waits: the tab whose
    # owner was checked is the tab the step works on.
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
    assert driver.calls[calls:] == [], "no page was opened, read or touched"
    assert driver.open_in == [], "nothing was loaded into a tab of before"
    asyncio.get_running_loop().call_later(0.05, provider.hand_back)
    spelt = await run(tools, "browser_follow", ref="e1", tab=" T1 ")
    assert not spelt.is_error
    assert spelt.text.startswith("The user had tab t1 for")
    assert "melo did not do this step" in spelt.text, "it waited, then read t1 again"
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

    def taken(tab: str, place: Any = "window", *, form: bool | None = None) -> bool:
        changed = original(tab, place, form=form)
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
    assert preview == "take over example.com: melo needs you for a one-time code"
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


# ── A real window for the person, and the run's hold ──


class Windowed(Takeable):
    """A browser that can put a page in a real window (or fail to), as Chrome does."""

    def __init__(self, *pages: Any, opens: bool = True, unsent: bool = False) -> None:
        super().__init__(*pages)
        self.opens = opens
        self.unsent = unsent
        self.shown: list[str] = []
        self.hidden = 0
        self.on_closed: Callable[[], None] | None = None

    async def show_window(self, tab: str, on_closed: Callable[[], None]) -> bool:
        self.shown.append(tab)
        self.on_closed = on_closed
        return self.opens

    async def hide_window(self) -> None:
        self.hidden += 1

    async def unsent_input(self, tab: str) -> bool:
        return self.unsent


def windowed(**kwargs: Any) -> Windowed:
    return Windowed(*browser().pages.values(), **kwargs)


@contextlib.contextmanager
def held_run() -> Iterator[list[tuple[str, Any]]]:
    """The run's hold, as the agent sets it around a tool call; what it reports."""
    said: list[tuple[str, Any]] = []
    hold = Hold(
        lambda what, detail: said.append(("paused", detail)),
        lambda what, how, held_s: said.append(("resumed", how)),
    )
    token = PAUSE.set(hold)
    try:
        yield said
    finally:
        PAUSE.reset(token)


async def test_take_over_puts_the_page_in_a_real_window_and_closing_it_is_done() -> (
    None
):
    driver = windowed()
    provider, tools, live = await started(driver)
    with asking(Asks()), held_run() as said:
        await run(tools, "browser_open", url=HOME)
        step = asyncio.ensure_future(run(tools, "browser_follow", ref="e1"))
        await until(lambda: bool(said))
        where = provider.owners.where("t1")
        assert driver.shown == ["t1"] and where is not None and where.place == "window"
        assert said == [
            ("paused", {"tab": "t1", "place": "window", "site": "example.com"})
        ]
        assert driver.on_closed is not None
        driver.on_closed()  # the person closed the window
        out = await asyncio.wait_for(step, 2)
    assert "closed its Chrome window, which gives it back to melo" in out.text
    assert "The page as it is now:" in out.text
    assert said[-1] == ("resumed", "closed") and driver.hidden == 1
    await provider.close()


async def test_a_form_the_person_began_stays_in_the_app_unless_they_pick_chrome() -> (
    None
):
    driver = windowed(unsent=True)
    provider, tools, live = await started(driver)
    asks = Asks()
    with asking(asks), held_run() as said:
        await run(tools, "browser_open", url=HOME)
        step = asyncio.ensure_future(run(tools, "browser_follow", ref="e1"))
        await until(lambda: bool(said))
        assert driver.shown == [], "the form stays: no window"
        assert said[0][1]["place"] == "app" and said[0][1]["form"] is True
        provider.show_to_person()  # "Open in Chrome, clear the form"
        await until(lambda: driver.shown == ["t1"])
        await until(lambda: said[-1][1].get("place") == "window")
        assert provider.hand_back() is True
        out = await asyncio.wait_for(step, 2)
    card = next(a for a in asks.asked if a[0].startswith("takeover:"))
    assert card[1]["form"] is True
    assert "gave it back to melo" in out.text and said[-1] == ("resumed", "done")
    await provider.close()


async def test_a_window_that_cannot_open_puts_the_page_in_the_app() -> None:
    driver = windowed(opens=False)
    provider, tools, live = await started(driver)
    viewer = await live.watch()
    with asking(Asks()), held_run() as said:
        await run(tools, "browser_open", url=HOME)
        step = asyncio.ensure_future(run(tools, "browser_follow", ref="e1"))
        await until(lambda: bool(said))
        assert said[0][1]["place"] == "app" and said[0][1]["failed"] is True
        framed(driver)
        await until(lambda: viewer.meta.get("failed") is True)
        assert viewer.meta["place"] == "app"
        provider.hand_back()
        await asyncio.wait_for(step, 2)
    await provider.close()


async def test_a_stop_while_the_person_has_the_page_takes_it_back() -> None:
    driver = windowed()
    provider, tools, live = await started(driver)
    with asking(Asks()), held_run():
        await run(tools, "browser_open", url=HOME)
        step = asyncio.ensure_future(run(tools, "browser_follow", ref="e1"))
        await until(lambda: driver.shown == ["t1"])
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step
    await until(lambda: driver.hidden == 1)
    assert provider.owners.owner("t1") == "melo"
    await provider.close()


class Wrote(Asks):
    """The person wrote to the model while a take-over card waited: their message ended it."""

    async def __call__(
        self, name: str, arguments: Any, preview: str
    ) -> ApprovalDecision:
        self.asked.append((name, arguments, preview))
        if name.startswith("takeover:"):
            return ApprovalDecision(False, "", origin="words")
        return ApprovalDecision(True, remember=True)


async def test_a_message_that_ends_a_sign_in_card_is_no_not_now_and_the_card_comes_again() -> (
    None
):
    driver = browser()
    provider, tools, live = await started(driver)
    wrote = Wrote()
    with asking(wrote):
        first = await run(tools, "browser_open", url=LOGIN)
        assert not provider._not_now and not provider._asked_you, "nothing was kept"
        again = await run(tools, "browser_read")
    assert "sent a message while melo waited" in first.text
    assert "they did not take it over" not in first.text and "<<page " in first.text
    assert [n for n in wrote.names if n.startswith("takeover:")] == [
        "takeover:https://example.com"
    ] * 2, "the next read of the page offers the card again"
    assert "sent a message while melo waited" in again.text
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


CHECK = "https://newassets.hcaptcha.com/captcha/v1/box"


@pytest.mark.parametrize(
    ("address", "in_table", "named"),
    [
        (CHECK, True, True),
        ("https://challenges.cloudflare.com/cdn-cgi/x", True, True),
        ("https://www.recaptcha.net/recaptcha/api2/anchor", True, True),
        # A label that starts with "captcha" names a check in words and lowers nothing.
        ("https://captcha.example.com/x", False, True),
        ("https://www.cloudflare.com/", False, False),
        ("https://example.com/hcaptcha.com", False, False),
        ("https://hcaptcha.com.example.net/x", False, False),
        # What stands before an "@" is a user part, not the host.
        ("https://newassets.hcaptcha.com:pw@evil.example/x", False, False),
        ("https://hcaptcha.com@evil.example/x", False, False),
        ("https://user@newassets.hcaptcha.com/x", True, True),
        ("https://newassets.hcaptcha.com./x", True, True),
        ("https://accounts.example.org/box", False, False),
        ("not an address", False, False),
    ],
)
def test_only_a_row_of_the_table_is_a_human_check_and_a_label_may_only_name_one(
    address: str, in_table: bool, named: bool
) -> None:
    assert is_human_check(address) is in_table
    assert names_a_check(address) is named


class Solving(Takeable):
    """A browser whose page shows an answer field filled once ``passes`` presses were made."""

    def __init__(self, *pages: Any, passes: int | None = None) -> None:
        super().__init__(*pages)
        self.passes = passes

    async def act(self, tab: str | None, act: Act) -> ActResult:
        done = await super().act(tab, act)
        if self.passes is not None and len(self.acts) >= self.passes:
            self.pages[HOME] = dataclasses.replace(self.pages[HOME], solved=True)
        return done


def check_page() -> RawPage:
    return page(HOME, "Check in.", button("I am human", site=CHECK, fp="iframe|1"))


async def test_melo_presses_a_human_check_like_any_control_and_says_what_it_did() -> (
    None
):
    """By point and by ref: the click's own question, no card; the step says passed or still there."""
    inside = Element("button", "", at=-1, fp="bare")
    driver = Solving(check_page(), passes=2)
    driver.shot_view = View("", "d1", 1000.0, 500.0)
    driver.hits[(500, 150)] = PointHit("frame", inside, site=CHECK, check=True)
    provider, tools, live = await started(driver)
    asks = Asks()
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        await run(tools, "browser_screenshot")
        point = await run(tools, "browser_click_point", x=500, y=300)
        ref = await run(tools, "browser_click", ref="e2")
    assert len(driver.acts) == 2 and driver.acts[0].point == (500, 150)
    assert not point.is_error and not ref.is_error
    assert point.text.startswith(
        "Clicked a human check box from newassets.hcaptcha.com at (500, 300) (via point)."
    )
    assert "The check is still there." in point.text
    assert "Clicked e2" in ref.text and "The check passed." in ref.text
    # The ordinary click question, named: no take-over card, no "box melo cannot check" card.
    assert [n for n in asks.names if n.startswith("browser-act:")]
    assert not [n for n in asks.names if n.startswith(("takeover:", "browser-point:"))]
    assert "melo never passes" not in point.text + ref.text
    await provider.close()


async def test_a_check_still_there_after_the_presses_is_offered_to_the_person() -> None:
    inside = Element("button", "", at=-1, fp="bare")
    driver = Solving(check_page())
    driver.shot_view = View("", "d1", 1000.0, 500.0)
    driver.hits[(500, 150)] = PointHit("frame", inside, site=CHECK, check=True)
    provider, tools, live = await started(driver)
    asks = Asks()
    with asking(asks):
        await run(tools, "browser_open", url=HOME)
        viewer = await live.watch()
        framed(driver)
        the_person(provider, live, viewer)
        await run(tools, "browser_screenshot")
        steps = [
            await run(tools, "browser_click_point", x=500, y=300)
            for _ in range(takeover.CHECK_PRESSES)
        ]
    for early in steps[:-1]:
        assert "The check is still there." in early.text
    last = steps[-1].text
    assert f"The check is still there after {takeover.CHECK_PRESSES} presses." in last
    assert "The user had tab t1 for 1 s and gave it back to melo." in last
    cards = [a for a in asks.asked if a[0].startswith("takeover:")]
    assert [(n, a["need"]) for n, a, _ in cards] == [
        ("takeover:https://example.com", "captcha")
    ], "one card, after the last press, never before"
    assert len(driver.acts) == takeover.CHECK_PRESSES
    await provider.close()
    # With nobody to ask, the step still says it, and carries on.
    driver = Solving(check_page())
    driver.shot_view = View("", "d1", 1000.0, 500.0)
    driver.hits[(500, 150)] = PointHit("frame", inside, site=CHECK, check=True)
    provider, tools, _ = await started(driver)
    with asking(Asks(deny=("takeover:",))):
        await run(tools, "browser_open", url=HOME)
        await run(tools, "browser_screenshot")
        for _ in range(takeover.CHECK_PRESSES):
            out = await run(tools, "browser_click_point", x=500, y=300)
    assert not out.is_error and "presses" in out.text
    assert "they did not take it over" in out.text
    await provider.close()


@pytest.mark.parametrize(
    "site",
    [
        "https://captcha.example.com/x",  # named a check by its label only
        CHECK,  # a table host, but nothing proved a frame of it sits there
        "https://ads.example.net/x",  # the browser's own word: another site's frame sits there
    ],
)
async def test_no_page_can_lower_the_card_for_a_box_melo_cannot_check(
    site: str,
) -> None:
    """A box is a check only where the browser proved a table host's frame is there; else the card, every time."""
    inside = Element("button", "", at=-1, fp="bare")
    # The page also loaded a check's frame, and the read names where that is.
    shown = dataclasses.replace(
        check_page(), unreached=(CHECK,), aims=((CHECK, 190, 423),)
    )
    driver = Solving(shown)
    driver.shot_view = View("", "d1", 1000.0, 500.0)
    driver.hits[(500, 150)] = PointHit("frame", inside, site=site)
    provider, tools, live = await started(driver)
    asks = Asks()
    with asking(asks):
        opened = await run(tools, "browser_open", url=HOME)
        await run(tools, "browser_screenshot")
        # The second press is on a site whose click grant the first one left: the card stays.
        outs = [await run(tools, "browser_click_point", x=500, y=300) for _ in range(2)]
    assert "a human check box from newassets.hcaptcha.com at (190, 423)" in opened.text
    cards = [a for a in asks.asked if a[0].startswith("browser-point:")]
    assert len(cards) == 2 and len(driver.acts) == 2
    for _, arguments, preview in cards:
        assert arguments["target"] == "unknown"
        assert "human check" not in f"{arguments['on']} {preview}"
        assert arguments["on"].endswith("that melo cannot check")
    for out in outs:
        assert not out.is_error
        assert "human check" not in out.text.split("\n", 1)[0]
        assert "The check" not in out.text and "presses" not in out.text
    await provider.close()


async def test_a_page_that_was_replaced_is_changed_not_passed_and_the_count_goes_on() -> (
    None
):
    driver = Solving(check_page())
    provider, tools, _ = await started(driver)
    await run(tools, "browser_open", url=HOME)
    key = ("t1", "https://newassets.hcaptcha.com")
    driver.docs["t1"] = "d2"  # the page was reloaded by the press
    said = [await provider._pressed("t1", CHECK, "d1") for _ in range(2)]
    assert said[0] == (
        " The page changed (press 1 of this check): the page below is the new one."
    )
    assert "press 2 of this check" in said[1] and "passed" not in said[0] + said[1]
    assert provider._presses[key] == 2 and provider._stuck is None
    for _ in range(takeover.CHECK_PRESSES - 2):
        await provider._pressed("t1", CHECK, "d1")
    assert provider._presses[key] == takeover.CHECK_PRESSES, "kept, never started over"
    assert provider._stuck == "t1"
    # Only an answer field that holds something is a pass, and it starts the count over.
    driver.pages[HOME] = dataclasses.replace(driver.pages[HOME], solved=True)
    assert await provider._pressed("t1", CHECK, "d2") == " The check passed."
    assert key not in provider._presses
    await provider.close()


async def test_a_check_still_there_is_not_pressed_again_once_the_card_cannot_come() -> (
    None
):
    inside = Element("button", "", at=-1, fp="bare")

    def stuck() -> Solving:
        driver = Solving(check_page())
        driver.shot_view = View("", "d1", 1000.0, 500.0)
        driver.hits[(500, 150)] = PointHit("frame", inside, site=CHECK, check=True)
        return driver

    async def press_all(tools: dict[str, Any]) -> Any:
        await run(tools, "browser_open", url=HOME)
        await run(tools, "browser_screenshot")
        for _ in range(takeover.CHECK_PRESSES):
            out = await run(tools, "browser_click_point", x=500, y=300)
        return out

    # They said not now: the count stays, and the next press is refused with the reason.
    driver = stuck()
    provider, tools, _ = await started(driver)
    with asking(Asks(deny=("takeover:",))):
        await press_all(tools)
        again = await run(tools, "browser_click_point", x=500, y=300)
        read = await run(tools, "browser_read")
        ref = re.search(r"\[(e\d+) button", read.text)
        assert ref is not None
        by_ref = await run(tools, "browser_click", ref=ref.group(1))
    for refused in (again, by_ref):
        assert refused.is_error and "already offered the page" in refused.text
        assert (
            f"pressed this human check {takeover.CHECK_PRESSES} times" in refused.text
        )
    assert len(driver.acts) == takeover.CHECK_PRESSES, "nothing more was pressed"
    await provider.close()
    # With no panel to hand the page to, the same.
    driver = stuck()
    provider = BrowserToolProvider(driver)  # type: ignore[arg-type]
    tools = {t.definition.name: t for t in await provider.start()}
    with asking(Asks()):
        await press_all(tools)
        again = await run(tools, "browser_click_point", x=500, y=300)
    assert again.is_error and "nobody can take the page over here" in again.text
    assert len(driver.acts) == takeover.CHECK_PRESSES
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
