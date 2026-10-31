"""Shift+Tab switches the chat between Ask and Auto, and the bar always names the mode.

The key does what ``/approvals ask|auto`` does: in the terminal's own chat in
place, in the main chat and its side chats on the gateway. It never reaches
Deny or ``--yolo``, never answers a question, and leaves one muted line.
"""

from __future__ import annotations

import asyncio

import pytest

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support.cli import theme
from localmelo.support.cli.app import MODE_KEY_MIN_WIDTH, StatusBar, status_fragments
from localmelo.support.cli.approvals import (
    APPROVAL_KEY_GUARD_S,
    MODE_SWITCHED,
    TerminalApprover,
    next_mode,
)
from localmelo.support.cli.session import MODE_BUSY, MODE_YOLO
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE

from .test_main_chat import MAIN, FakeGateway
from .test_main_chat import chat as chat
from .test_session import Setup, _until
from .test_session import setup as setup
from .ui_helpers import ENTER, ESC, SHIFT_TAB, FakeController, running_app

ASK_LINE = "Ask: melo asks before running commands or changing files."
AUTO_LINE = "Auto: routine steps run; risky ones are blocked with the reason."


# ── What the key switches to ──


@pytest.mark.parametrize(
    ("mode", "target"),
    [("ask", "auto"), ("auto", "ask"), ("deny", "ask"), ("yolo", None)],
)
def test_the_key_goes_ask_auto_ask_and_never_to_deny_or_yolo(
    mode: str, target: str | None
) -> None:
    assert next_mode(mode) == target  # type: ignore[arg-type]
    assert set(MODE_SWITCHED) == {"ask", "auto"}
    assert MODE_SWITCHED["ask"] == ASK_LINE and MODE_SWITCHED["auto"] == AUTO_LINE


# ── The bar ──


def _bar(mode: str, width: int, *, style: str = "class:muted", **kw: object) -> list:
    bar = StatusBar("Fix", mode, mode_style=style, mode_key="shift+tab", **kw)  # type: ignore[arg-type]
    return status_fragments(bar, width, "…")


def _text(fragments: list) -> str:
    return "".join(fragment[1] for fragment in fragments)


def test_the_bar_names_each_mode_with_its_key_at_the_right_edge() -> None:
    ask = _bar("ask", 80)
    assert _text(ask).rstrip().endswith("ask · shift+tab") and cells(_text(ask)) == 80
    assert ("class:muted", "ask") in ask and ("class:muted", " · shift+tab") in ask
    auto = _bar("auto", 80, style="class:mode_auto")
    assert _text(auto).endswith("auto · shift+tab") and cells(_text(auto)) == 80
    # Auto has its own colour; its key stays as quiet as Ask's.
    assert ("class:mode_auto", "auto") in auto and (
        "class:muted",
        " · shift+tab",
    ) in auto


def test_a_bar_below_sixty_columns_keeps_the_mode_and_drops_the_key() -> None:
    kept = _text(_bar("auto", MODE_KEY_MIN_WIDTH, style="class:mode_auto"))
    assert kept.endswith("auto · shift+tab") and cells(kept) == MODE_KEY_MIN_WIDTH
    short = _text(_bar("auto", MODE_KEY_MIN_WIDTH - 1, style="class:mode_auto"))
    assert short.endswith("auto") and "shift+tab" not in short
    assert cells(short) == MODE_KEY_MIN_WIDTH - 1


@pytest.mark.parametrize("width", range(12, 100, 3))
@pytest.mark.parametrize("mode", ["ask", "auto", "deny"])
def test_the_bar_with_a_key_never_overflows_and_keeps_the_mode(
    mode: str, width: int
) -> None:
    fragments = _bar(mode, width, usage=())
    line = _text(fragments)
    assert cells(line) == width, (width, line)
    assert " … " not in line and not line.strip().startswith("…")
    if width > len(mode) + 3:
        assert mode in line


def test_ascii_draws_the_hint_with_its_own_dot() -> None:
    bar = StatusBar("Fix", "auto", mode_key="shift+tab")
    line = _text(status_fragments(bar, 80, "...", f" {ASCII.dot} ", " | "))
    assert line.endswith("auto - shift+tab") and line.isascii()
    unicode_line = _text(status_fragments(bar, 80, "…", f" {UNICODE.dot} ", " │ "))
    assert unicode_line.endswith("auto · shift+tab")


def test_a_bar_with_no_key_draws_the_mode_alone() -> None:
    bar = StatusBar("Fix", "⚠ YOLO: no approvals", mode_style="class:yolo")
    line = _text(status_fragments(bar, 80, "…"))
    assert line.endswith("⚠ YOLO: no approvals") and "shift+tab" not in line


@pytest.mark.parametrize(
    ("depth", "polarity"),
    [
        ("none", "unknown"),
        ("16", "unknown"),
        ("16", "dark"),
        ("16", "light"),
        ("256", "dark"),
        ("truecolor", "dark"),
        ("truecolor", "light"),
    ],
)
def test_auto_stands_out_from_the_muted_ask_at_every_depth(
    depth: str, polarity: str
) -> None:
    style = theme.Palette(depth=depth, polarity=polarity).pt_style  # type: ignore[arg-type]
    auto = style.get_attrs_for_style_str("class:mode_auto")
    ask = style.get_attrs_for_style_str("class:muted")
    # Never by colour alone (NO_COLOR, colour-blind eyes): weight and colour both differ.
    assert auto.bold and not ask.bold
    assert (auto.color, auto.dim) != (ask.color, ask.dim)
    if depth != "none":
        assert auto.color and auto.color != ask.color
    assert not auto.dim
    # The transcript's line names Auto the same way: bold, and not muted.
    rich = theme.Palette(depth=depth, polarity=polarity).rich_theme.styles  # type: ignore[arg-type]
    assert rich["mode_auto"].bold and rich["mode_auto"] != rich["muted"]


# ── The session: the key and /approvals share one path ──


async def test_the_key_goes_ask_auto_ask_with_one_quiet_line_each(setup: Setup) -> None:
    session, out = setup.session, setup.out
    assert session.cycle_mode() is None
    await _until(lambda: AUTO_LINE in out.text())
    assert session.approvals.mode == "auto"
    assert session.status_bar().mode == "auto" and setup.app.invalidated
    # One line, Auto's name styled as the bar's, the rest muted.
    line = session._mode_line("auto")
    assert line.plain == AUTO_LINE
    assert [(span.start, span.end, span.style) for span in line.spans] == [
        (0, 5, "mode_auto"),
        (5, len(AUTO_LINE), "muted"),
    ]
    assert session.cycle_mode() is None
    await _until(lambda: ASK_LINE in out.text())
    assert session.approvals.mode == "ask" and session.status_bar().mode == "ask"
    assert out.text().count(AUTO_LINE) == 1 and out.text().count(ASK_LINE) == 1
    # The line is a status line, not a command's output: nothing hangs it.
    assert "└" not in out.text()


async def test_deny_goes_to_ask_and_yolo_is_never_left(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.approvals.set_mode("deny")
    assert session.cycle_mode() is None
    await _until(lambda: ASK_LINE in out.text())
    assert session.approvals.mode == "ask"
    session.approvals.mode = "yolo"
    assert session.cycle_mode() is None
    assert MODE_YOLO in out.text()
    assert session.approvals.mode == "yolo"  # unchanged, and told
    assert session.cycle_mode() is None
    assert out.text().count(MODE_YOLO) == 2 and AUTO_LINE not in out.text()


async def test_the_key_and_the_command_leave_the_same_state(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("/approvals auto")
    await _until(lambda: "Approvals: auto" in out.text())
    assert session.cycle_mode() is None
    await _until(lambda: ASK_LINE in out.text())
    assert session.approvals.mode == "ask"
    session.enter("/approvals auto")
    await _until(lambda: out.text().count("Approvals: auto") == 2)
    assert session.approvals.mode == "auto"


async def test_the_key_waits_for_the_turn_to_end_like_the_command(setup: Setup) -> None:
    session, out = setup.session, setup.out
    session.enter("hello")
    assert session.state == "running"
    assert session.cycle_mode() == MODE_BUSY
    assert session.approvals.mode == "ask" and AUTO_LINE not in out.text()
    await setup.idle()
    assert session.cycle_mode() is None
    await _until(lambda: session.approvals.mode == "auto")


async def test_a_press_while_the_switch_is_under_way_is_ignored(setup: Setup) -> None:
    session = setup.session
    assert session.cycle_mode() is None
    assert session.cycle_mode() is None  # no second spawn: the first has not run
    await _until(lambda: not session._mode_switching)
    assert session.approvals.mode == "auto"


# ── The app: when the key is heard ──


async def test_the_key_reaches_the_session_from_the_message_box() -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        app.type(SHIFT_TAB)
        await app.wait(lambda: controller.mode_presses == 1)
        app.type("a draft" + SHIFT_TAB)
        await app.wait(lambda: controller.mode_presses == 2)
        assert app.ui.buffer.text == "a draft"  # nothing is typed by the key


async def test_a_hint_says_why_the_key_did_nothing() -> None:
    controller = FakeController(mode_hint=MODE_BUSY)
    async with running_app(controller) as app:
        app.type(SHIFT_TAB)
        await app.wait(lambda: MODE_BUSY in app.text())


async def test_the_key_does_not_answer_a_question() -> None:
    controller = FakeController()
    async with running_app(controller) as app:
        asking = asyncio.ensure_future(app.ui.ask("Model", "1"))
        await app.wait(lambda: "Model [1]:" in app.text())
        app.type(SHIFT_TAB)
        await app.settle()
        assert controller.mode_presses == 0 and not asking.done()
        app.type(ENTER)
        assert await asyncio.wait_for(asking, 3) == "1"


async def test_the_key_does_not_touch_a_list_or_an_approval_panel() -> None:
    controller = FakeController(state="running")
    approver = TerminalApprover()
    async with running_app(controller, approver=approver) as app:
        picking = asyncio.ensure_future(
            app.ui.pick("Switch", ["one", "two"], default=0, escape=1)
        )
        await app.wait(lambda: "Switch" in app.text())
        app.type(SHIFT_TAB)
        await app.settle()
        assert controller.mode_presses == 0 and not picking.done()
        app.type(ENTER)
        assert await asyncio.wait_for(picking, 3) == 0
        # An approval question takes the keys: the mode is not switched under it.
        request = ApprovalRequest(
            request_id="r1",
            run_id="run",
            call_id="c1",
            tool_name="shell_exec",
            arguments={"command": "ls -1"},
            preview="",
            workspace_root="/tmp/w",
        )
        asking = asyncio.ensure_future(approver.ask(request))
        await app.wait(lambda: "Run a shell command?" in app.text())
        await asyncio.sleep(APPROVAL_KEY_GUARD_S + 0.1)
        app.type(SHIFT_TAB)
        await app.settle()
        assert controller.mode_presses == 0 and not asking.done()
        app.type("1")
        assert (await asyncio.wait_for(asking, 3)).allow


async def test_with_the_menu_open_the_key_still_goes_back_in_the_list() -> None:
    from localmelo.support.cli.completion import SlashCompleter

    controller = FakeController()
    async with running_app(controller, completer=SlashCompleter()) as app:
        app.type("/")
        await app.wait(lambda: app.ui.buffer.complete_state is not None)
        app.type("\x1b[B\x1b[B")  # down, down: the third row
        await app.wait(lambda: app.ui.buffer.text == "/new")
        app.type(SHIFT_TAB)  # up one row, as before: not a mode switch
        await app.wait(lambda: app.ui.buffer.text != "/new")
        assert controller.mode_presses == 0
        # With the menu closed the same key is the mode's.
        app.type(ESC)
        await app.wait(lambda: app.ui.buffer.complete_state is None)
        app.type(SHIFT_TAB)
        await app.wait(lambda: controller.mode_presses == 1)


# ── Help ──


def test_help_names_the_key() -> None:
    from localmelo.support.cli.commands import KEYS, MODE_KEY_LINE

    assert MODE_KEY_LINE in KEYS
    assert MODE_KEY_LINE.startswith("shift+tab") and len(MODE_KEY_LINE) <= 70


# ── The main chat: the key goes to the chat on the gateway ──


def _patches(gateway: FakeGateway) -> list[tuple[str, str, object]]:
    return [r for r in gateway.requests if r[0] == "PATCH"]


async def test_in_the_main_chat_the_key_switches_the_chat_on_the_gateway(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session, out = setup.session, setup.out
    assert session.status_bar().mode == "ask"
    assert session.cycle_mode() is None
    await _until(lambda: AUTO_LINE in out.text())
    assert _patches(gateway) == [
        ("PATCH", f"/api/sessions/{MAIN}", {"approvals": "auto"})
    ]
    assert gateway.sessions[MAIN]["approvals"] == "auto"
    bar = session.status_bar()
    assert (bar.mode, bar.mode_style, bar.mode_key) == (
        "auto",
        "class:mode_auto",
        "shift+tab",
    )
    # The terminal's own chat is untouched: its Approvals stay where they were.
    assert session.approvals.mode == "ask"
    assert session.cycle_mode() is None
    await _until(lambda: ASK_LINE in out.text())
    assert gateway.sessions[MAIN]["approvals"] == "ask"
    assert session.status_bar().mode == "ask"


async def test_the_main_chat_command_and_key_send_the_same_request(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session, out = setup.session, setup.out
    assert session.enter("/approvals auto") is None
    await _until(lambda: "Approvals: auto" in out.text())
    assert "works only in the terminal's own chat" not in out.text()
    via_command = _patches(gateway)
    assert session.cycle_mode() is None
    await _until(lambda: ASK_LINE in out.text())
    assert via_command == [("PATCH", f"/api/sessions/{MAIN}", {"approvals": "auto"})]
    assert _patches(gateway)[1:] == [
        ("PATCH", f"/api/sessions/{MAIN}", {"approvals": "ask"})
    ]
    assert session.enter("/approvals") is None
    await _until(lambda: out.text().count("Approvals: ask") >= 1)


async def test_the_main_chat_has_no_deny_or_forget(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    for command in ("/approvals deny", "/approvals forget", "/approvals nope"):
        assert setup.session.enter(command) is None
    await _until(lambda: setup.out.text().count("✗ Use /approvals ask or auto") == 3)
    assert not _patches(gateway)


async def test_a_refused_switch_says_so_and_changes_nothing(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    gateway.refuse_mode = True
    assert setup.session.cycle_mode() is None
    await _until(lambda: "✗ Could not switch to auto" in setup.out.text())
    assert "A task's mode is fixed." in setup.out.text()
    assert AUTO_LINE not in setup.out.text()
    assert setup.session.status_bar().mode == "ask"
    # The next press is heard again.
    gateway.refuse_mode = False
    assert setup.session.cycle_mode() is None
    await _until(lambda: AUTO_LINE in setup.out.text())


async def test_a_chat_opens_in_the_mode_the_gateway_stores_for_it(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    remote = setup.session.remote
    assert remote is not None
    gateway.sessions[MAIN]["approvals"] = "auto"
    await remote.main()  # already open: no change
    await remote.open(type(remote.chat)(MAIN, "Main chat", main=True))
    assert setup.session.status_bar().mode == "auto"
    # Another window's switch reaches the bar with the next look at the chat list.
    gateway.sessions[MAIN]["approvals"] = "ask"
    main, side = await remote.client.chats()
    remote._read_mode([main, *side])
    assert setup.session.status_bar().mode == "ask"


async def test_a_new_side_chat_starts_in_the_mode_picked_before_its_first_message(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session, out = setup.session, setup.out
    assert session.enter("/side") is None
    await _until(lambda: "your first message starts it" in out.text())
    assert session.status_bar().mode == "ask"
    assert session.cycle_mode() is None
    await _until(lambda: AUTO_LINE in out.text())
    assert not _patches(gateway)  # nothing exists on the gateway yet
    assert session.status_bar().mode == "auto"
    assert session.enter("plan it") is None
    await setup.idle()
    created = next(r for r in gateway.requests if r[:2] == ("POST", "/api/sessions"))
    assert created[2]["approvals"] == "auto"
    assert session.status_bar().mode == "auto"


async def test_a_side_chat_without_a_pick_leaves_the_mode_to_the_gateway(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("/side") is None
    await _until(lambda: "your first message starts it" in setup.out.text())
    assert session.enter("plan it") is None
    await setup.idle()
    created = next(r for r in gateway.requests if r[:2] == ("POST", "/api/sessions"))
    assert "approvals" not in created[2]


async def test_status_in_the_main_chat_names_the_chats_mode(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, _ = chat
    session, out = setup.session, setup.out
    assert session.cycle_mode() is None
    await _until(lambda: AUTO_LINE in out.text())
    session.enter("/status all")
    await _until(lambda: "version    melo " in out.text())
    assert "approvals  auto" in out.text()
