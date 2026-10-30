"""``/clear`` starts a new chat on a cleared screen; ``/new`` keeps the page.

The session tests use the scripted model and a recording output; the output
tests write to a real pseudo-terminal, where escape codes are wanted.
"""

from __future__ import annotations

import os
import sys

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.text import Text

from localmelo.support.cli.output import CLEAR_SCREEN, TerminalOut

from .test_session import Setup, answer
from .test_session import setup as setup

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

RULE = "─ New chat ─"


def _welcome(setup: Setup) -> list[int]:
    """Draw a one-line welcome on a /clear; the list counts the draws."""
    draws: list[int] = []

    def draw() -> None:
        draws.append(1)
        setup.out.print(Text("WELCOME"))

    setup.session.welcome = draw
    return draws


async def _chat_once(setup: Setup) -> None:
    setup.session.enter("one")
    await setup.idle()
    setup.session.approvals._policy._granted.add("shell_exec")


async def test_clear_starts_a_new_chat_on_an_empty_screen_with_the_welcome(
    setup: Setup,
) -> None:
    session = setup.session
    draws = _welcome(setup)
    await _chat_once(setup)
    assert "Paris." in setup.out.text()
    assert session.enter("/clear") is None
    await setup.idle()
    assert setup.out.cleared == 1 and draws == [1]
    assert setup.out.text().strip() == "WELCOME"  # nothing from before, no rule
    assert session.runner is None and session.approvals.allowed_tools == frozenset()
    assert session.agent.conversation() == [] and session.title == ""


async def test_the_next_message_after_clear_opens_a_new_stored_chat(
    setup: Setup,
) -> None:
    session = setup.session
    _welcome(setup)
    await _chat_once(setup)
    session.enter("/clear")
    await setup.idle()
    setup.factory.scripts["m1"] = [answer("again")]
    session.agent = await setup.factory.build()
    session.enter("two")
    await setup.idle()
    assert len(await setup.store.list_sessions()) == 2  # a new chat, not the old one


async def test_new_keeps_the_page_and_prints_the_rule(setup: Setup) -> None:
    draws = _welcome(setup)
    await _chat_once(setup)
    assert setup.session.enter("/new") is None
    await setup.idle()
    assert setup.out.cleared == 0 and draws == []
    assert "Paris." in setup.out.text() and RULE in setup.out.text()
    assert setup.session.approvals.allowed_tools == frozenset()


async def test_clear_is_new_where_the_terminal_takes_no_escape_codes(
    setup: Setup,
) -> None:
    setup.out.can_clear = False
    draws = _welcome(setup)
    await _chat_once(setup)
    assert setup.session.enter("/clear") is None
    await setup.idle()
    assert setup.out.cleared == 0 and draws == []
    assert "Paris." in setup.out.text() and RULE in setup.out.text()
    assert setup.session.approvals.allowed_tools == frozenset()


async def test_clear_without_a_welcome_still_clears(setup: Setup) -> None:
    await _chat_once(setup)
    assert setup.session.enter("/clear") is None
    await setup.idle()
    assert setup.out.cleared == 1 and setup.out.text().strip() == ""


async def test_clear_waits_for_an_idle_session_like_new(setup: Setup) -> None:
    session = setup.session
    setup.factory.scripts["m1"] = []
    session.agent = await setup.factory.build()
    session.enter("wait")
    assert session.enter("/clear") == (
        "/clear is not available while a turn runs (esc stops it)."
    )
    assert setup.out.cleared == 0


async def test_clear_is_refused_while_files_are_staged(setup: Setup) -> None:
    (setup.factory.root / "input.txt").write_text("text", encoding="utf-8")
    assert setup.session.enter("/attach input.txt") is None
    await setup.idle()
    for command in ("/clear", "/new"):
        assert (
            setup.session.enter(command)
            == "Detach staged files before switching chats."
        )
    assert setup.out.cleared == 0


# ── TerminalOut: the escape codes, and where they are held back ──


def _terminal(
    *, no_color: bool = False, pipe: bool = False
) -> tuple[TerminalOut, int, int]:
    """A TerminalOut on a pseudo-terminal (or a pipe): (out, read fd, write fd)."""
    read, write = os.pipe() if pipe else os.openpty()
    file = os.fdopen(write, "w", closefd=False)
    output = Vt100_Output(file, lambda: Size(30, 100), term="xterm-256color")
    return TerminalOut(output, no_color=no_color), read, write


def _read(fd: int) -> bytes:
    os.set_blocking(fd, False)
    try:
        return os.read(fd, 65536)
    except BlockingIOError:
        return b""


def test_the_screen_is_cleared_before_what_is_printed_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    out, read, write = _terminal()
    try:
        out.print(Text("before"))
        assert out.clear_screen() is True
        out.print(Text("after"))
        written = _read(read).decode()
        assert written.index("before") < written.index(CLEAR_SCREEN)
        assert written.index(CLEAR_SCREEN) < written.index("after")
        assert CLEAR_SCREEN == "\x1b[H\x1b[2J\x1b[3J"
    finally:
        os.close(read)
        os.close(write)


@pytest.mark.parametrize("why", ["no_color", "dumb", "pipe", "detached"])
def test_no_escape_code_is_written_where_none_is_wanted(
    monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    monkeypatch.setenv("TERM", "dumb" if why == "dumb" else "xterm-256color")
    out, read, write = _terminal(no_color=why == "no_color", pipe=why == "pipe")
    if why == "detached":
        out.detach()
    try:
        assert out.clear_screen() is False
        out.print(Text("text"))
        assert b"\x1b[2J" not in _read(read) and b"\x1b[3J" not in _read(read)
    finally:
        os.close(read)
        os.close(write)
