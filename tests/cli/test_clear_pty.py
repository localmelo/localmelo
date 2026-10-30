"""``/clear`` in a real terminal: the screen holds the welcome and the input box.

The terminal is read through pyte, history included, so what was wiped from the
screen or its scrollback is not there to find. ``NO_COLOR`` is left unset: with it
a terminal gets no escape codes and ``/clear`` is ``/new``.
"""

from __future__ import annotations

from collections.abc import Callable

import pyte

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub

COLORS = {"NO_COLOR": ""}
WIPE = b"\x1b[3J"


def _screen(term: Melo, columns: int = 100) -> str:
    """The visible rows and the scrollback, as the terminal holds them now."""
    display = pyte.HistoryScreen(columns, 30, history=500)
    pyte.Stream(display).feed(term.output.decode("utf-8", "replace"))
    rows = [*display.history.top, *display.history.bottom]
    lines = ["".join(row[c].data for c in range(columns)) for row in rows]
    return "\n".join([*(line.rstrip() for line in lines), *display.display])


def _pause(term: Melo, seconds: float) -> None:
    """Wait while reading: a terminal nobody reads stops the program's writes."""
    term.read_until(lambda _: False, timeout_s=seconds)


def _redrawn(term: Melo) -> None:
    """Wait until the welcome and the input box are drawn after the last wipe."""
    assert term.read_until(
        lambda _: WIPE in term.output
        and b"Try: " in term.output.rsplit(WIPE, 1)[1]
        and b"Ask melo" in term.output.rsplit(WIPE, 1)[1]
    ), term.text()
    _pause(term, 0.7)  # the bar settles


def test_clear_leaves_only_the_welcome_the_input_box_and_the_bar(
    melo: Callable[..., Melo],
) -> None:
    term = melo(env=COLORS)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    _pause(term, 1)  # the turn wraps up; a command waits for it
    before = _screen(term)
    assert "hello" in before and "Hello! I am melo" in before
    term.type("/clear\r")
    _redrawn(term)
    after = _screen(term)
    assert "hello" not in after and "Worked for" not in after, after
    assert "/clear" not in after and "New chat ─" not in after, after
    rows = [row for row in after.split("\n") if row.strip()]
    assert any("Try: " in row for row in rows) and any(
        "Ask melo" in row for row in rows
    )
    assert rows[0].lstrip().startswith(("▄", "▀", "█", "melo")) or "melo " in rows[0]
    # Nothing from before comes back when the screen is drawn again.
    term.type("abc")
    _pause(term, 0.7)
    typed = _screen(term)
    assert "abc" in typed and "Worked for" not in typed and "Hello! I am" not in typed
    term.type("\x15")  # ctrl+u
    assert term.quit() == 0, term.text()


def test_the_next_message_after_clear_starts_a_new_chat(
    melo: Callable[..., Melo],
) -> None:
    term = melo(env=COLORS)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    _pause(term, 1)  # the turn wraps up; a command waits for it
    term.type("/clear\r")
    _redrawn(term)
    term.type("hello\r")
    assert term.read_until(lambda out: out.count(b"Worked for") >= 2), term.text()
    screen = _screen(term)
    assert screen.count("Hello! I am melo") == 1, screen  # only the new chat's
    assert term.quit() == 0, term.text()


def test_clear_without_color_is_new_and_writes_no_escape_codes(
    melo: Callable[..., Melo],
) -> None:
    term = melo()  # NO_COLOR=1, as the other terminal tests
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    _pause(term, 1)
    term.type("/clear\r")
    term.see("─ New chat ─")  # the rule, and the page stays
    assert WIPE not in term.output and b"\x1b[2J" not in term.output
    assert "Hello! I am melo" in _screen(term)
    assert term.quit() == 0, term.text()


def test_new_keeps_the_page_in_color_too(melo: Callable[..., Melo]) -> None:
    term = melo(env=COLORS)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    _pause(term, 1)
    term.type("/new\r")
    term.see("─ New chat ─")
    assert WIPE not in term.output and b"\x1b[2J" not in term.output
    assert "Hello! I am melo" in _screen(term)
    assert term.quit() == 0, term.text()
