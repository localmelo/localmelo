"""Asking the terminal for its colours: the reply parser, the time cap, DA1, a pty."""

from __future__ import annotations

import os
import pty
import select
import termios
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from localmelo.support.cli.termcolors import (
    QUERY,
    QUERY_TIMEOUT_S,
    TerminalColors,
    blend,
    contrast,
    fit,
    nearest256,
    parse_reply,
    query_colors,
)

BG = b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07"
FG = b"\x1b]10;rgb:d4d4/d4d4/d4d4\x1b\\"
DA1 = b"\x1b[?62;22c"
ENV = {"TERM": "xterm-256color"}


@pytest.mark.parametrize(
    ("data", "colors", "done"),
    [
        (BG + FG + DA1, TerminalColors("#1e1e1e", "#d4d4d4"), True),
        (b"\x1b]11;rgb:ff/fa/f0\x07" + DA1, TerminalColors("#fffaf0"), True),
        (b"\x1b]11;rgba:0/f/8/f\x1b\\", TerminalColors("#00ff88"), False),
        (BG + b"\x1b]10;rgb:d4d4", TerminalColors("#1e1e1e"), False),  # not yet
        (DA1, TerminalColors(), True),  # answers DA1 only: no OSC 11 support
        (b"\x1b]11;?\x07", TerminalColors(), False),  # an echo of the query
        (b"", TerminalColors(), False),
    ],
)
def test_the_reply_parser(data: bytes, colors: TerminalColors, done: bool) -> None:
    assert parse_reply(data) == (colors, done)


@pytest.fixture
def terminal() -> Iterator[tuple[int, int]]:
    """A pty: the master is the terminal, the slave is melo's stdin and stdout."""
    master, slave = pty.openpty()
    yield master, slave
    os.close(master)
    os.close(slave)


def _answer(master: int, reply: bytes, delay: float = 0.0) -> threading.Thread:
    """Read the query on *master*, then write *reply* after *delay*."""

    def run() -> None:
        seen = b""
        while not seen.endswith(b"\x1b[c"):
            ready, _, _ = select.select([master], [], [], 2)
            if not ready:
                return
            seen += os.read(master, 1024)
        assert seen.endswith(QUERY)
        time.sleep(delay)
        if reply:
            os.write(master, reply)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _timed(call: Callable[[], TerminalColors | None]) -> tuple[object, float]:
    start = time.monotonic()
    result = call()
    return result, time.monotonic() - start


def _pending(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0.05)[0])


def test_a_terminal_that_answers_gives_its_exact_colors(
    terminal: tuple[int, int],
) -> None:
    master, slave = terminal
    before = termios.tcgetattr(slave)
    _answer(master, BG + FG + DA1)
    colors, took = _timed(lambda: query_colors(slave, slave, ENV, timeout=1.0))
    assert colors == TerminalColors("#1e1e1e", "#d4d4d4")
    assert took < 0.5  # DA1 ends the wait, not the cap
    # The terminal's mode is restored (the kernel may add PENDIN when canonical
    # mode comes back), and nothing is left to read.
    after = termios.tcgetattr(slave)
    pendin = getattr(termios, "PENDIN", 0)
    after[3] &= ~pendin
    before[3] &= ~pendin
    assert after == before
    assert not _pending(slave)


def test_a_terminal_without_osc_11_ends_the_wait_with_da1(
    terminal: tuple[int, int],
) -> None:
    master, slave = terminal
    _answer(master, DA1)
    colors, took = _timed(lambda: query_colors(slave, slave, ENV, timeout=1.0))
    assert colors is None and took < 0.5


def test_a_silent_terminal_costs_at_most_the_cap(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    _answer(master, b"")
    colors, took = _timed(lambda: query_colors(slave, slave, ENV))
    assert colors is None
    assert QUERY_TIMEOUT_S <= took < QUERY_TIMEOUT_S + 0.2


def test_a_reply_after_the_cap_is_not_used(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    answering = _answer(master, BG + FG + DA1, delay=QUERY_TIMEOUT_S + 0.3)
    colors, took = _timed(lambda: query_colors(slave, slave, ENV))
    assert colors is None and took < QUERY_TIMEOUT_S + 0.2
    answering.join()


def test_a_reply_cut_at_the_cap_is_used_and_its_rest_discarded(
    terminal: tuple[int, int],
) -> None:
    # The background arrived, DA1 never does: what came is used, and the
    # input is flushed so no half of a reply reaches the prompt.
    master, slave = terminal
    _answer(master, BG + b"\x1b]10;rgb:d4d4")
    colors, took = _timed(lambda: query_colors(slave, slave, ENV))
    assert colors == TerminalColors("#1e1e1e")
    assert took < 3 * QUERY_TIMEOUT_S + 0.2
    assert not _pending(slave)


def test_only_a_terminal_on_both_sides_is_asked(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    read, write = os.pipe()
    try:
        assert query_colors(read, slave, ENV) is None
        assert query_colors(slave, write, ENV) is None
        assert query_colors(slave, slave, {"TERM": "dumb"}) is None
    finally:
        os.close(read)
        os.close(write)
    # Nothing was written to the terminal.
    assert not _pending(master)


def test_the_arithmetic() -> None:
    assert round(contrast("#000000", "#ffffff"), 1) == 21.0
    assert blend("#ffffff", "#000000", 0.5) == "#808080"
    moved = fit("#91978a", ["#4a4a4a"], 4.5)
    assert contrast(moved, "#4a4a4a") >= 4.5
    assert fit("#91978a", ["#000000"], 4.5) == "#91978a"  # already reads
    assert nearest256("#ffffff") == 231
    assert nearest256("#b5cf9f", ["#000000"], 4.5) == 151
