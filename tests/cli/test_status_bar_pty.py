"""The status bar says the session's usage on a real terminal, with and without colors."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable

import pyte
import pytest

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

USAGE = re.compile(r"↑ \d+ ↓ \d+ · ctx \d+")


def _bar(term: Melo, columns: int) -> str:
    """The bottom row of the screen, which is the status bar, once it has settled."""
    term.read_until(lambda out: False, 1.0)  # what the turn's last event draws
    screen = pyte.Screen(columns, 30)
    pyte.Stream(screen).feed(term.output.decode("utf-8", "replace"))
    return [row.rstrip() for row in screen.display if row.strip()][-1]


@pytest.mark.parametrize("no_color", ["1", ""], ids=["NO_COLOR", "colors"])
@pytest.mark.parametrize("columns", [100, 80])
def test_the_bar_says_the_tokens_and_the_context(
    melo: Callable[..., Melo], columns: int, no_color: str
) -> None:
    term = melo(columns=columns, env={"NO_COLOR": no_color})
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Worked for")
    term.see("ctx ")
    bar = _bar(term, columns)
    assert len(bar) <= columns
    assert USAGE.search(bar) and bar.endswith("│ hello"), bar
    # The welcome and /status name the model and the folder; a cache the
    # server did not report is not shown at all.
    assert "stub-model" not in bar and "demo" not in bar and "cache" not in bar
    assert term.quit() == 0


def test_a_narrow_bar_drops_the_title_and_keeps_the_figures_whole(
    melo: Callable[..., Melo],
) -> None:
    term = melo(columns=28)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")  # the answer wraps at this width
    term.see("ctx ")
    bar = _bar(term, 28)
    assert len(bar) <= 28 and USAGE.search(bar), bar
    assert "hello" not in bar
    assert "…" not in bar and "..." not in bar, bar  # a figure is never cut
    assert term.quit() == 0
