"""The status bar says its figures in words on a real terminal, with and without colors."""

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

WORDS = re.compile(r"context (\d+% used|[\d.]+k? tokens)")


def _bar(term: Melo, columns: int) -> str:
    """The bottom row of the screen, which is the status bar, once it has settled."""
    term.read_until(lambda out: False, 1.0)  # what the turn's last event draws
    screen = pyte.Screen(columns, 30)
    pyte.Stream(screen).feed(term.output.decode("utf-8", "replace"))
    return [row.rstrip() for row in screen.display if row.strip()][-1]


@pytest.mark.parametrize("no_color", ["1", ""], ids=["NO_COLOR", "colors"])
@pytest.mark.parametrize("columns", [100, 80])
def test_the_bar_says_how_full_the_context_is(
    melo: Callable[..., Melo], columns: int, no_color: str
) -> None:
    term = melo(columns=columns, env={"NO_COLOR": no_color})
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Worked for")
    term.see("context ")
    bar = _bar(term, columns)
    assert len(bar) <= columns
    assert WORDS.search(bar), bar
    # The compact figures are developer mode's, and a cache the server did not
    # report is not shown at all.
    assert "ctx " not in bar and "cache" not in bar
    assert term.quit() == 0


def test_a_narrow_bar_drops_the_words_whole_and_never_cuts_them(
    melo: Callable[..., Melo],
) -> None:
    term = melo(columns=44)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Worked for")
    bar = _bar(term, 44)
    assert len(bar) <= 44 and bar.startswith("  stub-model")
    assert ("context" in bar) == bool(WORDS.search(bar)), bar
    assert "…" not in bar and "..." not in bar, bar  # a word is left out, never cut
    assert term.quit() == 0
