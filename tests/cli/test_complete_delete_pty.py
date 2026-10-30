"""The command list follows the draft while characters are deleted, on a real terminal (#277).

``/exi`` shows ``/exit``; deleting the ``i`` must show the ``/ex`` matches, and
deleting the ``/`` must close the list. Same for the names after an ``@``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pyte
import pytest

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

BACKSPACE = b"\x7f"
COLUMNS = 100


def _rows(term: Melo) -> list[str]:
    screen = pyte.Screen(COLUMNS, 30)
    pyte.Stream(screen).feed(term.output.decode("utf-8", "replace"))
    return [row.rstrip() for row in screen.display]


def _prompt(term: Melo) -> str:
    """The input row: the one that starts with the prompt mark."""
    return next((row for row in reversed(_rows(term)) if row.startswith("› ")), "")


def _menu(term: Melo) -> list[str]:
    """The names in the menu rows, which sit between the lower rule and the bar."""
    rows = _rows(term)
    rules = [i for i, row in enumerate(rows) if row.startswith("──────")]
    if len(rules) < 2:
        return []
    below = rows[rules[-1] + 1 :]
    return [row.split()[0] for row in below if row.strip() and "│" not in row]


def _wait(term: Melo, check: Callable[[], bool]) -> None:
    assert term.read_until(lambda _out: check(), 10), "\n".join(_rows(term))


def _wait_menu(term: Melo, prompt: str, names: list[str]) -> None:
    def shown() -> bool:
        return _prompt(term) == prompt and _menu(term)[: len(names)] == names

    _wait(term, shown)


def _wait_no_menu(term: Melo, prompt: str) -> None:
    _wait(term, lambda: _prompt(term).startswith(prompt) and _menu(term) == [])
    term.read_until(lambda _out: False, 0.5)  # a late menu would be drawn by now
    assert _menu(term) == [], "\n".join(_rows(term))


def _clear_and_quit(term: Melo) -> int:
    """Ctrl+U empties the draft (Ctrl+D quits only on an empty one)."""
    term.type(b"\x15")
    return term.quit()


def test_the_command_list_follows_the_draft_while_characters_are_deleted(
    melo: Callable[..., Melo],
) -> None:
    term = melo(columns=COLUMNS)
    term.see("Ask melo")
    term.type("/exi")
    _wait_menu(term, "› /exi", ["/exit"])
    term.type(BACKSPACE)
    _wait_menu(term, "› /ex", ["/exit", "/expand"])
    term.type(BACKSPACE)
    _wait_menu(term, "› /e", ["/exit", "/expand"])
    term.type(BACKSPACE)
    _wait_menu(term, "› /", ["/model", "/new", "/status", "/help", "/exit"])
    term.type(BACKSPACE)
    _wait_no_menu(term, "› Ask melo")  # the empty draft shows its hint
    assert _clear_and_quit(term) == 0


def test_esc_closes_the_list_until_the_next_change(melo: Callable[..., Melo]) -> None:
    term = melo(columns=COLUMNS)
    term.see("Ask melo")
    term.type("/exi")
    _wait_menu(term, "› /exi", ["/exit"])
    term.type(BACKSPACE)
    _wait_menu(term, "› /ex", ["/exit", "/expand"])
    term.type(b"\x1b")
    _wait_no_menu(term, "› /ex")
    term.type(b"\x1b[D")  # moving the cursor alone opens nothing
    term.read_until(lambda _out: False, 0.5)
    assert _menu(term) == [], "\n".join(_rows(term))
    term.type(b"\x1b[C")
    term.type(BACKSPACE)  # the next change does
    _wait_menu(term, "› /e", ["/exit", "/expand"])
    assert _clear_and_quit(term) == 0


def test_the_attachment_names_follow_the_draft_while_characters_are_deleted(
    melo: Callable[..., Melo],
) -> None:
    term = melo(columns=COLUMNS)
    (term.workspace / "notes.txt").write_text("x", encoding="utf-8")
    (term.workspace / "other.md").write_text("x", encoding="utf-8")
    term.see("Ask melo")
    term.type("look @no")
    _wait_menu(term, "› look @no", ["@notes.txt"])
    term.type(BACKSPACE)
    _wait_menu(term, "› look @n", ["@notes.txt"])
    term.type(BACKSPACE)
    _wait_menu(term, "› look @", ["@notes.txt", "@other.md"])
    term.type(BACKSPACE)
    _wait_no_menu(term, "› look")
    assert _clear_and_quit(term) == 0
