"""Drive session commands through the real terminal; no real clipboard writes."""

from __future__ import annotations

import base64
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub


def test_title_memory_usage_copy_and_undo_in_a_real_terminal(
    melo: Callable[..., Melo],
    tmp_path: Path,
) -> None:
    # The PTY is consumed by this test, not an actual terminal emulator.
    term = melo(env={"SSH_TTY": "synthetic-test-pty"})
    term.see("Ask melo")
    term.type("/title A terminal command test\r")
    term.see("Title: A terminal command test")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Worked for")
    term.type("/memory window\r")
    term.see("Working memory:")
    term.see("Long-term memory:")
    term.type("/usage\r")
    term.see("Totals include undone turns")
    term.type("/copy\r")
    term.see("Sent the answer of turn 1")
    term.see("OSC 52")
    match = re.search(rb"\x1b\]52;c;([^\x07]+)\x07", term.output)
    assert match is not None, term.text()
    assert base64.b64decode(match[1], validate=True).decode() == (
        "Hello! I am **melo**, running on a stub model."
    )
    term.type("/undo\r")
    term.see("Undid turn 1: hello")
    term.see("Files the tools changed stay changed.")
    assert term.quit() == 0, term.text()
    path = tmp_path / ".cache/localmelo/cli/sessions.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        assert db.execute("SELECT title FROM sessions").fetchone() == (
            "A terminal command test",
        )
        assert db.execute("SELECT undone FROM turns").fetchone() == (1,)
