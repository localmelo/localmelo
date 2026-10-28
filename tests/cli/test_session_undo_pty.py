"""A terminal undo/retry round-trip keeps the previous live turn."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from .test_e2e_pty import Melo
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub


def test_retry_after_undo_targets_the_undone_message_in_a_real_terminal(
    melo: Callable[..., Melo],
    tmp_path: Path,
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Worked for")
    term.type("/usage\r")
    term.see("Totals include undone turns")
    term.type("hello again\r")
    assert term.read_until(lambda output: output.count(b"Worked for") >= 2), term.text()
    assert term.quit() == 0, term.text()
    term = melo("-c")
    term.see("Resumed")
    term.type("/usage\r")
    term.see("Totals include undone turns")
    term.type("/undo\r")
    term.see("Undid turn 2: hello again.")
    term.type("/usage\r")
    assert term.read_until(
        lambda output: output.count(b"Totals include undone turns") >= 2
    ), term.text()
    term.type("/retry\r")
    term.see("Retrying turn 2: hello again")
    term.see("Worked for")
    assert term.quit() == 0, term.text()
    path = tmp_path / ".cache/localmelo/cli/sessions.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        assert db.execute(
            "SELECT turn, user_text, undone FROM turns ORDER BY turn"
        ).fetchall() == [
            (1, "hello", 0),
            (2, "hello again", 1),
            (3, "hello again", 0),
        ]
