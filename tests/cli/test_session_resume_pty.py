"""Saved sessions survive real terminal restarts without changing direct mode."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .test_e2e_pty import Melo, _turns
from .test_e2e_pty import melo as melo
from .test_e2e_pty import stub as stub


def test_continue_picker_and_exit_hint_across_terminal_processes(
    melo: Callable[..., Melo],
    tmp_path: Path,
) -> None:
    first = melo()
    first.see("Ask melo")
    first.type("hello\r")
    first.see("Hello! I am melo, running on a stub model.")
    first.see("Worked for")
    assert first.quit() == 0
    assert "Resume it with: melo -r" in first.text()
    resumed = melo("-c")
    resumed.see('Resumed "hello"')
    resumed.see("its tool calls are not")
    assert resumed.quit() == 0
    assert len(_turns(tmp_path)) == 1
    picker = melo("-r")
    picker.see("Resume a session")
    picker.type(b"\x1b")
    picker.see("Ask melo")
    assert picker.quit() == 0
    assert len(_turns(tmp_path)) == 1


def test_unknown_resume_reference_exits_two_on_a_terminal(
    melo: Callable[..., Melo],
) -> None:
    term = melo("-r", "unknown-reference")
    assert term.wait() == 2, term.text()
    assert "no session matches 'unknown-reference'" in term.text()
