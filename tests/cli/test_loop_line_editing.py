"""The `melo ""` loop on a real terminal: editing a line with wide characters.

A new pseudo-terminal does not set IUTF8, so without readline the
terminal's own Backspace removes one byte of a multi-byte character and
input() then fails to decode the line. These tests type Chinese text,
delete part of it and check what reaches the model.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from localmelo.tests.cli.test_direct_mode_process import (
    FakeChatServer,
    _env,
    chat_server,  # noqa: F401  (a fixture)
)

pty = pytest.importorskip("pty")

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

PROMPT = "❯ ".encode()
BACKSPACE = b"\x7f"

# Runs melo as `python -m localmelo` would, with the readline module hidden.
NO_READLINE = (
    "import sys, runpy; sys.modules['readline'] = None; "
    "sys.argv = ['melo'] + sys.argv[1:]; "
    "runpy.run_module('localmelo', run_name='__main__', alter_sys=True)"
)


class Terminal:
    """A process on a pseudo-terminal, and everything it has written so far."""

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self._master, slave = os.openpty()
        self.process = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        os.close(slave)
        self.output = b""

    def read_until(self, check: Callable[[bytes], bool], timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if check(self.output):
                return True
            ready, _, _ = select.select([self._master], [], [], 0.05)
            if ready:
                try:
                    self.output += os.read(self._master, 4096)
                except OSError:  # the process closed the terminal
                    return check(self.output)
        return check(self.output)

    def type(self, data: bytes) -> None:
        os.write(self._master, data)
        time.sleep(0.15)

    def wait(self, timeout_s: float) -> int | None:
        """Wait for the process to exit, reading what it writes meanwhile.

        readline restores the terminal with TCSADRAIN, which waits until
        the other end has read the output, so the reading must go on.
        """
        self.read_until(lambda _: self.process.poll() is not None, timeout_s)
        return self.process.poll()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        os.close(self._master)


def _prompts(output: bytes) -> int:
    return output.count(PROMPT)


def _user_text(server: FakeChatServer) -> str:
    messages = server.bodies[0]["messages"]
    return str(next(m for m in messages if m["role"] == "user")["content"])


def _type_and_exit(term: Terminal, server: FakeChatServer, line: bytes) -> None:
    assert term.read_until(lambda out: _prompts(out) >= 1, 30), term.output
    term.type("你好吗".encode())
    term.type(BACKSPACE * 2)
    term.type(line + b"\r")
    assert term.read_until(lambda out: _prompts(out) >= 2, 30), term.output
    term.type(b"/exit\r")
    assert term.wait(30) == 0, term.output


def test_backspace_deletes_whole_wide_characters(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    argv = [sys.executable, "-I", "-m", "localmelo"]
    argv += ["--base-url", chat_server.base_url, ""]
    term = Terminal(argv, _env(tmp_path), tmp_path)
    try:
        _type_and_exit(term, chat_server, b"ok")
    finally:
        term.close()
    assert term.process.returncode == 0, term.output
    assert b"Traceback" not in term.output
    assert _user_text(chat_server) == "你ok"


def test_a_line_that_is_not_utf8_is_asked_for_again(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    # Without readline, the terminal's Backspace leaves half a character.
    argv = [sys.executable, "-I", "-c", NO_READLINE]
    argv += ["--base-url", chat_server.base_url, ""]
    term = Terminal(argv, _env(tmp_path), tmp_path)
    try:
        assert term.read_until(lambda out: _prompts(out) >= 1, 30), term.output
        term.type("你好吗".encode())
        term.type(BACKSPACE * 2)
        term.type(b"ok\r")
        assert term.read_until(lambda out: b"not valid UTF-8" in out, 30), term.output
        assert term.read_until(lambda out: _prompts(out) >= 2, 30), term.output
        term.type(b"again\r")
        assert term.read_until(lambda out: _prompts(out) >= 3, 30), term.output
        term.type(b"/exit\r")
        assert term.wait(30) == 0, term.output
    finally:
        term.close()
    assert term.process.returncode == 0, term.output
    assert b"Traceback" not in term.output
    assert chat_server.requests >= 1
    assert _user_text(chat_server) == "again"
