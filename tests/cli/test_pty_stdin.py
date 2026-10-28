"""A direct run on a pty that is not its controlling terminal.

On macOS, ``bash``'s start-up ``check_dev_tty()`` opens ``ttyname(stdin)``
when ``/dev/tty`` fails, without ``O_NOCTTY``.  A shell tool's child is a
session leader, so it takes that pty as its controlling terminal; when the
shell exits, the pty is hung up and melo's standard streams return EIO.
The flush at interpreter shutdown then fails and CPython exits 120 after a
completed run.

The pty here is the slave of ``os.openpty()`` given to ``Popen`` with
``start_new_session=True`` and no ``TIOCSCTTY``, so it belongs to no session
of melo's and the shell child can acquire it.  The regression is that
``shell_exec`` must not pass melo's terminal on as the child's stdin.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from localmelo.tests.cli.test_direct_mode_process import _env
from localmelo.tests.fixtures.stub_llm import StubServer

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX pseudo-terminals"
)


class _Reader:
    """Reads a pty master until every writer has closed it."""

    def __init__(self, master: int) -> None:
        self._master = master
        self._chunks: list[bytes] = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            try:
                data = os.read(self._master, 65536)
            except OSError:  # EIO once the last slave fd is closed
                return
            if not data:
                return
            self._chunks.append(data)

    def finish(self, timeout_s: float = 10.0) -> str:
        self._thread.join(timeout_s)
        return b"".join(self._chunks).decode(errors="replace")


@pytest.fixture(scope="module")
def stub() -> Iterator[StubServer]:
    server = StubServer().start()
    try:
        yield server
    finally:
        server.close()


def _open_pty() -> tuple[int, int]:
    """A pty, or skip where the environment has none (a restricted sandbox)."""
    try:
        return os.openpty()
    except PermissionError as exc:
        pytest.skip(f"no pseudo-terminal in this environment: {exc}")


def test_a_completed_run_on_a_pty_it_does_not_control_exits_0(
    stub: StubServer, tmp_path: Path
) -> None:
    env = _env(tmp_path)
    env.update(TERM="xterm-256color", NO_COLOR="1", LOCALMELO_DETECT_URLS="")
    master, slave = _open_pty()
    reader = _Reader(master)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-m",
                "localmelo",
                "--base-url",
                stub.base_url,
                "--chat-model",
                "stub-model",
                "--tool-exposure",
                "all",
                "--builtins",
                "all",
                "--",
                "run echo hi",
            ],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=tmp_path,
            start_new_session=True,
        )
    finally:
        os.close(slave)
    assert process is not None
    try:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail("melo did not finish within 60 s")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        text = reader.finish()
        os.close(master)

    assert process.returncode == 0, text
    assert "Ran echo hi." in text
    assert "Traceback" not in text
