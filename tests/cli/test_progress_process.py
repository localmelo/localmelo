"""The real CLI with stderr on a pseudo-terminal: the progress line.

Each test starts ``python -m localmelo`` against a fake chat server, with
stdout on a pipe and stderr on a pseudo-terminal 200 columns wide, as in
a terminal where the result is piped on. A pseudo-terminal turns each
``\\n`` into ``\\r\\n``. Every process a test starts is killed should the
test fail.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from localmelo.__main__ import SIGNAL_CLEANUP_S
from localmelo.support.progress_line import ERASE
from localmelo.tests.cli.test_direct_mode_process import (
    FakeChatServer,
    _env,
    _process_gone,
    _tool_call,
    _wait_until,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pseudo-terminals and POSIX signals"
)

COLUMNS = 200


@pytest.fixture
def chat_server() -> Iterator[FakeChatServer]:
    server = FakeChatServer()
    try:
        yield server
    finally:
        server.close()


@dataclass
class Done:
    returncode: int
    stdout: bytes
    stderr: str  # what reached the pseudo-terminal

    def draws(self) -> list[str]:
        """The lines drawn, in order (each draw starts with ERASE)."""
        return [part for part in self.stderr.split(ERASE)[1:] if part]


class PtyReader:
    """Reads a pseudo-terminal's master side until every writer is gone."""

    def __init__(self, master: int) -> None:
        self._master = master
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            try:
                data = os.read(self._master, 4096)
            except OSError:  # EIO once the last writer closed it
                return
            if not data:
                return
            with self._lock:
                self._chunks.append(data)

    def text(self) -> str:
        with self._lock:
            return b"".join(self._chunks).decode(errors="replace")

    def type(self, keys: bytes) -> None:
        """Type *keys* on the terminal's keyboard."""
        os.write(self._master, keys)

    def finish(self) -> str:
        self._thread.join(timeout=10)
        return self.text()


def _set_columns(fd: int, columns: int) -> None:
    import fcntl
    import struct
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, columns, 0, 0))


@contextlib.contextmanager
def _melo_on_pty(
    chat_server: FakeChatServer,
    tmp_path: Path,
    args: Sequence[str],
    *,
    color: bool = False,
    stderr_on_pty: bool = True,
    typed: bytes = b"",
) -> Iterator[tuple[subprocess.Popen[bytes], PtyReader | None]]:
    """Start melo with *args* and a query; yield it and the terminal's reader.

    *typed* is typed on the terminal's keyboard before melo starts.
    """
    env = _env(tmp_path)
    env["TERM"] = "xterm-256color"
    env.pop("NO_COLOR", None)
    if not color:
        env["NO_COLOR"] = "1"
    master, slave = os.openpty()
    _set_columns(slave, COLUMNS)
    if typed:
        os.write(master, typed)
    try:
        melo = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-m",
                "localmelo",
                "--base-url",
                chat_server.base_url,
                "--tool-exposure",
                "all",
                *args,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=slave if stderr_on_pty else subprocess.PIPE,
            start_new_session=True,
            env=env,
            cwd=tmp_path,
        )
    finally:
        os.close(slave)
    reader = PtyReader(master)
    try:
        yield melo, reader if stderr_on_pty else None
    finally:
        if melo.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(melo.pid, signal.SIGKILL)
            melo.communicate(timeout=10)
        reader.finish()
        os.close(master)


def _run(
    chat_server: FakeChatServer,
    tmp_path: Path,
    args: Sequence[str],
    *,
    color: bool = False,
    stderr_on_pty: bool = True,
) -> Done:
    with _melo_on_pty(
        chat_server, tmp_path, args, color=color, stderr_on_pty=stderr_on_pty
    ) as (melo, reader):
        out, err = melo.communicate(timeout=60)
        stderr = reader.finish() if reader is not None else err.decode()
    return Done(melo.returncode, out, stderr)


def _wait_for_text(
    melo: subprocess.Popen[bytes], reader: PtyReader, pattern: str, timeout_s: float
) -> str:
    deadline = time.monotonic() + timeout_s
    while not re.search(pattern, reader.text()):
        if melo.poll() is not None or time.monotonic() > deadline:
            pytest.fail(f"no {pattern!r} on the terminal: {reader.text()!r}")
        time.sleep(0.05)
    return reader.text()


# A session: a test run that fails, an edit, a test run that passes, done.
SESSION = [
    _tool_call("shell_exec", {"command": "echo '1 failed, 2 passed' # pytest"}),
    _tool_call("python_exec", {"code": "print('patched')"}),
    _tool_call("shell_exec", {"command": "echo '3 passed' # pytest"}),
]


def _queue(chat_server: FakeChatServer, replies: list[dict[str, Any]]) -> None:
    chat_server.replies[:] = [dict(reply) for reply in replies]


# ── The line ────────────────────────────────────────────────────────────────


def test_the_line_on_a_terminal_and_the_same_stdout_without_it(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    _queue(chat_server, SESSION)
    drawn = _run(chat_server, tmp_path, ["fix the tests"], color=True)
    _queue(chat_server, SESSION)
    off = _run(chat_server, tmp_path, ["--progress", "off", "fix the tests"])

    assert (drawn.returncode, off.returncode) == (0, 0)
    assert drawn.stdout == off.stdout == b"done\n"  # byte for byte
    # Drawn, dim by default, with the step and the tests seen.
    assert drawn.stderr.startswith(ERASE + "\x1b[2mstarting · ")
    plain = [re.sub(r"\x1b\[[0-9]*m", "", draw) for draw in drawn.draws()]
    assert any(re.fullmatch(r"step 2 · Verify · tests 2/3 · \d+s", d) for d in plain)
    assert any(
        re.fullmatch(r"step 2 · Verify · Python 0:0\d · tests 2/3 · \d+s", d)
        for d in plain
    )
    assert any(re.fullmatch(r"step 4 · Finishing · tests 3/3 · \d+s", d) for d in plain)
    # Erased at the end, and never a newline.
    assert drawn.stderr.endswith(ERASE)
    assert "\n" not in drawn.stderr
    # Without the line, nothing on the terminal at all.
    assert off.stderr == ""


def test_the_line_is_erased_before_the_status_line(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    _queue(chat_server, SESSION)
    done = _run(chat_server, tmp_path, ["--max-steps", "2", "fix the tests"])
    assert done.returncode == 2
    assert done.stdout == b""
    status = "melo: run ended with status budget_exhausted: "
    before, found, after = done.stderr.rpartition(ERASE + status)
    assert found, done.stderr
    assert ERASE not in after and after.endswith("\r\n")
    assert before.count(ERASE) >= 2  # drawn at least once before


@pytest.mark.parametrize("stderr_on_pty", [True, False], ids=["tty", "pipe"])
def test_nothing_drawn_for_json_output(
    chat_server: FakeChatServer, tmp_path: Path, stderr_on_pty: bool
) -> None:
    _queue(chat_server, SESSION)
    done = _run(
        chat_server,
        tmp_path,
        ["--output-format", "json", "fix the tests"],
        stderr_on_pty=stderr_on_pty,
    )
    assert done.returncode == 0
    assert json.loads(done.stdout)["status"] == "completed"
    assert done.stderr == ""


def test_nothing_drawn_without_a_terminal(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    _queue(chat_server, SESSION)
    done = _run(chat_server, tmp_path, ["fix the tests"], stderr_on_pty=False)
    assert (done.returncode, done.stdout, done.stderr) == (0, b"done\n", "")


def test_the_timer_runs_while_python_exec_sleeps_in_its_child(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    """Without a workspace root, python_exec code runs in a child process."""
    code = "import time\ntime.sleep(2)\nprint('slept')"
    _queue(chat_server, [_tool_call("python_exec", {"code": code})])
    done = _run(chat_server, tmp_path, ["sleep a while"])
    assert done.returncode == 0
    timers = [re.search(r"Python (\d+:\d\d) · (\d+)s", draw) for draw in done.draws()]
    shown = {(m.group(1), m.group(2)) for m in timers if m is not None}
    # At least two redraws during the sleep, each with a later time.
    assert len({tool for tool, _ in shown}) >= 2, done.draws()
    assert len({run for _, run in shown}) >= 2, done.draws()


def test_a_warning_goes_on_its_own_line_above_the_line(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    code = (
        "import logging, time\n"
        "logging.getLogger('localmelo.tests').warning('careful now')\n"
        "time.sleep(1.2)\n"
    )
    _queue(chat_server, [_tool_call("python_exec", {"code": code})])
    done = _run(chat_server, tmp_path, ["warn me"])
    assert done.returncode == 0
    assert done.stdout == b"done\n"
    before, found, after = done.stderr.partition(ERASE + "careful now\r\n")
    assert found, done.stderr
    assert before.startswith(ERASE + "starting")
    # Drawn again right below it, then on until the end.
    assert after.startswith(ERASE + "step 1 · Python 0:0")
    assert after.endswith(ERASE)
    assert done.stderr.count("careful now") == 1


def test_sigterm_erases_the_line_before_melo_exits(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    pid_file = tmp_path / "shell.pid"
    command = f'echo $$ > "{pid_file}"; sleep 30'
    _queue(chat_server, [_tool_call("shell_exec", {"command": command})])
    shell: int | None = None
    try:
        with _melo_on_pty(chat_server, tmp_path, ["wait for it"]) as (melo, reader):
            assert reader is not None
            _wait_for_text(melo, reader, r"echo 0:0\d", 30)
            assert _wait_until(
                lambda: pid_file.exists() and pid_file.read_text().endswith("\n")
            )
            shell = int(pid_file.read_text())
            os.killpg(melo.pid, signal.SIGTERM)
            out, _ = melo.communicate(timeout=30)
            stderr = reader.finish()
        assert melo.returncode == -signal.SIGTERM, stderr
        assert out == b""
        assert stderr.endswith(ERASE), stderr
        assert "Traceback" not in stderr
        assert _wait_until(lambda: _process_gone(shell))  # melo stopped it
    finally:
        if shell is not None and not _process_gone(shell):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(shell, signal.SIGKILL)


# Ctrl+S and Ctrl+Q: a terminal stops and resumes its output (IXON).
XOFF, XON = b"\x13", b"\x11"


def test_sigterm_ends_melo_while_the_terminal_takes_no_output(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    """After Ctrl+S, the line fills the terminal's output queue (1 KiB on
    macOS), and a write then waits until Ctrl+Q. The run goes on meanwhile,
    and SIGTERM still ends melo within SIGNAL_CLEANUP_S."""
    reads = []
    for n in range(9):
        (tmp_path / f"notes{n}.txt").write_text(f"note {n}\n")
        reads.append(_tool_call("file_read", {"path": f"notes{n}.txt"}))
    _queue(chat_server, [*reads, _tool_call("shell_exec", {"command": "sleep 30"})])
    with _melo_on_pty(chat_server, tmp_path, ["read the notes"], typed=XOFF) as (
        melo,
        reader,
    ):
        assert reader is not None
        try:
            # Every step reached the model, with nothing drawn meanwhile.
            assert _wait_until(
                lambda: chat_server.requests >= 10, timeout_s=30
            ), f"{chat_server.requests} requests"
            assert reader.text() == ""
            os.killpg(melo.pid, signal.SIGTERM)
            started = time.monotonic()
            melo.wait(timeout=SIGNAL_CLEANUP_S + 2)
            assert time.monotonic() - started < SIGNAL_CLEANUP_S
        finally:
            with contextlib.suppress(OSError):  # EIO once melo has exited
                reader.type(XON)
        assert melo.returncode == -signal.SIGTERM


# ── history.db ──────────────────────────────────────────────────────────────

_ID = re.compile(r"\b[0-9a-f]{32}\b")
_DURATION_MS = re.compile(r'"duration_ms": [0-9.e+-]+')


def _history(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    """history.db's tasks, steps and llm_calls, with ids and timings masked.

    Ids become ``id0``, ``id1``, ... by first appearance; ``timestamp`` and
    ``duration_s`` columns and ``duration_ms`` values become ``<t>``.
    """
    ids: dict[str, str] = {}

    def mask(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        value = _ID.sub(lambda m: ids.setdefault(m.group(0), f"id{len(ids)}"), value)
        return _DURATION_MS.sub('"duration_ms": "<t>"', value)

    tables: dict[str, list[tuple[Any, ...]]] = {}
    with contextlib.closing(sqlite3.connect(path)) as db:
        for table, order in (
            ("tasks", "rowid"),
            ("steps", "seq"),
            ("llm_calls", "seq"),
        ):
            cursor = db.execute(f"SELECT * FROM {table} ORDER BY {order}")
            names = [column[0] for column in cursor.description]
            tables[table] = [
                tuple(
                    "<t>" if name in ("timestamp", "duration_s") else mask(value)
                    for name, value in zip(names, row, strict=True)
                )
                for row in cursor
            ]
    return tables


def test_history_is_the_same_with_and_without_the_line(
    chat_server: FakeChatServer, tmp_path: Path
) -> None:
    memory = {name: tmp_path / name for name in ("drawn", "off")}
    _queue(chat_server, SESSION)
    drawn = _run(
        chat_server,
        tmp_path,
        ["--memory-dir", str(memory["drawn"]), "fix the tests"],
    )
    _queue(chat_server, SESSION)
    off = _run(
        chat_server,
        tmp_path,
        ["--memory-dir", str(memory["off"]), "--progress", "off", "fix the tests"],
    )
    assert drawn.returncode == off.returncode == 0
    assert ERASE in drawn.stderr and off.stderr == ""
    rows = {name: _history(path / "history.db") for name, path in memory.items()}
    assert len(rows["drawn"]["steps"]) == 3
    assert len(rows["drawn"]["llm_calls"]) == 4
    assert rows["drawn"] == rows["off"]
