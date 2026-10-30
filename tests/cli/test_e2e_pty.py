"""Bare melo end to end, on a pseudo-terminal, against the stub model server.

Each test starts ``python -m localmelo`` with HOME and the workspace in a
temporary folder, a config naming the stub (except the no-config test), a
100x30 terminal, ``TERM=xterm-256color`` and ``NO_COLOR=1``, and reads what
it draws. The stub picks its reply by the message typed (its scenarios'
``match``).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import signal
import sqlite3
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyte
import pytest

from localmelo.tests.cli.test_direct_mode_process import _env
from localmelo.tests.fixtures.stub_llm import StubServer

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

READY = b"Ask melo"
GUARD_S = 0.6  # past the approval panel's key guard
_ESCAPES = re.compile(
    rb"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][0-9A-B]|\x1b[=>78]"
)


class Melo:
    """``python -m localmelo`` on a pseudo-terminal of its own."""

    def __init__(
        self,
        home: Path,
        argv: list[str],
        *,
        columns: int = 100,
        iutf8: bool | None = None,
        nohup: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        self.home = home
        self.workspace = home / "demo-project"
        self.workspace.mkdir(exist_ok=True)
        extra = env or {}
        env = _env(home)
        env.update(TERM="xterm-256color", NO_COLOR="1", LOCALMELO_DETECT_URLS="")
        env.update(extra)
        self._master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, columns, 0, 0))
        if iutf8 is not None:
            attrs = termios.tcgetattr(slave)
            flag = getattr(termios, "IUTF8", 0)
            attrs[0] = attrs[0] | flag if iutf8 else attrs[0] & ~flag
            termios.tcsetattr(slave, termios.TCSANOW, attrs)
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-m", "localmelo", *argv],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=self.workspace,
            start_new_session=True,
            preexec_fn=lambda: _child_setup(nohup),
        )
        os.close(slave)
        self.output = b""
        self._open = True

    def _read(self, timeout_s: float) -> bool:
        ready, _, _ = select.select([self._master], [], [], timeout_s)
        if not ready:
            return True
        try:
            chunk = os.read(self._master, 65536)
        except OSError:
            return False
        self.output += chunk
        return bool(chunk)

    def read_until(self, check: Callable[[bytes], bool], timeout_s: float = 30) -> bool:
        deadline = time.monotonic() + timeout_s
        while not check(_ESCAPES.sub(b"", self.output)):
            if time.monotonic() > deadline or not self._read(0.05):
                return check(_ESCAPES.sub(b"", self.output))
        return True

    def see(self, text: str, timeout_s: float = 30) -> None:
        assert self.read_until(lambda out: text.encode() in out, timeout_s), self.text()

    def type(self, text: str | bytes) -> None:
        os.write(self._master, text.encode() if isinstance(text, str) else text)

    def wait(self, timeout_s: float = 30) -> int:
        deadline = time.monotonic() + timeout_s
        while self.process.poll() is None and time.monotonic() < deadline:
            self._read(0.05)
        return self.process.wait(timeout=1)

    def text(self) -> str:
        return _ESCAPES.sub(b"", self.output).decode(errors="replace")

    def quit(self, timeout_s: float = 30) -> int:
        """Ctrl+D, again until it quits: right after an answer the turn
        still wraps up, and Ctrl+D then only says that a turn is running."""
        deadline = time.monotonic() + timeout_s
        while self.process.poll() is None and time.monotonic() < deadline:
            try:
                self.type(b"\x04")
            except OSError:
                break  # melo exited and closed the terminal (EIO on macOS)
            for _ in range(10):
                self._read(0.05)
        return self.wait(5)

    def hang_up(self) -> None:
        """Close the terminal, as closing its window does."""
        os.close(self._master)
        self._open = False

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        if self._open:
            os.close(self._master)


def _child_setup(nohup: bool) -> None:
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    if nohup:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)


@pytest.fixture(scope="module")
def stub() -> Iterator[StubServer]:
    server = StubServer().start()
    try:
        yield server
    finally:
        server.close()


def _configure(home: Path, url: str) -> None:
    folder = home / ".cache" / "localmelo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.toml").write_text(
        f'chat_backend = "vllm"\nembedding_backend = "none"\n\n[vllm]\n'
        f'chat_url = "{url}"\nchat_model = "stub-model"\n',
        encoding="utf-8",
    )


@pytest.fixture
def melo(tmp_path: Path, stub: StubServer) -> Iterator[Callable[..., Melo]]:
    started: list[Melo] = []

    def start(*argv: str, configured: bool = True, **kwargs: Any) -> Melo:
        if configured:
            _configure(tmp_path, stub.base_url)
        process = Melo(tmp_path, list(argv), **kwargs)
        started.append(process)
        return process

    yield start
    for process in started:
        process.close()


def _turns(home: Path) -> list[dict[str, Any]]:
    path = home / ".cache" / "localmelo" / "cli" / "sessions.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT user_text, status, cause, result FROM turns ORDER BY started_at"
        ).fetchall()
    return [
        {**dict(row), "error": json.loads(row["result"] or "{}").get("error", "")}
        for row in rows
    ]


def test_a_message_gets_its_answer_and_ctrl_d_quits(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    assert "╭─ melo" in term.text() and "stub-model" in term.text()
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Worked for")
    assert term.quit() == 0, term.text()
    [turn] = _turns(tmp_path)
    assert (turn["user_text"], turn["status"], turn["cause"]) == (
        "hello",
        "completed",
        "run",
    )
    # No cursor position request by default.
    assert b"\x1b[6n" not in term.output


def test_a_file_write_waits_for_the_answer(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("write the notes\r")
    term.see("Write a new file?")
    assert not (tmp_path / "demo-project" / "notes.txt").exists()
    time.sleep(GUARD_S)
    term.type("1")
    term.see("I wrote notes.txt.")
    assert (
        tmp_path / "demo-project" / "notes.txt"
    ).read_text() == "remember the milk\n"
    assert "file_write  notes.txt" in term.text() and "allowed" in term.text()
    assert term.quit() == 0


def test_esc_stops_a_turn(melo: Callable[..., Melo], tmp_path: Path) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.type(b"\x1b")
    term.see("■ Stopped")
    assert term.quit() == 0
    [turn] = _turns(tmp_path)
    assert (turn["status"], turn["cause"], turn["error"]) == (
        "cancelled",
        "user_stop",
        "Stopped by the user",
    )


def test_sigterm_during_a_turn_ends_the_process_by_it(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.process.send_signal(signal.SIGTERM)
    assert term.wait() == -signal.SIGTERM
    [turn] = _turns(tmp_path)
    assert (turn["status"], turn["cause"], turn["error"]) == (
        "cancelled",
        "shutdown",
        "Cancelled by SIGTERM",
    )
    assert "Traceback" not in term.text()


def test_sigterm_when_idle_leaves_no_traceback(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    term.process.send_signal(signal.SIGTERM)
    assert term.wait() == -signal.SIGTERM
    log = (tmp_path / ".cache" / "localmelo" / "logs" / "cli.log").read_text()
    assert "Traceback" not in log and "Traceback" not in term.text()


@pytest.mark.parametrize("iutf8", [False, True])
def test_deleting_chinese_text_leaves_the_right_characters(
    melo: Callable[..., Melo], tmp_path: Path, iutf8: bool
) -> None:
    text = "一二三四五六七八九十" * 3
    term = melo(columns=41, iutf8=iutf8)
    term.see("Ask melo")
    term.type(text)
    term.type(b"\x7f" * 10)
    term.type("吗\r")
    expected = text[:20] + "吗"
    term.see("Worked for")
    assert term.quit() == 0, term.text()
    [turn] = _turns(tmp_path)
    assert turn["user_text"] == expected
    assert f"› {expected[:18]}" in term.text()  # the echo, wrapped at 41 columns
    assert "Traceback" not in term.text()


def test_with_no_config_a_url_is_offered_and_declining_saves_nothing(
    melo: Callable[..., Melo], tmp_path: Path, stub: StubServer
) -> None:
    term = melo("--base-url", stub.base_url, configured=False)
    term.see("as the default backend? [Y/n]")
    term.type("n\r")
    term.see("Ask melo")
    assert f"stub-model · OpenAI-compatible · {stub.base_url}" in term.text()
    assert "from --url (not saved)" in term.text()
    assert term.quit() == 0
    assert not (tmp_path / ".cache" / "localmelo" / "config.toml").exists()


# ── Review fixes: whatever ends the session, it closes and melo exits ──


def test_sighup_during_a_turn_ends_the_process_by_it(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.process.send_signal(signal.SIGHUP)
    assert term.wait(20) == -signal.SIGHUP
    assert "Traceback" not in term.text()
    [turn] = _turns(tmp_path)
    assert (turn["status"], turn["cause"], turn["error"]) == (
        "cancelled",
        "shutdown",
        "Cancelled by SIGHUP",
    )


def test_a_closed_terminal_master_ends_the_process_by_sighup(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.hang_up()
    assert term.process.wait(timeout=20) == -signal.SIGHUP
    [turn] = _turns(tmp_path)
    assert turn["status"] != "running"
    assert (turn["status"], turn["cause"]) == ("cancelled", "shutdown")


def test_a_closed_terminal_under_nohup_takes_the_hangup_path(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo(nohup=True)
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.hang_up()
    assert term.process.wait(timeout=20) == -signal.SIGHUP
    [turn] = _turns(tmp_path)
    assert (turn["status"], turn["cause"], turn["error"]) == (
        "cancelled",
        "shutdown",
        "Cancelled by SIGHUP",
    )


@pytest.mark.parametrize("delay_s", [0.0, 0.01, 0.05])
def test_sigterm_right_after_exit_ends_the_process_by_it(
    melo: Callable[..., Melo], tmp_path: Path, delay_s: float
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.type("/exit\r")
    # The app has ended and the turn has stopped; the session closes.
    term.see("■ Stopped")
    time.sleep(delay_s)
    term.process.send_signal(signal.SIGTERM)
    # It exits by the signal, or with 0 when it had already finished
    # closing; never 1 with a traceback, as when the signal cancelled melo's
    # own task.
    assert term.wait() in (-signal.SIGTERM, 0), term.text()
    assert "Traceback" not in term.text()
    [turn] = _turns(tmp_path)
    assert (turn["status"], turn["cause"]) == ("cancelled", "user_stop")


def test_exit_during_a_turn_records_a_user_stop(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.type("/exit\r")
    assert term.wait() == 0
    [turn] = _turns(tmp_path)
    assert (turn["status"], turn["cause"], turn["error"]) == (
        "cancelled",
        "user_stop",
        "Stopped by the user",
    )


def test_yolo_runs_tools_without_asking(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    (tmp_path / "demo-project").mkdir(exist_ok=True)
    (tmp_path / "demo-project" / "notes.txt").write_text("x\n")
    term = melo("--yolo")
    term.see("Ask melo")
    # The status bar's right edge (its last cell is drawn on its own). It
    # comes after the input box in the same drawing, so it is waited for.
    term.see("⚠ YOLO: no approval")
    term.type("list the folder\r")
    term.see("The folder holds what")
    assert "Run a shell command?" not in term.text()
    assert "No approvals for this turn (--yolo)" in term.text()
    assert "notes.txt" in term.text()
    assert term.quit() == 0


def test_a_history_file_that_cannot_be_opened_is_not_fatal(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    (tmp_path / ".cache" / "localmelo" / "cli_history").mkdir(parents=True)
    term = melo()
    term.see("Ask melo")
    assert "The input history is not saved this time" in term.text()
    assert "Traceback" not in term.text()
    term.type("hello\r")
    term.see("Hello! I am melo")
    assert term.quit() == 0


def test_ascii_mode_prints_only_ascii(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo(env={"LOCALMELO_ASCII": "1"})
    term.see("Ask melo")
    term.type("hold on\r")
    term.see("esc to stop")
    term.type(b"\x1b")
    term.see("Stopped")
    time.sleep(1)  # the turn wraps up
    term.type("/help\r")
    term.see("ctrl+d quit")
    term.type("/model\r")
    term.see("/model NAME or /model N")
    term.type("/new\r")
    term.see("New session -")
    term.type("hello\r")
    term.see("Hello! I am melo")
    assert term.quit() == 0
    term.output.decode("ascii")  # nothing else reached the terminal


# ── Enter takes the completion picked with Tab or the arrows ──


@pytest.fixture
def listed(melo: Callable[..., Melo], tmp_path: Path, stub: StubServer) -> Melo:
    """melo on a server that lists stub-model, qwen3-8b and llama-3.1."""
    _configure(tmp_path, stub.base_url.replace("/v1", "/s/three_models/v1"))
    return melo(configured=False)


def test_tab_then_enter_runs_the_completed_command(listed: Melo) -> None:
    listed.see("Ask melo")
    listed.type("/he")
    listed.see("Show commands and keys")
    listed.type("\t")
    time.sleep(0.3)
    listed.type("\r")
    listed.see("esc stop the turn · ctrl+c clear, stop or quit")
    assert "Unknown command" not in listed.text()
    assert listed.quit() == 0


def test_tab_then_enter_switches_to_the_completed_model(
    listed: Melo, tmp_path: Path
) -> None:
    listed.see("Ask melo")
    time.sleep(0.5)  # the banner's check lists the models
    listed.type("/model q")
    listed.see("qwen3-8b")
    listed.type("\t")
    time.sleep(0.3)
    listed.type("\r")
    listed.see("Model: qwen3-8b (vLLM). Saved as the default.")
    saved = (tmp_path / ".cache" / "localmelo" / "config.toml").read_text()
    assert 'chat_model = "qwen3-8b"' in saved
    assert listed.quit() == 0


def test_arrows_then_enter_run_the_picked_command(listed: Melo) -> None:
    listed.see("Ask melo")
    listed.type("/")
    listed.see("Start a new session")
    listed.type("\x1b[B" * 3)  # /model, /backend, then /new
    time.sleep(0.3)
    listed.type("\r")
    listed.see("─ New session ─")
    assert "Unknown command" not in listed.text()
    assert listed.quit() == 0


def test_a_command_is_echoed_in_the_scrollback(melo: Callable[..., Melo]) -> None:
    term = melo()
    term.see("Ask melo")
    term.type("/help attach\r")
    term.see("During a turn: no")
    screen = pyte.Screen(100, 30)
    pyte.Stream(screen).feed(term.output.decode("utf-8", "replace"))
    rows = [row.rstrip() for row in screen.display]
    echo = rows.index("› /help attach")
    assert rows[echo - 1] == "" and rows[echo + 1] == "/attach PATH", rows
    assert term.quit() == 0
