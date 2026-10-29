"""``! command`` in the real terminal, against the stub model server.

Each test starts ``python -m localmelo`` on a pseudo-terminal with HOME and the
workspace in a temporary folder, as ``test_e2e_pty`` does, and types ``!``
commands into it. Nothing reaches a model server; the stub keeps every request
body, so what the model was sent can be read back.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import struct
import sys
import termios
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from localmelo.tests.cli.test_e2e_pty import Melo, _configure, _turns
from localmelo.tests.fixtures.stub_llm import StubServer

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer().start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def melo(tmp_path: Path, stub: StubServer) -> Iterator[Callable[..., Melo]]:
    started: list[Melo] = []

    def start(**kwargs: object) -> Melo:
        _configure(tmp_path, stub.base_url)
        term = Melo(tmp_path, [], env={"SHELL": "/bin/sh"}, **kwargs)  # type: ignore[arg-type]
        started.append(term)
        term.see("Ask melo")
        return term

    yield start
    for term in started:
        term.close()


def _alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _until(check: Callable[[], bool], timeout_s: float = 10) -> bool:
    deadline = time.monotonic() + timeout_s
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _settle(term: Melo, seconds: float = 0.5) -> None:
    """Keep reading for a moment: a turn that has answered still wraps up."""
    term.read_until(lambda out: False, seconds)


def _ready(term: Melo, after: str) -> None:
    """Wait for the prompt to be drawn again below the last ``after`` on screen."""

    def drawn(out: bytes) -> bool:
        at = out.rfind(after.encode())
        return at >= 0 and b"Ask melo" in out[at:]

    assert term.read_until(drawn), term.text()
    _settle(term, 0.2)


def _modes(term: Melo) -> list[int]:
    """The terminal's input, output and local mode flags, as the app sees them."""
    attrs = termios.tcgetattr(term._master)
    return [attrs[0], attrs[1], attrs[3]]


def test_a_command_shows_its_block_exit_code_and_output(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.type("! echo from-the-child; exit 3\r")
    term.see("└ exit 3")
    _ready(term, "/attach ! attaches")
    text = term.text()
    assert "! user shell" in text and "demo-project" in text
    assert "$ echo from-the-child; exit 3" in text
    assert text.index("$ echo from") < text.index("from-the-child\r\n")
    assert "from-the-child\r\n" in text and "└ exit 3 · " in text
    assert "/attach ! attaches" in text
    term.type("! true\r")
    term.see("└ exit 0")
    assert term.quit() == 0
    assert _turns(tmp_path) == []  # a shell command is no turn


def test_a_signal_is_named(melo: Callable[..., Melo]) -> None:
    term = melo()
    term.type("! kill -TERM $$\r")
    term.see("killed by SIGTERM")
    assert term.quit() == 0


def test_the_working_directory_is_the_workspace_and_a_pipeline_runs(
    melo: Callable[..., Melo],
) -> None:
    term = melo()
    term.type("! pwd -P; printf 'b\\na\\n' | sort | head -1\r")
    term.see("└ exit 0")
    lines = term.text().splitlines()
    assert os.path.realpath(term.workspace) in [line.strip() for line in lines]
    assert [line.strip() for line in lines].count("a") == 1
    assert term.quit() == 0


def test_a_program_reading_input_gets_what_is_typed(
    melo: Callable[..., Melo],
) -> None:
    term = melo()
    term.type("! printf 'name? '; read -r name; echo \"hello $name\"\r")
    term.see("name? ")
    term.type("Ada\r")
    term.see("hello Ada")
    term.see("└ exit 0")
    _ready(term, "└ exit 0")
    # Esc is the child's: it arrives as a byte, and stops nothing of melo's.
    term.type("! printf 'r%s' eady; read -r x; printf '%s' \"$x\" | od -An -c\r")
    term.see("ready")  # the child is running: the keys are its now
    term.type("\x1b\r")
    term.see("033")
    term.see("└ exit 0")
    assert term.quit() == 0


def test_ctrl_c_goes_to_the_child_and_not_to_melo(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.type("! sleep 30\r")
    time.sleep(0.5)
    term.type("\x03")
    assert term.read_until(lambda out: b"SIGINT" in out or b"exit 130" in out)
    assert term.process.poll() is None  # melo is still here
    # After the return the old precedence holds: a first Ctrl+C on an idle,
    # empty prompt only asks for a second one.
    term.see("sleep 30")
    time.sleep(0.3)
    term.type("\x03")
    term.see("Press ctrl+c again to quit")
    term.type("\x03")
    assert term.wait() == 0
    assert _turns(tmp_path) == []


def test_the_terminal_is_given_back_exactly_as_it_was(
    melo: Callable[..., Melo],
) -> None:
    term = melo()
    before = _modes(term)
    term.type("! stty -a >/dev/null; printf '\\033[?25l\\033[31m'; exit 1\r")
    term.see("└ exit 1")
    _ready(term, "└ exit 1")
    assert _until(lambda: _modes(term) == before)
    text = term.output.decode(errors="replace")
    # What the child left on is turned off, once, before the prompt comes back.
    assert text.rfind("\x1b[?25h") > text.rfind("\x1b[?25l")
    term.type("hello\r")
    term.see("Hello! I am melo")  # the prompt draws and works again
    assert term.quit() == 0


def test_output_stays_out_of_the_next_model_request_unless_attached(
    melo: Callable[..., Melo], stub: StubServer, tmp_path: Path
) -> None:
    term = melo()
    term.type("! echo KEEP_OUT_MARK\r")
    term.see("└ exit 0")
    _ready(term, "└ exit 0")
    term.type("hello\r")
    term.see("Worked for")
    _settle(term)
    assert stub.requests and "KEEP_OUT_MARK" not in json.dumps(stub.requests)
    assert "echo KEEP" not in json.dumps(stub.requests)
    # Explicitly attached, the snapshot goes with the next message.
    term.type("! echo ATTACH_ME_MARK\r")
    term.see("ATTACH_ME_MARK\r\n")
    term.see("/attach ! attaches")
    _ready(term, "/attach ! attaches")
    term.type("/attach !\r")
    term.see("✓ Attached shell output: echo ATTACH_ME_MARK")
    _ready(term, "Attached: shell output")
    sent = len(stub.requests)
    term.type("hello\r")
    term.read_until(lambda out: len(stub.requests) > sent, 15)  # reads the screen too
    assert len(stub.requests) > sent, term.text()
    assert "ATTACH_ME_MARK" in json.dumps(stub.requests[sent:])
    assert "KEEP_OUT_MARK" not in json.dumps(stub.requests)
    assert term.quit() == 0
    # Stored turns hold the message and the attachment only.
    assert [t["user_text"] for t in _turns(tmp_path)][0] == "hello"


def test_a_shell_command_is_not_saved_in_the_input_history(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    term.type("! echo SECRET_HISTORY_MARK\r")
    term.see("└ exit 0")
    term.type("hello\r")
    term.see("Worked for")
    assert term.quit() == 0
    saved = b"".join(
        p.read_bytes() for p in (tmp_path / ".cache" / "localmelo").rglob("*history*")
    )
    assert b"hello" in saved and b"SECRET_HISTORY_MARK" not in saved
    assert b"SECRET_HISTORY_MARK" not in b"".join(
        p.read_bytes()
        for p in (tmp_path / ".cache" / "localmelo").rglob("*")
        if p.is_file() and p.suffix in (".db", ".log", ".jsonl", ".toml")
    )


def test_bang_alone_shows_help_and_the_escape_reaches_the_model(
    melo: Callable[..., Melo], stub: StubServer
) -> None:
    term = melo()
    term.type("!\r")
    term.see("is not a sandbox")
    term.type("\\! rm -rf is dangerous\r")
    term.see("Worked for")
    sent = json.dumps(stub.requests)
    assert "! rm -rf is dangerous" in sent and "\\\\! rm" not in sent
    assert "user shell" not in term.text()
    assert term.quit() == 0


def test_bang_during_a_turn_is_refused_and_the_draft_is_kept(
    melo: Callable[..., Melo],
) -> None:
    term = melo()
    term.type("hold on\r")
    term.see("esc to stop")
    term.type("! echo REFUSED_MARK\r")
    term.see("! runs only when melo is idle")
    time.sleep(0.3)
    assert "! user shell" not in term.text()
    assert term.text().count("REFUSED_MARK") >= 1  # the kept draft is on screen
    term.type(b"\x1b")
    term.see("■ Stopped")
    term.type("\x03")  # the kept draft: the old precedence clears it first
    term.see("Draft cleared")
    assert term.quit() == 0


def test_a_resize_reaches_the_child(melo: Callable[..., Melo]) -> None:
    term = melo()
    term.type("! sleep 1.5; stty size\r")
    time.sleep(0.5)
    fcntl.ioctl(term._master, termios.TIOCSWINSZ, struct.pack("HHHH", 17, 61, 0, 0))
    term.see("17 61")
    term.see("└ exit 0")
    assert term.quit() == 0


def test_a_closed_terminal_ends_melo_and_the_childs_group(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    marker = tmp_path / "pid"
    term.type(f"! echo $$ > {marker}; sleep 60\r")
    assert _until(lambda: marker.exists() and marker.read_text().strip() != "")
    pgid = int(marker.read_text())
    assert _alive(pgid) and pgid != os.getpgid(term.process.pid)
    term.hang_up()
    assert term.process.wait(timeout=20) == -signal.SIGHUP
    assert _until(lambda: not _alive(pgid))


def test_sigterm_to_melo_ends_the_childs_group_and_leaves_the_terminal_cooked(
    melo: Callable[..., Melo], tmp_path: Path
) -> None:
    term = melo()
    before = _modes(term)
    marker = tmp_path / "pid"
    term.type(f"! echo $$ > {marker}; sleep 60 & sleep 60\r")
    assert _until(lambda: marker.exists() and marker.read_text().strip() != "")
    pgid = int(marker.read_text())
    term.process.send_signal(signal.SIGTERM)
    assert term.wait() == -signal.SIGTERM
    assert _until(lambda: not _alive(pgid))
    # The terminal is as the shell that started melo had it: not raw.
    assert _modes(term)[2] & termios.ICANON
    assert before[2] & termios.ICANON == 0
    assert re.search(r"Traceback", term.text()) is None
