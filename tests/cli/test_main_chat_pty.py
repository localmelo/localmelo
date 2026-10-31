"""Bare melo in the main chat, end to end: a real gateway started on demand.

Each test runs ``python -m localmelo`` on a pseudo-terminal with HOME in a
temporary folder, the stub model server, and ``[gateway] port`` set to a free
port the system picked. Bare melo finds no gateway there and starts one in the
background; the test stops that gateway (found by its owner lock, its command
line checked) when it ends. Every wait is bounded and waits for the row it
asserts.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.tests.fixtures.stub_llm import StubServer

from .test_e2e_pty import GUARD_S, Melo
from .test_e2e_pty import stub as stub

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

# Starting the gateway imports its web stack: give it time on a busy machine.
START_WAIT_S = 60.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _configure(home: Path, url: str, port: int) -> None:
    folder = home / ".cache" / "localmelo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.toml").write_text(
        f'chat_backend = "vllm"\nembedding_backend = "none"\n\n[vllm]\n'
        f'chat_url = "{url}"\nchat_model = "stub-model"\n\n'
        f"[gateway]\nport = {port}\n",
        encoding="utf-8",
    )


def _gateway_pid(home: Path, port: int) -> int | None:
    """The gateway this test's melo started: its owner lock's pid, if its command is ours."""
    owner = home / ".cache" / "localmelo" / "gateway" / "sessions.db.owner"
    try:
        pid = int(owner.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    command = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return pid if "--serve" in command and f"--port {port}" in command else None


def _stop_gateway(home: Path, port: int) -> None:
    pid = _gateway_pid(home, port)
    if pid is None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


class _Main:
    def __init__(self, home: Path, port: int) -> None:
        self.home, self.port = home, port


@pytest.fixture
def main_melo(
    tmp_path: Path, stub: StubServer
) -> Iterator[tuple[Callable[..., Melo], _Main]]:
    port = _free_port()
    _configure(tmp_path, stub.base_url, port)
    started: list[Melo] = []

    def start(*argv: str, **kwargs: Any) -> Melo:
        env = {"LOCALMELO_CHAT": "main", **kwargs.pop("env", {})}
        term = Melo(tmp_path, list(argv), env=env, **kwargs)
        started.append(term)
        return term

    try:
        yield start, _Main(tmp_path, port)
    finally:
        for term in started:
            term.close()
        _stop_gateway(tmp_path, port)


def _front_turns(home: Path) -> list[tuple[str, str]]:
    path = home / ".cache" / "localmelo" / "gateway" / "sessions.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT s.kind, t.user_text FROM turns t JOIN sessions s "
            "ON s.id = t.session_id ORDER BY t.started_at"
        ).fetchall()
    return [(str(kind), str(text)) for kind, text in rows]


def test_bare_melo_starts_the_gateway_and_talks_in_the_main_chat(
    main_melo: tuple[Callable[..., Melo], _Main],
) -> None:
    start, main = main_melo
    term = start()
    term.see("Starting the gateway in the background", START_WAIT_S)
    term.see(f"Started the gateway in the background on port {main.port}", START_WAIT_S)
    term.see("Ask melo")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.see("Main chat")  # the welcome and the status bar name the chat
    assert term.quit() == 0, term.text()
    assert ("front", "hello") in _front_turns(main.home)
    # The gateway keeps running after melo quits: the next melo finds it.
    again = start()
    again.see(f"Connected to the gateway on port {main.port}", START_WAIT_S)
    again.see("hello")  # the recap of the main chat's last turns
    assert "Starting the gateway" not in again.text()
    assert again.quit() == 0, again.text()


def test_a_side_chat_comes_back_to_the_main_chat_as_a_card(
    main_melo: tuple[Callable[..., Melo], _Main],
) -> None:
    start, _ = main_melo
    term = start()
    term.see("Ask melo", START_WAIT_S)
    term.see("Started the gateway", START_WAIT_S)
    term.type("/side\r")
    term.see("New side chat in")
    term.type("hello\r")
    term.see("Hello! I am melo, running on a stub model.")
    term.type("/main\r")
    term.see("◆ Main chat")
    # The card is written when the list is next read (every few seconds).
    term.see("Done: hello", 30)
    term.see("/chats opens it")
    term.type("/chats\r")
    term.see("Chats")
    term.see("Main chat (open)")
    term.see("New side chat")
    time.sleep(GUARD_S)
    term.type(b"\x1b")  # Esc: nothing changes
    term.read_until(lambda out: False, 0.5)
    assert term.quit() == 0, term.text()


class _NotAGateway(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        self.send_response(404)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<p>not here</p>")

    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_without_the_main_chat_melo_says_why_and_keeps_its_own_chat(
    main_melo: tuple[Callable[..., Melo], _Main],
) -> None:
    start, main = main_melo
    server = http.server.ThreadingHTTPServer(("127.0.0.1", main.port), _NotAGateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        term = start()
        term.see(
            f"No main chat (port {main.port} is used by another program)", START_WAIT_S
        )
        term.see("this is the terminal's own chat.")
        term.see("Ask melo")
        term.type("hello\r")
        term.see("Hello! I am melo, running on a stub model.")
        assert term.quit() == 0, term.text()
    finally:
        server.shutdown()
        server.server_close()
    assert "Starting the gateway" not in term.text()
