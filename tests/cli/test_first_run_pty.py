"""Bare `melo` on a real terminal: the first run's questions, then straight in.

The model server is a fake on a free port, which the first run finds
through LOCALMELO_DETECT_URLS; HOME is a temporary directory. After the
questions, the interactive UI opens; a later `melo` opens it at once.
"""

from __future__ import annotations

import fcntl
import re
import struct
import sys
import termios
from pathlib import Path

import pytest

from localmelo.tests.cli.test_direct_mode_process import (
    FakeChatServer,
    _env,
    chat_server,  # noqa: F401  (a fixture)
)
from localmelo.tests.cli.test_loop_line_editing import Terminal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")

# The input box's placeholder: the UI is up and waits for a message.
READY = b"Ask melo"
_ESCAPES = re.compile(
    rb"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][0-9A-B]|\x1b[=>78]"
)


def _melo(home: Path, server: FakeChatServer) -> Terminal:
    env = _env(home)
    env["LOCALMELO_DETECT_URLS"] = server.base_url
    env["TERM"] = "xterm-256color"  # whatever terminal runs the tests
    term = Terminal([sys.executable, "-I", "-m", "localmelo"], env, home)
    fcntl.ioctl(term._master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    return term


def _exit_code(term: Terminal) -> int:
    """melo's exit code; it closes the terminal a moment before it exits."""
    term.wait(30)
    return term.process.wait(timeout=30)


def _text(term: Terminal) -> str:
    return _ESCAPES.sub(b"", term.output).decode().replace("\r\n", "\n")


def test_the_first_run_asks_for_the_model_then_later_runs_go_straight_in(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    chat_server.models = ["nomic-embed-text", "chat-a", "chat-b"]
    url = chat_server.base_url

    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(
            lambda out: b"Model server URL [" in out, 30
        ), term.output
        term.type(b"\r")
        assert term.read_until(lambda out: b"Model: " in out, 30), term.output
        term.type(b"\r")  # no model is known to be loaded: Enter alone asks again
        assert term.read_until(
            lambda out: b"Type the number of the model to use." in out, 30
        ), term.output
        term.type(b"3\r")
        assert term.read_until(
            lambda out: b"Use nomic-embed-text for long-term memory? [Y/n]: " in out,
            30,
        ), term.output
        term.type(b"\r")
        assert term.read_until(lambda out: READY in out, 30), term.output
        term.type(b"hi\r")
        assert term.read_until(lambda out: b"Worked for" in out, 30), term.output
        term.type(b"/exit\r")
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()

    text = _text(term)
    assert f"Found a model server at {url} (3 models).\n" in text
    assert f"✓ {url} lists 3 models; using chat-b.\n" in text
    assert f"chat-b · vLLM · {url}" in text
    assert "saved as the default" in text
    assert chat_server.bodies[0]["model"] == "chat-b"
    saved = (tmp_path / ".cache" / "localmelo" / "config.toml").read_text()
    assert 'chat_backend = "vllm"' in saved
    assert f'chat_url = "{url}"' in saved and 'chat_model = "chat-b"' in saved
    assert 'embedding_backend = "vllm"' in saved
    assert 'embedding_model = "nomic-embed-text"' in saved

    chat_server.models = []  # the second run lists nothing and asks nothing
    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(lambda out: READY in out, 30), term.output
        term.type(b"/exit\r")
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()
    text = _text(term)
    assert f"chat-b · vLLM · {url}" in text
    assert "long-term memory · vLLM · nomic-embed-text" in text
    assert "Model server URL" not in text


def test_a_base_url_with_several_models_and_none_known_loaded_exits_2(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    chat_server.models = ["chat-a", "chat-b"]
    url = chat_server.base_url
    env = _env(tmp_path)
    env["TERM"] = "xterm-256color"
    term = Terminal(
        [sys.executable, "-I", "-m", "localmelo", "--base-url", url], env, tmp_path
    )
    try:
        assert _exit_code(term) == 2, term.output
    finally:
        term.close()
    text = _text(term)
    assert f"melo: error: {url} serves several models" in text
    assert "\n  chat-a\n  chat-b\n" in text
    assert "Pass --chat-model <id>." in text
    assert not (tmp_path / ".cache" / "localmelo" / "config.toml").exists()
