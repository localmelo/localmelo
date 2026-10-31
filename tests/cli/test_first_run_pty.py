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

from localmelo.support.backends import discovery
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


def _melo(home: Path, server: FakeChatServer, *args: str) -> Terminal:
    env = _env(home)
    env["LOCALMELO_DETECT_URLS"] = server.base_url
    env["TERM"] = "xterm-256color"  # whatever terminal runs the tests
    term = Terminal([sys.executable, "-I", "-m", "localmelo", *args], env, home)
    fcntl.ioctl(term._master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    return term


def _exit_code(term: Terminal) -> int:
    """melo's exit code; it closes the terminal a moment before it exits."""
    term.wait(30)
    return term.process.wait(timeout=30)


def _text(term: Terminal) -> str:
    return _ESCAPES.sub(b"", term.output).decode().replace("\r\n", "\n")


@pytest.mark.parametrize(
    ("owned_by", "key", "label"),
    [("", "vllm", "OpenAI-compatible server"), ("omlx", "omlx", "oMLX")],
    ids=["another-server", "an-omlx-server"],
)
def test_the_first_run_asks_for_the_model_then_later_runs_go_straight_in(
    chat_server: FakeChatServer,  # noqa: F811
    tmp_path: Path,
    owned_by: str,
    key: str,
    label: str,
) -> None:
    """A server that lists itself as owned by oMLX is saved, and shown, as oMLX."""
    chat_server.models = ["nomic-embed-text", "chat-a", "chat-b"]
    chat_server.owned_by = owned_by
    url = chat_server.base_url

    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(
            lambda out: b"Use this model server?" in out, 30
        ), term.output
        term.type(b"\r")
        assert term.read_until(lambda out: b"Choose a model" in out, 30), term.output
        # No model is known to be loaded, so none is highlighted and Enter alone
        # chooses nothing; the arrow keys move to a row, the filter narrows them.
        term.type(b"\r")
        assert term.read_until(lambda out: b"choose one, then enter" in out, 30)
        term.type(b"\x1b[B")  # chat-a
        assert term.read_until(lambda out: b"enter chooses" in out, 30), term.output
        term.type(b"-b")  # the filter leaves chat-b, and the cursor goes to it
        assert term.read_until(lambda out: b"1 model " in out, 30), term.output
        term.type(b"\r")
        assert term.read_until(
            lambda out: b"Remember across chats with nomic-embed-text?" in out,
            30,
        ), term.output
        term.type(b"\r")
        assert term.read_until(lambda out: READY in out, 30), term.output
        term.type(b"hi\r")
        assert term.read_until(lambda out: b"Worked for" in out, 30), term.output
        term.type(b"/status all\r")  # the welcome no longer names the server
        assert term.read_until(lambda out: b"version    melo" in out, 30), term.output
        term.type(b"/exit\r")
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()

    text = _text(term)
    assert text.startswith("melo uses a model server on this computer (")
    assert f"Found a model server at {url} (3 models).\n" in text
    # The note about loading a model is one row at 100 columns; the list is the
    # arrow-key picker (no numbers), with chat models only.
    assert f"  {discovery.LOAD_COST_NOTE}\n" in text and "[1] chat-a" not in text
    picker = text[text.index("Choose a model") : text.index("lists 3 models")]
    assert "chat-a" in picker and "chat-b" in picker and "nomic" not in picker
    assert f"✓ {url} lists 3 models; using chat-b.\n" in text
    assert "chat-b (/model to change)" in text and "Try: What is this project?" in text
    assert f"model      chat-b · {label} · {url}" in text
    assert chat_server.bodies[0]["model"] == "chat-b"
    saved = (tmp_path / ".cache" / "localmelo" / "config.toml").read_text()
    assert f'chat_backend = "{key}"' in saved
    assert f'chat_url = "{url}"' in saved and 'chat_model = "chat-b"' in saved
    assert f'embedding_backend = "{key}"' in saved
    assert 'embedding_model = "nomic-embed-text"' in saved

    chat_server.models = []  # the second run lists nothing and asks nothing
    term = _melo(tmp_path, chat_server, "--developer")  # the box names the server
    try:
        assert term.read_until(lambda out: READY in out, 30), term.output
        term.type(b"/exit\r")
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()
    text = _text(term)
    # A new process has listed nothing yet; the file alone names the server.
    assert f"chat-b · {label} · {url}" in text
    assert f"long-term memory · {label} · nomic-embed-text" in text
    assert "Server URL" not in text


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


def test_closing_the_model_list_cancels_the_first_run(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    chat_server.models = ["chat-a", "chat-b"]
    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(
            lambda out: b"Use this model server?" in out, 30
        ), term.output
        term.type(b"\r")
        assert term.read_until(lambda out: b"Choose a model" in out, 30), term.output
        term.type(b"\x1b")
        assert _exit_code(term) == 130, term.output
    finally:
        term.close()
    assert "Setup cancelled." in _text(term)
    assert not (tmp_path / ".cache" / "localmelo" / "config.toml").exists()
    assert chat_server.requests == 0  # no chat request: nothing was loaded


@pytest.mark.parametrize(
    ("keys", "code"),
    [(b"\x1b", 130), (b"\x03", 130), (b"\x04", 0)],
    ids=["esc", "ctrl-c", "ctrl-d"],
)
def test_closing_the_found_server_list_ends_the_first_run_like_a_typed_question(
    chat_server: FakeChatServer, tmp_path: Path, keys: bytes, code: int  # noqa: F811
) -> None:
    chat_server.models = ["chat-a"]
    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(lambda out: b"Use this model server?" in out, 30)
        term.type(keys)
        assert _exit_code(term) == code, term.output
    finally:
        term.close()
    assert "Setup cancelled." in _text(term)
    assert not (tmp_path / ".cache" / "localmelo" / "config.toml").exists()
    assert chat_server.requests == 0


def test_a_number_key_answers_the_found_server_list_at_once(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    """`2` is "Use another server": the text question follows, with the found URL."""
    chat_server.models = ["chat-a"]
    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(lambda out: b"Use this model server?" in out, 30)
        term.type(b"2")
        assert term.read_until(lambda out: b"Server URL [" in out, 30), term.output
        term.type(b"\x04")  # end of input at the text question
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()
    assert f"Server URL [{chat_server.base_url}]" in _text(term)


def test_the_first_run_offers_other_assistants_notes_once_as_a_list(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    """After the model is chosen, an arrow-key list offers the notes; "Not now" is not asked again."""
    chat_server.models = ["chat-a"]
    rules = tmp_path / ".codex" / "AGENTS.md"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        "# Rules\n- Prefer tabs in Makefiles.\n- Name branches after issues.\n"
    )

    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(lambda out: b"Use this model server?" in out, 30)
        term.type(b"\r")
        assert term.read_until(
            lambda out: b"melo found notes on this Mac from other assistants" in out, 30
        ), term.output
        assert term.read_until(lambda out: b"Bring in 2 notes from 1 file" in out, 30)
        term.type(b"\r")  # the list starts on Show what they say
        assert term.read_until(lambda out: b"Prefer tabs in Makefiles." in out, 30)
        assert term.read_until(lambda out: b"Not now" in out, 30), term.output
        term.type(b"\x1b[B\x1b[B\r")  # the list again: Not now
        assert term.read_until(lambda out: READY in out, 30), term.output
        term.type(b"/exit\r")
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()
    text = _text(term)
    assert "~/.codex/AGENTS.md · 2 notes" in text and "Brought in" not in text
    assert str(tmp_path) not in text.split("melo found notes")[1].split("Not now")[0]

    term = _melo(tmp_path, chat_server)  # a later run offers nothing
    try:
        assert term.read_until(lambda out: READY in out, 30), term.output
        term.type(b"/exit\r")
        assert _exit_code(term) == 0, term.output
    finally:
        term.close()
    assert "found notes on this Mac" not in _text(term)
