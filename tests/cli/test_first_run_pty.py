"""Bare `melo` on a real terminal: the first run's questions, then straight in.

The model server is a fake on a free port, which the first run finds
through LOCALMELO_DETECT_URLS; HOME is a temporary directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from localmelo.tests.cli.test_direct_mode_process import (
    FakeChatServer,
    _env,
    chat_server,  # noqa: F401  (a fixture)
)
from localmelo.tests.cli.test_loop_line_editing import PROMPT, Terminal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals")


def _melo(home: Path, server: FakeChatServer) -> Terminal:
    env = _env(home)
    env["LOCALMELO_DETECT_URLS"] = server.base_url
    env["TERM"] = "xterm-256color"  # whatever terminal runs the tests
    return Terminal([sys.executable, "-I", "-m", "localmelo"], env, home)


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
        assert term.read_until(lambda out: b"Model [2]: " in out, 30), term.output
        term.type(b"3\r")
        assert term.read_until(lambda out: PROMPT in out, 30), term.output
        term.type(b"hi\r")
        assert term.read_until(lambda out: out.count(PROMPT) >= 2, 30), term.output
        term.type(b"/exit\r")
        assert term.wait(30) == 0, term.output
    finally:
        term.close()

    text = term.output.decode().replace("\r\n", "\n")
    assert f"Found a model server at {url} (3 models).\n" in text
    assert f"✓ {url} lists 3 models; using chat-b.\n" in text
    assert f"Model: chat-b · {url} (melo --reconfigure changes it)\n" in text
    assert chat_server.bodies[0]["model"] == "chat-b"
    saved = (tmp_path / ".cache" / "localmelo" / "config.toml").read_text()
    assert 'chat_backend = "vllm"' in saved
    assert f'chat_url = "{url}"' in saved and 'chat_model = "chat-b"' in saved

    chat_server.models = []  # the second run lists nothing and asks nothing
    term = _melo(tmp_path, chat_server)
    try:
        assert term.read_until(lambda out: PROMPT in out, 30), term.output
        term.type(b"/exit\r")
        assert term.wait(30) == 0, term.output
    finally:
        term.close()
    text = term.output.decode().replace("\r\n", "\n")
    assert text.startswith(f"Model: chat-b · {url} (melo --reconfigure changes it)\n")
    assert "Model server URL" not in text
