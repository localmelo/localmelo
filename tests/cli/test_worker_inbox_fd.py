"""The inherited-pipe inbox: line rules, CLI acks and close-on-exec for tools."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support.cli.worker_inbox import (
    MAX_INBOX_BYTES,
    FdInbox,
    Inbox,
    InboxError,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="inherited pipe fds are POSIX"
)


def _frames(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines()]


def _acks(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        frame["event"]["data"]
        for frame in frames
        if frame.get("type") == "event" and frame["event"]["type"] == "inbox.accepted"
    ]


def test_fd_and_file_inboxes_return_the_same_messages(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    data = b"first\n\xe2\x82"
    path.write_bytes(data)
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)
        file_inbox, fd_inbox = Inbox(path), FdInbox(read_fd)
        assert fd_inbox.poll() == file_inbox.poll() == [(6, "first")]
        with pytest.raises(InboxError):
            fd_inbox.accept(5)  # only the next pending offset commits
        for inbox in (file_inbox, fd_inbox):
            inbox.accept(6)
        with path.open("ab") as file:
            file.write(b"\xac\nsecond\npartial")
        os.write(write_fd, b"\xac\nsecond\npartial")
        assert (
            fd_inbox.poll()
            == file_inbox.poll()
            == [
                (10, "€"),
                (17, "second"),
            ]
        )
        for inbox in (file_inbox, fd_inbox):
            inbox.accept(10)
            inbox.accept(17)
        assert fd_inbox.poll() == file_inbox.poll() == []
        with path.open("ab") as file:
            file.write(b"!\n")
        os.write(write_fd, b"!\n")
        assert fd_inbox.poll() == file_inbox.poll() == [(26, "partial!")]
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize(
    "content", [b"\n", b" \n", b"\xff\n", b"x" * 16385 + b"\n", b"x" * 16385]
)
def test_invalid_fd_message_fails(content: bytes) -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, content)
        with pytest.raises(InboxError):
            FdInbox(read_fd).poll()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_fd_inbox_keeps_its_total_byte_cap() -> None:
    read_fd, write_fd = os.pipe()
    try:
        inbox = FdInbox(read_fd)
        written = 0
        with pytest.raises(InboxError, match="exceeds 1 MiB"):
            while written <= MAX_INBOX_BYTES:
                line = b"x" * 4000 + b"\n"
                os.write(write_fd, line)
                written += len(line)
                inbox.poll()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_fd_inbox_rejects_a_non_pipe_and_a_closed_fd(tmp_path: Path) -> None:
    regular = tmp_path / "file"
    regular.write_text("x\n", encoding="utf-8")
    fd = os.open(regular, os.O_RDONLY)
    try:
        with pytest.raises(InboxError, match="not a pipe"):
            FdInbox(fd).validate()
    finally:
        os.close(fd)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        with pytest.raises(InboxError, match="cannot open inbox fd"):
            FdInbox(read_fd).validate()
    finally:
        os.close(write_fd)


def test_fd_inbox_is_close_on_exec() -> None:
    read_fd, write_fd = os.pipe()
    try:
        FdInbox(read_fd).validate()
        assert not os.get_inheritable(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def _build_agent(
    transport: httpx.MockTransport, *, builtins: frozenset[str] | None = None
) -> Any:
    def build(*args: Any, **kwargs: Any) -> Agent:
        return Agent(
            llm=OpenAICompatLLM(
                base_url="http://llm.test/v1", model="stub", transport=transport
            ),
            options=AgentOptions(
                memory=False,
                mode="direct",
                builtins=builtins,
                max_steps=2,
                attempt_policy="none",
            ),
        )

    return build


def test_cli_file_and_fd_inboxes_emit_the_same_acks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import localmelo.__main__ as cli

    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    lines = b"first\nsecond\n"
    file_inbox = tmp_path / "inbox"
    file_inbox.write_bytes(lines)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        )
    )
    monkeypatch.setattr(cli, "_build_direct_mode_agent", _build_agent(transport))
    base = ["melo", "--brief", str(brief), "--output-format", "stream-json"]
    monkeypatch.setattr(sys, "argv", [*base, "--inbox", str(file_inbox)])
    cli.main()
    file_frames = _frames(capsys.readouterr().out)

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, lines)
        monkeypatch.setattr(sys, "argv", [*base, "--inbox-fd", str(read_fd)])
        cli.main()
        fd_frames = _frames(capsys.readouterr().out)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert (
        _acks(fd_frames)
        == _acks(file_frames)
        == [
            {"offset": 6, "step": 1, "attempt": 1},
            {"offset": 13, "step": 1, "attempt": 1},
        ]
    )
    assert fd_frames[-1]["result"]["status"] == "completed"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--inbox-fd", "3"], "--inbox-fd requires --brief"),
        (
            ["--brief", "BRIEF", "--inbox", "FILE", "--inbox-fd", "3"],
            "--inbox and --inbox-fd cannot be combined",
        ),
        (["--brief", "BRIEF", "--inbox-fd", "x"], "not an integer"),
    ],
)
def test_inbox_fd_usage_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    import localmelo.__main__ as cli

    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    args = [brief.as_posix() if value == "BRIEF" else value for value in argv]
    args = [
        (tmp_path / "inbox").as_posix() if value == "FILE" else value for value in args
    ]
    monkeypatch.setattr(sys, "argv", ["melo", *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("tool", ["python_exec", "shell_exec"])
def test_a_tool_subprocess_cannot_write_the_inbox_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tool: str,
) -> None:
    import localmelo.__main__ as cli

    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    brief = tmp_path / "brief"
    brief.write_text("Task", encoding="utf-8")
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"first instruction\n")
    requests: list[dict[str, Any]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if tool == "python_exec":
                arguments = {
                    "code": f"import os\nos.write({read_fd}, b'tool injected\\n')"
                }
            else:
                arguments = {"command": f"printf 'tool injected\\n' >&{read_fd}"}
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "Done"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    transport = httpx.MockTransport(reply)
    monkeypatch.setattr(
        cli,
        "_build_direct_mode_agent",
        _build_agent(transport, builtins=frozenset({tool})),
    )
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "melo",
                "--brief",
                str(brief),
                "--inbox-fd",
                str(read_fd),
                "--output-format",
                "stream-json",
                "--builtins",
                tool,
            ],
        )
        cli.main()
    finally:
        os.close(read_fd)
        os.close(write_fd)
    frames = _frames(capsys.readouterr().out)
    assert [ack["offset"] for ack in _acks(frames)] == [18]
    messages = requests[-1]["messages"]
    assert not any(
        message.get("role") == "user" and "tool injected" in str(message.get("content"))
        for message in messages
    )
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert tool_messages and "Bad file descriptor" in tool_messages[-1]["content"]


async def test_an_mcp_server_cannot_write_the_inbox_fd(tmp_path: Path) -> None:
    from localmelo.support.mcp import McpError, McpServerConfig, StdioMcpClient

    read_fd, write_fd = os.pipe()
    sentinel = tmp_path / "seen"
    code = (
        "import os\n"
        "try:\n"
        f"    os.write({read_fd}, b'mcp injected\\n')\n"
        "except OSError:\n"
        f"    open({str(sentinel)!r}, 'w').write('blocked')\n"
        "else:\n"
        f"    open({str(sentinel)!r}, 'w').write('wrote')\n"
    )
    try:
        FdInbox(read_fd).validate()
        client = StdioMcpClient(
            McpServerConfig(
                name="srv",
                command=sys.executable,
                args=["-c", code],
                startup_timeout_s=1,
            )
        )
        with pytest.raises(McpError):
            await client.start()
        await client.close()
        assert sentinel.read_text(encoding="utf-8") == "blocked"
    finally:
        os.close(read_fd)
        os.close(write_fd)
