"""StdioMcpClient against the fake stdio server (a real subprocess)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import os
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.support.mcp import McpError, StdioMcpClient, tool_result_to_output
from localmelo.support.mcp.client import STDERR_LINE_CHARS
from localmelo.tests.mcp.conftest import (
    SpawnGate,
    fake_server_config,
    read_log,
    wait_held,
)
from localmelo.tests.mcp.fake_mcp_server import PNG_1X1, TOOLS


class TestHandshake:
    async def test_initialize_and_initialized(
        self, client: StdioMcpClient, log_path: Path
    ) -> None:
        assert client.protocol_version == "2025-06-18"
        assert client.server_info["name"] == "fake-mcp"
        methods = [m.get("method") for m in read_log(log_path)]
        assert methods[:2] == ["initialize", "notifications/initialized"]
        init = read_log(log_path)[0]["params"]
        assert init["clientInfo"]["name"] == "localmelo"

    async def test_tools_list_follows_pages(self, client: StdioMcpClient) -> None:
        tools = await client.list_tools()
        assert [t.name for t in tools] == [t["name"] for t in TOOLS]
        echo = tools[0]
        assert echo.description == "Echo text."
        assert echo.input_schema["properties"] == {"text": {"type": "string"}}

    async def test_stderr_is_drained(self, client: StdioMcpClient) -> None:
        # The fake server writes 200 KB to stderr before reading stdin; the
        # handshake only completes because the client drains it.
        assert any("fake-mcp: starting" in line for line in client.stderr_tail)

    async def test_stderr_tail_keeps_the_end_of_each_line(
        self, client: StdioMcpClient
    ) -> None:
        # The fake server's second stderr line is 200,000 dots long.
        assert list(client.stderr_tail) == [
            "fake-mcp: starting",
            "." * STDERR_LINE_CHARS,
        ]

    async def test_stderr_lines_across_chunks(self, log_path: Path) -> None:
        mcp = StdioMcpClient(fake_server_config(log_path))
        stderr = asyncio.StreamReader()
        # private: a stand-in process, so the reader sees exactly these chunks
        mcp._proc = cast(asyncio.subprocess.Process, SimpleNamespace(stderr=stderr))
        long_line = "x" * 100_000 + "end"

        async def write() -> None:
            for chunk in ("first\r\nsec", "ond\n", long_line, "\nno newline at EOF"):
                stderr.feed_data(chunk.encode())
                await asyncio.sleep(0)  # the reader takes each chunk alone
            stderr.feed_eof()

        await asyncio.wait_for(asyncio.gather(mcp._read_stderr(), write()), 5)
        assert list(mcp.stderr_tail) == [
            "first",
            "second",
            long_line[-STDERR_LINE_CHARS:],
            "no newline at EOF",
        ]


class TestToolCalls:
    async def test_text_result(self, client: StdioMcpClient) -> None:
        result = await client.call_tool("echo", {"text": "hello"})
        assert tool_result_to_output(result).text == "hello"

    async def test_image_result(self, client: StdioMcpClient) -> None:
        output = tool_result_to_output(await client.call_tool("screenshot", {}))
        assert output.text == "screen captured"
        assert len(output.images) == 1
        image = output.images[0]
        assert image.mime_type == "image/png"
        assert image.sha256 == hashlib.sha256(PNG_1X1).hexdigest()
        assert base64.b64decode(image.data) == PNG_1X1

    async def test_tool_level_error(self, client: StdioMcpClient) -> None:
        output = tool_result_to_output(await client.call_tool("fail", {}))
        assert output.is_error
        assert output.text == "element not found"

    async def test_protocol_error_raises(self, client: StdioMcpClient) -> None:
        with pytest.raises(McpError, match="Unknown tool") as info:
            await client.call_tool("no_such_tool", {})
        assert info.value.code == -32602

    async def test_large_image_fits_one_line(self, client: StdioMcpClient) -> None:
        result = await client.call_tool("big_image", {"size": 3_000_000})
        image = tool_result_to_output(result).images[0]
        assert image.size_bytes == 3_000_000

    async def test_answers_server_ping(self, client: StdioMcpClient) -> None:
        result = await client.call_tool("ping_client", {})
        assert tool_result_to_output(result).text == "pong ok"

    async def test_concurrent_calls(self, client: StdioMcpClient) -> None:
        results = await asyncio.gather(
            *(client.call_tool("echo", {"text": str(i)}) for i in range(5))
        )
        assert [tool_result_to_output(r).text for r in results] == [
            str(i) for i in range(5)
        ]


class TestCancellation:
    async def test_timeout_sends_cancelled(
        self, client: StdioMcpClient, log_path: Path
    ) -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.call_tool("slow", {"seconds": 3}), 0.2)
        # The connection stays usable.
        result = await client.call_tool("echo", {"text": "after"})
        assert tool_result_to_output(result).text == "after"
        entries = read_log(log_path)
        slow_id = next(
            m["id"]
            for m in entries
            if m.get("method") == "tools/call" and m["params"]["name"] == "slow"
        )
        cancels = [m for m in entries if m.get("method") == "notifications/cancelled"]
        assert [c["params"]["requestId"] for c in cancels] == [slow_id]


async def env_var(mcp: StdioMcpClient, name: str) -> str:
    """The value of *name* in the server's environment, or ``<unset>``."""
    return tool_result_to_output(await mcp.call_tool("env", {"name": name})).text


async def process_gone(pid: int, timeout: float = 3.0) -> bool:
    """Whether process *pid* is gone within *timeout* seconds.

    A killed process is a zombie until it is reaped, and kill(pid, 0) still
    succeeds on a zombie.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.02)


class TestEnvironment:
    async def test_only_default_vars_and_config_env_are_passed(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCALMELO_TEST_SECRET", "hunter2")
        monkeypatch.delenv("LC_ALL", raising=False)
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_FLAG": "42"}),
            shutdown_grace_s=0.5,
        )
        await mcp.start()
        try:
            seen = {
                name: await env_var(mcp, name)
                for name in ("LOCALMELO_TEST_SECRET", "FAKE_FLAG", "PATH", "LC_ALL")
            }
        finally:
            await mcp.close()
        assert seen == {
            "LOCALMELO_TEST_SECRET": "<unset>",  # not inherited
            "FAKE_FLAG": "42",  # config env, literal
            "PATH": os.environ["PATH"],  # a default variable
            "LC_ALL": "<unset>",  # a default variable unset here stays unset
        }

    async def test_config_env_overrides_a_default_var(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"LANG": "C"}), shutdown_grace_s=0.5
        )
        await mcp.start()
        try:
            lang = await env_var(mcp, "LANG")
        finally:
            await mcp.close()
        assert lang == "C"

    async def test_env_and_cwd(self, log_path: Path, tmp_path: Path) -> None:
        workdir = tmp_path / "work"
        workdir.mkdir()
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_FLAG": "42"}, cwd=str(workdir)),
            shutdown_grace_s=0.5,
        )
        await mcp.start()
        try:
            flag = await mcp.call_tool("env", {"name": "FAKE_FLAG"})
            path = await mcp.call_tool("env", {"name": "PATH"})
            cwd = await mcp.call_tool("cwd", {})
        finally:
            await mcp.close()
        assert tool_result_to_output(flag).text == "42"
        assert tool_result_to_output(path).text not in ("", "<unset>")
        assert Path(tool_result_to_output(cwd).text).resolve() == workdir.resolve()


class TestFailures:
    async def test_crash_fails_the_call_and_later_calls(
        self, client: StdioMcpClient
    ) -> None:
        with pytest.raises(McpError, match="exited"):
            await client.call_tool("crash", {})
        with pytest.raises(McpError, match="exited"):
            await client.call_tool("echo", {"text": "x"})
        assert client.returncode == 3

    async def test_missing_command(self, tmp_path: Path) -> None:
        cfg = fake_server_config(tmp_path / "log")
        cfg.command = str(tmp_path / "does-not-exist")
        mcp = StdioMcpClient(cfg)
        with pytest.raises(McpError, match="Cannot start"):
            await mcp.start()
        await mcp.close()

    @pytest.mark.parametrize(
        "change",
        [
            {"env": {"A=B": "1"}},
            {"args": ["-I", "a\0b"]},
            {"cwd": "/tmp\0"},
            {"env": {"TEXT": "\ud800"}},  # a lone surrogate cannot be encoded
        ],
        ids=[
            "equals_in_env_name",
            "nul_in_args",
            "nul_in_cwd",
            "unencodable_env_value",
        ],
    )
    async def test_spawn_value_error(
        self, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        # A config built in code skips parse_mcp_config's checks, and the
        # spawn raises ValueError for strings the OS cannot take.
        cfg = dataclasses.replace(fake_server_config(tmp_path / "log"), **change)
        mcp = StdioMcpClient(cfg)
        with pytest.raises(McpError, match="Cannot start MCP server 'fake'"):
            await mcp.start()
        await mcp.close()

    async def test_initialize_timeout(self, log_path: Path) -> None:
        mcp = StdioMcpClient(
            fake_server_config(
                log_path,
                env={"FAKE_MCP_MODE": "hang_initialize"},
                startup_timeout_s=0.3,
            ),
            shutdown_grace_s=0.5,
        )
        with pytest.raises(McpError, match="did not answer initialize"):
            await mcp.start()
        assert mcp.returncode is not None  # the process was shut down

    async def test_tools_list_timeout(self, log_path: Path) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "hang_tools_list"}),
            shutdown_grace_s=0.5,
        )
        await mcp.start()
        # Only the listing gets a short bound: start() ran with the default,
        # and list_tools() reads the config when it is called.
        mcp.config.startup_timeout_s = 0.3
        try:
            with pytest.raises(McpError) as info:
                await mcp.list_tools()
        finally:
            await mcp.close()
        # The server, the request, the bound and the stderr tail are named.
        assert str(info.value).startswith(
            "MCP server 'fake' did not answer tools/list within 0.3s; stderr: "
        )

    async def test_non_json_stdout_is_ignored(self, log_path: Path) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "garbage_stdout"}),
            shutdown_grace_s=0.5,
        )
        await mcp.start()
        try:
            result = await mcp.call_tool("echo", {"text": "ok"})
        finally:
            await mcp.close()
        assert tool_result_to_output(result).text == "ok"

    async def test_malformed_stdout_lines_are_skipped(self, log_path: Path) -> None:
        # Before every response the server writes a line that is not UTF-8
        # and a line of JSON nested deeper than the parser's recursion limit.
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "malformed_stdout"}),
            shutdown_grace_s=0.5,
        )
        await mcp.start()
        try:
            first = await mcp.call_tool("echo", {"text": "ok"})
            second = await mcp.call_tool("echo", {"text": "again"})
        finally:
            await mcp.close()
        assert tool_result_to_output(first).text == "ok"
        assert tool_result_to_output(second).text == "again"

    async def test_response_that_is_not_utf8_is_read(
        self, client: StdioMcpClient
    ) -> None:
        # The server encodes this response as Latin-1; the byte that is not
        # UTF-8 becomes U+FFFD instead of losing the response.
        result = await asyncio.wait_for(client.call_tool("latin1", {}), 5)
        assert tool_result_to_output(result).text == "caf\ufffd"

    async def test_a_message_that_cannot_be_handled_is_skipped(
        self,
        client: StdioMcpClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        handle = client._dispatch_one  # private: handles one parsed message

        def handle_then_fail(message: dict[str, Any]) -> None:
            handle(message)
            raise RuntimeError("handler failed")

        monkeypatch.setattr(client, "_dispatch_one", handle_then_fail)
        first = await client.call_tool("echo", {"text": "one"})
        # The reader outlived the failure, so the next response arrives too.
        second = await client.call_tool("echo", {"text": "two"})
        assert tool_result_to_output(first).text == "one"
        assert tool_result_to_output(second).text == "two"
        assert "handler failed" in caplog.text

    async def test_closed_stdout_is_not_reported_as_an_exit(
        self, client: StdioMcpClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The wait for an exit code once the output has ended.
        monkeypatch.setattr("localmelo.support.mcp.client.EXIT_WAIT_S", 0.1)
        # The server replaces its stdout with /dev/null and keeps running.
        with pytest.raises(McpError, match="stopped responding") as info:
            await client.call_tool("close_stdout", {})
        assert "exited" not in str(info.value)
        assert client.returncode is None
        with pytest.raises(McpError, match="stopped responding"):
            await client.call_tool("echo", {"text": "x"})


class TestShutdown:
    async def test_close_is_idempotent(self, client: StdioMcpClient) -> None:
        await client.close()
        await client.close()
        assert client.returncode is not None
        with pytest.raises(McpError):
            await client.call_tool("echo", {"text": "x"})

    async def test_stubborn_server_is_killed(self, log_path: Path) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"}),
            shutdown_grace_s=0.2,
        )
        await mcp.start()
        await asyncio.wait_for(mcp.close(), 5)
        assert mcp.returncode is not None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    async def test_server_behind_a_wrapper_is_killed(
        self, log_path: Path, tmp_path: Path
    ) -> None:
        pid_file = tmp_path / "server.pid"
        server = fake_server_config(
            log_path,
            env={"FAKE_MCP_MODE": "stubborn", "FAKE_MCP_PIDFILE": str(pid_file)},
        )
        # "; true" keeps sh from replacing itself with the server, so sh is
        # the client's child and the server is sh's child.
        wrapped = dataclasses.replace(
            server,
            command="/bin/sh",
            args=["-c", shlex.join([server.command, *server.args]) + "; true"],
        )
        mcp = StdioMcpClient(wrapped, shutdown_grace_s=0.2)
        await mcp.start()
        pid = int(pid_file.read_text())
        gone = False
        try:
            # The server ignores its closed stdin and SIGTERM, which ends sh,
            # so only a SIGKILL sent to the whole group stops it.
            await asyncio.wait_for(mcp.close(), 5)
            gone = await process_gone(pid)
        finally:
            if not gone:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        assert gone

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    async def test_close_raises_when_the_server_cannot_be_killed(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"}),
            shutdown_grace_s=0.1,
        )
        await mcp.start()

        def refuse(pgid: int, sig: int) -> None:
            raise PermissionError("Operation not permitted")

        # As for a server started through sudo, no signal gets through.
        monkeypatch.setattr("localmelo.support.mcp.client.os.killpg", refuse)
        try:
            with pytest.raises(McpError, match="still running after SIGKILL"):
                await asyncio.wait_for(mcp.close(), 5)
            assert mcp.returncode is None
        finally:
            monkeypatch.undo()
            # Once signals get through, the next close() finishes the shutdown.
            await asyncio.wait_for(mcp.close(), 5)
        assert mcp.returncode is not None

    @pytest.mark.parametrize(
        "cancel_after_s", [0.1, 0.3], ids=["during_grace", "after_terminate"]
    )
    async def test_cancelled_close_is_finished_by_the_next_one(
        self, log_path: Path, cancel_after_s: float
    ) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"}),
            shutdown_grace_s=0.2,
        )
        await mcp.start()
        readers = set(mcp._tasks)  # private: the stdout and stderr readers
        # The stubborn server outlives its closed stdin and SIGTERM; it is
        # only killed after two grace periods (0.4 s), so a close() cancelled
        # before that leaves it running ...
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(mcp.close(), cancel_after_s)
        assert mcp.returncode is None
        # ... and the next close() finishes the shutdown.
        await asyncio.wait_for(mcp.close(), 5)
        assert mcp.returncode is not None
        assert readers and all(task.done() for task in readers)


class TestCloseDuringStart:
    async def test_close_during_the_spawn_stops_the_new_process(
        self, log_path: Path, spawn_gate: SpawnGate
    ) -> None:
        spawn_gate.hold_from = 0
        mcp = StdioMcpClient(fake_server_config(log_path), shutdown_grace_s=0.5)
        starting = asyncio.create_task(mcp.start())
        await wait_held(spawn_gate, starting)
        # start() is still spawning, so close() has no process to stop and
        # returns at once, however often it is called ...
        await mcp.close()
        await mcp.close()
        spawn_gate.release()
        # ... and start() stops the process it then gets, without the
        # handshake.
        with pytest.raises(
            McpError,
            match="Cannot start MCP server 'fake': the client was closed",
        ):
            await starting
        (proc,) = spawn_gate.processes
        assert proc.returncode is not None
        assert mcp.returncode is not None
        assert not mcp._tasks  # private: no reader task is running
        assert read_log(log_path) == []  # the server never got initialize
        # The client stays closed.
        await mcp.close()
        with pytest.raises(McpError, match="the client is closed"):
            await mcp.start()
        assert len(spawn_gate.processes) == 1

    async def test_start_after_close_is_refused(
        self, log_path: Path, spawn_gate: SpawnGate
    ) -> None:
        mcp = StdioMcpClient(fake_server_config(log_path))
        await mcp.close()
        await mcp.close()
        with pytest.raises(
            McpError, match="Cannot start MCP server 'fake': the client is closed"
        ):
            await mcp.start()
        assert spawn_gate.processes == []
        assert mcp.returncode is None
