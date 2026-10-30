"""StdioMcpClient against the fake stdio server (a real subprocess)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from localmelo.melo.checker.validators import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_RESULT,
    MAX_OUTPUT_LEN,
)
from localmelo.support.mcp import McpError, StdioMcpClient, tool_result_to_output
from localmelo.support.mcp.client import STDERR_LINE_CHARS, STREAM_LIMIT, _parse_tool
from localmelo.tests.mcp.conftest import (
    Gate,
    SpawnGate,
    fake_server_config,
    read_log,
    wait_held,
)
from localmelo.tests.mcp.fake_mcp_server import PNG_1X1, TOOLS


class TestToolAnnotations:
    def test_annotations_are_kept_raw(self) -> None:
        tool = _parse_tool(
            {
                "name": "fetch",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "title": "Fetch",
                    "unknown": {"nested": 1},
                },
            }
        )
        assert tool is not None
        assert tool.annotations == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "title": "Fetch",
            "unknown": {"nested": 1},
        }

    @pytest.mark.parametrize(
        "annotations",
        [
            pytest.param(None, id="null"),
            pytest.param(["readOnlyHint", True], id="list"),
            pytest.param("readOnlyHint", id="string"),
            pytest.param(7, id="number"),
        ],
    )
    def test_absent_or_not_an_object_gives_an_empty_mapping(
        self, annotations: object
    ) -> None:
        item: dict[str, Any] = {"name": "fetch"}
        if annotations is not None:
            item["annotations"] = annotations
        tool = _parse_tool(item)
        assert tool is not None
        assert tool.annotations == {}

    def test_a_non_string_key_gives_an_empty_mapping(self) -> None:
        tool = _parse_tool({"name": "fetch", "annotations": {3: True}})
        assert tool is not None
        assert tool.annotations == {}

    def test_a_non_bool_hint_is_kept_and_does_not_fail_listing(self) -> None:
        tool = _parse_tool(
            {"name": "fetch", "annotations": {"readOnlyHint": "yes", "n": 1}}
        )
        assert tool is not None
        assert tool.annotations == {"readOnlyHint": "yes", "n": 1}

    async def test_the_server_annotation_reaches_the_tool(
        self, client: StdioMcpClient
    ) -> None:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert tools["cwd"].annotations == {
            "readOnlyHint": True,
            "destructiveHint": False,
        }
        assert tools["echo"].annotations == {}


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

    async def test_stderr_is_drained(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fake server writes 200 KB to stderr before it reads stdin.
        # asyncio reads a pipe into a buffer of about twice the stream limit
        # even if nobody reads that buffer, so with the default limit the
        # handshake would complete anyway. With a 16 KiB limit, asyncio and
        # the kernel's pipe buffer hold well under 200 KB: the server blocks
        # in its write and answers initialize only if the client drains
        # stderr. The startup timeout makes a regression fail, not hang.
        monkeypatch.setattr("localmelo.support.mcp.client.STREAM_LIMIT", 16 * 1024)
        mcp = StdioMcpClient(
            fake_server_config(log_path, startup_timeout_s=5), shutdown_grace_s=0.5
        )
        try:
            await mcp.start()
            assert any("fake-mcp: starting" in line for line in mcp.stderr_tail)
        finally:
            await mcp.close()

    async def test_stderr_tail_keeps_the_end_of_each_line(
        self, client: StdioMcpClient
    ) -> None:
        # The fake server's second stderr line is 200,000 characters long.
        assert list(client.stderr_tail) == [
            "fake-mcp: starting",
            "..." + "x" * STDERR_LINE_CHARS,
        ]

    async def test_stderr_lines_across_chunks(
        self, log_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="localmelo.support.mcp.client")
        mcp = StdioMcpClient(fake_server_config(log_path))
        stderr = asyncio.StreamReader()
        # private: a stand-in process, so the reader sees exactly these chunks
        mcp._proc = cast(asyncio.subprocess.Process, SimpleNamespace(stderr=stderr))
        full = "f" * STDERR_LINE_CHARS  # as long as a kept line can be
        long_line = "x" * 100_000 + "end"
        wide = "\U0001f600" * STDERR_LINE_CHARS  # 4 bytes a character in UTF-8

        async def write() -> None:
            chunks = (
                "first\r\nsec",
                "ond\n",
                full + "\r\n",  # fits
                "+" + full + "\n",  # one character too long
                "+" + wide + "\n",  # one byte too long
                long_line,
                # 6,000 bytes: before this line ends, the reader drops all
                # but its last 4,000 bytes, which are 1,000 characters.
                "\n" + "\U0001f600" * 500 + wide,
                "\nno newline at EOF",
            )
            for chunk in chunks:
                stderr.feed_data(chunk.encode())
                await asyncio.sleep(0)  # the reader takes each chunk alone
            stderr.feed_eof()

        await asyncio.wait_for(asyncio.gather(mcp._read_stderr(), write()), 5)
        # A line whose start was dropped begins with "...".
        expected = [
            "first",
            "second",
            full,
            "..." + full,
            "..." + wide,
            "..." + long_line[-STDERR_LINE_CHARS:],
            "..." + wide,
            "no newline at EOF",
        ]
        assert list(mcp.stderr_tail) == expected
        # The debug log shows the same text.
        logged = [
            record.getMessage()
            for record in caplog.records
            if record.name == "localmelo.support.mcp.client"
        ]
        assert logged == [f"[mcp fake] {line}" for line in expected]


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

    async def test_structured_screenshot_sends_no_base64_as_text(
        self, client: StdioMcpClient
    ) -> None:
        result = await client.call_tool("screenshot", {"structured": True})
        assert result["structuredContent"]["screenshot"]  # the server repeats it
        output = tool_result_to_output(result)
        assert output.text == ""
        assert [base64.b64decode(i.data) for i in output.images] == [PNG_1X1]

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

    def test_line_limit_holds_the_largest_canonical_result(self) -> None:
        # The executor's result check passes MAX_IMAGES_PER_RESULT images of
        # MAX_IMAGE_BYTES each and MAX_OUTPUT_LEN characters of text
        # unchanged. The line carrying such a successful result must fit
        # the limit when its images are canonical base64, as written by
        # base64.b64encode and json.dumps. Other forms are not covered
        # (see STREAM_LIMIT).
        data_len = len(base64.b64encode(bytes(MAX_IMAGE_BYTES)))
        image_len = data_len + len(
            json.dumps({"type": "image", "data": "", "mimeType": "image/webp"})
        )
        # json.dumps writes a character outside the BMP as two \u escapes.
        text = "\U0001f600" * MAX_OUTPUT_LEN
        text_len = len(json.dumps({"type": "text", "text": text}))
        envelope_len = len(
            json.dumps({"jsonrpc": "2.0", "id": 2**53, "result": {"content": []}})
        )
        parts = [text_len] + [image_len] * MAX_IMAGES_PER_RESULT
        line_len = envelope_len + sum(parts) + len(", ") * (len(parts) - 1)
        assert line_len <= STREAM_LIMIT

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

    async def test_output_that_starts_with_a_byte_order_mark_is_read(
        self, log_path: Path
    ) -> None:
        # The server's first line, the initialize response, starts with a
        # UTF-8 byte order mark. The startup timeout makes a client that
        # drops that line fail, not hang.
        mcp = StdioMcpClient(
            fake_server_config(
                log_path, env={"FAKE_MCP_MODE": "bom_stdout"}, startup_timeout_s=5
            ),
            shutdown_grace_s=0.5,
        )
        await mcp.start()
        try:
            result = await mcp.call_tool("echo", {"text": "ok"})
        finally:
            await mcp.close()
        assert mcp.protocol_version == "2025-06-18"
        assert tool_result_to_output(result).text == "ok"

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

    async def test_a_skipped_message_is_logged_without_a_traceback(
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
        with caplog.at_level(logging.DEBUG, logger="localmelo.support.mcp.client"):
            await client.call_tool("echo", {"text": "one"})
        [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning.getMessage() == (
            "MCP server 'fake': a message could not be handled "
            "(RuntimeError: handler failed); skipped"
        )
        # A traceback on stderr would read as a crash; the debug log has it.
        assert warning.exc_info is None
        [debug] = [r for r in caplog.records if r.exc_info]
        assert debug.levelno == logging.DEBUG

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

    async def test_server_that_exits_in_time_gets_no_signal(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signals: list[bool] = []

        def record(proc: asyncio.subprocess.Process, *, kill: bool) -> bool:
            signals.append(kill)
            return False

        monkeypatch.setattr("localmelo.support.mcp.client._signal_server", record)
        # The fake server exits as soon as its stdin closes, well within the
        # grace period, so neither it nor its process group gets a signal.
        mcp = StdioMcpClient(fake_server_config(log_path), shutdown_grace_s=5)
        await mcp.start()
        await asyncio.wait_for(mcp.close(), 5)
        assert mcp.returncode == 0
        assert signals == []

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

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX sessions")
    async def test_close_closes_the_pipes_a_helper_holds(
        self, log_path: Path, tmp_path: Path
    ) -> None:
        helper_pid_file = tmp_path / "helper.pid"
        mcp = StdioMcpClient(
            fake_server_config(
                log_path,
                env={
                    "FAKE_MCP_MODE": "stubborn_helper",
                    "FAKE_MCP_HELPER_PIDFILE": str(helper_pid_file),
                },
            ),
            shutdown_grace_s=0.2,
        )
        await mcp.start()
        helper = int(helper_pid_file.read_text())
        try:
            # SIGKILL to the group ends the server but not the helper, which
            # holds the server's stdout and stderr open from a session of
            # its own. Before Python 3.13 Process.wait() waits for those
            # pipes, so close() only sees the server exit if it closes them.
            await asyncio.wait_for(mcp.close(), 5)
            os.kill(helper, 0)  # the helper still holds the pipes
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(helper, signal.SIGKILL)
        assert mcp.returncode == -signal.SIGKILL
        # private: asyncio has no public API for the process's pipes
        transport = getattr(mcp._proc, "_transport", None)
        assert isinstance(transport, asyncio.SubprocessTransport)
        for fd in (1, 2):
            pipe = transport.get_pipe_transport(fd)
            assert pipe is not None and pipe.is_closing()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    @pytest.mark.parametrize(
        ("refuse", "message"),
        [
            (
                True,
                r"could not be signalled \(permission denied\) and is still running",
            ),
            (False, r"is still running after SIGKILL"),
        ],
        ids=["signals_refused", "signals_without_effect"],
    )
    async def test_close_raises_when_the_server_cannot_be_killed(
        self,
        log_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        refuse: bool,
        message: str,
    ) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"}),
            shutdown_grace_s=0.1,
        )
        await mcp.start()

        def killpg(pgid: int, sig: int) -> None:
            # Every signal is refused, as for a server started through sudo,
            # or delivered without effect, as to a process stuck in the
            # kernel.
            if refuse:
                raise PermissionError("Operation not permitted")

        monkeypatch.setattr("localmelo.support.mcp.client.os.killpg", killpg)
        try:
            with pytest.raises(
                McpError, match=rf"Cannot stop MCP server 'fake': process \d+ {message}"
            ):
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


class TestKill:
    async def test_kill_stops_a_stubborn_server_at_once(self, log_path: Path) -> None:
        # A grace period this long would make close() alone take 10 s.
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"}),
            shutdown_grace_s=5,
        )
        await mcp.start()
        loop = asyncio.get_running_loop()
        started = loop.time()
        mcp.kill()
        # close() finds the server gone and only finishes the shutdown.
        await asyncio.wait_for(mcp.close(), 5)
        assert loop.time() - started < 2.5
        assert mcp.returncode == -signal.SIGKILL

    async def test_kill_ends_a_close_that_is_waiting(self, log_path: Path) -> None:
        mcp = StdioMcpClient(
            fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"}),
            shutdown_grace_s=5,
        )
        await mcp.start()
        closing = asyncio.create_task(mcp.close())
        await asyncio.sleep(0.1)  # close() waits for the server to exit
        assert not closing.done()
        mcp.kill()
        await asyncio.wait_for(closing, 2.5)
        assert mcp.returncode == -signal.SIGKILL

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    async def test_kill_reaches_a_server_behind_a_wrapper(
        self, log_path: Path, tmp_path: Path
    ) -> None:
        pid_file = tmp_path / "server.pid"
        server = fake_server_config(
            log_path,
            env={"FAKE_MCP_MODE": "stubborn", "FAKE_MCP_PIDFILE": str(pid_file)},
        )
        # "; true" keeps sh from replacing itself with the server.
        wrapped = dataclasses.replace(
            server,
            command="/bin/sh",
            args=["-c", shlex.join([server.command, *server.args]) + "; true"],
        )
        mcp = StdioMcpClient(wrapped, shutdown_grace_s=5)
        await mcp.start()
        pid = int(pid_file.read_text())
        gone = False
        try:
            mcp.kill()
            gone = await process_gone(pid)
        finally:
            if not gone:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            await asyncio.wait_for(mcp.close(), 5)
        assert gone

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    async def test_kill_reaches_a_server_whose_wrapper_died_of_close(
        self, log_path: Path, tmp_path: Path
    ) -> None:
        pid_file = tmp_path / "server.pid"
        server = fake_server_config(
            log_path,
            env={"FAKE_MCP_MODE": "stubborn", "FAKE_MCP_PIDFILE": str(pid_file)},
        )
        # "; true" keeps sh from replacing itself with the server.
        wrapped = dataclasses.replace(
            server,
            command="/bin/sh",
            args=["-c", shlex.join([server.command, *server.args]) + "; true"],
        )
        mcp = StdioMcpClient(wrapped, shutdown_grace_s=1.0)
        await mcp.start()
        pid = int(pid_file.read_text())
        closing = asyncio.create_task(mcp.close())
        gone = False
        try:
            # close() sends SIGTERM a grace period after closing stdin. That
            # ends sh, while the server ignores it and runs on in the group
            # until close() sends SIGKILL, another grace period later.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            while mcp.returncode is None and loop.time() < deadline:
                await asyncio.sleep(0.02)
            assert mcp.returncode == -signal.SIGTERM
            assert not closing.done()
            # An exit that cannot wait for that kills the server at once,
            # even though the process started (sh) has exited, and then
            # abandons close() (asyncio.run cancels it on the way out).
            mcp.kill()
            closing.cancel()
            gone = await process_gone(pid)
        finally:
            if not gone:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(asyncio.CancelledError):
                await closing
            await asyncio.wait_for(mcp.close(), 5)
        assert gone

    async def test_kill_without_a_running_server_does_nothing(
        self, log_path: Path
    ) -> None:
        mcp = StdioMcpClient(fake_server_config(log_path), shutdown_grace_s=0.5)
        mcp.kill()  # no process yet
        await mcp.start()
        await mcp.close()
        mcp.kill()  # the server has exited
        assert mcp.returncode == 0


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

    async def test_close_after_the_handshake_fails_start(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = Gate()
        mcp = StdioMcpClient(fake_server_config(log_path), shutdown_grace_s=0.5)
        notify = mcp._notify  # private: sends one notification

        async def notify_then_hold(
            method: str, params: dict[str, Any] | None = None
        ) -> None:
            await notify(method, params)
            if method == "notifications/initialized":
                await gate.hold()

        monkeypatch.setattr(mcp, "_notify", notify_then_hold)
        starting = asyncio.create_task(mcp.start())
        # The handshake is done, but start() has not returned yet: a close()
        # now stops the server ...
        await wait_held(gate, starting)
        await mcp.close()
        assert mcp.returncode is not None
        gate.release()
        # ... and start() fails instead of returning a closed client.
        with pytest.raises(
            McpError, match="Cannot start MCP server 'fake': the client was closed"
        ):
            await starting
        assert not mcp._tasks  # private: no reader task is running
        assert [m.get("method") for m in read_log(log_path)] == [
            "initialize",
            "notifications/initialized",
        ]

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
