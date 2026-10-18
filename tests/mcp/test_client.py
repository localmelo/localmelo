"""StdioMcpClient against the fake stdio server (a real subprocess)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path

import pytest

from localmelo.support.mcp import McpError, StdioMcpClient, tool_result_to_output
from localmelo.tests.mcp.conftest import fake_server_config, read_log
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


class TestEnvironment:
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
