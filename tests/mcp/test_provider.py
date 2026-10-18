"""McpToolProvider against the fake stdio server (real subprocesses)."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.tools import ProvidedTool
from localmelo.melo.executor.models import ToolOutput
from localmelo.support.mcp import (
    McpError,
    McpServerConfig,
    McpTool,
    McpToolProvider,
    StdioMcpClient,
    exposed_tool_name,
)
from localmelo.tests.mcp.conftest import fake_server_config, read_log
from localmelo.tests.mcp.fake_mcp_server import PNG_1X1, TOOLS


def recording_factory(
    created: list[StdioMcpClient],
) -> Callable[[McpServerConfig], StdioMcpClient]:
    """Real clients with a short shutdown grace, kept for later assertions."""

    def make(server: McpServerConfig) -> StdioMcpClient:
        client = StdioMcpClient(server, shutdown_grace_s=0.5)
        created.append(client)
        return client

    return make


class BlankDescriptionClient(StdioMcpClient):
    """Lists the fake server's tools with their descriptions removed."""

    async def list_tools(self) -> list[McpTool]:
        return [
            dataclasses.replace(tool, description="")
            for tool in await super().list_tools()
        ]


async def invoke(tool: ProvidedTool, **arguments: Any) -> ToolOutput:
    output = await tool.function(**arguments)
    assert isinstance(output, ToolOutput)
    return output


@pytest.fixture
def created() -> list[StdioMcpClient]:
    return []


@pytest.fixture
async def provider(
    log_path: Path, created: list[StdioMcpClient]
) -> AsyncIterator[McpToolProvider]:
    mcp = McpToolProvider(
        [fake_server_config(log_path, timeout_s=7.5)],
        client_factory=recording_factory(created),
    )
    try:
        yield mcp
    finally:
        await mcp.close()


@pytest.fixture
async def tools(provider: McpToolProvider) -> dict[str, ProvidedTool]:
    return {tool.definition.name: tool for tool in await provider.start()}


class TestStart:
    async def test_tools_are_namespaced_with_server_metadata(
        self, provider: McpToolProvider, created: list[StdioMcpClient]
    ) -> None:
        tools = await provider.start()
        names = [tool.definition.name for tool in tools]
        assert names == ["mcp__fake__" + t["name"].replace(".", "_") for t in TOOLS]
        assert "mcp__fake__read_file" in names  # "read.file", sanitized
        assert [tool.definition.description for tool in tools] == [
            t["description"] for t in TOOLS
        ]
        echo = tools[0]
        assert echo.definition.name == "mcp__fake__echo"
        assert echo.definition.parameters == TOOLS[0]["inputSchema"]
        assert echo.definition.semantic_tags == ["fake", "echo"]
        assert {tool.timeout_ms for tool in tools} == {7500.0}
        assert provider.clients == created

    async def test_missing_description_gets_a_fallback(self, log_path: Path) -> None:
        provider = McpToolProvider(
            [fake_server_config(log_path)],
            client_factory=lambda server: BlankDescriptionClient(
                server, shutdown_grace_s=0.5
            ),
        )
        try:
            tools = await provider.start()
        finally:
            await provider.close()
        assert tools[0].definition.description == "echo (MCP server fake)"

    async def test_include_tools_filters(
        self, log_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = McpToolProvider(
            [
                fake_server_config(
                    log_path, include_tools=["read.file", "echo", "no_such_tool"]
                )
            ],
            client_factory=recording_factory([]),
        )
        try:
            tools = await provider.start()
        finally:
            await provider.close()
        # The server's listing order is kept, not the include list's.
        assert [tool.definition.name for tool in tools] == [
            "mcp__fake__echo",
            "mcp__fake__read_file",
        ]
        assert "does not offer no_such_tool" in caplog.text

    async def test_name_collision_raises(self, tmp_path: Path) -> None:
        # parse_mcp_config would reject these two server names; the
        # provider must still refuse to expose two tools under one name.
        created: list[StdioMcpClient] = []
        provider = McpToolProvider(
            [
                fake_server_config(tmp_path / "a.log", name="a.b"),
                fake_server_config(tmp_path / "b.log", name="a_b"),
            ],
            client_factory=recording_factory(created),
        )
        with pytest.raises(McpError, match="both map to 'mcp__a_b__echo'"):
            await provider.start()
        assert provider.clients == []
        assert len(created) == 2
        assert all(client.returncode is not None for client in created)

    async def test_failing_server_closes_the_started_ones(
        self, log_path: Path, tmp_path: Path
    ) -> None:
        broken = fake_server_config(tmp_path / "broken.log", name="broken")
        broken.command = str(tmp_path / "does-not-exist")
        created: list[StdioMcpClient] = []
        provider = McpToolProvider(
            [fake_server_config(log_path), broken],
            client_factory=recording_factory(created),
        )
        with pytest.raises(McpError, match="Cannot start"):
            await provider.start()
        first = created[0]
        # The first server was up and had listed its tools ...
        assert "tools/list" in [m.get("method") for m in read_log(log_path)]
        # ... and the failure of the second one shut it down.
        assert first.returncode is not None
        assert provider.clients == []
        with pytest.raises(McpError):
            await first.call_tool("echo", {"text": "x"})


class TestCalls:
    async def test_text_result(self, tools: dict[str, ProvidedTool]) -> None:
        output = await invoke(tools["mcp__fake__echo"], text="hello")
        assert output == ToolOutput(text="hello")

    async def test_image_result(self, tools: dict[str, ProvidedTool]) -> None:
        output = await invoke(tools["mcp__fake__screenshot"])
        assert output.text == "screen captured"
        assert [(image.mime_type, image.sha256) for image in output.images] == [
            ("image/png", hashlib.sha256(PNG_1X1).hexdigest())
        ]
        assert not output.is_error

    async def test_call_uses_the_original_tool_name(
        self, tools: dict[str, ProvidedTool], log_path: Path
    ) -> None:
        output = await invoke(tools["mcp__fake__read_file"])
        assert output.text == "dotted name ok"
        calls = [
            m["params"] for m in read_log(log_path) if m.get("method") == "tools/call"
        ]
        assert calls == [{"name": "read.file", "arguments": {}}]

    async def test_tool_error_maps_to_is_error(
        self, tools: dict[str, ProvidedTool]
    ) -> None:
        output = await invoke(tools["mcp__fake__fail"])
        assert output.is_error
        assert output.text == "element not found"


class TestClose:
    async def test_close_is_idempotent(
        self,
        provider: McpToolProvider,
        tools: dict[str, ProvidedTool],
        created: list[StdioMcpClient],
    ) -> None:
        await provider.close()
        await provider.close()
        assert provider.clients == []
        assert created[0].returncode is not None
        with pytest.raises(McpError):
            await tools["mcp__fake__echo"].function(text="late")

    async def test_close_before_start(self) -> None:
        provider = McpToolProvider([])
        await provider.close()
        assert provider.clients == []


def test_long_names_are_capped_with_a_digest() -> None:
    server = McpServerConfig(name="s" * 70, command="unused")
    click = exposed_tool_name(server, "click")
    scroll = exposed_tool_name(server, "scroll")
    assert len(click) == len(scroll) == 64
    assert click.startswith("mcp__sss")
    assert click != scroll
    assert click == exposed_tool_name(server, "click")
