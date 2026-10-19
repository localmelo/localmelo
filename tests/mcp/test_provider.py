"""McpToolProvider against the fake stdio server (real subprocesses)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.contracts.tools import ProvidedTool
from localmelo.melo.executor import Executor
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
    ToolOutput,
)
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import RAW_ARGUMENTS_KEY
from localmelo.support.mcp import (
    McpError,
    McpServerConfig,
    McpTool,
    McpToolProvider,
    StdioMcpClient,
    exposed_tool_name,
)
from localmelo.tests.mcp.conftest import (
    Gate,
    SpawnGate,
    fake_server_config,
    read_log,
    wait_held,
)
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


def gated_factory(
    cls: Callable[[McpServerConfig, Gate], StdioMcpClient],
    gate: Gate,
    created: list[StdioMcpClient],
) -> Callable[[McpServerConfig], StdioMcpClient]:
    """Clients of *cls* that share *gate*, kept for later assertions."""

    def make(server: McpServerConfig) -> StdioMcpClient:
        client = cls(server, gate)
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


class HeldListingClient(StdioMcpClient):
    """Lists the server's tools, then holds at the gate before returning them."""

    def __init__(self, server: McpServerConfig, gate: Gate) -> None:
        super().__init__(server, shutdown_grace_s=0.5)
        self._gate = gate

    async def list_tools(self) -> list[McpTool]:
        tools = await super().list_tools()
        await self._gate.hold()
        return tools


class HeldShutdownClient(StdioMcpClient):
    """Holds close() at the gate before each wait for the process to exit."""

    def __init__(self, server: McpServerConfig, gate: Gate) -> None:
        super().__init__(server, shutdown_grace_s=0.2)
        self._gate = gate

    async def _wait_exit(self, proc: asyncio.subprocess.Process) -> bool:
        await self._gate.hold()
        return await super()._wait_exit(proc)


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

    async def test_failed_start_can_be_retried(
        self, log_path: Path, tmp_path: Path
    ) -> None:
        # A failed start() closes the clients it started but leaves the
        # provider open, so once the cause is fixed start() can run again.
        broken = fake_server_config(tmp_path / "broken.log", name="broken")
        broken.command = str(tmp_path / "does-not-exist")
        created: list[StdioMcpClient] = []
        provider = McpToolProvider(
            [fake_server_config(log_path), broken],
            client_factory=recording_factory(created),
        )
        try:
            with pytest.raises(McpError, match="Cannot start"):
                await provider.start()
            assert provider.clients == []
            assert created[0].returncode is not None
            broken.command = sys.executable
            tools = await provider.start()
            assert [tool.definition.name for tool in tools] == [
                f"mcp__{server}__" + t["name"].replace(".", "_")
                for server in ("fake", "broken")
                for t in TOOLS
            ]
            # Only the second attempt's clients are listed.
            assert provider.clients == created[2:]
        finally:
            await provider.close()


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

    async def test_cancelled_close_is_finished_by_the_next_one(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        provider = McpToolProvider(
            [fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"})],
            client_factory=recording_factory(created),
        )
        await provider.start()
        (client,) = created
        readers = set(client._tasks)  # private: the stdout and stderr readers
        # The stubborn server ignores its closed stdin and SIGTERM. A close()
        # cancelled within the first grace period (0.5 s) re-raises the
        # cancellation and leaves the server running and its client listed ...
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(provider.close(), 0.2)
        assert client.returncode is None
        assert provider.clients == [client]
        # ... so the next close() finishes the shutdown.
        await asyncio.wait_for(provider.close(), 5)
        assert client.returncode is not None
        assert provider.clients == []
        assert readers and all(task.done() for task in readers)


class TestCloseDuringStart:
    async def test_close_during_a_spawn(
        self, tmp_path: Path, spawn_gate: SpawnGate, created: list[StdioMcpClient]
    ) -> None:
        spawn_gate.hold_from = 1  # server a starts; b is held in its spawn
        provider = McpToolProvider(
            [
                fake_server_config(tmp_path / f"{name}.log", name=name)
                for name in ("a", "b", "c")
            ],
            client_factory=recording_factory(created),
        )
        starting = asyncio.create_task(provider.start())
        await wait_held(spawn_gate, starting)
        readers = set(created[0]._tasks)  # private: server a's readers
        # close() stops a. b has no process yet, so close() only marks it
        # closing and drops it ...
        await provider.close()
        assert provider.clients == []
        spawn_gate.release()
        # ... and b's start() stops the process it then gets.
        with pytest.raises(McpError, match="Cannot start MCP server 'b'"):
            await starting
        assert [client.name for client in created] == ["a", "b"]  # c never ran
        assert len(spawn_gate.processes) == 2
        assert all(proc.returncode is not None for proc in spawn_gate.processes)
        assert readers and all(task.done() for task in readers)
        assert not created[1]._tasks
        assert provider.clients == []
        assert read_log(tmp_path / "b.log") == []  # b got no handshake

    @pytest.mark.parametrize(
        "names", [("a",), ("a", "b")], ids=["last_server", "more_servers"]
    )
    async def test_close_during_a_listing(
        self, tmp_path: Path, names: tuple[str, ...], created: list[StdioMcpClient]
    ) -> None:
        gate = Gate()
        provider = McpToolProvider(
            [fake_server_config(tmp_path / f"{name}.log", name=name) for name in names],
            client_factory=gated_factory(HeldListingClient, gate, created),
        )
        starting = asyncio.create_task(provider.start())
        try:
            # Server a has listed its tools; close() stops it while start()
            # is held before it uses the listing.
            await wait_held(gate, starting)
            readers = set(created[0]._tasks)  # private: server a's readers
            await provider.close()
            gate.release()
            with pytest.raises(McpError, match="the tool provider is closed"):
                await starting
        finally:
            await provider.close()
        # start() neither started another server nor returned a's tools.
        assert [client.name for client in created] == ["a"]
        assert created[0].returncode is not None
        assert readers and all(task.done() for task in readers)
        assert provider.clients == []

    async def test_start_after_close_is_refused(
        self, log_path: Path, spawn_gate: SpawnGate, created: list[StdioMcpClient]
    ) -> None:
        provider = McpToolProvider(
            [fake_server_config(log_path)], client_factory=recording_factory(created)
        )
        await provider.close()
        with pytest.raises(McpError, match="the tool provider is closed"):
            await provider.start()
        assert created == []
        assert spawn_gate.processes == []

    async def test_cancelled_start_still_stops_a_late_process(
        self, log_path: Path, spawn_gate: SpawnGate, created: list[StdioMcpClient]
    ) -> None:
        # close() drops a client that has no process yet, and the client's
        # start() stops the process it then gets. If the provider's start()
        # is cancelled during that shutdown, no later close() can reach the
        # client, so start() has to finish the shutdown in its clean-up.
        spawn_gate.hold_from = 0
        exit_gate = Gate()
        provider = McpToolProvider(
            # The stubborn server outlives its closed stdin and SIGTERM.
            [fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"})],
            client_factory=gated_factory(HeldShutdownClient, exit_gate, created),
        )
        starting = asyncio.create_task(provider.start())
        await wait_held(spawn_gate, starting)
        await provider.close()
        assert provider.clients == []
        spawn_gate.release()
        await wait_held(exit_gate, starting)  # stopping the new process
        starting.cancel()
        exit_gate.release()
        with pytest.raises(asyncio.CancelledError):
            await starting
        (proc,) = spawn_gate.processes
        assert proc.returncode is not None
        assert not created[0]._tasks


class TestThroughExecutor:
    async def test_unparsed_arguments_never_reach_the_server(
        self, tools: dict[str, ProvidedTool], log_path: Path
    ) -> None:
        hippo = Hippo()
        executor = Executor(hippo, Checker())
        for tool in tools.values():
            hippo.register_tool(tool.definition)
            executor.register(
                tool.definition.name, tool.function, timeout_ms=tool.timeout_ms
            )

        raw = '{"text": "hello"'  # cut off before the closing brace
        rejected = await executor.execute_structured(
            ExecutionRequest(
                tool_name="mcp__fake__echo", arguments={RAW_ARGUMENTS_KEY: raw}
            )
        )
        accepted = await executor.execute_structured(
            ExecutionRequest(tool_name="mcp__fake__echo", arguments={"text": "hello"})
        )

        assert rejected.status == ExecutionStatus.ERROR
        assert rejected.error_category == ErrorCategory.INVALID_ARGUMENTS
        assert rejected.error == (
            f"Arguments for mcp__fake__echo are not a JSON object: {raw}"
        )
        assert accepted.output == "hello"
        # Only the parsed call reached the server.
        calls = [
            m["params"] for m in read_log(log_path) if m.get("method") == "tools/call"
        ]
        assert calls == [{"name": "echo", "arguments": {"text": "hello"}}]


def test_long_names_are_capped_with_a_digest() -> None:
    server = McpServerConfig(name="s" * 70, command="unused")
    click = exposed_tool_name(server, "click")
    scroll = exposed_tool_name(server, "scroll")
    assert len(click) == len(scroll) == 64
    assert click.startswith("mcp__sss")
    assert click != scroll
    assert click == exposed_tool_name(server, "click")
