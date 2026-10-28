"""McpToolProvider against the fake stdio server (real subprocesses)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import signal
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.checker import Checker, tier_of
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

    async def test_annotations_reach_the_tool_definition(
        self, tools: dict[str, ProvidedTool]
    ) -> None:
        assert tools["mcp__fake__cwd"].definition.annotations == {
            "readOnlyHint": True,
            "destructiveHint": False,
        }
        assert tools["mcp__fake__echo"].definition.annotations == {}

    async def test_a_read_only_tool_exposes_read(
        self, tools: dict[str, ProvidedTool]
    ) -> None:
        checker = Checker()
        assert checker.tier(tools["mcp__fake__cwd"].definition) == "read"
        assert tier_of(tools["mcp__fake__cwd"].definition) == "read"

    async def test_an_unannotated_tool_exposes_write(
        self, tools: dict[str, ProvidedTool]
    ) -> None:
        checker = Checker()
        assert checker.tier(tools["mcp__fake__echo"].definition) == "write"
        assert tier_of(tools["mcp__fake__echo"].definition) == "write"

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

    @pytest.mark.parametrize(
        ("mode", "asked"),
        [("prompts_only", False), ("no_tools_list", True)],
        ids=["no_tools_capability", "tools_list_not_found"],
    )
    async def test_server_without_tools_adds_none(
        self,
        tmp_path: Path,
        mode: str,
        asked: bool,
        created: list[StdioMcpClient],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A server that offers only prompts or resources answers tools/list
        # with METHOD_NOT_FOUND. It neither fails start() nor keeps the
        # next server's tools from being registered.
        bare_log = tmp_path / "bare.log"
        provider = McpToolProvider(
            [
                fake_server_config(bare_log, name="bare", env={"FAKE_MCP_MODE": mode}),
                fake_server_config(tmp_path / "fake.log"),
            ],
            client_factory=recording_factory(created),
        )
        try:
            tools = await provider.start()
            # Both servers are still listed and running.
            assert [client.name for client in provider.clients] == ["bare", "fake"]
            assert all(client.returncode is None for client in created)
        finally:
            await provider.close()
        assert [tool.definition.name for tool in tools] == [
            "mcp__fake__" + t["name"].replace(".", "_") for t in TOOLS
        ]
        # Only a server that declares the tools capability is asked for them.
        methods = [m.get("method") for m in read_log(bare_log)]
        assert ("tools/list" in methods) is asked
        assert "MCP server 'bare'" in caplog.text

    async def test_other_tools_list_errors_fail_start(
        self, tmp_path: Path, created: list[StdioMcpClient]
    ) -> None:
        # Only METHOD_NOT_FOUND means the server has no tools.
        provider = McpToolProvider(
            [
                fake_server_config(tmp_path / "fake.log"),
                fake_server_config(
                    tmp_path / "bad.log",
                    name="bad",
                    env={"FAKE_MCP_MODE": "tools_list_error"},
                ),
            ],
            client_factory=recording_factory(created),
        )
        with pytest.raises(McpError, match="MCP server 'bad': Internal error"):
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


async def _marks(log_path: Path, **server: Any) -> list[tuple[str, bool]]:
    """``(name, requires_approval)`` of each tool the server's provider starts."""
    provider = McpToolProvider(
        [fake_server_config(log_path, **server)],
        client_factory=recording_factory([]),
    )
    try:
        tools = await provider.start()
    finally:
        await provider.close()
    return [(tool.definition.name, tool.requires_approval) for tool in tools]


class TestRequireApproval:
    async def test_no_tool_requires_approval_by_default(
        self, tools: dict[str, ProvidedTool]
    ) -> None:
        assert len(tools) == len(TOOLS)
        assert not any(tool.requires_approval for tool in tools.values())

    async def test_true_marks_every_tool(self, log_path: Path) -> None:
        marks = await _marks(log_path, require_approval=True)
        assert len(marks) == len(TOOLS)
        assert all(marked for _, marked in marks)

    async def test_a_list_marks_the_tools_it_names(
        self, log_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        marks = await _marks(
            log_path, require_approval=["read.file", "echo", "no_such_tool"]
        )
        # Spelled as the server names them; registered under the exposed name.
        assert {name for name, marked in marks if marked} == {
            "mcp__fake__echo",
            "mcp__fake__read_file",
        }
        assert len(marks) == len(TOOLS)
        assert (
            "MCP server 'fake' does not offer no_such_tool (named in "
            "require_approval)"
        ) in caplog.text

    async def test_with_include_tools(
        self, log_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        marks = await _marks(
            log_path, include_tools=["echo", "fail"], require_approval=["fail", "slow"]
        )
        assert marks == [("mcp__fake__echo", False), ("mcp__fake__fail", True)]
        # slow is offered, only not registered: nothing to warn about.
        assert "does not offer" not in caplog.text


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

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    async def test_a_server_that_cannot_be_stopped_is_logged_without_a_traceback(
        self,
        log_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        created: list[StdioMcpClient] = []

        def make(server: McpServerConfig) -> StdioMcpClient:
            created.append(StdioMcpClient(server, shutdown_grace_s=0.1))
            return created[-1]

        provider = McpToolProvider(
            [fake_server_config(log_path, env={"FAKE_MCP_MODE": "stubborn"})],
            client_factory=make,
        )
        await provider.start()
        (client,) = created

        def killpg(pgid: int, sig: int) -> None:
            # Every signal is refused, as for a server started through sudo.
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr("localmelo.support.mcp.client.os.killpg", killpg)
        try:
            with caplog.at_level(logging.DEBUG):
                await asyncio.wait_for(provider.close(), 5)
        finally:
            monkeypatch.undo()
            await asyncio.wait_for(client.close(), 5)
        assert client.returncode is not None
        # The server outlived close(): one warning that says so, and no
        # traceback, since no run crashed.
        [record] = [
            r for r in caplog.records if r.getMessage().startswith("Cannot stop")
        ]
        assert record.levelno == logging.WARNING
        assert "Cannot stop MCP server 'fake': process" in record.getMessage()
        assert [r for r in caplog.records if r.exc_info] == []


class TestKill:
    async def test_kill_stops_every_server_at_once(self, tmp_path: Path) -> None:
        created: list[StdioMcpClient] = []

        def make(server: McpServerConfig) -> StdioMcpClient:
            # A grace period this long would make close() alone take 10 s.
            created.append(StdioMcpClient(server, shutdown_grace_s=5))
            return created[-1]

        provider = McpToolProvider(
            [
                fake_server_config(
                    tmp_path / f"{name}.log",
                    name=name,
                    env={"FAKE_MCP_MODE": "stubborn"},
                )
                for name in ("a", "b")
            ],
            client_factory=make,
        )
        await provider.start()
        loop = asyncio.get_running_loop()
        started = loop.time()
        provider.kill()
        await asyncio.wait_for(provider.close(), 5)
        assert loop.time() - started < 2.5
        assert [client.returncode for client in created] == [-signal.SIGKILL] * 2
        assert provider.clients == []

    def test_kill_before_start_does_nothing(self, log_path: Path) -> None:
        provider = McpToolProvider([fake_server_config(log_path)])
        provider.kill()
        assert provider.clients == []


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
