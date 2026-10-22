"""The agent with an McpToolProvider against the fake stdio server (real subprocesses)."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.mcp import McpServerConfig, McpToolProvider, StdioMcpClient
from localmelo.tests.mcp.conftest import (
    SpawnGate,
    fake_server_config,
    read_log,
    wait_held,
)
from localmelo.tests.mcp.fake_mcp_server import TOOLS

# Every tool of the fake server under the name the agent registers, sorted.
FAKE_TOOL_NAMES = sorted("mcp__fake__" + t["name"].replace(".", "_") for t in TOOLS)


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request."""

    def __init__(self, replies: list[Message] | None = None) -> None:
        self.replies: deque[Message] = deque(replies or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            return Message(role="assistant", content="(no response queued)")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


class SignallingProvider(McpToolProvider):
    """An McpToolProvider that sets ``closed`` whenever a close() returns."""

    def __init__(
        self,
        servers: Sequence[McpServerConfig],
        *,
        client_factory: Callable[[McpServerConfig], StdioMcpClient],
    ) -> None:
        super().__init__(servers, client_factory=client_factory)
        self.closed = asyncio.Event()

    async def close(self) -> None:
        await super().close()
        self.closed.set()


@pytest.fixture
def created() -> list[StdioMcpClient]:
    """Every client the provider under test created, in order."""
    return []


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _mcp_agent(
    llm: FakeLLM,
    servers: Sequence[McpServerConfig],
    created: list[StdioMcpClient],
    **options: Any,
) -> tuple[Agent, SignallingProvider]:
    """An agent whose only tools are the servers' (as with ``--mcp-only``)."""

    def make(server: McpServerConfig) -> StdioMcpClient:
        client = StdioMcpClient(server, shutdown_grace_s=0.5)
        created.append(client)
        return client

    provider = SignallingProvider(servers, client_factory=make)
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), tool_exposure="all", **options),
        tool_providers=[provider],
    )
    return agent, provider


def _call(name: str, arguments: dict[str, Any]) -> Message:
    return Message(
        role="assistant", tool_call=ToolCall(tool_name=name, arguments=arguments)
    )


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _offered(call: dict[str, Any]) -> list[str]:
    return [t.name for t in call["tools"] or []]


def _results(call: dict[str, Any]) -> list[str]:
    return [m.content for m in call["messages"] if m.role == "tool"]


def _tool_calls(log_path: Path) -> list[dict[str, Any]]:
    return [m["params"] for m in read_log(log_path) if m.get("method") == "tools/call"]


class TestMcpTools:
    async def test_every_request_offers_exactly_the_mcp_tools(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        llm = FakeLLM([_call("mcp__fake__echo", {"text": "hello"}), _answer("done")])
        agent, provider = _mcp_agent(llm, [fake_server_config(log_path)], created)
        try:
            result = await agent.run_task("say hello")
        finally:
            await agent.close()

        assert (result.status, result.result) == ("completed", "done")
        # Every planning request offered the MCP tools, sorted by name, and
        # nothing else: no built-in is registered.
        assert [_offered(call) for call in llm.calls] == [FAKE_TOOL_NAMES] * 2
        assert result.config["tools"] == FAKE_TOOL_NAMES
        assert result.config["builtins"] == []
        # The call reached the server under the tool's own name, and its
        # result reached the model.
        assert _tool_calls(log_path) == [
            {"name": "echo", "arguments": {"text": "hello"}}
        ]
        assert _results(llm.calls[1]) == ["hello"]
        # close() stopped the server.
        (client,) = created
        assert client.returncode is not None
        assert provider.clients == []

    async def test_the_server_starts_once_for_several_runs(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        llm = FakeLLM([_answer("one"), _answer("two")])
        agent, _ = _mcp_agent(llm, [fake_server_config(log_path)], created)
        try:
            first = await agent.run_task("q1")
            second = await agent.run_task("q2")
        finally:
            await agent.close()
        assert (first.status, second.status) == ("completed", "completed")
        assert len(created) == 1
        methods = [m.get("method") for m in read_log(log_path)]
        assert methods.count("initialize") == 1


class TestSetupFailures:
    async def test_a_server_that_cannot_start(
        self, log_path: Path, tmp_path: Path, created: list[StdioMcpClient]
    ) -> None:
        broken = fake_server_config(tmp_path / "broken.log", name="broken")
        broken.command = str(tmp_path / "does-not-exist")
        llm = FakeLLM([_answer("never sent")])
        servers = [fake_server_config(log_path), broken]
        agent, _ = _mcp_agent(llm, servers, created)
        try:
            result = await agent.run_task("q")
            # The first server was up and had listed its tools; the failure
            # of the second one stopped it before the run ended.
            assert "tools/list" in [m.get("method") for m in read_log(log_path)]
            assert created[0].returncode is not None
        finally:
            await agent.close()
        assert result.status == "setup_error"
        assert result.error.startswith("Cannot start MCP server 'broken'")
        assert result.config["tools"] == []
        assert llm.calls == []

    async def test_a_name_collision_stops_the_server(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        llm = FakeLLM([_answer("never sent")])
        agent, _ = _mcp_agent(llm, [fake_server_config(log_path)], created)
        agent.hippo.register_tool(
            ToolDef(name="mcp__fake__echo", description="taken", parameters={})
        )
        try:
            result = await agent.run_task("q")
            (client,) = created
            assert client.returncode is not None
        finally:
            await agent.close()
        assert (result.status, result.error) == (
            "setup_error",
            "Provided tool 'mcp__fake__echo' collides with a registered tool",
        )
        assert result.config["tools"] == ["mcp__fake__echo"]

    async def test_timeout_s_during_the_start_up_stops_the_server(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        # The server never answers initialize; its own start-up timeout is
        # far longer than timeout_s.
        server = fake_server_config(
            log_path, env={"FAKE_MCP_MODE": "hang_initialize"}, startup_timeout_s=10.0
        )
        llm = FakeLLM([_answer("never sent")])
        agent, _ = _mcp_agent(llm, [server], created, timeout_s=0.5)
        try:
            result = await agent.run_task("q")
            (client,) = created
            assert client.returncode is not None
        finally:
            await agent.close()
        assert (result.status, result.error) == ("timeout", "Timed out after 0.5 s")
        assert llm.calls == []


class TestCloseDuringSetup:
    async def test_close_returns_once_the_setup_stopped_its_server(
        self, log_path: Path, spawn_gate: SpawnGate, created: list[StdioMcpClient]
    ) -> None:
        spawn_gate.hold_from = 0  # the server's spawn is held
        llm = FakeLLM([_answer("never sent")])
        agent, provider = _mcp_agent(llm, [fake_server_config(log_path)], created)
        running = asyncio.create_task(agent.run_task("q"))
        await wait_held(spawn_gate, running)

        closing = asyncio.create_task(agent.close())
        await provider.closed.wait()
        # The provider is closed, but the run's setup is still spawning the
        # server, so close() waits ...
        assert not closing.done()
        spawn_gate.release()
        await closing
        # ... and returns only once the server the setup spawned is stopped.
        (proc,) = spawn_gate.processes
        assert proc.returncode is not None

        result = await running
        assert (result.status, result.error) == (
            "setup_error",
            "Cannot start MCP server 'fake': the client was closed during start-up",
        )
        assert read_log(log_path) == []  # the server got no handshake
