"""The agent with an McpToolProvider against the fake stdio server (real subprocesses)."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import signal
import sqlite3
import sys
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import localmelo.melo.agent.agent as agent_mod
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    DenyAllPolicy,
)
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


class AllowAndRecord:
    """An approval policy that allows every call and keeps each request."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision(True)


class TestRequireApproval:
    async def test_the_named_tools_ask_the_policy(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        llm = FakeLLM(
            [
                _call("mcp__fake__echo", {"text": "hello"}),
                _call("mcp__fake__cwd", {}),
                _answer("done"),
            ]
        )
        server = fake_server_config(log_path, require_approval=["echo"])
        agent, _ = _mcp_agent(llm, [server], created)
        policy = AllowAndRecord()
        try:
            result = await agent.run_task("say hello", approval=policy)
            marked = [
                name
                for name in FAKE_TOOL_NAMES
                if agent.executor.requires_approval(name)
            ]
        finally:
            await agent.close()

        assert result.status == "completed"
        assert marked == ["mcp__fake__echo"]
        # Only echo asked, with the exposed name and the model's arguments.
        [request] = policy.requests
        assert (request.tool_name, dict(request.arguments), request.preview) == (
            "mcp__fake__echo",
            {"text": "hello"},
            '{"text":"hello"}',
        )
        assert [call["name"] for call in _tool_calls(log_path)] == ["echo", "cwd"]

    async def test_a_denied_call_never_reaches_the_server(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        llm = FakeLLM([_call("mcp__fake__echo", {"text": "hello"}), _answer("done")])
        server = fake_server_config(log_path, require_approval=True)
        agent, _ = _mcp_agent(llm, [server], created)
        try:
            result = await agent.run_task(
                "say hello", approval=DenyAllPolicy("no MCP calls today")
            )
        finally:
            await agent.close()

        assert result.status == "completed"
        assert _tool_calls(log_path) == []
        assert _results(llm.calls[1]) == [
            "Error: The approval policy denied this call: no MCP calls today. Do "
            "not run it again or reach the same result another way unless the "
            "user asks."
        ]

    async def test_without_a_policy_the_named_tools_run(
        self, log_path: Path, created: list[StdioMcpClient]
    ) -> None:
        llm = FakeLLM([_call("mcp__fake__echo", {"text": "hello"}), _answer("done")])
        server = fake_server_config(log_path, require_approval=True)
        agent, _ = _mcp_agent(llm, [server], created)
        try:
            result = await agent.run_task("say hello")
        finally:
            await agent.close()

        assert result.status == "completed"
        assert _tool_calls(log_path) == [
            {"name": "echo", "arguments": {"text": "hello"}}
        ]
        assert _results(llm.calls[1]) == ["hello"]


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


async def _until(check: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Whether *check* holds within *timeout* seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not check():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # also succeeds for a zombie not yet reaped
    except ProcessLookupError:
        return True
    return False


class TestExitDuringSetup:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
    async def test_the_clean_up_of_a_start_it_cuts_short_keeps_to_the_limit(
        self,
        tmp_path: Path,
        created: list[StdioMcpClient],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Server a has started; b has not answered initialize. Both ignore
        # SIGTERM and their closed stdin, so each takes two grace periods
        # of 0.5 s to stop, and the clean-up that the cancelled start()
        # does itself, b and then a, takes about 2 s: more than the exit
        # allows.
        monkeypatch.setattr(agent_mod, "EXIT_SAVE_RESERVE_S", 0.3)
        pid_files = {name: tmp_path / f"{name}.pid" for name in "ab"}
        modes = {"a": "stubborn", "b": "stubborn_hang_initialize"}
        servers = [
            fake_server_config(
                tmp_path / f"{name}.log",
                name=name,
                env={"FAKE_MCP_MODE": mode, "FAKE_MCP_PIDFILE": str(pid_files[name])},
            )
            for name, mode in modes.items()
        ]
        memory_dir = tmp_path / "memory"
        agent, _ = _mcp_agent(
            FakeLLM([_answer("never sent")]),
            servers,
            created,
            memory_dir=str(memory_dir),
        )
        running = asyncio.create_task(agent.run_task("q"))
        b_log = tmp_path / "b.log"
        assert await _until(
            lambda: "initialize" in [m.get("method") for m in read_log(b_log)]
        )
        pids = [int(pid_files[name].read_text()) for name in "ab"]
        try:
            loop = asyncio.get_running_loop()
            began = loop.time()
            agent.begin_exit("Cancelled by SIGTERM", within_s=1.0)
            running.cancel()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await running
                # The waiting ended 0.3 s before the limit and the provider
                # was killed, so there was still time to save the task record.
                assert loop.time() - began < 1.0
            finally:
                await agent.close()
            for pid in pids:
                assert await _until(functools.partial(_gone, pid)), f"{pid} runs"
        finally:
            for pid in pids:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            # private: the start() and close() calls the exit left running
            assert await _until(lambda: not agent._abandoned)
        with contextlib.closing(sqlite3.connect(memory_dir / "history.db")) as db:
            rows = db.execute("SELECT status, result FROM tasks").fetchall()
        assert rows == [("crashed", "Cancelled by SIGTERM")]
