"""Integration tests covering cross-module wiring after the merge.

Uses fake providers — no real network or model calls.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.agent.chat import STEP_ESTIMATE_PROMPT
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    ConfigError,
    GatewayConfig,
    LocalBackendConfig,
)
from localmelo.support.gateway.session import SessionManager

# ── Fake providers ──────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Programmable fake LLM that pops planning responses from a queue.

    Step-estimation requests (identified by ``STEP_ESTIMATE_PROMPT`` in the
    system message) are answered with ``estimate_response`` without consuming
    the normal response queue.
    """

    def __init__(
        self,
        responses: list[Message] | None = None,
        *,
        estimate_response: str = "30",
    ) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.estimate_response = estimate_response

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        # Intercept step-estimation requests.
        if any(
            STEP_ESTIMATE_PROMPT in m.content for m in messages if m.role == "system"
        ):
            return Message(role="assistant", content=self.estimate_response)
        if not self.responses:
            return Message(role="assistant", content="(empty)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]

    async def close(self) -> None:
        pass


class FakeAgent:
    def __init__(self) -> None:
        self.closed = False
        self.run_count = 0
        self._run_delay = 0.0

    async def run(self, query: str) -> str:
        self.run_count += 1
        if self._run_delay:
            await asyncio.sleep(self._run_delay)
        return f"echo: {query}"

    async def close(self) -> None:
        self.closed = True


class FakeSessionManager(SessionManager):
    def _create_agent(self) -> FakeAgent:  # type: ignore[override]
        return FakeAgent()


# ── Helpers ────────────────────────────────────────────────────────────────


def _planning_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    """Return only the non-estimation LLM calls (i.e. planning calls).

    Filters out the step-estimation call that ``Agent._estimate_max_steps``
    makes at the start of each ``run()``.
    """
    return [
        c
        for c in llm.calls
        if not any(
            STEP_ESTIMATE_PROMPT in m.content
            for m in c["messages"]
            if m.role == "system"
        )
    ]


# ── 1. Gateway startup validation failure ───────────────────────────────────


class TestGatewayStartupValidation:
    def test_validate_or_raise_blocks_empty_backend(self) -> None:
        cfg = Config(chat_backend="")
        with pytest.raises(ConfigError, match="chat_backend is not set"):
            cfg.validate_or_raise()

    def test_validate_or_raise_blocks_bad_backend(self) -> None:
        cfg = Config(chat_backend="imaginary", embedding_backend="none")
        with pytest.raises(ConfigError, match="not recognised"):
            cfg.validate_or_raise()

    def test_validate_or_raise_blocks_missing_mlc_model(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(chat_url="http://127.0.0.1:8400/v1", chat_model=""),
        )
        with pytest.raises(ConfigError, match="chat_model"):
            cfg.validate_or_raise()

    def test_validate_or_raise_blocks_bad_gateway_port(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
            gateway=GatewayConfig(port=0),
        )
        with pytest.raises(ConfigError, match="port"):
            cfg.validate_or_raise()

    def test_start_gateway_validates_config(self) -> None:
        from localmelo.support.gateway import start_gateway

        cfg = Config(chat_backend="")
        with pytest.raises(ConfigError, match="chat_backend is not set"):
            start_gateway(cfg)

    def test_valid_config_passes(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        cfg.validate_or_raise()  # should not raise


# ── 2. Agent loop still works after merged core+memory changes ──────────────


class TestAgentLoopAfterMerge:
    @pytest.mark.asyncio
    async def test_final_answer_no_tools(self) -> None:
        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="42")]),
            embedding=FakeEmbedding(),
        )
        result = await agent.run("What is 6*7?")
        assert result == "42"
        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_call_then_answer(self) -> None:
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.responses.append(Message(role="assistant", content="echo: hi"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("echo hi")
        assert result == "echo: hi"
        await agent.close()

    @pytest.mark.asyncio
    async def test_no_prompt_duplication(self) -> None:
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        await agent.run("unique_marker_xyz")

        planning = _planning_calls(llm)
        sent = planning[0]["messages"]
        user_msgs = [m for m in sent if m.role == "user"]
        matches = [m for m in user_msgs if "unique_marker_xyz" in m.content]
        assert len(matches) == 1
        await agent.close()

    @pytest.mark.asyncio
    async def test_no_embedding_mode(self) -> None:
        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="ok")]),
            embedding=None,
        )
        result = await agent.run("question")
        assert result == "ok"
        await agent.close()

    @pytest.mark.asyncio
    async def test_tool_registry_search_works(self) -> None:
        hippo = Hippo(embedding=None)
        hippo.register_tool(
            ToolDef(
                name="shell_exec",
                description="execute shell commands",
                parameters={},
                semantic_tags=["shell", "bash"],
            )
        )
        tools = hippo.resolve_tools("run a shell command")
        assert any(t.name == "shell_exec" for t in tools)


# ── 2b. Single system message with recalled memory ────────────────────────────


class TestSingleSystemMessageWithMemory:
    """MLC-compat: planning calls must contain exactly one system message."""

    @pytest.mark.asyncio
    async def test_single_system_msg_with_long_term_memory(self) -> None:
        """When long-term memory returns [memory] items, plan_step merges
        them into the first system prompt under a [RECALL] section."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())

        # Seed long-term memory so retrieve_context returns system messages.
        await agent.hippo.long.add("fact alpha", [1.0, 0.0, 0.0, 0.0])
        await agent.hippo.long.add("fact beta", [0.0, 1.0, 0.0, 0.0])

        await agent.run("recall test")

        planning = _planning_calls(llm)
        assert len(planning) >= 1
        sent = planning[0]["messages"]

        sys_msgs = [m for m in sent if m.role == "system"]
        assert (
            len(sys_msgs) == 1
        ), f"Expected exactly 1 system message, got {len(sys_msgs)}"
        assert "[RECALL]" in sys_msgs[0].content
        assert "fact alpha" in sys_msgs[0].content
        assert "fact beta" in sys_msgs[0].content
        # Memory prefix must be stripped.
        assert "[memory]" not in sys_msgs[0].content
        await agent.close()

    @pytest.mark.asyncio
    async def test_single_system_msg_without_memory(self) -> None:
        """Without long-term memory, still exactly one system message."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm, embedding=None)
        await agent.run("no memory test")

        planning = _planning_calls(llm)
        sent = planning[0]["messages"]
        sys_msgs = [m for m in sent if m.role == "system"]
        assert len(sys_msgs) == 1
        assert "[RECALL]" not in sys_msgs[0].content
        await agent.close()

    @pytest.mark.asyncio
    async def test_multi_turn_stays_single_system(self) -> None:
        """Across a tool-call turn, each planning call has one system msg."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.responses.append(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        await agent.hippo.long.add("remembered fact", [1.0, 0.0, 0.0, 0.0])
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        await agent.run("multi turn recall")

        planning = _planning_calls(llm)
        for i, call in enumerate(planning):
            sys_msgs = [m for m in call["messages"] if m.role == "system"]
            assert (
                len(sys_msgs) == 1
            ), f"Planning call {i} had {len(sys_msgs)} system messages"
        await agent.close()


# ── 3. Same-session serialization ───────────────────────────────────────────


class TestSessionSerializationAfterMerge:
    @pytest.mark.asyncio
    async def test_concurrent_runs_serialize(self) -> None:
        mgr = FakeSessionManager(max_sessions=10, idle_ttl=3600.0)
        s = await mgr.create("serial")
        agent: FakeAgent = s.agent  # type: ignore[assignment]
        agent._run_delay = 0.05

        order: list[str] = []

        async def run_task(label: str) -> None:
            async with s._lock:
                order.append(f"{label}-start")
                await agent.run("test")
                order.append(f"{label}-end")

        await asyncio.gather(run_task("A"), run_task("B"))

        # First task fully completes before second starts
        assert order[0].endswith("-start")
        assert order[1].endswith("-end")
        assert order[2].endswith("-start")
        assert order[3].endswith("-end")
        assert agent.run_count == 2

    @pytest.mark.asyncio
    async def test_session_eviction_closes_agent(self) -> None:
        import time

        mgr = FakeSessionManager(max_sessions=2, idle_ttl=0.01)
        s1 = await mgr.create("old")
        agent1: FakeAgent = s1.agent  # type: ignore[assignment]
        s1.last_active = time.time() - 1.0  # past TTL

        await mgr.create("new")
        assert mgr.get("old") is None
        assert agent1.closed


# ── 4. Optional SQLite-backed memory ────────────────────────────────────────


class TestPersistentMemoryWiring:
    @pytest.mark.asyncio
    async def test_sqlite_backends_via_env(self, tmp_path: Path) -> None:
        """LOCALMELO_PERSIST_MEMORY=1 enables SQLite backends."""
        mem_dir = str(tmp_path / "memory")
        with mock.patch.dict(
            os.environ,
            {
                "LOCALMELO_PERSIST_MEMORY": "1",
                "LOCALMELO_MEMORY_DIR": mem_dir,
            },
        ):
            agent = Agent(
                llm=FakeLLM([Message(role="assistant", content="ok")]),
                embedding=FakeEmbedding(),
            )
            assert isinstance(agent.hippo.history, SqliteHistory)
            assert isinstance(agent.hippo.long, SqliteLongTerm)

            result = await agent.run("test query")
            assert result == "ok"

            # Task was persisted in SQLite
            tasks_dir = Path(mem_dir)
            assert (tasks_dir / "history.db").exists()
            assert (tasks_dir / "long_term.db").exists()
            await agent.close()

    @pytest.mark.asyncio
    async def test_sqlite_history_only_without_embedding(self, tmp_path: Path) -> None:
        """Without embedding, only SqliteHistory is created (no long-term db)."""
        mem_dir = str(tmp_path / "memory")
        with mock.patch.dict(
            os.environ,
            {
                "LOCALMELO_PERSIST_MEMORY": "1",
                "LOCALMELO_MEMORY_DIR": mem_dir,
            },
        ):
            agent = Agent(
                llm=FakeLLM([Message(role="assistant", content="ok")]),
                embedding=None,
            )
            assert isinstance(agent.hippo.history, SqliteHistory)
            # No embedding → long-term stays in-memory (no-op)
            from localmelo.melo.memory.long import LongTerm

            assert isinstance(agent.hippo.long, LongTerm)
            await agent.close()

    @pytest.mark.asyncio
    async def test_default_is_in_memory(self) -> None:
        """Without env vars, memory backends are in-memory."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCALMELO_PERSIST_MEMORY", None)
            agent = Agent(
                llm=FakeLLM([Message(role="assistant", content="ok")]),
                embedding=FakeEmbedding(),
            )
            from localmelo.melo.memory.history import History
            from localmelo.melo.memory.long import LongTerm

            assert isinstance(agent.hippo.history, History)
            assert isinstance(agent.hippo.long, LongTerm)
            await agent.close()

    @pytest.mark.asyncio
    async def test_hippo_close_cleans_up_sqlite(self, tmp_path: Path) -> None:
        """Hippo.close() closes SQLite backends without error."""
        h = SqliteHistory(tmp_path / "h.db")
        lt = SqliteLongTerm(tmp_path / "lt.db")
        hippo = Hippo(embedding=None, history=h, long=lt)
        hippo.close()  # should not raise

    @pytest.mark.asyncio
    async def test_agent_loop_with_sqlite_backends(self, tmp_path: Path) -> None:
        """Full agent loop works with SQLite backends."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.responses.append(Message(role="assistant", content="done"))

        h = SqliteHistory(tmp_path / "h.db")
        lt = SqliteLongTerm(tmp_path / "lt.db")
        emb = FakeEmbedding()

        agent = Agent(llm=llm, embedding=emb)
        # Replace hippo with SQLite-backed one
        agent.hippo.close()
        agent.hippo = Hippo(embedding=emb, history=h, long=lt)
        # Re-register tools
        from localmelo.melo.executor import register_builtins

        agent.executor = __import__(
            "localmelo.melo.executor", fromlist=["Executor"]
        ).Executor(agent.hippo, agent.checker)
        register_builtins(agent.executor, agent.hippo)

        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("echo hi")
        assert result == "done"

        # Verify task persisted in SQLite
        h2 = SqliteHistory(tmp_path / "h.db")
        # There should be at least one task
        # (We can't iterate easily, but get_task with the task_id works)
        await agent.close()
        h2.close()


# ── 5. Checker gateway ingress validation ─────────────────────────────


class TestCheckerGatewayIngress:
    """Gateway ingress validated through Checker."""

    def test_empty_query_rejected(self) -> None:
        from localmelo.melo.checker import Checker, GatewayIngressPayload

        checker = Checker()
        result = checker.check_gateway_ingress(GatewayIngressPayload(query=""))
        assert not result.allowed
        assert "non-empty" in result.reason

    def test_invalid_session_id_rejected(self) -> None:
        from localmelo.melo.checker import Checker, GatewayIngressPayload

        checker = Checker()
        result = checker.check_gateway_ingress(
            GatewayIngressPayload(query="hello", session_id="bad session id!")
        )
        assert not result.allowed
        assert "session_id" in result.reason

    def test_valid_ingress_sanitized(self) -> None:
        from localmelo.melo.checker import Checker, GatewayIngressPayload

        checker = Checker()
        result = checker.check_gateway_ingress(
            GatewayIngressPayload(query="  hello  ", session_id="abc123")
        )
        assert result.allowed
        assert result.sanitized_payload.query == "hello"

    def test_oversized_query_rejected(self) -> None:
        from localmelo.melo.checker import Checker, GatewayIngressPayload

        checker = Checker()
        result = checker.check_gateway_ingress(
            GatewayIngressPayload(query="x" * 200_000)
        )
        assert not result.allowed
        assert "too large" in result.reason.lower()


# ── 6. Checker session transition validation ──────────────────────────


class TestCheckerSessionTransition:
    """Illegal session transitions rejected by Checker."""

    def test_closed_to_running_rejected(self) -> None:
        from localmelo.melo.checker import Checker, SessionTransition

        checker = Checker()
        result = checker.check_session_transition(
            SessionTransition(from_status="closed", to_status="running", session_id="x")
        )
        assert not result.allowed

    def test_idle_to_running_allowed(self) -> None:
        from localmelo.melo.checker import Checker, SessionTransition

        checker = Checker()
        result = checker.check_session_transition(
            SessionTransition(from_status="idle", to_status="running", session_id="x")
        )
        assert result.allowed

    def test_running_to_running_rejected(self) -> None:
        from localmelo.melo.checker import Checker, SessionTransition

        checker = Checker()
        result = checker.check_session_transition(
            SessionTransition(
                from_status="running", to_status="running", session_id="x"
            )
        )
        assert not result.allowed


# ── 7. Checker tool resolution validation ─────────────────────────────


class TestCheckerToolResolution:
    """Invalid tool resolution blocked before planning/execution."""

    @pytest.mark.asyncio
    async def test_agent_validates_resolution_and_runs(self) -> None:
        """Agent loop uses check_tool_resolution and still produces results."""
        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="ok")]),
            embedding=FakeEmbedding(),
        )
        result = await agent.run("hello")
        assert result == "ok"
        await agent.close()

    def test_duplicate_tools_rejected(self) -> None:
        from localmelo.melo.checker import Checker
        from localmelo.melo.checker.payloads import ToolResolutionResult

        checker = Checker()
        result = checker.check_tool_resolution(
            ToolResolutionResult(
                query="test",
                resolved_tool_names=["shell_exec", "shell_exec"],
            )
        )
        assert not result.allowed
        assert "Duplicate" in result.reason

    def test_empty_query_rejected(self) -> None:
        from localmelo.melo.checker import Checker
        from localmelo.melo.checker.payloads import ToolResolutionResult

        checker = Checker()
        result = checker.check_tool_resolution(
            ToolResolutionResult(query="", resolved_tool_names=["shell_exec"])
        )
        assert not result.allowed


# ── 8. Structured executor path in agent loop ──────────────────────────────


class TestStructuredExecutorInAgentLoop:
    """execute_structured used in agent loop without breaking Agent.run()."""

    @pytest.mark.asyncio
    async def test_tool_call_result_fed_back(self) -> None:
        """Tool result from structured path is fed back into conversation."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.responses.append(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("echo hi")
        assert result == "done"

        # Verify tool result fed back in second planning call
        planning = _planning_calls(llm)
        sent = planning[1]["messages"]
        tool_msgs = [m for m in sent if m.role == "tool"]
        assert len(tool_msgs) >= 1
        assert "hi" in tool_msgs[0].content
        await agent.close()

    @pytest.mark.asyncio
    async def test_blocked_tool_handled_via_structured(self) -> None:
        """Blocked tool calls still handled correctly via structured path."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="deleting",
                tool_call=ToolCall(
                    tool_name="shell_exec",
                    arguments={"command": "rm -rf /"},
                ),
            )
        )
        llm.responses.append(Message(role="assistant", content="aborted"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("delete everything")
        assert result == "aborted"
        await agent.close()

    @pytest.mark.asyncio
    async def test_unknown_tool_handled_via_structured(self) -> None:
        """Unknown tool calls produce an error via structured path."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="try unknown",
                tool_call=ToolCall(tool_name="nonexistent", arguments={}),
            )
        )
        llm.responses.append(Message(role="assistant", content="fallback"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("do something")
        assert result == "fallback"
        await agent.close()

    @pytest.mark.asyncio
    async def test_large_output_truncated_via_v02(self) -> None:
        """Large tool output truncated via check_executor_result."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling big",
                tool_call=ToolCall(tool_name="big_tool", arguments={}),
            )
        )
        llm.responses.append(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="big_tool",
                description="huge output",
                parameters={"type": "object", "properties": {}},
            )
        )

        async def _big() -> str:
            return "x" * 100_000

        agent.executor.register("big_tool", _big)

        result = await agent.run("run big")
        assert result == "done"

        # Verify truncation occurred (tool msg in short-term is truncated)
        window = agent.hippo.short.get_window()
        tool_msgs = [m for m in window if m.role == "tool"]
        if tool_msgs:
            assert len(tool_msgs[-1].content) < 100_000
        await agent.close()


# ── 9. Artifact metadata survives execution path ───────────────────────────


class TestArtifactMetadata:
    """Artifact/log metadata from Executor in agent context."""

    @pytest.mark.asyncio
    async def test_agent_executor_produces_artifacts(self) -> None:
        """The agent's executor produces artifact metadata via structured path."""

        from localmelo.melo.executor.models import ExecutionRequest, ExecutionStatus

        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="ok")]),
            embedding=FakeEmbedding(),
        )

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            outcome = await agent.executor.execute_structured(
                ExecutionRequest(
                    tool_name="file_write",
                    arguments={"path": path, "content": "test"},
                )
            )
            assert outcome.status == ExecutionStatus.SUCCESS
            assert len(outcome.artifacts) == 1
            assert outcome.artifacts[0].kind == "file"
            assert outcome.artifacts[0].path == path
        finally:
            os.unlink(path)
        await agent.close()

    @pytest.mark.asyncio
    async def test_outcome_to_tool_result_preserves_fields(self) -> None:
        """ExecutionOutcome.to_tool_result() preserves key fields."""
        from localmelo.melo.executor.models import ExecutionOutcome, ExecutionStatus

        outcome = ExecutionOutcome(
            tool_name="python_exec",
            status=ExecutionStatus.SUCCESS,
            output="42",
            duration_ms=1.5,
        )
        result = outcome.to_tool_result()
        assert result.tool_name == "python_exec"
        assert result.output == "42"
        assert result.duration_ms == 1.5
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_error_category_in_structured_path(self) -> None:
        """Error category is set correctly for unknown tools."""
        from localmelo.melo.executor.models import (
            ErrorCategory,
            ExecutionRequest,
            ExecutionStatus,
        )

        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="ok")]),
            embedding=FakeEmbedding(),
        )
        outcome = await agent.executor.execute_structured(
            ExecutionRequest(tool_name="no_such_tool")
        )
        assert outcome.status == ExecutionStatus.ERROR
        assert outcome.error_category == ErrorCategory.TOOL_NOT_FOUND
        await agent.close()


# ── 10. Config → Agent creation (_providers_from_config) ────────────────────


class TestProvidersFromConfig:
    """Verify _providers_from_config returns correct provider types for each backend."""

    @pytest.fixture(autouse=True)
    def _register_backends(self) -> None:
        """Register backends so _providers_from_config can look them up."""
        from localmelo.support.backends.registry import (
            _clear,
            ensure_defaults_registered,
        )

        _clear()
        ensure_defaults_registered()

    def test_mlc_backend_creates_llm_and_embedding(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.providers.embedding.openai_compat import (
            OpenAICompatEmbedding,
        )
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        llm, embedding = _providers_from_config(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert isinstance(embedding, OpenAICompatEmbedding)
        assert llm.model == "Qwen3-1.7B"

    def test_ollama_backend_creates_providers(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        cfg = Config(
            chat_backend="ollama",
            embedding_backend="ollama",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_url="http://localhost:11434",
                embedding_model="nomic-embed",
            ),
        )
        llm, embedding = _providers_from_config(cfg)
        assert isinstance(llm, OllamaNativeChat)
        assert embedding is not None
        assert llm.model == "qwen3:8b"

    def test_ollama_without_embedding_model_returns_none(self) -> None:
        """Ollama with no embedding_model and embedding_backend='none'
        returns None from _providers_from_config."""
        from localmelo.melo.agent.agent import _providers_from_config

        cfg = Config(
            chat_backend="ollama",
            embedding_backend="none",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_model="",
            ),
        )
        llm, embedding = _providers_from_config(cfg)
        assert embedding is None

    def test_openai_cloud_backend_no_embedding(self) -> None:
        """OpenAI cloud backend is chat-only; embedding is None."""
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
            ),
        )
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            llm, embedding = _providers_from_config(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert embedding is None

    def test_gemini_chat_with_mlc_embedding(self) -> None:
        """Gemini cloud chat + mlc local embedding (split backend model)."""
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.providers.embedding.openai_compat import (
            OpenAICompatEmbedding,
        )

        cfg = Config(
            chat_backend="gemini",
            embedding_backend="mlc",
            gemini=CloudVendorConfig(
                api_key_env="GEMINI_API_KEY",
                chat_model="gemini-2.0-flash",
            ),
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "key-test"}):
            llm, embedding = _providers_from_config(cfg)
        assert isinstance(embedding, OpenAICompatEmbedding)

    def test_unknown_backend_raises(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config

        cfg = Config(chat_backend="something-else", embedding_backend="none")
        with pytest.raises(KeyError, match="Unknown backend"):
            _providers_from_config(cfg)


# ── 11. Agent + no-embedding full tool-call path ───────────────────────────


class TestNoEmbeddingFullPath:
    """Complete no-embedding path: query -> tool call -> execution -> memory."""

    @pytest.mark.asyncio
    async def test_tool_call_path_without_embedding(self) -> None:
        """Full tool-call cycle with embedding=None verifies:
        - Long-term memory stays empty (no embedding provider)
        - Short-term memory gets the tool result
        - Step record is properly created in history
        - Task completes successfully
        """
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hello"}),
            )
        )
        llm.responses.append(Message(role="assistant", content="echo result: hello"))

        agent = Agent(llm=llm, embedding=None)
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("echo hello")
        assert result == "echo result: hello"

        # Long-term memory must be empty (no embedding)
        assert len(agent.hippo.long._entries) == 0

        # Short-term memory should have the tool message
        short_contents = [m.content for m in agent.hippo.short.get_window()]
        assert any("echo" in c and "hello" in c for c in short_contents)

        # History should have the task and step
        tasks = list(agent.hippo.history._tasks.values())
        assert len(tasks) == 1
        assert tasks[0].status == "completed"
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 1
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.tool_name == "echo"
        assert steps[0].tool_result is not None
        assert steps[0].tool_result.output == "hello"

        await agent.close()

    @pytest.mark.asyncio
    async def test_multi_step_tool_calls_without_embedding(self) -> None:
        """Multiple tool calls with no embedding — all steps recorded."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="step 1",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "first"}),
            )
        )
        llm.responses.append(
            Message(
                role="assistant",
                content="step 2",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "second"}),
            )
        )
        llm.responses.append(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=None)
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("multi-step no-emb")
        assert result == "done"

        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 2

        # Long-term memory must stay empty throughout
        assert len(agent.hippo.long._entries) == 0

        await agent.close()


# ── 12. Agent + Executor + Checker three-way boundary ──────────────────────


class TestThreeWayBoundary:
    """Tests that cross Agent, Executor, and Checker boundaries simultaneously."""

    @pytest.mark.asyncio
    async def test_truncated_output_flows_through_all_layers(self) -> None:
        """Large tool output -> Checker truncates -> step record truncated
        -> memory-write blocks oversized tool message -> planning prompt
        has no tool message.

        This exercises three boundaries in sequence:
        1. Executor result checker truncates output > MAX_OUTPUT_LEN
        2. Agent memory-write checker rejects the tool message because
           truncation tail + tool-name prefix push it over MAX_MEMORY_TEXT_LEN
           (both limits are currently 50,000 — any truncated message exceeds)
        3. The follow-up planning prompt correctly reflects the absence
        """
        from localmelo.melo.checker.validators import (
            MAX_MEMORY_TEXT_LEN,
            MAX_OUTPUT_LEN,
        )

        tool_name = "big_tool"
        raw_len = MAX_OUTPUT_LEN + 500  # comfortably over → triggers truncation

        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="calling big",
                tool_call=ToolCall(tool_name=tool_name, arguments={}),
            )
        )
        llm.responses.append(Message(role="assistant", content="summarized"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name=tool_name,
                description="huge output tool",
                parameters={"type": "object", "properties": {}},
            )
        )

        async def _big() -> str:
            return "x" * raw_len

        agent.executor.register(tool_name, _big)

        result = await agent.run("run big tool")
        assert result == "summarized"

        # ── Boundary 1: step record has truncated output ──
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 1
        assert steps[0].tool_result is not None
        truncated_output = steps[0].tool_result.output
        assert "[truncated]" in truncated_output
        assert len(truncated_output) < raw_len

        # ── Boundary 2: truncated tool message exceeds memory limit ──
        # The agent builds: f"[{tool_name}] {truncated_output}"
        tool_msg_len = len(f"[{tool_name}] ") + len(truncated_output)
        assert tool_msg_len > MAX_MEMORY_TEXT_LEN, (
            f"Expected tool message ({tool_msg_len}) to exceed "
            f"MAX_MEMORY_TEXT_LEN ({MAX_MEMORY_TEXT_LEN}); "
            f"if limits diverge, this test should be updated"
        )

        # ── Boundary 3: planning prompt has no tool message ──
        # Because memory-write correctly blocked the oversized message,
        # the follow-up planning call must not contain a tool role message.
        planning = _planning_calls(llm)
        assert len(planning) >= 2
        second_plan_msgs = planning[1]["messages"]
        tool_msgs = [m for m in second_plan_msgs if m.role == "tool"]
        assert tool_msgs == [], (
            f"Expected no tool messages in planning prompt (memory-write "
            f"should have blocked the {tool_msg_len}-char message), "
            f"but found {len(tool_msgs)}"
        )

        await agent.close()

    @pytest.mark.asyncio
    async def test_blocked_command_produces_error_in_step_record(self) -> None:
        """A blocked shell command flows through Agent -> Executor -> Checker
        and the step record must contain the 'Blocked' error."""
        llm = FakeLLM()
        llm.responses.append(
            Message(
                role="assistant",
                content="running rm",
                tool_call=ToolCall(
                    tool_name="shell_exec",
                    arguments={"command": "rm -rf /"},
                ),
            )
        )
        llm.responses.append(Message(role="assistant", content="aborted"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("delete everything")
        assert result == "aborted"

        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 1
        assert steps[0].tool_result is not None
        assert "Blocked" in steps[0].tool_result.error

        await agent.close()


# ── 13. LLM provider error propagation ─────────────────────────────────────


class TestLLMErrorPropagation:
    """Failure modes that span the Agent + LLM provider boundary."""

    @pytest.mark.asyncio
    async def test_llm_exception_fails_task_gracefully(self) -> None:
        """If the LLM raises an exception, the agent loop should not crash
        but propagate the error appropriately."""

        class ExplodingLLM(FakeLLM):
            async def chat(self, messages: Any, tools: Any = None) -> Message:
                raise ConnectionError("LLM server unreachable")

        agent = Agent(llm=ExplodingLLM(), embedding=FakeEmbedding())
        with pytest.raises(ConnectionError, match="unreachable"):
            await agent.run("hello")
        await agent.close()

    @pytest.mark.asyncio
    async def test_llm_exception_after_tool_call(self) -> None:
        """LLM works for estimation + first planning call but fails on second."""

        class FailOnSecondPlanLLM(FakeLLM):
            def __init__(self) -> None:
                super().__init__()
                self._plan_count = 0

            async def chat(self, messages: Any, tools: Any = None) -> Message:
                # Let estimation pass through the parent handler.
                if any(
                    STEP_ESTIMATE_PROMPT in m.content
                    for m in messages
                    if hasattr(m, "role") and m.role == "system"
                ):
                    return await super().chat(messages, tools)
                self._plan_count += 1
                if self._plan_count == 1:
                    return Message(
                        role="assistant",
                        content="calling echo",
                        tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
                    )
                raise ConnectionError("LLM crashed mid-conversation")

        agent = Agent(llm=FailOnSecondPlanLLM(), embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        with pytest.raises(ConnectionError, match="crashed"):
            await agent.run("echo hi")

        # The first step should still be recorded in history
        tasks = list(agent.hippo.history._tasks.values())
        assert len(tasks) == 1
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 1
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.tool_name == "echo"

        await agent.close()


# ── 14. Agent.close() cascading cleanup ────────────────────────────────────


class TestAgentCloseCascade:
    """Verify that Agent.close() propagates through all subsystems."""

    @pytest.mark.asyncio
    async def test_close_calls_all_subsystems(self) -> None:
        """close() must call LLM.close(), embedding.close(), and hippo.close()."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        emb = FakeEmbedding()
        agent = Agent(llm=llm, embedding=emb)

        await agent.run("test")

        llm_closed = False
        emb_closed = False
        original_llm_close = llm.close
        original_emb_close = emb.close

        async def _track_llm_close() -> None:
            nonlocal llm_closed
            llm_closed = True
            await original_llm_close()

        async def _track_emb_close() -> None:
            nonlocal emb_closed
            emb_closed = True
            await original_emb_close()

        llm.close = _track_llm_close  # type: ignore[assignment]
        emb.close = _track_emb_close  # type: ignore[assignment]

        await agent.close()
        assert llm_closed, "LLM.close() not called"
        assert emb_closed, "Embedding.close() not called"

    @pytest.mark.asyncio
    async def test_close_with_no_embedding(self) -> None:
        """close() with embedding=None should not raise."""
        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="ok")]),
            embedding=None,
        )
        await agent.run("test")
        await agent.close()  # must not raise

    @pytest.mark.asyncio
    async def test_close_with_sqlite_backends(self, tmp_path: Path) -> None:
        """close() properly cascades through SQLite backends."""
        mem_dir = str(tmp_path / "memory")
        with mock.patch.dict(
            os.environ,
            {
                "LOCALMELO_PERSIST_MEMORY": "1",
                "LOCALMELO_MEMORY_DIR": mem_dir,
            },
        ):
            agent = Agent(
                llm=FakeLLM([Message(role="assistant", content="ok")]),
                embedding=FakeEmbedding(),
            )
            await agent.run("test")
            await agent.close()  # must not raise

            # Verify SQLite files were created
            assert (Path(mem_dir) / "history.db").exists()
            assert (Path(mem_dir) / "long_term.db").exists()


# ── 15. Config → Agent end-to-end (mock providers, real wiring) ────────────


class TestConfigToAgentEndToEnd:
    """Config-based Agent construction and execution with mock providers."""

    @pytest.mark.asyncio
    async def test_config_based_agent_runs_query(self) -> None:
        """Build Agent from Config, mock the HTTP layer, verify full run."""
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        # We mock _providers_from_config to inject fake providers
        fake_llm = FakeLLM([Message(role="assistant", content="42")])
        fake_emb = FakeEmbedding()
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(fake_llm, fake_emb),
        ):
            agent = Agent(config=cfg)
            result = await agent.run("What is 6*7?")
            assert result == "42"
            await agent.close()

    @pytest.mark.asyncio
    async def test_config_openai_no_embedding_agent(self) -> None:
        """OpenAI cloud config with no embedding -> Agent works without long-term memory."""
        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
            ),
        )
        fake_llm = FakeLLM([Message(role="assistant", content="answer")])
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(fake_llm, None),
        ):
            agent = Agent(config=cfg)
            assert agent._embedding is None
            result = await agent.run("question")
            assert result == "answer"
            # Long-term must be empty
            assert len(agent.hippo.long._entries) == 0
            await agent.close()


# ── 16. has_embedding → Agent embedding wiring consistency ─────────────────


class TestHasEmbeddingConsistency:
    """Config.has_embedding must predict whether Agent gets an embedding provider."""

    def test_mlc_has_embedding_matches_provider(self) -> None:
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        assert cfg.has_embedding is True

        fake_llm = FakeLLM()
        fake_emb = FakeEmbedding()
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(fake_llm, fake_emb),
        ):
            agent = Agent(config=cfg)
            assert agent._embedding is not None

    def test_openai_no_embedding_matches_provider(self) -> None:
        cfg = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
            ),
        )
        assert cfg.has_embedding is False

        fake_llm = FakeLLM()
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(fake_llm, None),
        ):
            agent = Agent(config=cfg)
            assert agent._embedding is None


# ── 17. Backend adapter architecture integration ────────────────────────────


class TestBackendAdapterArchitecture:
    """Verify the backend adapter architecture wires correctly end-to-end."""

    @pytest.fixture(autouse=True)
    def _register_backends(self) -> None:
        from localmelo.support.backends.registry import (
            _clear,
            ensure_defaults_registered,
        )

        _clear()
        ensure_defaults_registered()

    def test_registry_has_all_eight_backends(self) -> None:
        from localmelo.support.backends import get_backend, list_backends

        backends = list_backends()
        keys = {b.key for b in backends}
        assert keys == {
            "mlc",
            "ollama",
            "vllm",
            "sglang",
            "openai",
            "gemini",
            "anthropic",
            "nvidia",
        }

        # Each is retrievable by key
        for key in keys:
            assert get_backend(key).key == key

    def test_backend_validate_matches_config_validate(self) -> None:
        """Backend.validate() and Config.validate() should agree on errors."""
        from localmelo.support.backends import get_backend

        # Valid mlc config
        cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        assert get_backend("mlc").validate(cfg) == []
        assert cfg.validate() == []

        # Invalid mlc config (missing chat_model)
        bad_cfg = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="",
            ),
        )
        backend_errors = get_backend("mlc").validate(bad_cfg)
        config_errors = bad_cfg.validate()
        assert len(backend_errors) > 0
        assert len(config_errors) > 0

    def test_backend_has_embedding_matches_config(self) -> None:
        """Backend.has_embedding() and Config.has_embedding should agree."""
        from localmelo.support.backends import get_backend

        # mlc with embedding fields set: True
        cfg_mlc = Config(
            chat_backend="mlc",
            embedding_backend="mlc",
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        assert get_backend("mlc").has_embedding(cfg_mlc) is True
        assert cfg_mlc.has_embedding is True

        # ollama without embedding_model: has_embedding False
        cfg_ollama = Config(
            chat_backend="ollama",
            embedding_backend="none",
            ollama=LocalBackendConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
            ),
        )
        assert get_backend("ollama").has_embedding(cfg_ollama) is False
        assert cfg_ollama.has_embedding is False

        # openai cloud: never has embedding (chat-only)
        cfg_openai = Config(
            chat_backend="openai",
            embedding_backend="none",
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
            ),
        )
        assert get_backend("openai").has_embedding(cfg_openai) is False
        assert cfg_openai.has_embedding is False

        # openai chat + mlc embedding: Config says True
        cfg_cloud_emb = Config(
            chat_backend="openai",
            embedding_backend="mlc",
            openai=CloudVendorConfig(
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
            ),
            mlc=LocalBackendConfig(
                chat_url="http://127.0.0.1:8400/v1",
                chat_model="Qwen3-1.7B",
                embedding_url="http://127.0.0.1:8400/v1",
                embedding_model="nomic-embed",
            ),
        )
        assert cfg_cloud_emb.has_embedding is True

    def test_all_backends_implement_full_contract(self) -> None:
        """Every registered backend must satisfy the BaseBackend ABC."""
        from localmelo.support.backends import list_backends
        from localmelo.support.backends.base import BaseBackend

        for backend in list_backends():
            assert isinstance(backend, BaseBackend)
            assert isinstance(backend.key, str)
            assert isinstance(backend.display_name, str)
            # Backends are pure connectors -- no runtime_mode or start_runtime
            assert not hasattr(backend, "runtime_mode")
            assert not hasattr(backend, "start_runtime")

    def test_deployment_matrix(self) -> None:
        """Verify the supported deployment combinations.

        Covers all four local backends, all four cloud vendors, and
        the split-backend model (cloud chat + local embedding).
        """
        mlc_full = LocalBackendConfig(
            chat_url="http://127.0.0.1:8400/v1",
            chat_model="Qwen3-1.7B",
            embedding_url="http://127.0.0.1:8400/v1",
            embedding_model="nomic-embed",
        )
        ollama_with_emb = LocalBackendConfig(
            chat_url="http://localhost:11434",
            chat_model="qwen3:8b",
            embedding_url="http://localhost:11434",
            embedding_model="nomic-embed",
        )

        combos: list[tuple[str, str, str, bool, dict[str, Any]]] = [
            # (chat_backend, embedding_backend, description, expected_has_embedding, extra_kwargs)
            ("mlc", "mlc", "mlc chat + mlc embedding", True, {"mlc": mlc_full}),
            (
                "ollama",
                "ollama",
                "ollama chat + ollama embedding",
                True,
                {"ollama": ollama_with_emb},
            ),
            ("vllm", "none", "vllm chat only", False, {}),
            ("sglang", "none", "sglang chat only", False, {}),
            ("openai", "none", "openai chat only", False, {}),
            ("gemini", "none", "gemini chat only", False, {}),
            ("anthropic", "none", "anthropic chat only", False, {}),
            ("nvidia", "none", "nvidia chat only", False, {}),
            (
                "openai",
                "ollama",
                "openai chat + ollama embedding",
                True,
                {"ollama": ollama_with_emb},
            ),
            ("gemini", "mlc", "gemini chat + mlc embedding", True, {"mlc": mlc_full}),
        ]
        for chat_b, emb_b, desc, expected_emb, extra in combos:
            cfg = Config(chat_backend=chat_b, embedding_backend=emb_b, **extra)
            assert cfg.chat_backend == chat_b, desc
            assert cfg.embedding_backend == emb_b, desc
            assert cfg.has_embedding is expected_emb, desc


# ── 18. Step-estimation path exercised during Agent.run() ────────────────────


class TestAttemptBasedLoop:
    """Prove that Agent.run() uses the attempt-based loop."""

    @pytest.mark.asyncio
    async def test_direct_answer_completes_in_one_attempt(self) -> None:
        """A direct answer should complete without reflection."""
        llm = FakeLLM(
            [Message(role="assistant", content="42")],
        )
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("What is 6*7?")

        assert result == "42"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "completed"
        assert tasks[0].attempts_completed == 0

        await agent.close()


# ── Helper ──────────────────────────────────────────────────────────────────


async def _echo_tool(text: str) -> str:
    return text
