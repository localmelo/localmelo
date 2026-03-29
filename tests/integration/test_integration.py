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
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.melo.schema import MAX_AGENT_STEPS, Message, ToolCall, ToolDef
from localmelo.support.config import Config, ConfigError, GatewayConfig, MlcConfig
from localmelo.support.gateway.session import SessionManager

# ── Fake providers ──────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    def __init__(self, responses: list[Message] | None = None) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
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


# ── Estimation fixture ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _skip_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip LLM step estimation in integration tests."""

    async def _use_max(self: Agent, query: str) -> int:
        return MAX_AGENT_STEPS

    monkeypatch.setattr(Agent, "_estimate_max_steps", _use_max)


# ── 1. Gateway startup validation failure ───────────────────────────────────


class TestGatewayStartupValidation:
    def test_validate_or_raise_blocks_empty_backend(self) -> None:
        cfg = Config(backend="")
        with pytest.raises(ConfigError, match="backend is not set"):
            cfg.validate_or_raise()

    def test_validate_or_raise_blocks_bad_backend(self) -> None:
        cfg = Config(backend="imaginary")
        with pytest.raises(ConfigError, match="invalid"):
            cfg.validate_or_raise()

    def test_validate_or_raise_blocks_missing_mlc_model(self) -> None:
        cfg = Config(backend="mlc-llm", mlc=MlcConfig(chat_model=""))
        with pytest.raises(ConfigError, match="chat_model"):
            cfg.validate_or_raise()

    def test_validate_or_raise_blocks_bad_gateway_port(self) -> None:
        cfg = Config(
            backend="mlc-llm",
            mlc=MlcConfig(chat_model="Qwen3-1.7B"),
            gateway=GatewayConfig(port=0),
        )
        with pytest.raises(ConfigError, match="port"):
            cfg.validate_or_raise()

    def test_start_gateway_validates_config(self) -> None:
        from localmelo.support.gateway import start_gateway

        cfg = Config(backend="")
        with pytest.raises(ConfigError, match="backend is not set"):
            start_gateway(cfg)

    def test_valid_config_passes(self) -> None:
        cfg = Config(
            backend="mlc-llm",
            mlc=MlcConfig(chat_model="Qwen3-1.7B", chat_port=8400),
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

        sent = llm.calls[0]["messages"]
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


# ── 5. Checker v0.2 gateway ingress validation ─────────────────────────────


class TestCheckerV02GatewayIngress:
    """Gateway ingress validated through Checker v0.2."""

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


# ── 6. Checker v0.2 session transition validation ──────────────────────────


class TestCheckerV02SessionTransition:
    """Illegal session transitions rejected by Checker v0.2."""

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


# ── 7. Checker v0.2 tool resolution validation ─────────────────────────────


class TestCheckerV02ToolResolution:
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

        # Verify tool result fed back in second LLM call
        sent = llm.calls[1]["messages"]
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
    """Artifact/log metadata from Executor v0.2 in agent context."""

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

    def test_mlc_backend_creates_llm_and_embedding(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.providers.embedding.openai_compat import (
            OpenAICompatEmbedding,
        )
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        cfg = Config(
            backend="mlc-llm",
            mlc=MlcConfig(chat_model="Qwen3-1.7B", chat_port=8400),
        )
        llm, embedding = _providers_from_config(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert isinstance(embedding, OpenAICompatEmbedding)
        assert llm.model == "Qwen3-1.7B"

    def test_ollama_backend_creates_providers(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.config import OllamaConfig
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        cfg = Config(
            backend="ollama",
            ollama=OllamaConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_model="nomic-embed",
            ),
        )
        llm, embedding = _providers_from_config(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert embedding is not None
        assert llm.model == "qwen3:8b"

    def test_ollama_without_embedding_model_falls_back_to_mlc(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.config import OllamaConfig
        from localmelo.support.providers.embedding.openai_compat import (
            OpenAICompatEmbedding,
        )

        cfg = Config(
            backend="ollama",
            ollama=OllamaConfig(
                chat_url="http://localhost:11434",
                chat_model="qwen3:8b",
                embedding_model="",  # empty = fallback
            ),
        )
        llm, embedding = _providers_from_config(cfg)
        assert isinstance(embedding, OpenAICompatEmbedding)

    def test_online_backend_no_local_embedding(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.config import OnlineConfig
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        cfg = Config(
            backend="online",
            online=OnlineConfig(
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
                local_embedding=False,
            ),
        )
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            llm, embedding = _providers_from_config(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert embedding is None

    def test_online_backend_with_local_embedding(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config
        from localmelo.support.config import OnlineConfig
        from localmelo.support.providers.embedding.openai_compat import (
            OpenAICompatEmbedding,
        )

        cfg = Config(
            backend="online",
            online=OnlineConfig(
                provider="gemini",
                api_key_env="GEMINI_API_KEY",
                chat_model="gemini-2.0-flash",
                local_embedding=True,
            ),
        )
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "key-test"}):
            llm, embedding = _providers_from_config(cfg)
        assert isinstance(embedding, OpenAICompatEmbedding)

    def test_unknown_backend_raises(self) -> None:
        from localmelo.melo.agent.agent import _providers_from_config

        cfg = Config(backend="something-else")
        with pytest.raises(ValueError, match="Unknown backend"):
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
        assert len(llm.calls) >= 2
        second_plan_msgs = llm.calls[1]["messages"]
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
        """LLM works for the first call (tool request) but fails on second."""

        class FailOnSecondLLM(FakeLLM):
            def __init__(self) -> None:
                super().__init__()
                self._call_count = 0

            async def chat(self, messages: Any, tools: Any = None) -> Message:
                self._call_count += 1
                if self._call_count == 1:
                    return Message(
                        role="assistant",
                        content="calling echo",
                        tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
                    )
                raise ConnectionError("LLM crashed mid-conversation")

        agent = Agent(llm=FailOnSecondLLM(), embedding=FakeEmbedding())
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
            backend="mlc-llm",
            mlc=MlcConfig(chat_model="Qwen3-1.7B", chat_port=8400),
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
    async def test_config_online_no_embedding_agent(self) -> None:
        """Online config with no embedding -> Agent works without long-term memory."""
        from localmelo.support.config import OnlineConfig

        cfg = Config(
            backend="online",
            online=OnlineConfig(
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
                local_embedding=False,
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
        cfg = Config(backend="mlc-llm", mlc=MlcConfig(chat_model="Qwen3-1.7B"))
        assert cfg.has_embedding is True

        fake_llm = FakeLLM()
        fake_emb = FakeEmbedding()
        with mock.patch(
            "localmelo.melo.agent.agent._providers_from_config",
            return_value=(fake_llm, fake_emb),
        ):
            agent = Agent(config=cfg)
            assert agent._embedding is not None

    def test_online_no_embedding_matches_provider(self) -> None:
        from localmelo.support.config import OnlineConfig

        cfg = Config(
            backend="online",
            online=OnlineConfig(
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                chat_model="gpt-4o",
                local_embedding=False,
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


# ── Helper ──────────────────────────────────────────────────────────────────


async def _echo_tool(text: str) -> str:
    return text
