"""Integration tests covering cross-module wiring after the merge.

Uses fake providers — no real network or model calls.
"""

from __future__ import annotations

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
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.backends import providers_for
from localmelo.support.config import (
    CloudVendorConfig,
    Config,
    ConfigError,
    LocalBackendConfig,
)

# ── Fake providers ──────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Programmable fake LLM that pops planning responses from a queue."""

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


# ── Helpers ────────────────────────────────────────────────────────────────


# ── 1. Gateway startup validation failure ───────────────────────────────────


class TestGatewayStartupValidation:
    def test_start_gateway_validates_config(self) -> None:
        from localmelo.support.gateway import start_gateway

        cfg = Config(chat_backend="")
        with pytest.raises(ConfigError, match="chat_backend is not set"):
            start_gateway(cfg)


# ── 2b. Single system message ─────────────────────────────────────────────────


class TestSingleSystemMessage:
    """MLC-compat: planning calls must contain exactly one system message."""

    @pytest.mark.asyncio
    async def test_single_system_msg_without_memory(self) -> None:
        """Without long-term memory, still exactly one system message."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm, embedding=None)
        await agent.run("no memory test")

        planning = llm.calls
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

        await agent.run("echo multi turn recall")

        planning = llm.calls
        for i, call in enumerate(planning):
            sys_msgs = [m for m in call["messages"] if m.role == "system"]
            assert (
                len(sys_msgs) == 1
            ), f"Planning call {i} had {len(sys_msgs)} system messages"
        await agent.close()


# ── 4. Optional SQLite-backed memory ────────────────────────────────────────


class TestPersistentMemoryWiring:
    @pytest.mark.asyncio
    async def test_hippo_close_cleans_up_sqlite(self, tmp_path: Path) -> None:
        """Hippo.close() closes SQLite backends without error."""
        h = SqliteHistory(tmp_path / "h.db")
        lt = SqliteLongTerm(tmp_path / "lt.db")
        hippo = Hippo(embedding=None, history=h, long=lt)
        hippo.close()  # should not raise


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


# ── 10. Config → providers (providers_for) ──────────────────────────────────


class TestProvidersFor:
    """Verify providers_for returns correct provider types for each backend."""

    @pytest.fixture(autouse=True)
    def _register_backends(self) -> None:
        """Register backends so providers_for can look them up."""
        from localmelo.support.backends.registry import (
            _clear,
            ensure_defaults_registered,
        )

        _clear()
        ensure_defaults_registered()

    def test_mlc_backend_creates_llm_and_embedding(self) -> None:
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
        llm, embedding = providers_for(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert isinstance(embedding, OpenAICompatEmbedding)
        assert llm.model == "Qwen3-1.7B"

    def test_ollama_backend_creates_providers(self) -> None:
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
        llm, embedding = providers_for(cfg)
        assert isinstance(llm, OllamaNativeChat)
        assert embedding is not None
        assert llm.model == "qwen3:8b"

    def test_openai_cloud_backend_no_embedding(self) -> None:
        """OpenAI cloud backend is chat-only; embedding is None."""
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
            llm, embedding = providers_for(cfg)
        assert isinstance(llm, OpenAICompatLLM)
        assert embedding is None

    def test_gemini_chat_with_mlc_embedding(self) -> None:
        """Gemini cloud chat + mlc local embedding (split backend model)."""
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
            llm, embedding = providers_for(cfg)
        assert isinstance(embedding, OpenAICompatEmbedding)


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

        # Short-term memory holds the echo call and its paired result
        window = agent.hippo.working.get_window()
        calls = [tc for m in window for tc in m.tool_calls]
        results = [m for m in window if m.role == "tool"]
        assert [tc.tool_name for tc in calls] == ["echo"]
        assert [(m.tool_call_id, m.content) for m in results] == [
            (calls[0].id, "hello")
        ]

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

        result = await agent.run("echo multi-step no-emb")
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
        -> the tool message is too long for one memory write -> planning
        prompt carries the start of the output, cut to fit, bound to the
        call.

        This exercises three boundaries in sequence:
        1. Executor result checker truncates output > MAX_OUTPUT_LEN
        2. The truncation tail pushes the tool message over
           MAX_MEMORY_TEXT_LEN (both limits are currently 50,000, so any
           truncated message exceeds it). The agent's memory-write check
           passes the message once it is cut to that length
        3. The turn budget cuts it to the room the turn has left, and the
           follow-up planning prompt answers the call with the output's
           start
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
        # The tool message is the truncated output itself.
        tool_msg_len = len(truncated_output)
        assert tool_msg_len > MAX_MEMORY_TEXT_LEN, (
            f"Expected tool message ({tool_msg_len}) to exceed "
            f"MAX_MEMORY_TEXT_LEN ({MAX_MEMORY_TEXT_LEN}); "
            f"if limits diverge, this test should be updated"
        )

        # ── Boundary 3: planning prompt carries the output, cut to fit ──
        # The result is too long for one memory write, and the tool call
        # still needs exactly one answer (OpenAI-compatible servers reject
        # an unanswered call). The memory-write check passes the result
        # cut to that length, and the turn budget cuts it to the room the
        # turn has left, so the prompt holds the output's start, bound to
        # the same tool_call_id.
        planning = llm.calls
        assert len(planning) >= 2
        second_plan_msgs = planning[1]["messages"]
        calls = [tc for m in second_plan_msgs for tc in m.tool_calls]
        tool_msgs = [m for m in second_plan_msgs if m.role == "tool"]
        assert len(calls) == 1 and len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == calls[0].id
        content = tool_msgs[0].content
        assert content.startswith("x" * 49_000)
        assert content.endswith(" more characters]")
        assert len(content) < MAX_MEMORY_TEXT_LEN, (
            f"Expected the {tool_msg_len}-char result cut to fit working "
            f"memory, got {len(content)} chars"
        )

        await agent.close()


# ── 13. LLM provider error propagation ─────────────────────────────────────


class TestLLMErrorPropagation:
    """Failure modes that span the Agent + LLM provider boundary."""

    @pytest.mark.asyncio
    async def test_llm_exception_after_tool_call(self) -> None:
        """LLM works for the first planning call but fails on the second."""

        class FailOnSecondPlanLLM(FakeLLM):
            def __init__(self) -> None:
                super().__init__()
                self._plan_count = 0

            async def chat(self, messages: Any, tools: Any = None) -> Message:
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

        result = await agent.run_task("echo hi")
        assert result.status == "crashed"
        assert result.error == "ConnectionError: LLM crashed mid-conversation"

        # The first step should still be recorded in history
        tasks = list(agent.hippo.history._tasks.values())
        assert len(tasks) == 1
        assert tasks[0].status == "crashed"
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


# ── 16. has_embedding → embedding provider consistency ─────────────────────


class TestHasEmbeddingConsistency:
    """Config.has_embedding must predict whether providers_for builds an embedding."""

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

        _, embedding = providers_for(cfg)
        assert embedding is not None

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

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            _, embedding = providers_for(cfg)
        assert embedding is None


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


# ── Helper ──────────────────────────────────────────────────────────────────


async def _echo_tool(text: str) -> str:
    return text
