"""Agent-loop tests with fake providers — no real network or model calls."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import (
    MAX_AGENT_STEPS,
    MIN_AGENT_STEPS,
    Message,
    ToolCall,
    ToolDef,
)

# ── Fake providers ──────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Programmable fake LLM that pops responses from a queue."""

    def __init__(self, responses: list[Message] | None = None) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, msg: Message) -> None:
        self.responses.append(msg)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    """Returns deterministic 4-dim embeddings."""

    def __init__(self) -> None:
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]

    async def close(self) -> None:
        pass


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_agent(
    responses: list[Message],
    embedding: BaseEmbeddingProvider | None = None,
) -> Agent:
    """Build an Agent wired to fake providers."""
    llm = FakeLLM(responses)
    return Agent(llm=llm, embedding=embedding)


# ── Estimation fixture ─────────────────────────────────────────────────────

# Save the real method before any patching so estimation tests can restore it.
_orig_estimate_max_steps = Agent._estimate_max_steps


@pytest.fixture(autouse=True)
def _skip_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip LLM step estimation in tests — always use MAX_AGENT_STEPS."""

    async def _use_max(self: Agent, query: str) -> int:
        return MAX_AGENT_STEPS

    monkeypatch.setattr(Agent, "_estimate_max_steps", _use_max)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestFinalAnswer:
    """LLM returns a direct answer (no tool call) on the first turn."""

    @pytest.mark.asyncio
    async def test_returns_content(self) -> None:
        agent = _make_agent(
            [Message(role="assistant", content="42")],
            embedding=FakeEmbedding(),
        )
        result = await agent.run("What is 6*7?")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_task_completed(self) -> None:
        agent = _make_agent(
            [Message(role="assistant", content="done")],
            embedding=FakeEmbedding(),
        )
        await agent.run("hello")
        # The latest task should be completed
        tasks = list(agent.hippo.history._tasks.values())
        assert len(tasks) == 1
        assert tasks[0].status == "completed"


class TestToolCallLoop:
    """LLM requests a tool call, gets the result, then answers."""

    @pytest.mark.asyncio
    async def test_tool_executes_and_feeds_back(self) -> None:
        """Step 1: LLM calls echo tool → Step 2: LLM answers with result."""
        llm = FakeLLM()
        # Turn 1: request tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="Let me run echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hello"}),
            )
        )
        # Turn 2: final answer
        llm.enqueue(Message(role="assistant", content="Echo said: hello"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())

        # Register a trivial echo tool
        echo_def = ToolDef(
            name="echo",
            description="echo text back",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
            semantic_tags=["echo"],
        )
        agent.hippo.register_tool(echo_def)
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("echo hello")
        assert result == "Echo said: hello"

        # Verify the tool result was fed back into the conversation
        assert len(llm.calls) == 2
        second_call_msgs = llm.calls[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_msgs) >= 1
        assert "hello" in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_step_recorded_in_history(self) -> None:
        """Verify the tool-call step is recorded."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.enqueue(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        await agent.run("echo hi")

        tasks = list(agent.hippo.history._tasks.values())
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status == "completed"
        steps = await agent.hippo.history.get_steps(task.task_id)
        assert len(steps) == 1
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.tool_name == "echo"


class TestBlockedToolCall:
    """Checker blocks a dangerous tool call — agent must fail cleanly."""

    @pytest.mark.asyncio
    async def test_blocked_shell_command(self) -> None:
        """rm -rf / must be blocked by the checker."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="deleting",
                tool_call=ToolCall(
                    tool_name="shell_exec",
                    arguments={"command": "rm -rf /"},
                ),
            )
        )
        # After blocked result, LLM provides final answer
        llm.enqueue(Message(role="assistant", content="aborted"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("delete everything")

        # The built-in shell_exec is registered, but rm -rf / is blocked.
        # The agent should still produce a result (the second LLM turn).
        assert result == "aborted"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self) -> None:
        """Calling an unregistered tool must return an error result."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="try unknown",
                tool_call=ToolCall(
                    tool_name="nonexistent_tool",
                    arguments={},
                ),
            )
        )
        llm.enqueue(Message(role="assistant", content="fallback"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("do something")
        assert result == "fallback"


class TestNoEmbeddingProvider:
    """Agent with embedding=None (online mode without local embedding)."""

    @pytest.mark.asyncio
    async def test_runs_without_long_term_memory(self) -> None:
        agent = _make_agent(
            [Message(role="assistant", content="answer")],
            embedding=None,
        )
        result = await agent.run("question")
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_hippo_retrieve_returns_empty(self) -> None:
        """retrieve_context() returns no long-term results without embedding."""
        hippo = Hippo(embedding=None)
        hippo.short.append(Message(role="user", content="hello"))
        long_ctx = await hippo.retrieve_context("hello")
        assert long_ctx == []


class TestPromptAssembly:
    """Verify short-term window is not injected twice into the prompt."""

    @pytest.mark.asyncio
    async def test_no_duplicate_short_term(self) -> None:
        """Short-term messages must appear exactly once in the LLM call."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        await agent.run("unique_query_marker")

        # Inspect the messages sent to the LLM
        assert len(llm.calls) == 1
        sent = llm.calls[0]["messages"]
        user_msgs = [m for m in sent if m.role == "user"]
        matches = [m for m in user_msgs if "unique_query_marker" in m.content]
        assert (
            len(matches) == 1
        ), f"Expected user query exactly once, found {len(matches)}"


class TestCheckedMemoryWrite:
    """Memory writes must be routed through Checker.pre_memory_write."""

    @pytest.mark.asyncio
    async def test_huge_summary_blocked(self) -> None:
        """A step summary exceeding MAX_OUTPUT_LEN must not be memorized."""
        llm = FakeLLM()
        # Tool call that produces a huge output
        llm.enqueue(
            Message(
                role="assistant",
                content="x" * 60_000,  # huge thought
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.enqueue(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("test")
        assert result == "done"

        # The huge summary should have been blocked by pre_memory_write,
        # so long-term memory should be empty.
        assert len(agent.hippo.long._entries) == 0


class TestRetrievalSeparation:
    """Retrieval results must not include tool resolution artefacts."""

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_only_long_term(self) -> None:
        """retrieve_context must not include short-term messages."""
        emb = FakeEmbedding()
        hippo = Hippo(embedding=emb)

        # Populate short-term
        hippo.short.append(Message(role="user", content="short term msg"))

        # Populate long-term
        await hippo.long.add("long term entry", [1.0, 0.0, 0.0, 0.0])

        long_ctx = await hippo.retrieve_context("anything")

        # Should contain only the long-term entry
        assert len(long_ctx) == 1
        assert "[memory]" in long_ctx[0].content
        assert "long term entry" in long_ctx[0].content

        # Must NOT contain the short-term message
        contents = " ".join(m.content for m in long_ctx)
        assert "short term msg" not in contents

    @pytest.mark.asyncio
    async def test_extract_tool_hints_is_separate(self) -> None:
        """extract_tool_hints is a pure text scan, separate from retrieval."""
        hippo = Hippo(embedding=None)
        hippo.register_tool(
            ToolDef(name="shell_exec", description="run shell", parameters={})
        )

        msgs = [
            Message(role="system", content="[memory] used shell_exec before"),
            Message(role="user", content="run a command"),
        ]
        hints = hippo.extract_tool_hints(msgs)
        assert "shell_exec" in hints


class TestMaxStepsTermination:
    """Loop terminates cleanly when estimated step budget is exhausted."""

    @pytest.mark.asyncio
    async def test_max_steps_sets_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Agent that always requests tool calls must hit the step limit."""

        async def _return_3(self: Agent, query: str) -> int:
            return 3

        monkeypatch.setattr(Agent, "_estimate_max_steps", _return_3)

        llm = FakeLLM()
        # Queue more tool-call responses than the estimated budget.
        for _ in range(5):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="again",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("loop forever")
        assert result == "Max steps reached"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestNoEmbeddingToolLoop:
    """Tool-call loop works correctly with embedding=None."""

    @pytest.mark.asyncio
    async def test_tool_loop_without_embedding(self) -> None:
        """Full tool-call -> answer cycle with no embedding provider."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="calling echo",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "hi"}),
            )
        )
        llm.enqueue(Message(role="assistant", content="got it"))

        agent = Agent(llm=llm, embedding=None)
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("echo hi")
        assert result == "got it"

        # Long-term memory must stay empty (no embedding).
        assert len(agent.hippo.long._entries) == 0

        # Short-term should contain the tool message.
        short_contents = [m.content for m in agent.hippo.short.get_window()]
        assert any("echo" in c and "hi" in c for c in short_contents)


class TestPrePlanCheckFailure:
    """Pre-plan check failure terminates the loop with clear status."""

    @pytest.mark.asyncio
    async def test_huge_context_blocks_plan(self) -> None:
        """Context exceeding 100k chars triggers pre_plan rejection."""
        llm = FakeLLM()
        llm.enqueue(Message(role="assistant", content="should not reach"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())

        # Inject a huge message into short-term to trigger pre_plan block.
        agent.hippo.short.append(Message(role="system", content="x" * 110_000))

        result = await agent.run("test")
        assert "Plan check failed" in result
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"

        # LLM should never have been called for planning.
        assert len(llm.calls) == 0


class TestSanitizedOutputInMemory:
    """Executor result truncation flows correctly into step records."""

    @pytest.mark.asyncio
    async def test_truncated_output_in_step_record(self) -> None:
        """When executor output exceeds MAX_OUTPUT_LEN, the ToolResult
        stored in the step record uses the truncated (sanitized) version."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="reading big file",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a" * 60_000}),
            )
        )
        llm.enqueue(Message(role="assistant", content="done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("big output")
        assert result == "done"

        # The step record must contain the truncated output, not the raw 60k.
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 1
        tool_result = steps[0].tool_result
        assert tool_result is not None
        assert "[truncated]" in tool_result.output
        assert len(tool_result.output) < 55_000
        # Must not be the raw 60k output.
        assert len(tool_result.output) < 60_000

    @pytest.mark.asyncio
    async def test_truncated_tool_msg_blocked_from_short_memory(self) -> None:
        """When both executor truncation and memory-write check apply,
        the oversized tool message is correctly blocked from short-term."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="big read",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "b" * 60_000}),
            )
        )
        llm.enqueue(Message(role="assistant", content="ok"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("huge output test")
        assert result == "ok"

        # The truncated output (~50k) plus the "[echo] " prefix exceeds
        # the memory-write limit, so no tool message in short-term.
        tool_msgs = [m for m in agent.hippo.short.get_window() if m.role == "tool"]
        assert len(tool_msgs) == 0


class TestToolResultMemoryWriteBlocked:
    """Tool result memory write is checked and can be blocked."""

    @pytest.mark.asyncio
    async def test_huge_tool_error_blocked(self) -> None:
        """A tool returning a huge error string must not be written to
        short-term memory if the memory-write check blocks it."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="calling fail_tool",
                tool_call=ToolCall(tool_name="fail_tool", arguments={}),
            )
        )
        llm.enqueue(Message(role="assistant", content="handled"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="fail_tool",
                description="always fails with huge error",
                parameters={"type": "object", "properties": {}},
            )
        )

        async def _fail_tool() -> str:
            raise RuntimeError("E" * 60_000)

        agent.executor.register("fail_tool", _fail_tool)

        result = await agent.run("do fail")
        assert result == "handled"

        # The huge error should be blocked from short-term memory by the
        # memory-write checker (text > 50k chars).
        tool_msgs = [
            m
            for m in agent.hippo.short.get_window()
            if m.role == "tool" and "fail_tool" in m.content
        ]
        assert len(tool_msgs) == 0


class TestDirectAnswerEdgeCases:
    """Edge cases for the direct-answer (no tool_call) path."""

    @pytest.mark.asyncio
    async def test_empty_content_direct_answer(self) -> None:
        """Direct answer with empty content still completes the task."""
        agent = _make_agent(
            [Message(role="assistant", content="")],
            embedding=FakeEmbedding(),
        )
        result = await agent.run("hello")
        assert result == ""
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "completed"

    @pytest.mark.asyncio
    async def test_task_saved_on_direct_answer(self) -> None:
        """The task record is persisted after a direct answer."""
        agent = _make_agent(
            [Message(role="assistant", content="saved")],
            embedding=FakeEmbedding(),
        )
        await agent.run("save test")
        tasks = list(agent.hippo.history._tasks.values())
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status == "completed"
        assert task.result == "saved"


class TestMultiStepToolCalls:
    """Agent handles multiple sequential tool calls before final answer."""

    @pytest.mark.asyncio
    async def test_two_tool_calls_then_answer(self) -> None:
        """LLM calls tool twice, then gives final answer."""
        llm = FakeLLM()
        # Turn 1: first tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="step 1",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "first"}),
            )
        )
        # Turn 2: second tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="step 2",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "second"}),
            )
        )
        # Turn 3: final answer
        llm.enqueue(Message(role="assistant", content="all done"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("multi step")
        assert result == "all done"

        # Should have 3 LLM calls total
        assert len(llm.calls) == 3

        # 2 steps recorded in history
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert len(steps) == 2

        # Both tool results should appear in the third LLM call's messages
        third_call_msgs = llm.calls[2]["messages"]
        tool_msgs = [m for m in third_call_msgs if m.role == "tool"]
        assert len(tool_msgs) >= 2


class TestPostPlanCheckFailure:
    """Post-plan check failure terminates cleanly."""

    @pytest.mark.asyncio
    async def test_empty_tool_name_blocked(self) -> None:
        """An LLM response with an empty tool_name is caught by post_plan."""
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="bad response",
                tool_call=ToolCall(tool_name="", arguments={}),
            )
        )
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent.run("test")
        assert "Response check failed" in result
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestStepEstimation:
    """Dynamic step-budget estimation before the agent loop."""

    @pytest.fixture(autouse=True)
    def _restore_estimation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use real estimation for tests in this class."""
        monkeypatch.setattr(Agent, "_estimate_max_steps", _orig_estimate_max_steps)

    @pytest.mark.asyncio
    async def test_parse_step_estimate(self) -> None:
        """_parse_step_estimate extracts integers correctly."""
        from localmelo.melo.agent.chat import _parse_step_estimate

        assert _parse_step_estimate("5") == 5
        assert _parse_step_estimate("I think 12 steps") == 12
        assert _parse_step_estimate("no number") == -1
        assert _parse_step_estimate("") == -1

    @pytest.mark.asyncio
    async def test_clamps_to_min(self) -> None:
        """Estimate below MIN_AGENT_STEPS is clamped up."""
        llm = FakeLLM([Message(role="assistant", content="1")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent._estimate_max_steps("tiny")
        assert result == MIN_AGENT_STEPS

    @pytest.mark.asyncio
    async def test_clamps_to_max(self) -> None:
        """Estimate above MAX_AGENT_STEPS is clamped down."""
        llm = FakeLLM([Message(role="assistant", content="100")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent._estimate_max_steps("huge")
        assert result == MAX_AGENT_STEPS

    @pytest.mark.asyncio
    async def test_within_bounds(self) -> None:
        """Estimate within bounds is used as-is."""
        llm = FakeLLM([Message(role="assistant", content="10")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent._estimate_max_steps("medium")
        assert result == 10

    @pytest.mark.asyncio
    async def test_unparseable_falls_back(self) -> None:
        """Non-numeric LLM response falls back to MAX_AGENT_STEPS."""
        llm = FakeLLM([Message(role="assistant", content="I don't know")])
        agent = Agent(llm=llm, embedding=FakeEmbedding())
        result = await agent._estimate_max_steps("query")
        assert result == MAX_AGENT_STEPS

    @pytest.mark.asyncio
    async def test_estimated_budget_limits_loop(self) -> None:
        """Estimation of 3 steps limits the agent to 3 loop iterations."""
        llm = FakeLLM()
        # Estimation response
        llm.enqueue(Message(role="assistant", content="3"))
        # More tool calls than the estimate
        for _ in range(5):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="again",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        agent.executor.register("echo", _echo_tool)

        result = await agent.run("loop task")
        assert result == "Max steps reached"


# ── Test helper ─────────────────────────────────────────────────────────────


async def _echo_tool(text: str) -> str:
    return text
