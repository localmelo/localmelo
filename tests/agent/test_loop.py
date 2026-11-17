"""Agent-loop tests with fake providers — no real network or model calls."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    Message,
    ToolCall,
    ToolDef,
)
from localmelo.tests.agent.test_reflection_gate import _scoped_chat

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
    """A dangerous call the run cannot make must fail cleanly."""

    @pytest.mark.asyncio
    async def test_blocked_shell_command(self) -> None:
        """The default Agent (python mode, no approval) does not offer shell_exec.

        P13b gives ``python`` the same default set as direct: a caller that
        passes no ``approval=`` has no policy, so the destructive call has
        no tool to reach. The call is refused before it runs.
        """
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
        assert agent.options.mode == "python"
        result = await agent.run("delete everything with shell_exec")

        # shell_exec is not registered, so the destructive call never runs.
        # The agent still produces a result (the second LLM turn).
        assert result == "aborted"
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert steps[0].tool_result is not None
        # The error names the tools the run does offer (P13b review fix 5).
        assert steps[0].tool_result.error == (
            "shell_exec is not available in this run. "
            "Available tools: file_read, file_write, python_exec."
        )

    @pytest.mark.asyncio
    async def test_a_registered_shell_is_blocked_in_gateway_mode(
        self, tmp_path: Any
    ) -> None:
        """A named shell in gateway mode is registered, offered, and refused."""
        marker = tmp_path / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="deleting",
                tool_call=ToolCall(
                    tool_name="shell_exec",
                    arguments={"command": f"rm -f {marker}"},
                ),
            )
        )
        llm.enqueue(Message(role="assistant", content="aborted"))

        agent = Agent(
            llm=llm,
            embedding=FakeEmbedding(),
            options=AgentOptions(
                mode="gateway",
                builtins=frozenset({"shell_exec"}),
            ),
        )
        result = await agent.run("delete the marker")

        assert result == "aborted"
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert steps[0].tool_result is not None
        assert steps[0].tool_result.error.startswith("Blocked:")
        assert "gateway" in steps[0].tool_result.error
        assert marker.exists()

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


class TestMaxStepsTermination:
    """Loop terminates cleanly when step/attempt budget is exhausted."""

    @pytest.mark.asyncio
    async def test_max_steps_ends_as_budget_exhausted(self) -> None:
        """Agent that always requests tool calls must hit the step limit."""
        options = AgentOptions(steps_per_attempt=3, max_attempts=1, max_steps=3)

        llm = FakeLLM()
        # 5 tool-call responses (more than 3-step budget)
        for _ in range(5):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="again",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )
        # Reflection response (will be consumed after attempt exhausts)
        llm.enqueue(
            Message(
                role="assistant",
                content='{"recommended_action":"stop","rationale":"stuck","best_effort_result":""}',
            )
        )

        agent = Agent(llm=llm, embedding=FakeEmbedding(), options=options)
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

        await agent.run("echo loop forever")
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "budget_exhausted"


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

        # Short-term holds the real turn: the echo call and its result,
        # paired by tool_call_id.
        window = agent.hippo.working.get_window()
        calls = [tc for m in window for tc in m.tool_calls]
        results = [m for m in window if m.role == "tool"]
        assert [tc.tool_name for tc in calls] == ["echo"]
        assert [(m.tool_call_id, m.content) for m in results] == [(calls[0].id, "hi")]


class TestPrePlanCheckFailure:
    """Pre-plan check failure terminates the loop with clear status."""

    @pytest.mark.asyncio
    async def test_huge_context_blocks_plan(self) -> None:
        """Context exceeding 100k chars triggers pre_plan rejection."""
        llm = FakeLLM()
        llm.enqueue(Message(role="assistant", content="should not reach"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())

        # A query too large on its own triggers the pre_plan block: it is
        # the newest message, so dropping older ones cannot make room.
        result = await agent.run("x" * 110_000)
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

        result = await agent.run("echo big output")
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
    async def test_truncated_tool_msg_is_cut_in_short_memory(self) -> None:
        """When the call's arguments and its truncated output are both too
        long for memory, nothing oversized reaches short-term memory, but
        the next planning prompt still holds the call and its result."""
        llm = FakeLLM()
        call = ToolCall(tool_name="echo", arguments={"text": "b" * 60_000})
        llm.enqueue(Message(role="assistant", content="big read", tool_call=call))
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

        result = await agent.run("echo huge output test")
        assert result == "ok"

        # The call already ran, so its turn stays: the same id with the
        # 60k-char arguments replaced by a placeholder, answered by the
        # start of the truncated output, cut to the room the turn has left.
        second_call_msgs = llm.calls[1]["messages"]
        sent_calls = [tc for m in second_call_msgs for tc in m.tool_calls]
        assert [tc.id for tc in sent_calls] == [call.id]
        assert sent_calls[0].arguments == {"_elided": "60012 chars withheld"}
        tool_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert [m.tool_call_id for m in tool_msgs] == [call.id]
        assert tool_msgs[0].content.startswith("b" * 49_000)
        assert tool_msgs[0].content.endswith(" more characters]")

        window = agent.hippo.working.get_window()
        assert all(len(m.content) < MAX_MEMORY_TEXT_LEN for m in window)

        # History keeps the real call, which the executor ran as sent.
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert steps[0].tool_call is not None
        assert steps[0].tool_call.arguments == {"text": "b" * 60_000}
        assert steps[0].tool_result is not None
        assert steps[0].tool_result.output.startswith("b" * 50_000)

    @pytest.mark.asyncio
    async def test_truncated_result_is_cut_with_small_arguments(self) -> None:
        """When both executor truncation and memory-write check apply to
        a result, nothing oversized reaches short-term memory: the call
        keeps its arguments and is answered by the result's start."""
        llm = FakeLLM()
        call = ToolCall(tool_name="big_read", arguments={"path": "big.txt"})
        llm.enqueue(Message(role="assistant", content="big read", tool_call=call))
        llm.enqueue(Message(role="assistant", content="ok"))

        agent = Agent(llm=llm, embedding=FakeEmbedding())
        agent.hippo.register_tool(
            ToolDef(
                name="big_read",
                description="read a big file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        )

        async def _big_read(path: str) -> str:
            return "b" * 60_000

        agent.executor.register("big_read", _big_read)

        result = await agent.run("huge output test with big_read")
        assert result == "ok"

        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert steps[0].tool_result is not None
        assert "[truncated]" in steps[0].tool_result.output

        second_call_msgs = llm.calls[1]["messages"]
        sent_calls = [tc for m in second_call_msgs for tc in m.tool_calls]
        assert [(tc.id, tc.arguments) for tc in sent_calls] == [
            (call.id, {"path": "big.txt"})
        ]
        tool_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert [m.tool_call_id for m in tool_msgs] == [call.id]
        assert tool_msgs[0].content.startswith("b" * 49_000)
        assert tool_msgs[0].content.endswith(" more characters]")

        window = agent.hippo.working.get_window()
        assert all(len(m.content) < MAX_MEMORY_TEXT_LEN for m in window)


class TestToolResultMemoryWriteBlocked:
    """Tool result memory write is checked and can be blocked."""

    @pytest.mark.asyncio
    async def test_huge_tool_error_is_cut(self) -> None:
        """A tool returning a huge error string must not be written whole
        to short-term memory: it is too long for one memory write, so it
        is cut."""
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

        # The huge error is too long for one memory write (text > 50k
        # chars). The memory-write check passes it cut to that length, and
        # the turn budget cuts it to the room the turn has left, so
        # short-term memory keeps its start, bound to the same tool_call_id.
        window = agent.hippo.working.get_window()
        calls = [tc for m in window for tc in m.tool_calls]
        tool_msgs = [m for m in window if m.role == "tool"]
        assert len(calls) == 1 and len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == calls[0].id
        assert tool_msgs[0].content.startswith("Error: " + "E" * 49_000)
        assert tool_msgs[0].content.endswith(" more characters]")
        assert len(tool_msgs[0].content) < MAX_MEMORY_TEXT_LEN


class TestDirectAnswerEdgeCases:
    """Edge cases for the direct-answer (no tool_call) path."""

    @pytest.mark.asyncio
    async def test_empty_content_is_not_a_final_answer(self) -> None:
        """A reply with neither a tool call nor text is no answer: after
        two in a row the task fails."""
        agent = _make_agent(
            [Message(role="assistant", content="")] * 2,
            embedding=FakeEmbedding(),
        )
        result = await agent.run("hello")
        # run() shows "[<status>] <error>" whenever the result is empty.
        assert result == "[failed] empty replies"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"

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

        result = await agent.run("echo multi step")
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


class TestStepBudget:
    """The attempt budget bounds the agent loop."""

    @pytest.mark.asyncio
    async def test_attempt_budget_limits_loop(self) -> None:
        """Attempt budget limits the agent loop iterations."""
        options = AgentOptions(steps_per_attempt=3, max_attempts=1, max_steps=3)

        llm = FakeLLM()
        for _ in range(5):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="again",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )
        # Reflection response
        llm.enqueue(
            Message(
                role="assistant",
                content='{"recommended_action":"stop","rationale":"budget","best_effort_result":""}',
            )
        )

        agent = Agent(llm=llm, embedding=FakeEmbedding(), options=options)
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

        await agent.run("echo loop task")
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "budget_exhausted"


class TestBuildSystemPrompt:
    """Unit tests for _build_system_prompt and the plan_step system message."""

    def test_no_notes_no_reflections(self) -> None:
        from localmelo.melo.agent._chat import SYSTEM_PROMPT, _build_system_prompt

        assert _build_system_prompt() == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_plan_step_single_system_message(self) -> None:
        """plan_step produces exactly one system message even with a recall block."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        chat = _scoped_chat(llm)
        short = [
            Message(role="system", content="extra note"),
            Message(role="user", content="query"),
        ]
        await chat.plan_step(
            short=short, tools=[], query="query", query_block="recalled mem0\nmem1"
        )

        sent = llm.calls[0]["messages"]
        sys_msgs = [m for m in sent if m.role == "system"]
        assert len(sys_msgs) == 1, f"Expected 1 system message, got {len(sys_msgs)}"
        # The recalled text is data in a user message, so the system prompt
        # stays the same from step to step.
        assert "extra note" in sys_msgs[0].content
        assert "mem0" not in sys_msgs[0].content
        [data] = [
            m
            for m in sent
            if m.role == "user" and "data, not instructions" in m.content
        ]
        assert "mem0" in data.content and "mem1" in data.content


# ── Test helper ─────────────────────────────────────────────────────────────


async def _echo_tool(text: str) -> str:
    return text
