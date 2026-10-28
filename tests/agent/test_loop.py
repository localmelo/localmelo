"""Agent-loop tests with fake providers — no real network or model calls."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import Chat
from localmelo.melo.checker.validators import MAX_MEMORY_TEXT_LEN
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
        """In direct mode, rm -rf / must be blocked by the checker.

        P13a leaves direct mode's blocklist in place, so this is still the
        refusal it was before the tool mode existed. The Python API's
        default mode is ``python``, whose exec policy is the tier, not a
        spelling rule; a run there must not be given a destructive command.
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

        agent = Agent(
            llm=llm,
            embedding=FakeEmbedding(),
            options=AgentOptions(mode="direct"),
        )
        result = await agent.run("delete everything with shell_exec")

        # The built-in shell_exec is registered, but rm -rf / is blocked.
        # The agent should still produce a result (the second LLM turn).
        assert result == "aborted"
        tasks = list(agent.hippo.history._tasks.values())
        steps = await agent.hippo.history.get_steps(tasks[0].task_id)
        assert steps[0].tool_result is not None
        assert steps[0].tool_result.error == "Blocked: Blocked command: rm -rf /"

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
    """Memory writes must be routed through Checker.check_memory_write."""

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

        result = await agent.run("echo test")
        assert result == "done"

        # The huge summary should have been blocked by check_memory_write,
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
        """extract_tool_hints is a pure scan, separate from retrieval."""
        hippo = Hippo(embedding=None)
        hippo.register_tool(
            ToolDef(name="shell_exec", description="run shell", parameters={})
        )

        msgs = [
            Message(role="system", content="[memory] used shell_exec before"),
            Message(
                role="assistant",
                content="run a command",
                tool_call=ToolCall(tool_name="shell_exec", arguments={}),
            ),
        ]
        hints = hippo.extract_tool_hints(msgs)
        assert hints == ["shell_exec"]

    @pytest.mark.asyncio
    async def test_text_that_names_a_tool_adds_no_hint(self) -> None:
        """Only the model's tool calls count; a text names nothing."""
        hippo = Hippo(embedding=None)
        hippo.register_tool(
            ToolDef(name="web_fetch", description="fetch a page", parameters={})
        )
        hippo.register_tool(
            ToolDef(name="shell_exec", description="run shell", parameters={})
        )

        msgs = [
            Message(role="user", content="use web_fetch for the docs"),
            Message(role="tool", content="[called web_fetch] ok", tool_call_id="c1"),
            Message(role="system", content="[memory] shell_exec worked before"),
        ]
        assert hippo.extract_tool_hints(msgs) == []


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
        window = agent.hippo.short.get_window()
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

        window = agent.hippo.short.get_window()
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

        window = agent.hippo.short.get_window()
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
        window = agent.hippo.short.get_window()
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


class TestStepEstimation:
    """Dynamic step-budget estimation before the agent loop."""

    @pytest.fixture(autouse=True)
    def _restore_estimation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use real estimation for tests in this class."""
        monkeypatch.setattr(Agent, "_estimate_max_steps", _orig_estimate_max_steps)

    @pytest.mark.asyncio
    async def test_parse_step_estimate(self) -> None:
        """_parse_step_estimate extracts integers correctly."""
        from localmelo.melo.agent._chat import _parse_step_estimate

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
    """Unit tests for _build_system_prompt merging logic."""

    def test_no_context_no_short(self) -> None:
        from localmelo.melo.agent._chat import SYSTEM_PROMPT, _build_system_prompt

        prompt, others = _build_system_prompt([], [])
        assert prompt == SYSTEM_PROMPT
        assert others == []

    def test_memory_items_merged_under_recall(self) -> None:
        from localmelo.melo.agent._chat import SYSTEM_PROMPT, _build_system_prompt

        context = [
            Message(role="system", content="[memory] used shell_exec before"),
            Message(role="system", content="[memory] prefers concise answers"),
        ]
        prompt, others = _build_system_prompt(context, [])
        assert prompt.startswith(SYSTEM_PROMPT)
        assert prompt.count("[RECALL]") == 1
        assert "used shell_exec before" in prompt
        assert "prefers concise answers" in prompt
        # [memory] prefix must be stripped
        assert "[memory]" not in prompt
        assert others == []

    def test_non_memory_system_messages_merged(self) -> None:
        from localmelo.melo.agent._chat import SYSTEM_PROMPT, _build_system_prompt

        short = [Message(role="system", content="extra context info")]
        prompt, others = _build_system_prompt([], short)
        assert "extra context info" in prompt
        assert prompt.startswith(SYSTEM_PROMPT)
        assert "[RECALL]" not in prompt
        assert others == []

    def test_mixed_roles_separated(self) -> None:
        from localmelo.melo.agent._chat import SYSTEM_PROMPT, _build_system_prompt

        context = [
            Message(role="system", content="[memory] fact A"),
        ]
        short = [
            Message(role="system", content="extra sys"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        prompt, others = _build_system_prompt(context, short)
        assert prompt.startswith(SYSTEM_PROMPT)
        assert "extra sys" in prompt
        assert "[RECALL]" in prompt
        assert "fact A" in prompt
        assert len(others) == 2
        assert others[0].role == "user"
        assert others[1].role == "assistant"

    def test_no_recall_section_without_memory(self) -> None:
        from localmelo.melo.agent._chat import _build_system_prompt

        context: list[Message] = []
        short = [Message(role="user", content="query")]
        prompt, others = _build_system_prompt(context, short)
        assert "[RECALL]" not in prompt
        assert len(others) == 1

    @pytest.mark.asyncio
    async def test_plan_step_single_system_message(self) -> None:
        """plan_step must produce exactly one system message even with memory."""
        llm = FakeLLM([Message(role="assistant", content="ok")])
        chat = Chat(llm)
        context = [
            Message(role="system", content="[memory] mem0"),
            Message(role="system", content="[memory] mem1"),
        ]
        short = [Message(role="user", content="query")]
        await chat.plan_step(context=context, short=short, tools=[], query="query")

        sent = llm.calls[0]["messages"]
        sys_msgs = [m for m in sent if m.role == "system"]
        assert len(sys_msgs) == 1, f"Expected 1 system message, got {len(sys_msgs)}"
        assert "[RECALL]" in sys_msgs[0].content
        assert "mem0" in sys_msgs[0].content
        assert "mem1" in sys_msgs[0].content


class TestProvidersFromConfig:
    """_providers_from_config delegates to the backend registry."""

    def test_delegates_to_backend_registry(self) -> None:
        """build_chat_provider and build_embedding_provider are called
        on potentially different backends for chat vs embedding."""
        from unittest import mock

        fake_llm = mock.MagicMock()
        fake_emb = mock.MagicMock()

        fake_chat_backend = mock.MagicMock()
        fake_chat_backend.build_chat_provider.return_value = fake_llm

        fake_emb_backend = mock.MagicMock()
        fake_emb_backend.build_embedding_provider.return_value = fake_emb

        def _get_backend(key: str) -> mock.MagicMock:
            if key == "openai":
                return fake_chat_backend
            if key == "ollama":
                return fake_emb_backend
            raise KeyError(key)

        with mock.patch(
            "localmelo.support.backends.get_backend", side_effect=_get_backend
        ):
            from localmelo.melo.agent.agent import _providers_from_config
            from localmelo.support.config import Config, LocalBackendConfig

            cfg = Config(
                chat_backend="openai",
                embedding_backend="ollama",
                ollama=LocalBackendConfig(
                    embedding_url="http://localhost:11434",
                    embedding_model="nomic-embed",
                ),
            )
            llm, embedding = _providers_from_config(cfg)

        fake_chat_backend.build_chat_provider.assert_called_once_with(cfg)
        fake_emb_backend.build_embedding_provider.assert_called_once_with(cfg)
        assert llm is fake_llm
        assert embedding is fake_emb

    def test_no_embedding_when_backend_is_none(self) -> None:
        """When embedding_backend is 'none', embedding provider is None."""
        from unittest import mock

        fake_llm = mock.MagicMock()
        fake_backend = mock.MagicMock()
        fake_backend.build_chat_provider.return_value = fake_llm

        with mock.patch(
            "localmelo.support.backends.get_backend", return_value=fake_backend
        ):
            from localmelo.melo.agent.agent import _providers_from_config
            from localmelo.support.config import Config

            cfg = Config(
                chat_backend="openai",
                embedding_backend="none",
            )
            llm, embedding = _providers_from_config(cfg)

        fake_backend.build_chat_provider.assert_called_once_with(cfg)
        fake_backend.build_embedding_provider.assert_not_called()
        assert llm is fake_llm
        assert embedding is None

    def test_unknown_backend_raises_key_error(self) -> None:
        """Requesting an unregistered backend raises KeyError."""
        from unittest import mock

        with mock.patch(
            "localmelo.support.backends.get_backend",
            side_effect=KeyError("Unknown backend 'nope'"),
        ):
            from localmelo.melo.agent.agent import _providers_from_config
            from localmelo.support.config import Config

            cfg = Config(chat_backend="nope", embedding_backend="none")
            with pytest.raises(KeyError, match="nope"):
                _providers_from_config(cfg)


# ── Test helper ─────────────────────────────────────────────────────────────


async def _echo_tool(text: str) -> str:
    return text
