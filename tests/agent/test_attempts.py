"""Tests for the attempt-based agent loop with structured reflection."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.agent import _providers_from_config  # noqa: F401
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    MAX_REFLECTION_CHARS,
    Message,
    ToolCall,
    ToolDef,
)
from localmelo.tests.agent.test_reflection_gate import _said

# ── Fake providers ──────────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    def __init__(self, responses: list[Message] | None = None) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, msg: Message) -> None:
        self.responses.append(msg)

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]

    async def close(self) -> None:
        pass


async def _echo_tool(text: str) -> str:
    return text


def _reflection_json(
    action: str = "stop",
    rationale: str = "done",
    best_effort: str = "",
    progress: bool = False,
    feasible: bool = True,
    summary: str = "tried things",
    info_gain: float = 0.5,
    cost: float = 0.1,
    repeat_risk: float = 0.1,
    novelty: float = 0.8,
    feasibility_score: float = 0.8,
    concrete: bool = True,
    novel: bool = True,
    **kwargs: Any,
) -> str:
    data = {
        "recommended_action": action,
        "rationale": rationale,
        "best_effort_result": best_effort,
        "progress_made": progress,
        "task_still_feasible": feasible,
        "summary": summary,
        "next_step_is_concrete": concrete,
        "next_step_is_novel": novel,
        "failed_hypotheses": kwargs.get("failed", []),
        "useful_evidence": kwargs.get("evidence", []),
        "recommended_avoids": kwargs.get("avoids", []),
        "next_promising_directions": kwargs.get("directions", []),
        "estimated_info_gain": info_gain,
        "estimated_cost": cost,
        "repeat_risk": repeat_risk,
        "novelty": novelty,
        "feasibility": feasibility_score,
    }
    return json.dumps(data)


def _make_echo_agent(llm: FakeLLM, options: AgentOptions | None = None) -> Agent:
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
    return agent


# ── Tests ───────────────────────────────────────────────────────────────────


class TestAttemptDirectAnswer:
    """Direct answer in attempt 0 completes the task normally."""

    @pytest.mark.asyncio
    async def test_direct_answer_no_reflection(self) -> None:
        agent = Agent(
            llm=FakeLLM([Message(role="assistant", content="42")]),
            embedding=FakeEmbedding(),
        )
        result = await agent.run("What is 6*7?")
        assert result == "42"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "completed"
        assert tasks[0].attempts_completed == 0


class TestAttemptBudgetTriggersReflection:
    """Exhausting attempt budget triggers reflection."""

    @pytest.mark.asyncio
    async def test_reflection_fires_after_budget(self) -> None:
        options = AgentOptions(steps_per_attempt=2, max_attempts=1, max_steps=10)

        llm = FakeLLM()
        # 2 tool calls (exactly the budget — for-else fires)
        for _ in range(2):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="calling",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )
        # Reflection: stop
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop", best_effort="partial answer"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        result = await agent.run("echo test")
        assert result == "partial answer"
        tasks = list(agent.hippo.history._tasks.values())
        # The only attempt was the last one, so its reflection could only
        # finalize the answer.
        assert tasks[0].status == "budget_exhausted"


class TestReflectionContinue:
    """Reflection returning 'continue' causes a second attempt."""

    @pytest.mark.asyncio
    async def test_continue_then_answer(self) -> None:
        options = AgentOptions(steps_per_attempt=2, max_attempts=3, max_steps=20)

        llm = FakeLLM()
        # Attempt 0: 2 tool calls → budget exhausted
        for _ in range(2):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="try",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
                )
            )
        # Reflection: continue
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    progress=True,
                    feasible=True,
                    directions=["try approach B"],
                ),
            )
        )
        # Attempt 1: direct answer
        llm.enqueue(Message(role="assistant", content="final answer"))

        agent = _make_echo_agent(llm, options=options)
        result = await agent.run("echo test")
        assert result == "final answer"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "completed"
        assert tasks[0].attempts_completed == 1


class TestReflectionStop:
    """Reflection returning 'stop' terminates gracefully."""

    @pytest.mark.asyncio
    async def test_stop_with_best_effort(self) -> None:
        options = AgentOptions(steps_per_attempt=2, max_attempts=3)

        llm = FakeLLM()
        for _ in range(2):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="try",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop", best_effort="best I can do"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        result = await agent.run("echo hard task")
        assert result == "best I can do"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "gate_stopped"


class TestMaxAttemptsExhausted:
    """All MAX_ATTEMPTS exhausted triggers failure."""

    @pytest.mark.asyncio
    async def test_all_attempts_used(self) -> None:
        # Without the request for an answer, the run reports the error.
        options = AgentOptions(
            steps_per_attempt=1, max_attempts=2, max_steps=10, answer_on_stop=False
        )

        llm = FakeLLM()
        # Attempt 0: 1 tool call → budget exhausted
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "1"}),
            )
        )
        # Reflection: continue
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("continue", progress=True),
            )
        )
        # Attempt 1: 1 tool call → budget exhausted
        llm.enqueue(
            Message(
                role="assistant",
                content="try again",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "2"}),
            )
        )
        # Reflection: continue (but no more attempts)
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("continue"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        result = await agent.run("echo endless")
        assert result == "[budget_exhausted] Max attempts reached"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "budget_exhausted"
        assert tasks[0].result == "Max attempts reached"


class TestStuckDetection:
    """Stuck detection catches repeated patterns."""

    @pytest.mark.asyncio
    async def test_repeated_tool_calls_detected(self) -> None:
        options = AgentOptions(
            steps_per_attempt=5, max_attempts=1, answer_on_stop=False
        )

        llm = FakeLLM()
        # Same tool+args 3 times → stuck
        for _ in range(3):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="same",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )
        # Reflection after stuck detection
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop", rationale="stuck"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        await agent.run("echo stuck task")
        # Three planning calls, then stuck detection fired: one reflection.
        assert len(llm.calls) == 4
        tasks = list(agent.hippo.history._tasks.values())
        # The only attempt was the last one.
        assert tasks[0].status == "budget_exhausted"

    def test_detect_stuck_unit(self) -> None:
        fp = ("echo", "abc123", "")
        assert Agent._detect_stuck([fp, fp, fp]) == "stuck"
        assert Agent._detect_stuck([fp, fp]) == ""
        assert Agent._detect_stuck([fp, ("other", "def", ""), fp]) == ""


class TestReflectionInPrompt:
    """Reflection context appears in the system prompt after reflection."""

    @pytest.mark.asyncio
    async def test_reflection_section_in_prompt(self) -> None:
        options = AgentOptions(steps_per_attempt=1, max_attempts=3, max_steps=10)

        llm = FakeLLM()
        # Attempt 0: tool call → budget
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        # Reflection: continue with avoids
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    summary="tried echo a",
                    avoids=["echo a"],
                    directions=["try echo b"],
                ),
            )
        )
        # Attempt 1: direct answer
        llm.enqueue(Message(role="assistant", content="done"))

        agent = _make_echo_agent(llm, options=options)
        await agent.run("echo test")

        # The last LLM call (attempt 1 planning) should have [REFLECTION]
        text = _said(llm.calls[-1])
        assert "[REFLECTION]" in text
        assert "echo a" in text


class TestReflectionPromotion:
    """Reflections are promoted to long-term memory at terminal state."""

    @pytest.mark.asyncio
    async def test_promoted_on_completion(self) -> None:
        options = AgentOptions(steps_per_attempt=1, max_attempts=3, max_steps=10)

        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("continue", summary="found clue"),
            )
        )
        llm.enqueue(Message(role="assistant", content="answer"))

        agent = _make_echo_agent(llm, options=options)
        result = await agent.run("echo test")
        assert result == "answer"

        # Reflections cleared from working memory after promotion
        assert len(agent.hippo.working.get_reflections()) == 0

        # Promoted to long-term memory
        assert len(agent.hippo.long._entries) > 0

    @pytest.mark.asyncio
    async def test_no_promotion_without_embedding(self) -> None:
        options = AgentOptions(steps_per_attempt=1, max_attempts=2)

        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop", best_effort="partial"),
            )
        )

        agent = Agent(llm=llm, embedding=None, options=options)
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

        await agent.run("echo test")
        # No crash, reflections cleared even without embedding
        assert len(agent.hippo.working.get_reflections()) == 0


class TestReflectionFromObject:
    """Unit tests for _reflection_from_object (the reading is in test_reflection_gate)."""

    def test_valid_object(self) -> None:
        entry, decision = Agent._reflection_from_object(
            json.loads(_reflection_json("continue", summary="ok", progress=True)),
            attempt_id=0,
            failure_type="budget",
        )
        assert entry.attempt_id == 0
        assert entry.summary == "ok"
        assert entry.failure_type == "budget"
        assert decision.recommended_action == "continue"
        assert decision.progress_made is True

    def test_malformed_types_coerced(self) -> None:
        """Valid JSON with wrong types should not crash."""
        data = {
            "summary": 123,
            "failed_hypotheses": "single string",
            "progress_made": "true",
            "confidence": "high",
            "recommended_action": "continue",
            "task_still_feasible": 1,
            "next_step_is_concrete": "yes",
            "next_step_is_novel": "yes",
        }
        entry, decision = Agent._reflection_from_object(data, 0, "budget")
        assert entry.summary == "123"
        assert entry.failed_hypotheses == ["single string"]
        assert entry.confidence == 0.0  # "high" -> default 0.0
        assert decision.progress_made is True  # "true" -> True
        assert decision.task_still_feasible is True

    def test_unknown_action_coerced_to_stop(self) -> None:
        _, decision = Agent._reflection_from_object(
            {"recommended_action": "yolo"}, 0, ""
        )
        assert decision.recommended_action == "stop"


class TestReflectionReceivesPriorReflections:
    """The reflect call should see prior reflection entries."""

    @pytest.mark.asyncio
    async def test_prior_reflections_in_reflect_call(self) -> None:
        options = AgentOptions(steps_per_attempt=1, max_attempts=3, max_steps=10)

        llm = FakeLLM()
        # Attempt 0: tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="try a",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        # Reflection 0: continue
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    summary="tried approach A",
                    failed=["approach A"],
                ),
            )
        )
        # Attempt 1: tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="try b",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "b"}),
            )
        )
        # Reflection 1: stop
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop", best_effort="done"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        await agent.run("echo test")

        # The second reflection call (4th LLM call) should contain
        # prior reflection info ("tried approach A")
        reflect_call_2 = llm.calls[3]  # 0=plan, 1=reflect, 2=plan, 3=reflect
        text = _said(reflect_call_2)
        assert "tried approach A" in text
        assert "approach A" in text


class TestReflectionInfluencesRetrieval:
    """A reflection's text must not add tools to the offered list."""

    @pytest.mark.asyncio
    async def test_reflection_text_does_not_add_a_tool(self) -> None:
        options = AgentOptions(steps_per_attempt=1, max_attempts=3, max_steps=10)

        llm = FakeLLM()
        # Attempt 0: tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        # Reflection: continue, mention "web_fetch" in directions
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    summary="echo didn't help",
                    directions=["try web_fetch instead"],
                ),
            )
        )
        # Attempt 1: direct answer
        llm.enqueue(Message(role="assistant", content="done"))

        agent = _make_echo_agent(llm, options=options)
        # Also register web_fetch so that a hint from the reflection would
        # offer it; only a tool call is a hint now.
        agent.hippo.register_tool(
            ToolDef(
                name="web_fetch",
                description="fetch a web page",
                parameters={"type": "object", "properties": {}},
            )
        )
        await agent.run("echo test")

        # The planning call for attempt 1 offers the built-ins alone: the
        # reflection names web_fetch in prose, which resolves nothing.
        plan_call_2 = llm.calls[2]  # 0=plan, 1=reflect, 2=plan
        tool_names = [t.name for t in plan_call_2.get("tools") or []]
        assert "echo" in tool_names
        assert "web_fetch" not in tool_names


class TestPrePlanIncludesReflections:
    """Pre-plan size check must account for reflection content."""

    @pytest.mark.asyncio
    async def test_oversized_reflection_is_trimmed(self) -> None:
        """An oversized reflection keeps the model's ``continue``: its summary
        is cut to fit :data:`MAX_REFLECTION_CHARS`, and the next attempt starts
        and plans with it."""
        options = AgentOptions(steps_per_attempt=1, max_attempts=3, max_steps=10)

        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        bloat = "x" * 4000
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    summary=bloat,
                    rationale="model wants to continue",
                ),
            )
        )
        llm.enqueue(Message(role="assistant", content="final answer"))

        agent = _make_echo_agent(llm, options=options)
        result = await agent.run("echo test")

        assert result == "final answer"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "completed"
        assert tasks[0].attempts_completed == 1  # the second attempt answered
        # The second attempt's prompt carries the summary as cut to fit
        # "[reflection] Attempt 0: <summary>" in the budget.
        room = MAX_REFLECTION_CHARS - len("[reflection] Attempt 0: ")
        cut = "x" * (room - 1) + "…"
        [block] = [
            m.content
            for m in llm.calls[-1]["messages"]
            if m.content.startswith("[Reflections; data")
        ]
        assert f"[REFLECTION]\nAttempt 0: {cut}\n[end of reflections]" in block


class TestReflectionFailureTypeInPrompt:
    """Reflection prompt must describe the real failure mode."""

    @pytest.mark.asyncio
    async def test_stuck_failure_type_in_reflection(self) -> None:
        options = AgentOptions(steps_per_attempt=5, max_attempts=1)

        llm = FakeLLM()
        # Same tool+args 3 times → stuck detection
        for _ in range(3):
            llm.enqueue(
                Message(
                    role="assistant",
                    content="same",
                    tool_call=ToolCall(tool_name="echo", arguments={"text": "x"}),
                )
            )
        # Reflection after stuck detection
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop", rationale="stuck"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        await agent.run("echo stuck task")

        # The reflection call should mention "repeated" not "budget"
        reflect_call = [
            c
            for c in llm.calls
            if any(
                "Reflect and decide" in m.content
                for m in c["messages"]
                if m.role == "user"
            )
        ]
        assert len(reflect_call) == 1
        reflect_user_msgs = [
            m
            for m in reflect_call[0]["messages"]
            if m.role == "user" and "Reflect and decide" in m.content
        ]
        assert len(reflect_user_msgs) == 1
        assert "repeated" in reflect_user_msgs[0].content.lower()
        assert "budget" not in reflect_user_msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_budget_failure_type_in_reflection(self) -> None:
        options = AgentOptions(steps_per_attempt=2, max_attempts=1, max_steps=10)

        llm = FakeLLM()
        # Different tool calls → no stuck, just budget exhaustion
        llm.enqueue(
            Message(
                role="assistant",
                content="try a",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        llm.enqueue(
            Message(
                role="assistant",
                content="try b",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "b"}),
            )
        )
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json("stop"),
            )
        )

        agent = _make_echo_agent(llm, options=options)
        await agent.run("echo budget test")

        reflect_call = [
            c
            for c in llm.calls
            if any(
                "Reflect and decide" in m.content
                for m in c["messages"]
                if m.role == "user"
            )
        ]
        assert len(reflect_call) == 1
        reflect_user_msgs = [
            m
            for m in reflect_call[0]["messages"]
            if m.role == "user" and "Reflect and decide" in m.content
        ]
        assert len(reflect_user_msgs) == 1
        content = reflect_user_msgs[0].content.lower()
        assert "budget" in content or "exhausted" in content
