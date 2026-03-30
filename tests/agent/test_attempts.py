"""Tests for the attempt-based agent loop with structured reflection."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.agent.agent import _providers_from_config  # noqa: F401
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    Message,
    ReflectionDecision,
    ReflectionEntry,
    ToolCall,
    ToolDef,
)

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


def _make_echo_agent(llm: FakeLLM) -> Agent:
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
    async def test_reflection_fires_after_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 2)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 1)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

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

        agent = _make_echo_agent(llm)
        result = await agent.run("test")
        assert result == "partial answer"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestReflectionContinue:
    """Reflection returning 'continue' causes a second attempt."""

    @pytest.mark.asyncio
    async def test_continue_then_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 2)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 20)

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

        agent = _make_echo_agent(llm)
        result = await agent.run("test")
        assert result == "final answer"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "completed"
        assert tasks[0].attempts_completed == 1


class TestReflectionStop:
    """Reflection returning 'stop' terminates gracefully."""

    @pytest.mark.asyncio
    async def test_stop_with_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 2)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)

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

        agent = _make_echo_agent(llm)
        result = await agent.run("hard task")
        assert result == "best I can do"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestMaxAttemptsExhausted:
    """All MAX_ATTEMPTS exhausted triggers failure."""

    @pytest.mark.asyncio
    async def test_all_attempts_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 2)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

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

        agent = _make_echo_agent(llm)
        result = await agent.run("endless")
        assert "Max attempts reached" in result
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestStuckDetection:
    """Stuck detection catches repeated patterns."""

    @pytest.mark.asyncio
    async def test_repeated_tool_calls_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 5)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 1)

        llm = FakeLLM()
        # Same tool+args 3 times → stuck
        for _ in range(4):
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

        agent = _make_echo_agent(llm)
        await agent.run("stuck task")
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"

    def test_detect_stuck_unit(self) -> None:
        fp = ("echo", "abc123", "")
        assert Agent._detect_stuck([fp, fp, fp]) == "stuck"
        assert Agent._detect_stuck([fp, fp]) == ""
        assert Agent._detect_stuck([fp, ("other", "def", ""), fp]) == ""


class TestReflectionInPrompt:
    """Reflection context appears in the system prompt after reflection."""

    @pytest.mark.asyncio
    async def test_reflection_section_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

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

        agent = _make_echo_agent(llm)
        await agent.run("test")

        # The last LLM call (attempt 1 planning) should have [REFLECTION]
        last_plan_call = llm.calls[-1]
        system_msg = last_plan_call["messages"][0]
        assert "[REFLECTION]" in system_msg.content
        assert "echo a" in system_msg.content


class TestReflectionPromotion:
    """Reflections are promoted to long-term memory at terminal state."""

    @pytest.mark.asyncio
    async def test_promoted_on_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

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

        agent = _make_echo_agent(llm)
        result = await agent.run("test")
        assert result == "answer"

        # Reflections cleared from working memory after promotion
        assert len(agent.hippo.working.get_reflections()) == 0

        # Promoted to long-term memory
        assert len(agent.hippo.long._entries) > 0

    @pytest.mark.asyncio
    async def test_no_promotion_without_embedding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 2)

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

        await agent.run("test")
        # No crash, reflections cleared even without embedding
        assert len(agent.hippo.working.get_reflections()) == 0


class TestParseReflection:
    """Unit tests for _parse_reflection."""

    def test_valid_json(self) -> None:
        entry, decision = Agent._parse_reflection(
            _reflection_json("continue", summary="ok", progress=True),
            attempt_id=0,
            failure_type="budget",
        )
        assert entry.attempt_id == 0
        assert entry.summary == "ok"
        assert entry.failure_type == "budget"
        assert decision.recommended_action == "continue"
        assert decision.progress_made is True

    def test_invalid_json_falls_back_to_stop(self) -> None:
        entry, decision = Agent._parse_reflection(
            "not json at all",
            attempt_id=1,
            failure_type="stuck",
        )
        assert entry.attempt_id == 1
        assert decision.recommended_action == "stop"
        assert "parse" in decision.rationale.lower()

    def test_empty_string(self) -> None:
        _, decision = Agent._parse_reflection("", 0, "")
        assert decision.recommended_action == "stop"

    def test_malformed_types_coerced(self) -> None:
        """Valid JSON with wrong types should not crash."""
        raw = json.dumps(
            {
                "summary": 123,
                "failed_hypotheses": "single string",
                "progress_made": "true",
                "confidence": "high",
                "recommended_action": "continue",
                "task_still_feasible": 1,
                "next_step_is_concrete": "yes",
                "next_step_is_novel": "yes",
                "estimated_info_gain": "0.7",
                "novelty": None,
                "feasibility": "bad",
            }
        )
        entry, decision = Agent._parse_reflection(raw, 0, "budget")
        assert entry.summary == "123"
        assert entry.failed_hypotheses == ["single string"]
        assert entry.confidence == 0.0  # "high" -> default 0.0
        assert decision.progress_made is True  # "true" -> True
        assert decision.task_still_feasible is True
        assert decision.estimated_info_gain == 0.7
        assert decision.novelty == 0.0  # None -> 0.0
        assert decision.feasibility == 0.0  # "bad" -> 0.0

    def test_json_array_falls_back_to_stop(self) -> None:
        """JSON that parses to a list (not dict) should stop."""
        _, decision = Agent._parse_reflection("[1,2,3]", 0, "")
        assert decision.recommended_action == "stop"

    def test_unknown_action_coerced_to_stop(self) -> None:
        raw = json.dumps({"recommended_action": "yolo"})
        _, decision = Agent._parse_reflection(raw, 0, "")
        assert decision.recommended_action == "stop"


class TestUtilityContinuationGate:
    """Active-learning utility gate controls continuation."""

    def test_high_utility_allows_continue(self) -> None:
        from localmelo.melo.agent.agent import _should_continue

        decision = ReflectionDecision(
            recommended_action="continue",
            task_still_feasible=True,
            next_step_is_concrete=True,
            next_step_is_novel=True,
            estimated_info_gain=0.8,
            feasibility=0.9,
            novelty=0.8,
            estimated_cost=0.1,
            repeat_risk=0.1,
        )
        assert _should_continue(decision) is True

    def test_low_utility_blocks_continue(self) -> None:
        from localmelo.melo.agent.agent import _should_continue

        decision = ReflectionDecision(
            recommended_action="continue",
            task_still_feasible=True,
            next_step_is_concrete=True,
            next_step_is_novel=True,
            estimated_info_gain=0.1,
            feasibility=0.1,
            novelty=0.1,
            estimated_cost=0.5,
            repeat_risk=0.5,
        )
        assert _should_continue(decision) is False

    def test_stop_action_always_blocks(self) -> None:
        from localmelo.melo.agent.agent import _should_continue

        decision = ReflectionDecision(
            recommended_action="stop",
            task_still_feasible=True,
            next_step_is_concrete=True,
            next_step_is_novel=True,
            estimated_info_gain=1.0,
            feasibility=1.0,
            novelty=1.0,
        )
        assert _should_continue(decision) is False

    def test_not_feasible_blocks(self) -> None:
        from localmelo.melo.agent.agent import _should_continue

        decision = ReflectionDecision(
            recommended_action="continue",
            task_still_feasible=False,
            next_step_is_concrete=True,
            next_step_is_novel=True,
            estimated_info_gain=1.0,
            feasibility=1.0,
            novelty=1.0,
        )
        assert _should_continue(decision) is False

    def test_not_concrete_blocks(self) -> None:
        from localmelo.melo.agent.agent import _should_continue

        decision = ReflectionDecision(
            recommended_action="continue",
            task_still_feasible=True,
            next_step_is_concrete=False,
            next_step_is_novel=True,
            estimated_info_gain=1.0,
            feasibility=1.0,
            novelty=1.0,
        )
        assert _should_continue(decision) is False

    @pytest.mark.asyncio
    async def test_utility_blocks_continuation_in_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with recommended_action=continue, low utility stops the loop."""
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

        llm = FakeLLM()
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        # Reflection says "continue" but utility is terrible
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    info_gain=0.01,
                    cost=0.9,
                    repeat_risk=0.9,
                    novelty=0.01,
                    feasibility_score=0.01,
                    best_effort="low utility answer",
                ),
            )
        )

        agent = _make_echo_agent(llm)
        result = await agent.run("test")
        assert result == "low utility answer"
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestReflectionReceivesPriorReflections:
    """The reflect call should see prior reflection entries."""

    @pytest.mark.asyncio
    async def test_prior_reflections_in_reflect_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

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

        agent = _make_echo_agent(llm)
        await agent.run("test")

        # The second reflection call (4th LLM call) should contain
        # prior reflection info ("tried approach A") in the system prompt
        reflect_call_2 = llm.calls[3]  # 0=plan, 1=reflect, 2=plan, 3=reflect
        system_msg = reflect_call_2["messages"][0]
        assert "tried approach A" in system_msg.content
        assert "approach A" in system_msg.content


class TestReflectionInfluencesRetrieval:
    """Reflection context should be included in tool hint extraction."""

    @pytest.mark.asyncio
    async def test_reflection_messages_in_hint_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

        llm = FakeLLM()
        # Attempt 0: tool call
        llm.enqueue(
            Message(
                role="assistant",
                content="try",
                tool_call=ToolCall(tool_name="echo", arguments={"text": "a"}),
            )
        )
        # Reflection: continue, mention "shell_exec" in directions
        llm.enqueue(
            Message(
                role="assistant",
                content=_reflection_json(
                    "continue",
                    summary="echo didn't help",
                    directions=["try shell_exec instead"],
                ),
            )
        )
        # Attempt 1: direct answer
        llm.enqueue(Message(role="assistant", content="done"))

        agent = _make_echo_agent(llm)
        # Also register shell_exec so it can be found
        agent.hippo.register_tool(
            ToolDef(
                name="shell_exec",
                description="run shell commands",
                parameters={"type": "object", "properties": {}},
            )
        )
        await agent.run("test")

        # The planning call for attempt 1 should have tools that include
        # shell_exec (found via reflection hint) — verify it was offered
        plan_call_2 = llm.calls[2]  # 0=plan, 1=reflect, 2=plan
        tools_offered = plan_call_2.get("tools")
        if tools_offered:
            tool_names = [t.name for t in tools_offered]
            assert "shell_exec" in tool_names


class TestPrePlanIncludesReflections:
    """Pre-plan size check must account for reflection content."""

    @pytest.mark.asyncio
    async def test_large_reflection_triggers_pre_plan_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reflection content that pushes total size over the limit
        should cause pre_plan to reject the prompt."""
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 1)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 3)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

        llm = FakeLLM()
        # Attempt 0: tool call → budget
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
                content=_reflection_json("continue", summary="ok"),
            )
        )
        # Attempt 1 should fail pre_plan due to huge reflection content

        agent = _make_echo_agent(llm)

        # Inject a huge reflection entry that will push total content over limit
        huge_reflection = ReflectionEntry(
            attempt_id=99,
            summary="x" * 110_000,  # exceeds the 100k pre_plan limit
        )
        agent.hippo.working.add_reflection(huge_reflection)

        # Attempt 0: tool call succeeds, budget exhausts, reflection fires
        # Then attempt 1: retrieval includes huge reflection → pre_plan blocks
        result = await agent.run("test")
        assert "Plan check failed" in result
        tasks = list(agent.hippo.history._tasks.values())
        assert tasks[0].status == "failed"


class TestReflectionFailureTypeInPrompt:
    """Reflection prompt must describe the real failure mode."""

    @pytest.mark.asyncio
    async def test_stuck_failure_type_in_reflection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 5)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 1)

        llm = FakeLLM()
        # Same tool+args 3 times → stuck detection
        for _ in range(4):
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

        agent = _make_echo_agent(llm)
        await agent.run("stuck task")

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
    async def test_budget_failure_type_in_reflection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as _agent_mod

        monkeypatch.setattr(_agent_mod, "STEPS_PER_ATTEMPT", 2)
        monkeypatch.setattr(_agent_mod, "MAX_ATTEMPTS", 1)
        monkeypatch.setattr(_agent_mod, "MAX_AGENT_STEPS", 10)

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

        agent = _make_echo_agent(llm)
        await agent.run("budget test")

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


class TestUnitFloatHardening:
    """Utility score floats must be clamped to [0, 1] and handle bad inputs."""

    def test_values_clamped_to_unit_range(self) -> None:
        raw = json.dumps(
            {
                "recommended_action": "continue",
                "estimated_info_gain": 2.5,
                "estimated_cost": -1.0,
                "repeat_risk": 3.0,
                "novelty": -0.5,
                "feasibility": 1.5,
            }
        )
        _, decision = Agent._parse_reflection(raw, 0, "budget")
        assert decision.estimated_info_gain == 1.0
        assert decision.estimated_cost == 0.0
        assert decision.repeat_risk == 1.0
        assert decision.novelty == 0.0
        assert decision.feasibility == 1.0

    def test_nan_defaults_safely(self) -> None:
        raw = json.dumps(
            {
                "recommended_action": "continue",
                "estimated_info_gain": float("nan"),
                "novelty": float("nan"),
            }
        )
        _, decision = Agent._parse_reflection(raw, 0, "")
        assert decision.estimated_info_gain == 0.0
        assert decision.novelty == 0.0

    def test_inf_defaults_safely(self) -> None:
        raw = json.dumps(
            {
                "estimated_info_gain": float("inf"),
                "estimated_cost": float("-inf"),
            }
        )
        _, decision = Agent._parse_reflection(raw, 0, "")
        assert decision.estimated_info_gain == 0.0
        assert decision.estimated_cost == 0.0

    def test_string_nan_defaults_safely(self) -> None:
        raw = json.dumps(
            {
                "estimated_info_gain": "nan",
                "novelty": "inf",
            }
        )
        _, decision = Agent._parse_reflection(raw, 0, "")
        # float("nan") and float("inf") are parseable but non-finite
        assert decision.estimated_info_gain == 0.0
        assert decision.novelty == 0.0

    def test_normal_values_pass_through(self) -> None:
        raw = json.dumps(
            {
                "estimated_info_gain": 0.7,
                "estimated_cost": 0.3,
                "repeat_risk": 0.1,
                "novelty": 0.9,
                "feasibility": 0.8,
            }
        )
        _, decision = Agent._parse_reflection(raw, 0, "")
        assert decision.estimated_info_gain == 0.7
        assert decision.estimated_cost == 0.3
        assert decision.novelty == 0.9
