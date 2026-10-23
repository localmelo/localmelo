"""AgentOptions.reflection_gate: when the next attempt starts after a
reflection, under "flags" (the default) and "utility" — no real model."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import REFLECTION_PROMPT
from localmelo.melo.agent.agent import (
    CONTINUATION_UTILITY_THRESHOLD,
    _should_continue,
    _utility,
)
from localmelo.melo.agent.options import ReflectionGate
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.schema import (
    Message,
    ReflectionDecision,
    ToolCall,
    ToolDef,
)

GATES: list[ReflectionGate] = ["flags", "utility"]

# The three checks of a reflection decision that both gates require.
CHECKS = ["task_still_feasible", "next_step_is_concrete", "next_step_is_novel"]

# Estimates that give a utility of -0.5: 0.0 * 0.8 * 0.8 - 0.25 - 0.25.
LOW_UTILITY = {"estimated_info_gain": 0.0, "estimated_cost": 0.25, "repeat_risk": 0.25}


# ── Fakes and helpers ───────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued replies and records every request; fails when none is left."""

    def __init__(self, replies: list[Message]) -> None:
        self.replies: deque[Message] = deque(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            raise AssertionError("the test queued no further reply")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


class FakeEmbedding(BaseEmbeddingProvider):
    """Embeds every text alike and keeps each text it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


async def _echo(text: str) -> str:
    return text


def _agent(
    llm: BaseLLMProvider,
    *,
    embedding: BaseEmbeddingProvider | None = None,
    **options: Any,
) -> Agent:
    """An agent whose only tool is echo (named in every query below)."""
    agent = Agent(
        llm=llm,
        embedding=embedding,
        options=AgentOptions(builtins=frozenset(), **options),
    )
    agent.hippo.register_tool(
        ToolDef(
            name="echo",
            description="echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    )
    agent.executor.register("echo", _echo)
    return agent


def _echo_call(text: str, content: str = "") -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_call=ToolCall(tool_name="echo", arguments={"text": text}),
    )


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _decision(action: str, **fields: Any) -> dict[str, Any]:
    """A reflection decision; with ``continue`` it passes both gates."""
    data: dict[str, Any] = {
        "recommended_action": action,
        "rationale": "test",
        "best_effort_result": "",
        "summary": "tried echo",
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "estimated_info_gain": 0.5,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    data.update(fields)
    return data


def _reflection(action: str, **fields: Any) -> Message:
    return Message(role="assistant", content=json.dumps(_decision(action, **fields)))


def _is_reflection(call: dict[str, Any]) -> bool:
    system: Message = call["messages"][0]
    return system.content.startswith(REFLECTION_PROMPT)


def _reflection_count(llm: FakeLLM) -> int:
    return sum(1 for call in llm.calls if _is_reflection(call))


def _passing_decision(**fields: Any) -> ReflectionDecision:
    """A ``continue`` decision whose three checks hold."""
    values: dict[str, Any] = {
        "recommended_action": "continue",
        "task_still_feasible": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
    }
    values.update(fields)
    return ReflectionDecision(**values)


# ── The gate ────────────────────────────────────────────────────────────────


class TestShouldContinue:
    def test_the_utility_comes_from_the_five_estimates(self) -> None:
        decision = _passing_decision(
            estimated_info_gain=0.5,
            feasibility=0.5,
            novelty=0.5,
            estimated_cost=0.25,
            repeat_risk=0.125,
        )
        assert _utility(decision) == 0.5 * 0.5 * 0.5 - 0.25 - 0.125

    def test_flags_ignore_the_utility(self) -> None:
        decision = _passing_decision(estimated_cost=1.0, repeat_risk=1.0)
        assert _utility(decision) == -2.0
        assert _should_continue(decision) is True
        assert _should_continue(decision, "flags") is True
        assert _should_continue(decision, "utility") is False

    def test_utility_continues_from_the_threshold_on(self) -> None:
        at = _passing_decision(estimated_info_gain=0.1, feasibility=1.0, novelty=1.0)
        assert _utility(at) == CONTINUATION_UTILITY_THRESHOLD
        assert _should_continue(at, "utility") is True
        below = _passing_decision(
            estimated_info_gain=0.1, feasibility=1.0, novelty=1.0, repeat_risk=0.01
        )
        assert _should_continue(below, "utility") is False

    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("action", ["stop", "decompose", "done"])
    def test_only_continue_continues(self, gate: ReflectionGate, action: str) -> None:
        decision = _passing_decision(
            recommended_action=action,
            estimated_info_gain=1.0,
            feasibility=1.0,
            novelty=1.0,
        )
        assert _should_continue(decision, gate) is False

    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("check", CHECKS)
    def test_every_check_must_hold(self, gate: ReflectionGate, check: str) -> None:
        decision = _passing_decision(
            estimated_info_gain=1.0, feasibility=1.0, novelty=1.0, **{check: False}
        )
        assert _should_continue(decision, gate) is False


class TestGateInTheLoop:
    @pytest.mark.asyncio
    async def test_a_low_utility_continue_starts_the_next_attempt_under_flags(
        self,
    ) -> None:
        reply = _reflection(
            "continue", rationale="run the tests", best_effort_result="7", **LOW_UTILITY
        )
        llm = FakeLLM([_echo_call("a"), reply, _answer("42")])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == ("completed", "42", "")
        assert (result.steps, result.attempts) == (2, 2)
        assert (result.n_main, result.n_aux) == (2, 1)

    @pytest.mark.asyncio
    async def test_a_low_utility_continue_is_gate_stopped_under_utility(
        self,
    ) -> None:
        reply = _reflection(
            "continue", rationale="run the tests", best_effort_result="7", **LOW_UTILITY
        )
        llm = FakeLLM([_echo_call("a"), reply, _answer("42")])
        agent = _agent(
            llm, steps_per_attempt=1, max_attempts=3, reflection_gate="utility"
        )
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "run the tests",
        )
        assert (result.steps, result.attempts) == (1, 1)
        assert (result.n_main, result.n_aux) == (1, 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("check", CHECKS)
    async def test_a_continue_with_a_check_false_is_gate_stopped(
        self, gate: ReflectionGate, check: str
    ) -> None:
        reply = _reflection(
            "continue",
            rationale="not this way",
            best_effort_result="7",
            estimated_info_gain=1.0,
            novelty=1.0,
            feasibility=1.0,
            **{check: False},
        )
        llm = FakeLLM([_echo_call("a"), reply])
        agent = _agent(llm, steps_per_attempt=1, max_attempts=3, reflection_gate=gate)
        result = await agent.run_task("echo the answer")
        assert (result.status, result.result, result.error) == (
            "gate_stopped",
            "7",
            "not this way",
        )
        assert result.attempts == 1
        assert _reflection_count(llm) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", GATES)
    async def test_the_run_result_reports_the_gate(self, gate: ReflectionGate) -> None:
        agent = _agent(FakeLLM([_answer("42")]), reflection_gate=gate)
        result = await agent.run_task("what is the answer")
        assert result.config["reflection_gate"] == gate
