"""Long attempts: 50 steps, one reflection, then on to 100; reflection as an option.

A run reflects only after 50 steps without an answer, and has 100 steps in all.
``attempt_policy="none"`` (what ``[agent] reflection = false`` sets) never reflects
and runs straight to the total. The earlier lengths (``max_steps=30``,
``steps_per_attempt=10``) still give the earlier requests: that is pinned with the
recorded goldens in ``test_legacy_golden.py`` and ``test_legacy_freeze.py``. No real
model.
"""

from __future__ import annotations

from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.schema import (
    MAX_AGENT_STEPS,
    MAX_ATTEMPTS,
    STEPS_PER_ATTEMPT,
    Message,
    ToolCall,
    ToolDef,
)
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _answer,
    _echo,
    _reflection,
)


def _call(n: int) -> Message:
    """An echo call that differs from every other, so no run is stuck."""
    return Message(
        role="assistant",
        tool_call=ToolCall(tool_name="echo", arguments={"text": f"step {n}"}),
    )


def _calls(first: int, count: int) -> list[Message]:
    return [_call(n) for n in range(first, first + count)]


def _agent(llm: FakeLLM, **options: Any) -> Agent:
    """An agent with the default length and gate, and echo as its only tool."""
    agent = Agent(
        llm=llm,
        options=AgentOptions(builtins=frozenset(), memory=False, **options),
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


class Sink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    def count(self, kind: str) -> int:
        return sum(1 for event in self.events if event.type == kind)


def test_the_defaults_are_a_run_of_100_steps_in_attempts_of_50() -> None:
    assert (MAX_AGENT_STEPS, STEPS_PER_ATTEMPT, MAX_ATTEMPTS) == (100, 50, 5)
    options = AgentOptions()
    assert options.max_steps == 100
    assert options.steps_per_attempt == 50
    assert options.attempt_policy == "reflect"


@pytest.mark.asyncio
async def test_a_task_of_49_steps_never_reflects() -> None:
    llm = FakeLLM([*_calls(1, 49), _answer("done")])
    sink = Sink()
    result = await _agent(llm).run_task("echo", on_event=sink)
    assert (result.status, result.result) == ("completed", "done")
    assert sink.count("reflection.started") == 0
    assert sink.count("step.started") == 50


@pytest.mark.asyncio
async def test_a_task_not_done_after_50_steps_reflects_once_then_goes_on() -> None:
    llm = FakeLLM(
        [
            *_calls(1, 50),
            _reflection("continue", summary="fifty steps in"),
            *_calls(51, 20),
            _answer("found it at step 71"),
        ]
    )
    sink = Sink()
    result = await _agent(llm).run_task("echo", on_event=sink)
    assert (result.status, result.result) == ("completed", "found it at step 71")
    assert sink.count("reflection.started") == 1
    assert sink.count("step.started") == 71
    assert result.attempts == 2
    reflecting = next(
        i for i, e in enumerate(sink.events) if e.type == "reflection.started"
    )
    steps_before = sum(1 for e in sink.events[:reflecting] if e.type == "step.started")
    assert steps_before == 50


@pytest.mark.asyncio
async def test_a_task_that_never_ends_stops_at_100_steps_with_its_last_word() -> None:
    llm = FakeLLM(
        [
            *_calls(1, 50),
            _reflection("continue"),
            *_calls(51, 50),
            _reflection("stop", best_effort_result="partial"),
        ]
    )
    sink = Sink()
    result = await _agent(llm).run_task("echo", on_event=sink)
    assert result.status == "budget_exhausted"
    assert result.result == "partial"
    assert sink.count("step.started") == 100
    # The first reflection decides to go on; the last one only finalizes the answer.
    assert sink.count("reflection.started") == 2
    assert len(llm.calls) == 102


@pytest.mark.asyncio
async def test_without_reflection_a_run_goes_straight_to_100_steps() -> None:
    llm = FakeLLM([*_calls(1, 100), _answer("answer from the evidence")])
    sink = Sink()
    agent = _agent(llm, attempt_policy="none")
    result = await agent.run_task("echo", on_event=sink)
    assert sink.count("reflection.started") == 0
    assert sink.count("step.started") == 100
    assert result.status == "budget_exhausted"
    assert result.result == "answer from the evidence"
    assert result.config["attempt_policy"] == "none"
    assert result.config["max_steps"] == 100
    # 100 planning requests and the one for an answer; nothing asks for a reflection.
    assert len(llm.calls) == 101


@pytest.mark.asyncio
async def test_the_earlier_lengths_reflect_every_10_steps_of_30() -> None:
    llm = FakeLLM(
        [
            *_calls(1, 10),
            _reflection("continue"),
            *_calls(11, 10),
            _reflection("continue"),
            *_calls(21, 10),
            _reflection("stop", best_effort_result="partial"),
        ]
    )
    sink = Sink()
    result = await _agent(llm, max_steps=30, steps_per_attempt=10).run_task(
        "echo", on_event=sink
    )
    assert result.status == "budget_exhausted"
    assert sink.count("reflection.started") == 3
    assert sink.count("step.started") == 30
    assert result.config["max_steps"] == 30
    assert result.config["steps_per_attempt"] == 10


@pytest.mark.asyncio
async def test_a_routine_learns_per_set_of_length_options() -> None:
    """The new lengths start their own template; the earlier ones are not mixed in."""
    from localmelo.melo.memory.procedures.normalize import length_options, template_id

    async def template(**options: Any) -> str:
        result = await _agent(FakeLLM([_answer("ok")]), **options).run_task("echo")
        return template_id("routine", "model", length_options(result.config))

    now = await template()
    assert await template() == now
    assert await template(max_steps=30, steps_per_attempt=10) != now
    assert await template(attempt_policy="none") != now
    assert await template(steps_per_attempt=20) != now
