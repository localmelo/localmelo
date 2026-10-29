"""The ``reflection`` event names the gate's flags and why a run stopped:
one stop reason per check, always consistent with ``continue_run``."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import pytest

from localmelo.melo.agent.agent import _gate_stop_reason, _should_continue, _stop_reason
from localmelo.melo.agent.options import ReflectionGate
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.schema import ReflectionDecision
from localmelo.tests.agent.test_reflection_gate import (
    CHECKS,
    GATES,
    LOW_UTILITY,
    FakeLLM,
    _agent,
    _answer,
    _echo_call,
    _reflection,
)

REASON_OF = {
    "task_still_feasible": "not_feasible",
    "next_step_is_concrete": "not_concrete",
    "next_step_is_novel": "not_novel",
}
PROSE = "I would rather say it in words."
STILL_PROSE = "Still in words, no object."


async def _run(
    replies: list[Any], **options: Any
) -> tuple[Any, list[Mapping[str, Any]]]:
    """The result and the data of every ``reflection`` event of a run."""
    events: list[AgentEvent] = []
    agent = _agent(FakeLLM(replies), steps_per_attempt=1, **options)
    result = await agent.run_task("echo the answer", on_event=events.append)
    return result, [e.data for e in events if e.type == "reflection"]


class TestStopReasonInTheEvent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", GATES)
    async def test_a_continue_that_passes_carries_the_flags_and_no_reason(
        self, gate: ReflectionGate
    ) -> None:
        result, [first, *_] = await _run(
            [_echo_call("a"), _reflection("continue"), _answer("42")],
            max_attempts=3,
            reflection_gate=gate,
        )
        assert result.status == "completed"
        assert first["flags"] == {
            "task_still_feasible": True,
            "next_step_is_concrete": True,
            "next_step_is_novel": True,
        }
        assert (first["gate"], first["stop_reason"]) == (gate, None)
        assert first["continue_run"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", GATES)
    @pytest.mark.parametrize("check", CHECKS)
    async def test_a_failed_check_names_itself(
        self, gate: ReflectionGate, check: str
    ) -> None:
        result, [event] = await _run(
            [
                _echo_call("a"),
                _reflection("continue", rationale="go on", **{check: False}),
            ],
            max_attempts=3,
            reflection_gate=gate,
        )
        assert result.status == "gate_stopped"
        assert event["flags"][check] is False
        assert [v for v in event["flags"].values() if v is False] == [False]
        assert (event["gate"], event["continue_run"]) == (gate, False)
        assert event["stop_reason"] == REASON_OF[check]
        assert result.error == "go on"

    @pytest.mark.asyncio
    async def test_a_blank_rationale_error_names_the_check(self) -> None:
        result, [event] = await _run(
            [
                _echo_call("a"),
                _reflection("continue", rationale="", next_step_is_novel=False),
            ],
            max_attempts=3,
        )
        assert result.error == "Stopped after attempt 1 (stop reason: not_novel)"
        assert event["stop_reason"] == "not_novel"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["stop", "decompose"])
    async def test_an_action_other_than_continue(self, action: str) -> None:
        result, [event] = await _run(
            [_echo_call("a"), _reflection(action, rationale="no way")],
            max_attempts=3,
        )
        assert result.status == "gate_stopped"
        assert event["stop_reason"] == "action_not_continue"
        # The rationale already says why: the error is unchanged.
        assert result.error == "no way"

    @pytest.mark.asyncio
    async def test_done_is_not_continue_either(self) -> None:
        result, [event] = await _run(
            [_echo_call("a"), _reflection("done", best_effort_result="7")],
            max_attempts=3,
        )
        assert result.status == "completed"
        assert (event["action"], event["stop_reason"]) == (
            "done",
            "action_not_continue",
        )

    @pytest.mark.asyncio
    async def test_low_utility_stops_only_under_utility(self) -> None:
        replies = [_echo_call("a"), _reflection("continue", **LOW_UTILITY)]
        result, [event] = await _run(replies, max_attempts=3, reflection_gate="utility")
        assert result.status == "gate_stopped"
        assert event["stop_reason"] == "utility_below_threshold"
        assert event["utility"] == pytest.approx(-0.5)
        result, [event, *_] = await _run(
            [*replies, _answer("42")], max_attempts=3, reflection_gate="flags"
        )
        assert (event["stop_reason"], event["utility"]) == (None, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("options", "error", "reason"),
        [
            ({"max_attempts": 1}, "Max attempts reached", "no_attempt_left"),
            ({"max_steps": 1}, "Max steps reached", "step_budget_used"),
        ],
        ids=["attempts", "steps"],
    )
    async def test_a_used_up_budget(
        self, options: dict[str, Any], error: str, reason: str
    ) -> None:
        result, [event] = await _run(
            [_echo_call("a"), _reflection("continue", best_effort_result="7")],
            **options,
        )
        assert (result.status, result.error) == ("budget_exhausted", error)
        assert (event["continue_run"], event["stop_reason"]) == (False, reason)

    @pytest.mark.asyncio
    async def test_an_unreadable_reply_with_nothing_left(self) -> None:
        result, [event] = await _run(
            [_echo_call("a"), _answer(PROSE), _answer(STILL_PROSE)], max_attempts=1
        )
        assert result.status == "budget_exhausted"
        assert (event["readable"], event["flags"]) == (False, None)
        assert (event["continue_run"], event["stop_reason"]) == (
            False,
            "unreadable_reply",
        )

    @pytest.mark.asyncio
    async def test_an_unreadable_reply_with_an_attempt_left_goes_on(self) -> None:
        _, [event, *_] = await _run(
            [_echo_call("a"), _answer(PROSE), _answer(STILL_PROSE), _answer("42")],
            max_attempts=3,
        )
        assert (event["readable"], event["continue_run"]) == (False, True)
        assert event["stop_reason"] is None


FLAG_VALUES = list(itertools.product([True, False], repeat=3))


class TestAlwaysConsistent:
    """``continue_run`` is whether there is no stop reason, for every input."""

    @pytest.mark.parametrize("gate", GATES)
    def test_over_every_decision(self, gate: ReflectionGate) -> None:
        estimates = [0.0, 0.5, 1.0]
        for action, flags, out_of, gain, cost in itertools.product(
            ["continue", "stop", "decompose", "done"],
            FLAG_VALUES,
            ["", "Max attempts reached", "Max steps reached"],
            estimates,
            estimates,
        ):
            decision = ReflectionDecision(
                recommended_action=action,
                task_still_feasible=flags[0],
                next_step_is_concrete=flags[1],
                next_step_is_novel=flags[2],
                estimated_info_gain=gain,
                estimated_cost=cost,
                novelty=0.5,
                feasibility=0.5,
                repeat_risk=0.0,
            )
            reason = _stop_reason(decision, gate, out_of)
            old = not out_of and action != "done" and _should_continue(decision, gate)
            assert (reason is None) == old, (action, flags, out_of, gain, cost)
            assert (_gate_stop_reason(decision, gate) is None) == _should_continue(
                decision, gate
            )
            if reason is not None:
                assert reason in KNOWN_REASONS

    def test_the_checks_are_read_in_a_fixed_order(self) -> None:
        decision = ReflectionDecision(
            recommended_action="continue",
            task_still_feasible=False,
            next_step_is_concrete=False,
            next_step_is_novel=False,
        )
        assert _gate_stop_reason(decision, "flags") == "not_feasible"
        decision.task_still_feasible = True
        assert _gate_stop_reason(decision, "flags") == "not_concrete"
        decision.next_step_is_concrete = True
        assert _gate_stop_reason(decision, "flags") == "not_novel"
        decision.recommended_action = "stop"
        assert _gate_stop_reason(decision, "flags") == "action_not_continue"


KNOWN_REASONS = {
    "action_not_continue",
    "not_feasible",
    "not_concrete",
    "not_novel",
    "utility_below_threshold",
    "no_attempt_left",
    "step_budget_used",
    "unreadable_reply",
}
