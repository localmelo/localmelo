"""Brief runs only (``AgentOptions.brief_probe``): a ``continue`` reflection
goes on whatever its flags say, and an attempt boundary that finds the
workspace unchanged adds one line for the next attempt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from localmelo.melo.agent.agent import NO_CHANGE_PUSH, _stop_reason
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.schema import ReflectionDecision
from localmelo.tests.agent.test_reflection_gate import (
    CHECKS,
    FakeLLM,
    _agent,
    _answer,
    _decision,
    _echo_call,
    _reflection,
)


def _digest(*values: str | None) -> Any:
    """A probe that returns *values* in turn, then the last one again."""
    seen = list(values)

    def probe(workspace: str) -> str | None:
        return seen.pop(0) if len(seen) > 1 else seen[0]

    return probe


async def _run(
    replies: list[Any], **options: Any
) -> tuple[Any, FakeLLM, list[AgentEvent]]:
    events: list[AgentEvent] = []
    llm = FakeLLM(replies)
    agent = _agent(llm, steps_per_attempt=1, **options)
    result = await agent.run_task("echo the answer", on_event=events.append)
    return result, llm, events


def _data(events: list[AgentEvent], type_: str) -> list[Mapping[str, Any]]:
    return [e.data for e in events if e.type == type_]


class TestContinueContinues:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("check", CHECKS)
    async def test_a_brief_run_goes_on_with_a_false_flag(self, check: str) -> None:
        result, _llm, events = await _run(
            [
                _echo_call("a"),
                _reflection("continue", **{check: False}),
                _answer("42"),
            ],
            max_attempts=3,
            brief_probe=_digest(None),
        )
        assert result.status == "completed"
        [event] = _data(events, "reflection")
        assert event["flags"][check] is False  # still recorded
        assert (event["continue_run"], event["stop_reason"]) == (True, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("check", CHECKS)
    async def test_the_same_decision_without_a_brief_stops(self, check: str) -> None:
        result, _llm, events = await _run(
            [_echo_call("a"), _reflection("continue", **{check: False})],
            max_attempts=3,
        )
        assert result.status == "gate_stopped"
        [event] = _data(events, "reflection")
        assert event["continue_run"] is False and event["stop_reason"] is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["stop", "decompose"])
    async def test_another_action_still_stops(self, action: str) -> None:
        result, _llm, events = await _run(
            [_echo_call("a"), _reflection(action, rationale="no way")],
            max_attempts=3,
            brief_probe=_digest(None),
        )
        assert result.status == "gate_stopped"
        [event] = _data(events, "reflection")
        assert event["stop_reason"] == "action_not_continue"

    @pytest.mark.asyncio
    async def test_a_spent_budget_still_ends_the_run(self) -> None:
        result, _llm, events = await _run(
            [_echo_call("a"), _reflection("continue", next_step_is_novel=False)],
            max_steps=1,
            max_attempts=3,
            brief_probe=_digest(None),
        )
        assert result.status == "budget_exhausted"
        [event] = _data(events, "reflection")
        assert event["stop_reason"] == "step_budget_used"

    def test_the_decision_function_agrees(self) -> None:
        decision = ReflectionDecision(
            **{
                k: v
                for k, v in _decision("continue", next_step_is_concrete=False).items()
                if k in ReflectionDecision.__dataclass_fields__
            }
        )
        assert _stop_reason(decision, "flags", "") == "not_concrete"
        assert _stop_reason(decision, "flags", "", brief=True) is None
        assert _stop_reason(decision, "utility", "", brief=True) is None
        assert (
            _stop_reason(decision, "flags", "Max steps reached", brief=True)
            == "step_budget_used"
        )


def _pushes(llm: FakeLLM) -> int:
    return sum(
        message.content == NO_CHANGE_PUSH
        for call in llm.calls
        for message in call["messages"]
    )


def _first_push_request(llm: FakeLLM) -> int | None:
    for index, call in enumerate(llm.calls):
        if any(message.content == NO_CHANGE_PUSH for message in call["messages"]):
            return index
    return None


class TestNoChangePush:
    REPLIES = [_echo_call("a"), _reflection("continue"), _answer("42")]

    @pytest.mark.asyncio
    async def test_a_clean_tree_gets_the_line_in_the_next_first_request(self) -> None:
        result, llm, events = await _run(
            list(self.REPLIES), max_attempts=3, brief_probe=_digest("same")
        )
        assert result.status == "completed"
        # plan, reflect, then the next attempt's first planning request.
        assert _first_push_request(llm) == 2
        assert _pushes(llm) == 1
        [notice] = [d for d in _data(events, "notice") if d["kind"] == "no_change"]
        assert notice["message"] == "Nothing has changed in the workspace yet."
        assert (notice["attempt"], notice["detail"]) == (1, {})

    @pytest.mark.asyncio
    async def test_a_changed_tree_gets_no_line(self) -> None:
        result, llm, events = await _run(
            list(self.REPLIES), max_attempts=3, brief_probe=_digest("a", "b")
        )
        assert result.status == "completed"
        assert _pushes(llm) == 0
        assert not [d for d in _data(events, "notice") if d["kind"] == "no_change"]

    @pytest.mark.asyncio
    async def test_a_tree_that_was_dirty_and_is_unchanged_gets_it(self) -> None:
        # The digest, not a dirty flag: a dirty start is the baseline.
        _result, llm, _events = await _run(
            list(self.REPLIES), max_attempts=3, brief_probe=_digest("dirty")
        )
        assert _pushes(llm) == 1

    @pytest.mark.asyncio
    async def test_no_git_tree_gets_no_line(self) -> None:
        _result, llm, events = await _run(
            list(self.REPLIES), max_attempts=3, brief_probe=_digest(None)
        )
        assert _pushes(llm) == 0
        assert not [d for d in _data(events, "notice") if d["kind"] == "no_change"]

    @pytest.mark.asyncio
    async def test_a_run_without_a_brief_gets_no_line(self) -> None:
        _result, llm, events = await _run(list(self.REPLIES), max_attempts=3)
        assert _pushes(llm) == 0
        assert not [d for d in _data(events, "notice") if d["kind"] == "no_change"]

    @pytest.mark.asyncio
    async def test_the_line_goes_with_each_unchanged_boundary(self) -> None:
        result, llm, events = await _run(
            [
                _echo_call("a"),
                _reflection("continue"),
                _echo_call("b"),
                _reflection("continue"),
                _answer("42"),
            ],
            max_attempts=3,
            brief_probe=_digest("same"),
        )
        assert result.status == "completed"
        notices = [d for d in _data(events, "notice") if d["kind"] == "no_change"]
        assert [d["attempt"] for d in notices] == [1, 2]

    def test_the_line_is_the_one_in_the_brief(self) -> None:
        assert NO_CHANGE_PUSH == (
            "Nothing has changed in the workspace yet. Start this attempt by writing "
            "the part of the task you can already write, then continue."
        )
