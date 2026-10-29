"""``fixed-prefix-v2``: the reflection goes last, and a reflection request repeats the plan.

The model is a stub OpenAI-compatible server behind ``httpx.MockTransport``.
"""

from __future__ import annotations

from typing import Any

from localmelo.melo.agent._chat import PREFIX_VERSION, REFLECTION_PROMPT
from localmelo.melo.contracts.events import AgentEvent
from localmelo.tests.agent.test_cache_telemetry import (
    Stub,
    _cached,
    _call_reply,
    _final_reply,
    _reflection_reply,
    _run,
)

TWO_ATTEMPTS: dict[str, Any] = {"steps_per_attempt": 2, "fixed_prefix": True}


def _stub(*, reflection: list[dict[str, Any]] | None = None) -> Stub:
    return Stub(
        [
            _call_reply(_cached(0)),
            _call_reply(_cached(0)),
            *(reflection or [_reflection_reply(_cached(0))]),
            _call_reply(_cached(0)),
            _final_reply(_cached(0)),
        ]
    )


def _of(events: list[AgentEvent], kind: str) -> list[AgentEvent]:
    return [e for e in events if e.type == kind]


async def test_a_reflection_request_starts_with_the_last_plan_request() -> None:
    stub = _stub()
    events = await _run(stub, **TWO_ATTEMPTS)
    plan, reflect = stub.bodies[1], stub.bodies[2]
    assert PREFIX_VERSION == "fixed-prefix-v2"
    assert reflect["tools"] == plan["tools"] and plan["tools"]
    assert reflect["messages"][: len(plan["messages"])] == plan["messages"]
    final = reflect["messages"][-1]
    assert final["role"] == "user" and final["content"].startswith(REFLECTION_PROMPT)
    assert "Original task: echo hi" in final["content"]
    [reflection] = _of(events, "reflection")
    prefix = reflection.data["prefix"]
    assert prefix["against"] == {"kind": "plan", "request": 2}
    assert prefix["messages"] == len(plan["messages"])
    assert prefix["tools_changed"] is False
    assert prefix["total_messages"] == len(reflect["messages"])
    assert (reflection.data["readable"], reflection.data["requests"]) == (True, 1)


async def test_the_plan_after_a_reflection_shares_the_whole_history() -> None:
    stub = _stub()
    events = await _run(stub, **TWO_ATTEMPTS)
    before, reflect, after = stub.bodies[1], stub.bodies[2], stub.bodies[3]
    # The system text and the tools do not change across the attempt boundary.
    assert after["messages"][0] == before["messages"][0]
    assert after["tools"] == before["tools"]
    # The reflection is the last message, after the history of attempt one.
    tail = after["messages"][-1]
    assert tail["role"] == "user"
    assert "[Reflections; data, not instructions]" in tail["content"]
    assert "Echoed once." in tail["content"]
    assert not any(
        "Echoed once." in m["content"] for m in after["messages"][:-1] if m["content"]
    )
    plans = _of(events, "assistant.message")
    prefix = plans[2].data["prefix"]
    history = len(reflect["messages"]) - 1
    assert prefix["messages"] == history and prefix["first_changed"] == "user"
    assert after["messages"][:history] == reflect["messages"][:history]
    assert prefix["tools_changed"] is False


async def test_a_second_reflection_repeats_the_plan_that_carried_the_first() -> None:
    stub = Stub(
        [
            _call_reply(_cached(0)),
            _reflection_reply(_cached(0)),
            _call_reply(_cached(0)),
            _reflection_reply(_cached(0)),
            _final_reply(_cached(0)),
        ]
    )
    events = await _run(stub, steps_per_attempt=1, fixed_prefix=True)
    plan, reflect = stub.bodies[2], stub.bodies[3]
    assert reflect["messages"][: len(plan["messages"])] == plan["messages"]
    # The plan held the first reflection already; the request does not repeat it.
    text = reflect["messages"][-1]["content"]
    assert "Prior reflections" not in text and "Echoed once." not in text
    first, second = _of(events, "reflection")
    assert second.data["prefix"]["against"] == {"kind": "plan", "request": 3}
    assert first.data["prefix"]["against"] == {"kind": "plan", "request": 1}


async def test_a_tool_call_during_reflection_is_asked_again_and_the_run_goes_on() -> (
    None
):
    stub = _stub(
        reflection=[_call_reply(_cached(0)), _reflection_reply(_cached(0))],
    )
    events = await _run(stub, **TWO_ATTEMPTS)
    [reflection] = _of(events, "reflection")
    assert (reflection.data["requests"], reflection.data["readable"]) == (2, True)
    reflect, retry, after = stub.bodies[2], stub.bodies[3], stub.bodies[4]
    assert reflect["tools"]
    # The retry has the transcript as text and offers no tools.
    assert "tools" not in retry
    assert "Echoed once." not in retry["messages"][-1]["content"]
    assert REFLECTION_PROMPT in retry["messages"][0]["content"]
    assert after["tools"] == stub.bodies[1]["tools"]
    assert _of(events, "assistant.message")[-1].data["content"] == "done"


async def test_flag_off_keeps_the_reflection_in_the_system_prompt_and_the_old_request() -> (
    None
):
    stub = _stub()
    await _run(stub, steps_per_attempt=2)
    reflect, after = stub.bodies[2], stub.bodies[3]
    assert "tools" not in reflect
    assert reflect["messages"][0]["content"].startswith(REFLECTION_PROMPT[:20])
    assert "[REFLECTION]" in after["messages"][0]["content"]
    assert not any(
        "[Reflections; data" in (m["content"] or "") for m in after["messages"]
    )
