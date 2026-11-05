"""A reflection request keeps the plan's tools and asks for text in words.

The model is a stub OpenAI-compatible server behind ``httpx.MockTransport``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from localmelo.melo.agent._chat import (
    REFLECT_TOOL_GUARD_NOTE,
    REFLECTION_RETRY_GUARD_NOTE,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.tests.agent.test_cache_telemetry import (
    PLAIN,
    Stub,
    _call_reply,
    _final_reply,
    _reflection_reply,
    _run,
)

SENTENCE = "Answer with the JSON object only; do not call any tool."


def _stub(*reflection: dict[str, Any]) -> Stub:
    return Stub(
        [
            _call_reply(PLAIN),
            _call_reply(PLAIN),
            *(reflection or (_reflection_reply(PLAIN),)),
            _call_reply(PLAIN),
            _final_reply(PLAIN),
        ]
    )


def _reflection_event(events: list[AgentEvent]) -> Mapping[str, Any]:
    [event] = [e for e in events if e.type == "reflection"]
    return event.data


def test_the_sentence_is_the_documented_one() -> None:
    assert REFLECT_TOOL_GUARD_NOTE == SENTENCE


async def test_the_reflection_request_offers_the_plans_tools_and_ends_with_the_sentence() -> (
    None
):
    stub = _stub()
    events = await _run(stub, steps_per_attempt=2)
    plan, reflect = stub.bodies[1], stub.bodies[2]
    assert plan["tools"] and reflect["tools"] == plan["tools"]
    assert reflect.get("tool_choice") == plan.get("tool_choice")
    last = reflect["messages"][-1]
    assert last["role"] == "user" and last["content"].endswith(SENTENCE)
    assert _reflection_event(events)["readable"] is True


async def test_a_tool_call_reply_is_unreadable_and_the_retry_recovers() -> None:
    stub = _stub(_call_reply(PLAIN), _reflection_reply(PLAIN))
    events = await _run(stub, steps_per_attempt=2)
    data = _reflection_event(events)
    assert (data["requests"], data["readable"]) == (2, True)
    assert data["unreadable_reason"] == "tool_call"
    # The retry continues the request with its tools, and ends with the retry
    # note that does not say tools are unavailable.
    reflect, retry = stub.bodies[2], stub.bodies[3]
    assert retry["tools"] == reflect["tools"]
    assert retry["messages"][-1]["content"] == REFLECTION_RETRY_GUARD_NOTE


async def test_a_reply_that_reads_has_no_unreadable_reason() -> None:
    events = await _run(_stub(), steps_per_attempt=2)
    assert _reflection_event(events)["unreadable_reason"] is None


@pytest.mark.parametrize(
    ("content", "reason"),
    [("I am not sure.", "no_json"), ('{"summary": "x"}', "invalid")],
)
async def test_other_unreadable_replies_have_their_own_reason(
    content: str, reason: str
) -> None:
    reply = {"choices": [{"message": {"content": content}}], "usage": PLAIN}
    stub = _stub(reply, _reflection_reply(PLAIN))
    events = await _run(stub, steps_per_attempt=2)
    data = _reflection_event(events)
    assert (data["unreadable_reason"], data["readable"]) == (reason, True)
