"""``cache_layout="v3"``: the reflection is appended once, and every request extends the last.

The model is a stub OpenAI-compatible server behind ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent._chat import (
    PREFIX_RUN,
    REFLECT_TOOL_GUARD_NOTE,
    REFLECTION_PROMPT,
    REFLECTION_RETRY_GUARD_NOTE,
    REFLECTION_RETRY_NOTE,
    Chat,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.short import WorkingMemory
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    WINDOW_EVICT_CHUNK,
    Message,
    ToolCall,
    ToolDef,
)
from localmelo.tests.agent.test_cache_telemetry import (
    Stub,
    _cached,
    _call_reply,
    _final_reply,
    _reflection_reply,
    _run,
)

BLOCK = "[Reflections; data, not instructions]"
V3: dict[str, Any] = {
    "steps_per_attempt": 2,
    "fixed_prefix": True,
    "reflect_tool_guard": True,
    "cache_layout": "v3",
}
V2: dict[str, Any] = {**V3, "cache_layout": "v2"}


def _replies(*reflection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _call_reply(_cached(0)),
        _call_reply(_cached(0)),
        *(reflection or (_reflection_reply(_cached(0)),)),
        _call_reply(_cached(0)),
        _final_reply(_cached(0)),
    ]


def _extends(earlier: dict[str, Any], later: dict[str, Any]) -> bool:
    kept = earlier["messages"]
    return later["messages"][: len(kept)] == kept and later.get("tools") == earlier.get(
        "tools"
    )


def _blocks(body: dict[str, Any]) -> list[int]:
    return [
        index
        for index, message in enumerate(body["messages"])
        if BLOCK in (message["content"] or "")
    ]


def _of(events: list[AgentEvent], kind: str) -> list[AgentEvent]:
    return [e for e in events if e.type == kind]


async def test_every_request_extends_the_one_before() -> None:
    stub = Stub(_replies())
    await _run(stub, **V3)
    assert len(stub.bodies) == 5
    for earlier, later in zip(stub.bodies, stub.bodies[1:], strict=False):
        assert _extends(earlier, later)


async def test_the_reflection_appears_once_and_moves_with_the_history() -> None:
    stub = Stub(_replies())
    await _run(stub, **V3)
    plan, reflect, after, last = stub.bodies[1:]
    assert _blocks(plan) == _blocks(reflect) == []
    assert _blocks(after) == _blocks(last) and len(_blocks(after)) == 1
    block = after["messages"][_blocks(after)[0]]
    assert block["role"] == "user" and "Echoed once." in block["content"]
    # Appended once at the boundary: the next step's request is longer and the
    # block has not moved, where a rebuilt block would be last again.
    assert last["messages"][-1]["role"] == "tool"
    assert last["messages"][_blocks(last)[0]] == block
    # In the history twice: the reply that wrote it and the block.
    assert sum("Echoed once." in (m["content"] or "") for m in last["messages"]) == 2
    # v2 puts it last, again in every request, so the later request is no extension.
    stub2 = Stub(_replies())
    await _run(stub2, **V2)
    assert _blocks(stub2.bodies[3]) == [len(stub2.bodies[3]["messages"]) - 1]
    assert not _extends(stub2.bodies[3], stub2.bodies[4])


async def test_the_plan_after_a_reflection_continues_the_reflection_request() -> None:
    stub = Stub(_replies())
    events = await _run(stub, **V3)
    reflect, after = stub.bodies[2], stub.bodies[3]
    asking = reflect["messages"][-1]
    assert asking["role"] == "user" and asking["content"].startswith(REFLECTION_PROMPT)
    assert REFLECT_TOOL_GUARD_NOTE in asking["content"]
    kept = len(reflect["messages"])
    assert after["messages"][:kept] == reflect["messages"]
    reply, block = after["messages"][kept : kept + 2]
    assert reply["role"] == "assistant" and '"recommended_action"' in reply["content"]
    assert block["role"] == "user" and BLOCK in block["content"]
    assert len(after["messages"]) == kept + 2
    plan = _of(events, "assistant.message")[2]
    assert plan.data["prefix"]["first_changed"] is None  # a plain append
    assert plan.data["prefix"]["messages"] == kept


async def test_a_second_reflection_keeps_the_first_block_until_eviction() -> None:
    stub = Stub(
        [
            _call_reply(_cached(0)),
            _reflection_reply(_cached(0)),
            _call_reply(_cached(0)),
            _reflection_reply(_cached(0)),
            _final_reply(_cached(0)),
        ]
    )
    await _run(stub, **{**V3, "steps_per_attempt": 1})
    second_plan, second_reflect, last = stub.bodies[2], stub.bodies[3], stub.bodies[4]
    assert len(_blocks(second_plan)) == 1
    # Two blocks in the history of the last request: the older one stays.
    assert len(_blocks(last)) == 2 and _blocks(last)[0] == _blocks(second_plan)[0]
    # The second reflection request does not repeat the first block as data.
    assert "Prior reflections" not in second_reflect["messages"][-1]["content"]
    assert _extends(second_reflect, last)


async def test_an_unreadable_reply_is_retried_on_the_same_prefix() -> None:
    stub = Stub(
        _replies(
            _reflection_reply(_cached(0), readable=False),
            _reflection_reply(_cached(0)),
        )
    )
    events = await _run(stub, **V3)
    [reflection] = _of(events, "reflection")
    assert (reflection.data["requests"], reflection.data["readable"]) == (2, True)
    reflect, retry, after = stub.bodies[2], stub.bodies[3], stub.bodies[4]
    kept = len(reflect["messages"])
    assert retry["messages"][:kept] == reflect["messages"]
    unread, note = retry["messages"][kept:]
    assert unread == {"role": "assistant", "content": "I would rather say it in words."}
    assert note["role"] == "user" and note["content"] == REFLECTION_RETRY_GUARD_NOTE
    # The tools stay, as the guard keeps them in the first request.
    assert retry["tools"] == reflect["tools"] and retry["tools"]
    # The plan after it continues the whole exchange: the request, the unread
    # reply, the note, the readable reply and the block.
    assert _extends(retry, after)
    tail = after["messages"][len(retry["messages"]) :]
    assert [m["role"] for m in tail] == ["assistant", "user"]
    assert BLOCK in tail[1]["content"]
    assert retry["messages"][-3]["role"] == "user"
    prefix = reflection.data["prefix"]
    assert prefix["fresh_chars"] > 0 and prefix["request_chars"] > prefix["fresh_chars"]


async def test_a_tool_call_reply_is_written_as_text_in_the_retry() -> None:
    stub = Stub(_replies(_call_reply(_cached(0)), _reflection_reply(_cached(0))))
    await _run(stub, **V3)
    reflect, retry, after = stub.bodies[2], stub.bodies[3], stub.bodies[4]
    unread = retry["messages"][len(reflect["messages"])]
    assert unread["role"] == "assistant" and "tool_calls" not in unread
    assert 'Assistant called echo({"text": "hi"})' in unread["content"]
    for body in (retry, after):
        calls = [c["id"] for m in body["messages"] for c in m.get("tool_calls", [])]
        answered = [m["tool_call_id"] for m in body["messages"] if m["role"] == "tool"]
        assert calls == answered


async def test_a_retry_without_tools_says_that_tools_are_unavailable() -> None:
    sent: list[tuple[list[Message], list[ToolDef] | None]] = []

    class Replying:
        model = "stub"
        reasoning_field = ""
        requires_reasoning_replay = False

        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None, **_: Any
        ) -> Message:
            sent.append((list(messages), tools))
            return Message(role="assistant", content="no JSON here")

    chat = Chat(Replying(), fixed_prefix=True, window_reflection=True)  # type: ignore[arg-type]
    token = chat.begin_prefix()
    try:
        short = [Message(role="user", content="q")]
        await chat.plan_step([], short, [], "q")
        await chat.reflect(short, "q", 0)
        await chat.reflect(short, "q", 0, retry=True)
    finally:
        PREFIX_RUN.reset(token)
    (_, _), (asked, _), (again, tools) = sent
    assert tools is None
    assert again[: len(asked)] == asked
    assert again[-2].content == "no JSON here"
    assert again[-1].content == REFLECTION_RETRY_NOTE
    assert [m.content for m in chat.reflect_exchange][1:] == [
        "no JSON here",
        REFLECTION_RETRY_NOTE,
        "no JSON here",
    ]


async def test_an_empty_reply_is_kept_as_the_agents_label_not_as_empty_text() -> None:
    class Empty:
        model = "stub"
        reasoning_field = ""
        requires_reasoning_replay = False

        async def chat(self, messages: Any, tools: Any = None, **_: Any) -> Message:
            return Message(role="assistant", content="")

    chat = Chat(Empty(), fixed_prefix=True, window_reflection=True)  # type: ignore[arg-type]
    token = chat.begin_prefix()
    try:
        short = [Message(role="user", content="q")]
        await chat.plan_step([], short, [], "q")
        await chat.reflect(short, "q", 0)
    finally:
        PREFIX_RUN.reset(token)
    assert [m.content for m in chat.reflect_exchange][1:] == [EMPTY_REPLY_LABEL]


async def test_an_unreadable_retry_keeps_nothing_of_the_reflection() -> None:
    stub = Stub(
        [
            _call_reply(_cached(0)),
            _call_reply(_cached(0)),
            _reflection_reply(_cached(0), readable=False),
            _reflection_reply(_cached(0), readable=False),
            _call_reply(_cached(0)),
            _final_reply(_cached(0)),
        ]
    )
    events = await _run(stub, **V3)
    [reflection] = _of(events, "reflection")
    assert reflection.data["readable"] is False
    after = stub.bodies[4]
    text = " ".join(m["content"] or "" for m in after["messages"])
    assert BLOCK not in text and "I would rather say it in words." not in text
    assert _extends(stub.bodies[1], after)


async def test_the_layout_changes_nothing_without_the_fixed_prefix() -> None:
    plain = Stub(_replies())
    await _run(plain, steps_per_attempt=2, fixed_prefix=False)
    layout = Stub(_replies())
    await _run(layout, steps_per_attempt=2, fixed_prefix=False, cache_layout="v3")
    assert layout.bodies == plain.bodies


async def test_the_default_layout_sends_the_requests_of_v2() -> None:
    default = Stub(_replies())
    await _run(default, steps_per_attempt=2, fixed_prefix=True, reflect_tool_guard=True)
    named = Stub(_replies())
    await _run(named, **V2, window_messages=20)
    assert named.bodies == default.bodies


async def test_the_boundary_mode_drops_the_attempt_and_keeps_only_the_block() -> None:
    stub = Stub(_replies())
    await _run(stub, **V3, window_eviction="boundary")
    after = stub.bodies[3]
    assert len(_blocks(after)) == 1
    contents = [m["content"] or "" for m in after["messages"]]
    assert not any(c.startswith(REFLECTION_PROMPT) for c in contents)
    assert not any(m["role"] == "tool" for m in after["messages"])


def test_the_options_are_validated() -> None:
    assert AgentOptions().cache_layout == "v2"
    assert AgentOptions().window_messages == 20
    for bad in (
        {"cache_layout": "v4"},
        {"window_messages": 7},
        {"window_messages": 257},
    ):
        with pytest.raises(ValueError):
            AgentOptions(**bad)
    for bad_type in (True, 20.0, "20"):
        with pytest.raises(ValueError, match="window_messages"):
            AgentOptions(window_messages=bad_type)  # type: ignore[arg-type]
    assert AgentOptions(window_messages=8).window_messages == 8
    assert AgentOptions(window_messages=256).window_messages == 256


# ── window_messages ──


def _big_call(n: int, size: int) -> dict[str, Any]:
    arguments = json.dumps({"text": f"{n:03d}" + "x" * size})
    call = {
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": "echo", "arguments": arguments},
    }
    return {
        "choices": [{"message": {"content": "", "tool_calls": [call]}}],
        "usage": _cached(0),
    }


def _pairs_whole(body: dict[str, Any]) -> bool:
    """Every tool call is answered by the messages right after it, and only so."""
    messages = body["messages"]
    index = 0
    while index < len(messages):
        message = messages[index]
        if message["role"] == "tool":
            return False
        calls = [c["id"] for c in message.get("tool_calls", [])]
        answers = messages[index + 1 : index + 1 + len(calls)]
        if [a.get("tool_call_id") for a in answers if a["role"] == "tool"] != calls:
            return False
        index += 1 + len(calls)
    return True


def _starts_the_same(earlier: dict[str, Any], later: dict[str, Any]) -> bool:
    kept = earlier["messages"]
    return bool(later["messages"][: len(kept)] == kept)


async def test_a_large_cap_leaves_the_size_fit_to_drop_history_in_chunks() -> None:
    steps = 26
    stub = Stub([*(_big_call(n, 4000) for n in range(steps)), _final_reply(_cached(0))])
    await _run(
        stub,
        max_steps=30,
        steps_per_attempt=30,
        fixed_prefix=True,
        window_eviction="chunk",
        window_messages=64,
    )
    bodies = stub.bodies
    assert len(bodies) == steps + 1
    # The count cap is never what drops: the prompt-size fit reaches 100k chars first.
    assert max(len(b["messages"]) for b in bodies) < 64
    assert all(_pairs_whole(b) for b in bodies)
    resets = [
        i
        for i in range(1, len(bodies))
        if not _starts_the_same(bodies[i - 1], bodies[i])
    ]
    assert len(resets) >= 2
    # Each drop is a chunk: down to 70% of the cap, so several units go at once.
    for i in resets:
        assert len(bodies[i]["messages"]) <= len(bodies[i - 1]["messages"]) - 4
    # Between drops the history holds for several steps.
    gaps = [b - a for a, b in zip([0, *resets], resets, strict=False)]
    assert all(gap >= 3 for gap in gaps[1:]), gaps


async def test_the_slide_mode_drops_a_unit_per_step_once_the_cap_is_reached() -> None:
    steps = 26
    stub = Stub([*(_big_call(n, 4000) for n in range(steps)), _final_reply(_cached(0))])
    await _run(
        stub,
        max_steps=30,
        steps_per_attempt=30,
        fixed_prefix=True,
        window_eviction="slide",
        window_messages=64,
    )
    bodies = stub.bodies
    assert all(_pairs_whole(b) for b in bodies)
    resets = [
        i
        for i in range(1, len(bodies))
        if not _starts_the_same(bodies[i - 1], bodies[i])
    ]
    assert len(resets) > 8  # nearly every step after the cap, so no cache holds


@pytest.mark.parametrize("cap", [8, 64, 256])
def test_the_message_cap_drops_a_chunk_of_whole_units_at_the_cap(cap: int) -> None:
    working = WorkingMemory(cap, evict_chunk=WINDOW_EVICT_CHUNK)
    working.append(Message("user", "query"))
    sizes = []
    for n in range(cap * 2):
        call = ToolCall("echo", {"n": n}, id=f"c{n}")
        working.append_turn(
            Message("assistant", tool_calls=[call]),
            [Message("tool", "r", tool_call_id=call.id)],
        )
        window = working.get_window()
        sizes.append(len(window))
        calls = {c.id for m in window for c in m.tool_calls}
        assert calls == {m.tool_call_id for m in window if m.role == "tool"}
        assert window[0].role != "tool"
    assert max(sizes) <= cap + 1  # the newest unit may be what crosses it
    drops = [i for i in range(1, len(sizes)) if sizes[i] < sizes[i - 1]]
    assert drops
    for i in drops:
        assert sizes[i] <= max(cap - WINDOW_EVICT_CHUNK, 0) + 2  # to within a unit
