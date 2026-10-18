"""Tool-call protocol tests for the core runtime — no real network or model."""

from __future__ import annotations

import dataclasses

import pytest

from localmelo.melo.schema import Message, ToolCall, ToolResult

# ── Schema: tool call ids ──────────────────────────────────────────────────


class TestToolCallId:
    def test_generated_when_missing(self) -> None:
        tc = ToolCall(tool_name="echo")
        assert tc.id.startswith("call_")
        assert len(tc.id) == len("call_") + 24

    def test_generated_when_empty(self) -> None:
        assert ToolCall(tool_name="echo", id="").id.startswith("call_")

    def test_kept_when_given(self) -> None:
        assert ToolCall(tool_name="echo", id="call_xyz").id == "call_xyz"

    def test_generated_ids_are_unique(self) -> None:
        assert len({ToolCall(tool_name="echo").id for _ in range(50)}) == 50

    def test_equality_ignores_id(self) -> None:
        a = ToolCall(tool_name="echo", arguments={"t": 1}, id="call_a")
        b = ToolCall(tool_name="echo", arguments={"t": 1}, id="call_b")
        assert a == b


# ── Schema: multiple tool calls per message ────────────────────────────────


class TestMessageToolCalls:
    def test_single_tool_call_kwarg_still_works(self) -> None:
        tc = ToolCall(tool_name="echo", arguments={"text": "hi"})
        msg = Message(role="assistant", content="x", tool_call=tc)
        assert msg.tool_calls == [tc]
        assert msg.tool_call is tc

    def test_tool_call_returns_first_of_many(self) -> None:
        first = ToolCall(tool_name="a")
        second = ToolCall(tool_name="b")
        msg = Message(role="assistant", tool_calls=[first, second])
        assert msg.tool_call is first
        assert [tc.tool_name for tc in msg.tool_calls] == ["a", "b"]

    def test_tool_call_is_none_without_calls(self) -> None:
        msg = Message(role="assistant", content="answer")
        assert msg.tool_calls == []
        assert msg.tool_call is None

    def test_tool_call_tracks_later_mutation(self) -> None:
        """The compat accessor is derived, so it cannot go stale."""
        msg = Message(role="assistant")
        msg.tool_calls.append(ToolCall(tool_name="late"))
        assert msg.tool_call is not None
        assert msg.tool_call.tool_name == "late"

    def test_tool_call_setter(self) -> None:
        msg = Message(role="assistant", tool_calls=[ToolCall("a"), ToolCall("b")])
        msg.tool_call = ToolCall(tool_name="c")
        assert [tc.tool_name for tc in msg.tool_calls] == ["c"]
        msg.tool_call = None
        assert msg.tool_calls == []

    def test_positional_signature_unchanged(self) -> None:
        tc = ToolCall(tool_name="echo")
        result = ToolResult(tool_name="echo", output="ok")
        msg = Message("assistant", "text", tc, result, 1.5, {"total_tokens": 3}, "t")
        assert msg.tool_call is tc
        assert msg.tool_result is result
        assert msg.timestamp == 1.5
        assert msg.usage == {"total_tokens": 3}
        assert msg.thinking == "t"

    def test_both_forms_rejected(self) -> None:
        with pytest.raises(TypeError):
            Message(
                role="assistant",
                tool_call=ToolCall(tool_name="a"),
                tool_calls=[ToolCall(tool_name="b")],
            )

    def test_tool_calls_list_is_copied(self) -> None:
        calls = [ToolCall(tool_name="a")]
        msg = Message(role="assistant", tool_calls=calls)
        calls.append(ToolCall(tool_name="b"))
        assert len(msg.tool_calls) == 1

    def test_dataclass_replace_keeps_protocol_fields(self) -> None:
        msg = Message(role="tool", content="out", tool_call_id="call_1", timestamp=2.0)
        copy = dataclasses.replace(msg, content="new")
        assert copy.content == "new"
        assert copy.tool_call_id == "call_1"
        assert copy.timestamp == 2.0

    def test_tool_result_message_carries_id(self) -> None:
        msg = Message(role="tool", content="hello", tool_call_id="call_9")
        assert msg.tool_call_id == "call_9"
        assert msg.tool_calls == []
