"""Tool-call protocol tests for the core runtime — no real network or model."""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.agent._chat import REFLECTION_PROMPT
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.memory.short import WorkingMemory, group_turns, tail_turns
from localmelo.melo.schema import Message, ToolCall, ToolDef, ToolResult
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

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


# ── Agent-level helpers ────────────────────────────────────────────────────


class FakeLLM(BaseLLMProvider):
    """Pops queued responses and records every request."""

    def __init__(self, responses: list[Message] | None = None) -> None:
        self.responses: deque[Message] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return Message(role="assistant", content="(no response queued)")
        return self.responses.popleft()

    async def close(self) -> None:
        pass


def _echo_def(name: str = "echo") -> ToolDef:
    return ToolDef(
        name=name,
        description="echo text back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def _agent(llm: BaseLLMProvider, executed: list[str] | None = None) -> Agent:
    agent = Agent(llm=llm, embedding=None)
    agent.hippo.register_tool(_echo_def())

    async def _echo(text: str) -> str:
        if executed is not None:
            executed.append(text)
        return text

    agent.executor.register("echo", _echo)
    return agent


def _echo_call(text: str, call_id: str = "") -> ToolCall:
    return ToolCall(tool_name="echo", arguments={"text": text}, id=call_id)


def assert_protocol_valid(messages: list[Message]) -> None:
    """Every tool call is answered once, right after it; no orphan results."""
    pending: set[str] = set()
    for msg in messages:
        if msg.role == "tool":
            assert msg.tool_call_id in pending, f"orphan result {msg.tool_call_id!r}"
            pending.remove(msg.tool_call_id)
            continue
        assert not pending, f"unanswered tool calls {pending}"
        if msg.role == "assistant" and msg.tool_calls:
            pending = {tc.id for tc in msg.tool_calls}
    assert not pending, f"unanswered tool calls {pending}"


def _is_reflection(call: dict[str, Any]) -> bool:
    return bool(call["messages"]) and call["messages"][0].content.startswith(
        REFLECTION_PROMPT
    )


def _planning_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    return [c for c in llm.calls if not _is_reflection(c)]


def _reflection_calls(llm: FakeLLM) -> list[dict[str, Any]]:
    return [c for c in llm.calls if _is_reflection(c)]


# ── Multiple tool calls per step ───────────────────────────────────────────


class TestMultiCallExecution:
    @pytest.mark.asyncio
    async def test_all_calls_run_in_order_and_are_answered(self) -> None:
        calls = [_echo_call("a", "call_a"), _echo_call("b", "call_b")]
        calls.append(_echo_call("c", "call_c"))
        llm = FakeLLM(
            [
                Message(role="assistant", content="three echoes", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        executed: list[str] = []
        agent = _agent(llm, executed)
        assert await agent.run("echo a, b and c") == "done"
        assert executed == ["a", "b", "c"]

        second = _planning_calls(llm)[1]["messages"]
        assert_protocol_valid(second)
        turn = [m for m in second if m.tool_calls or m.role == "tool"]
        assert [tc.id for tc in turn[0].tool_calls] == ["call_a", "call_b", "call_c"]
        assert turn[0].content == "three echoes"
        assert [(m.tool_call_id, m.content) for m in turn[1:]] == [
            ("call_a", "a"),
            ("call_b", "b"),
            ("call_c", "c"),
        ]

    @pytest.mark.asyncio
    async def test_history_gets_one_step_per_call(self) -> None:
        calls = [_echo_call("x", "call_x"), _echo_call("y", "call_y")]
        llm = FakeLLM(
            [
                Message(role="assistant", content="thought", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)
        await agent.run("two")
        task = next(iter(agent.hippo.history._tasks.values()))
        steps = await agent.hippo.history.get_steps(task.task_id)
        assert [s.tool_call.id for s in steps if s.tool_call] == ["call_x", "call_y"]
        assert [s.thought for s in steps] == ["thought", ""]
        assert [s.tool_result.output for s in steps if s.tool_result] == ["x", "y"]

    @pytest.mark.asyncio
    async def test_blocked_and_unknown_calls_do_not_stop_the_batch(self) -> None:
        calls = [
            ToolCall(tool_name="shell_exec", arguments={"command": "rm -rf /"}),
            ToolCall(tool_name="no_such_tool", arguments={}),
            _echo_call("still runs"),
        ]
        llm = FakeLLM(
            [
                Message(role="assistant", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        executed: list[str] = []
        agent = _agent(llm, executed)
        await agent.run("mixed batch")
        assert executed == ["still runs"]
        second = _planning_calls(llm)[1]["messages"]
        assert_protocol_valid(second)
        results = {m.tool_call_id: m.content for m in second if m.role == "tool"}
        assert results[calls[0].id].startswith("Error: Blocked")
        assert results[calls[1].id].startswith("Error: Tool not found")
        assert results[calls[2].id] == "still runs"

    @pytest.mark.asyncio
    async def test_empty_output_is_marked(self) -> None:
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=_echo_call("", "call_e")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)
        await agent.run("echo nothing")
        second = _planning_calls(llm)[1]["messages"]
        assert [m.content for m in second if m.role == "tool"] == ["(no output)"]


# ── Checked writes keep the turn valid ─────────────────────────────────────


class TestCheckedTurnWrites:
    @pytest.mark.asyncio
    async def test_rejected_result_becomes_bound_notice(self) -> None:
        big = "z" * 60_000
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=_echo_call("x", "call_big")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)

        async def _big(text: str) -> str:
            return big

        agent.executor.register("echo", _big)
        await agent.run("big")
        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        notice = [m for m in window if m.role == "tool"]
        assert len(notice) == 1
        assert notice[0].tool_call_id == "call_big"
        assert notice[0].content.startswith("[Tool result withheld")
        assert len(notice[0].content) < 1_000

    @pytest.mark.asyncio
    async def test_rejected_text_keeps_the_calls(self) -> None:
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    content="t" * 60_000,
                    tool_call=_echo_call("hi", "call_t"),
                ),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)
        await agent.run("long thought")
        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        assistant = [m for m in window if m.tool_calls]
        assert len(assistant) == 1
        assert assistant[0].content == ""
        assert assistant[0].tool_calls[0].id == "call_t"

    @pytest.mark.asyncio
    async def test_summary_goes_to_long_term_not_working(self) -> None:
        class Emb(BaseEmbeddingProvider):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

            async def close(self) -> None:
                pass

        llm = FakeLLM(
            [
                Message(role="assistant", content="go", tool_call=_echo_call("v")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = Agent(llm=llm, embedding=Emb())
        agent.hippo.register_tool(_echo_def())

        async def _echo(text: str) -> str:
            return text

        agent.executor.register("echo", _echo)
        await agent.run("summary")
        assert any("[called echo]" in e.text for e in agent.hippo.long._entries)
        window = agent.hippo.working.get_window()
        assert not any("[called echo]" in m.content for m in window)


# ── Window truncation keeps pairs ──────────────────────────────────────────


def _turn(n_calls: int, tag: str) -> tuple[Message, list[Message]]:
    calls = [_echo_call(f"{tag}{i}", f"call_{tag}{i}") for i in range(n_calls)]
    results = [
        Message(role="tool", content=tc.arguments["text"], tool_call_id=tc.id)
        for tc in calls
    ]
    return Message(role="assistant", tool_calls=calls), results


class TestWorkingMemoryTurns:
    def test_plain_messages_still_evict_one_by_one(self) -> None:
        wm = WorkingMemory(max_len=2)
        for text in "abc":
            wm.append(Message(role="user", content=text))
        assert [m.content for m in wm.get_window()] == ["b", "c"]

    def test_eviction_drops_whole_turns(self) -> None:
        wm = WorkingMemory(max_len=5)
        wm.append(Message(role="user", content="q"))
        wm.append_turn(*_turn(2, "a"))  # 1 + 3 = 4
        wm.append_turn(*_turn(1, "b"))  # 6 > 5: drop "q"
        assert [m.role for m in wm.get_window()][0] == "assistant"
        wm.append_turn(*_turn(2, "c"))  # 8 > 5: drop the whole "a" turn
        window = wm.get_window()
        assert len(window) == 5
        assert_protocol_valid(window)
        assert [m.tool_call_id for m in window if m.role == "tool"] == [
            "call_b0",
            "call_c0",
            "call_c1",
        ]

    def test_newest_turn_kept_even_if_too_large(self) -> None:
        wm = WorkingMemory(max_len=2)
        wm.append(Message(role="user", content="q"))
        wm.append_turn(*_turn(3, "a"))
        window = wm.get_window()
        assert len(window) == 4
        assert_protocol_valid(window)

    def test_window_is_always_valid_under_churn(self) -> None:
        wm = WorkingMemory(max_len=7)
        for i in range(40):
            if i % 5 == 0:
                wm.append(Message(role="user", content=f"u{i}"))
            else:
                wm.append_turn(*_turn(i % 4, f"t{i}_"))
            window = wm.get_window()
            assert len(window) <= 7 or len(group_turns(window)) == 1
            assert window[0].role != "tool"
            assert_protocol_valid(window)

    def test_tail_turns_never_splits(self) -> None:
        msgs: list[Message] = [Message(role="user", content="q")]
        a, a_results = _turn(2, "a")
        b, b_results = _turn(1, "b")
        msgs += [a, *a_results, b, *b_results]
        assert tail_turns(msgs, 3) == [b, *b_results]
        assert tail_turns(msgs, 5) == [a, *a_results, b, *b_results]
        assert tail_turns(msgs, 1) == [b, *b_results]  # newest turn is kept
        assert tail_turns(msgs, 100) == msgs


class TestAgentWindowTruncation:
    @pytest.mark.asyncio
    async def test_every_planning_prompt_is_protocol_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as agent_mod

        monkeypatch.setattr(agent_mod, "STEPS_PER_ATTEMPT", 8)
        monkeypatch.setattr(agent_mod, "MAX_ATTEMPTS", 1)
        responses = [
            Message(
                role="assistant",
                tool_calls=[_echo_call(f"s{i}c{j}") for j in range(i % 3 + 1)],
            )
            for i in range(6)
        ]
        responses.append(Message(role="assistant", content="done"))
        llm = FakeLLM(responses)
        agent = _agent(llm)
        agent.hippo.working = WorkingMemory(max_len=4)
        assert await agent.run("long task") == "done"

        planning = _planning_calls(llm)
        assert len(planning) == 7
        for call in planning:
            sent = call["messages"]
            assert_protocol_valid(sent)
            # The task query opens the conversation even after eviction.
            assert sent[1].role == "user"
            assert sent[1].content == "long task"

    @pytest.mark.asyncio
    async def test_reflection_window_is_protocol_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as agent_mod

        monkeypatch.setattr(agent_mod, "STEPS_PER_ATTEMPT", 3)
        monkeypatch.setattr(agent_mod, "MAX_ATTEMPTS", 1)
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    tool_calls=[_echo_call(f"r{i}c{j}") for j in range(4)],
                )
                for i in range(3)
            ]
        )
        llm.responses.append(
            Message(role="assistant", content='{"recommended_action": "stop"}')
        )
        agent = _agent(llm)
        await agent.run("needs reflection")
        reflection = _reflection_calls(llm)
        assert len(reflection) == 1
        sent = reflection[0]["messages"]
        assert_protocol_valid(sent)
        assert sent[-1].role == "user"
        assert any(m.tool_calls for m in sent)


# ── Stuck detection with several calls per step ────────────────────────────


class TestStuckDetectionMultiCall:
    def test_same_batch_in_any_order_matches(self) -> None:
        ok = ToolResult(tool_name="echo", output="")
        a, b = _echo_call("a"), _echo_call("b")
        assert Agent._make_fingerprint([a, b], [ok, ok]) == Agent._make_fingerprint(
            [b, a], [ok, ok]
        )

    def test_three_repeated_batches_are_stuck(self) -> None:
        ok = ToolResult(tool_name="echo", output="")
        fp = Agent._make_fingerprint([_echo_call("a"), _echo_call("b")], [ok, ok])
        assert Agent._detect_stuck([fp, fp, fp]) == "stuck"

    def test_one_step_with_repeated_calls_is_not_stuck(self) -> None:
        ok = ToolResult(tool_name="echo", output="")
        fp = Agent._make_fingerprint([_echo_call("a")] * 3, [ok] * 3)
        assert Agent._detect_stuck([fp]) == ""

    def test_same_errors_across_steps_are_stuck(self) -> None:
        err = ToolResult(tool_name="echo", error="boom")
        fps = [
            Agent._make_fingerprint([_echo_call(str(i)), _echo_call("x")], [err, err])
            for i in range(3)
        ]
        assert Agent._detect_stuck(fps) == "stuck"

    @pytest.mark.asyncio
    async def test_loop_stops_on_repeated_batches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localmelo.melo.agent.agent as agent_mod

        monkeypatch.setattr(agent_mod, "STEPS_PER_ATTEMPT", 6)
        monkeypatch.setattr(agent_mod, "MAX_ATTEMPTS", 1)
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    tool_calls=[_echo_call("a"), _echo_call("b")][:: (-1) ** i],
                )
                for i in range(6)
            ]
        )
        llm.responses.append(
            Message(role="assistant", content='{"recommended_action": "stop"}')
        )
        agent = _agent(llm)
        await agent.run("repeat")
        # 3 planning steps, then stuck -> one reflection call.
        assert len(_planning_calls(llm)) == 3
        assert len(_reflection_calls(llm)) == 1


# ── End to end over the real OpenAI-compatible provider ────────────────────


class TestEndToEndOpenAIWire:
    @pytest.mark.asyncio
    async def test_agent_requests_follow_the_openai_format(self) -> None:
        replies: deque[dict[str, Any]] = deque(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "two echoes",
                    "tool_calls": [
                        {
                            "id": "chatcmpl-tool-a",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"text": "one"}',
                            },
                        },
                        {
                            "id": "chatcmpl-tool-b",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"text": "two"}',
                            },
                        },
                    ],
                },
                {"role": "assistant", "content": "one two"},
            ]
        )
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": replies.popleft()}]}
            )

        llm = OpenAICompatLLM(
            base_url="http://omlx.test/v1",
            model="Qwen3.8-27B-8bit",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            transport=httpx.MockTransport(handler),
        )
        agent = _agent(llm)
        assert await agent.run("echo one and two") == "one two"
        await agent.close()

        assert len(bodies) == 2
        assert all(
            b["chat_template_kwargs"] == {"enable_thinking": False} for b in bodies
        )
        second = bodies[1]["messages"]
        assert second[0]["role"] == "system"
        assert second[1] == {"role": "user", "content": "echo one and two"}
        assert second[2] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "chatcmpl-tool-a",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "one"}'},
                },
                {
                    "id": "chatcmpl-tool-b",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "two"}'},
                },
            ],
        }
        assert second[3:] == [
            {"role": "tool", "tool_call_id": "chatcmpl-tool-a", "content": "one"},
            {"role": "tool", "tool_call_id": "chatcmpl-tool-b", "content": "two"},
        ]
        assert "reasoning_content" not in json.dumps(bodies[1])
