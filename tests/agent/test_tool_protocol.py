"""Tool-call protocol tests for the core runtime — no real network or model."""

from __future__ import annotations

import dataclasses
import json
import re
from collections import deque
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import REFLECTION_PROMPT
from localmelo.melo.agent.agent import (
    ASSISTANT_TEXT_SHARE,
    WITHHELD_PREVIEW_CHARS,
    _cut_to,
)
from localmelo.melo.checker.validators import (
    MAX_MEMORY_TEXT_LEN,
    MAX_OUTPUT_LEN,
    TRUNCATION_SUFFIX,
)
from localmelo.melo.contracts.providers import BaseEmbeddingProvider, BaseLLMProvider
from localmelo.melo.executor import ToolOutput
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.short import WorkingMemory, group_turns, tail_turns
from localmelo.melo.schema import (
    EMPTY_REPLY_LABEL,
    EMPTY_REPLY_NUDGE,
    ImagePart,
    Message,
    ToolCall,
    ToolDef,
    ToolResult,
)
from localmelo.support.providers.llm.ollama_chat import (
    OllamaNativeChat,
)
from localmelo.support.providers.llm.ollama_chat import (
    _tool_def_to_openai as _ollama_tool_def,
)
from localmelo.support.providers.llm.openai_compat import (
    OpenAICompatLLM,
)
from localmelo.support.providers.llm.openai_compat import (
    _tool_def_to_openai as _openai_tool_def,
)

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


def test_annotations_do_not_reach_a_request_body() -> None:
    # A ToolDef with annotations serializes exactly as one without: the
    # request body the model server reads does not change.
    plain = _echo_def()
    annotated = ToolDef(
        name=plain.name,
        description=plain.description,
        parameters=plain.parameters,
        annotations={"readOnlyHint": True, "openWorldHint": False, "title": "Echo"},
    )
    body = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "echo text back",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    assert _openai_tool_def(annotated) == body == _openai_tool_def(plain)
    assert _ollama_tool_def(annotated) == body == _ollama_tool_def(plain)


def _agent(
    llm: BaseLLMProvider,
    executed: list[str] | None = None,
    *,
    options: AgentOptions | None = None,
) -> Agent:
    agent = Agent(llm=llm, embedding=None, options=options)
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


_CUT_MARKER = re.compile(r"\n\.\.\. \[truncated: (\d+) more characters\]$")


def _cut_parts(content: str) -> tuple[str, int]:
    """Split a result cut to fit working memory into the start it kept and
    the number of characters it left out."""
    match = _CUT_MARKER.search(content)
    assert match, f"not a cut result: ...{content[-80:]!r}"
    return content[: match.start()], int(match.group(1))


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
        await agent.run("echo two")
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
        await agent.run("mixed batch of shell_exec and echo")
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["\n\n", "   ", " \t\n"])
    async def test_whitespace_output_is_marked(self, blank: str) -> None:
        # e.g. file_read on a blank file: nothing was withheld, so the
        # result must not read as a withheld-result notice.
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=_echo_call(blank, "call_w")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)
        await agent.run("read a blank file with echo")
        second = _planning_calls(llm)[1]["messages"]
        assert [m.content for m in second if m.role == "tool"] == ["(no output)"]


# ── Checked writes keep the turn valid ─────────────────────────────────────


class TestCutTo:
    @pytest.mark.parametrize("length", [250, 999, 1_000, 1_001, 60_000, 100_000])
    @pytest.mark.parametrize("room", [WITHHELD_PREVIEW_CHARS, 1_000, 49_950])
    def test_fits_the_room_and_counts_what_it_left_out(
        self, length: int, room: int
    ) -> None:
        text = "".join(chr(ord("a") + i % 26) for i in range(length))
        cut = _cut_to(text, room)
        if length <= room:
            assert cut == text
            return
        # The marker's count may have fewer digits than the length.
        assert room - len(str(length)) <= len(cut) <= room
        kept, left_out = _cut_parts(cut)
        assert text.startswith(kept)
        assert len(kept) + left_out == length

    @pytest.mark.parametrize("cut_before", [1, 9_984, 999_984])
    def test_counts_what_the_result_check_cut_before(self, cut_before: int) -> None:
        # What the executor-result check left of a longer output: the start
        # of it, then its own suffix. The marker counts the whole output.
        text = "x" * MAX_OUTPUT_LEN + TRUNCATION_SUFFIX
        room = MAX_MEMORY_TEXT_LEN - 100
        cut = _cut_to(text, room, cut_before=cut_before)
        assert len(cut) <= room
        kept, left_out = _cut_parts(cut)
        assert kept == "x" * len(kept)
        assert len(kept) + left_out == len(text) + cut_before


async def _text_beside_a_long_result(thought: str) -> tuple[Message, Message]:
    """Run one call whose result has 120,000 characters, with *thought*
    as the assistant text beside it; return the working-memory turn."""
    llm = FakeLLM(
        [
            Message(
                role="assistant",
                content=thought,
                thinking="plan",
                tool_call=_echo_call("x", "call_r"),
            ),
            Message(role="assistant", content="done"),
        ]
    )
    agent = _agent(llm)

    async def _read(text: str) -> str:
        return "r" * 120_000

    agent.executor.register("echo", _read)
    assert await agent.run("read it with echo") == "done"

    window = agent.hippo.working.get_window()
    assert_protocol_valid(window)
    assistant, result = [m for m in window if m.tool_calls or m.role == "tool"]
    # History keeps the text, and the result as the executor-result check
    # left it, whatever working memory keeps.
    task = next(iter(agent.hippo.history._tasks.values()))
    [step] = await agent.hippo.history.get_steps(task.task_id)
    assert step.thought == thought
    assert step.tool_result is not None
    assert step.tool_result.output == "r" * MAX_OUTPUT_LEN + TRUNCATION_SUFFIX
    # Working memory keeps the start of that output, with a marker that
    # counts every character of the tool's output left out, those the
    # executor-result check had cut included.
    kept, left_out = _cut_parts(result.content)
    assert kept == "r" * len(kept)
    assert len(kept) + left_out == 120_000
    return assistant, result


class TestCheckedTurnWrites:
    @pytest.mark.asyncio
    async def test_too_long_result_is_cut_to_the_room_left(self) -> None:
        # Cut to MAX_OUTPUT_LEN by the executor-result check, the result is
        # still too long for one memory write. Working memory keeps as much
        # of it as the turn has room for, under the same tool_call_id.
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
        await agent.run("echo big")
        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        [result] = [m for m in window if m.role == "tool"]
        assert result.tool_call_id == "call_big"

        # History keeps the checked text; working memory its start, with a
        # marker that counts every character of the tool's output left out,
        # those the executor-result check had cut included.
        task = next(iter(agent.hippo.history._tasks.values()))
        [step] = await agent.hippo.history.get_steps(task.task_id)
        assert step.tool_result is not None
        checked = step.tool_result.output
        assert checked == "z" * MAX_OUTPUT_LEN + TRUNCATION_SUFFIX
        kept, left_out = _cut_parts(result.content)
        assert checked.startswith(kept)
        assert len(kept) + left_out == len(big)
        # The call keeps its share of the turn; the result gets the rest.
        room = MAX_MEMORY_TEXT_LEN - len('echo({"text": "x"})')
        assert room - 10 <= len(result.content) <= room

    @pytest.mark.asyncio
    async def test_step_results_share_one_budget(self) -> None:
        # Each 34k result fits the memory-write check on its own; together
        # they would push the next planning prompt over the pre_plan cap.
        part = "x" * 34_000
        calls = [_echo_call(tag, f"call_{tag}") for tag in "abc"]
        llm = FakeLLM(
            [
                Message(role="assistant", content="read three", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)

        async def _read(text: str) -> str:
            return part

        agent.executor.register("echo", _read)
        assert await agent.run("summarize three parts with echo") == "done"

        # The second planning call got past pre_plan and answers every call.
        planning = _planning_calls(llm)
        assert len(planning) == 2
        second = planning[1]["messages"]
        assert_protocol_valid(second)
        ids = ["call_a", "call_b", "call_c"]
        assert [tc.id for m in second for tc in m.tool_calls] == ids
        results = [(m.tool_call_id, m.content) for m in second if m.role == "tool"]
        assert [call_id for call_id, _ in results] == ids
        contents = [content for _, content in results]
        assert sum(len(c) for c in contents) <= MAX_MEMORY_TEXT_LEN
        assert contents.count(part) == 1
        # One result is left less room than a preview and is withheld; the
        # next is cut to the room the others leave it.
        withheld = [c for c in contents if c.startswith("[Tool result withheld")]
        cut = [c for c in contents if c != part and c not in withheld]
        assert (len(withheld), len(cut)) == (1, 1)
        kept, left_out = _cut_parts(cut[0])
        assert kept == "x" * len(kept)
        assert len(kept) + left_out == len(part)

    @pytest.mark.asyncio
    async def test_largest_results_are_cut_first(self) -> None:
        sizes = {"a": 40_000, "b": 10_000, "c": 30_000}
        calls = [_echo_call(tag, f"call_{tag}") for tag in sizes]
        llm = FakeLLM(
            [
                Message(role="assistant", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)

        async def _read(text: str) -> str:
            return text * sizes[text]

        agent.executor.register("echo", _read)
        await agent.run("read three parts with echo")
        window = agent.hippo.working.get_window()
        results = {m.tool_call_id: m.content for m in window if m.role == "tool"}
        # Cutting the 40k result is enough; the others stay whole. It gets
        # what the 59 chars of rendered calls and the others leave.
        room = MAX_MEMORY_TEXT_LEN - 59 - 10_000 - 30_000
        assert room - 10 <= len(results["call_a"]) <= room
        kept, left_out = _cut_parts(results["call_a"])
        assert kept == "a" * len(kept)
        assert len(kept) + left_out == 40_000
        assert results["call_b"] == "b" * 10_000
        assert results["call_c"] == "c" * 30_000

    @pytest.mark.asyncio
    async def test_oversized_arguments_are_elided_not_dropped(self) -> None:
        # Each call fits the memory-write check on its own; together they
        # do not, so the largest arguments give way to a placeholder.
        texts = {"a": "p" * 15_000, "b": "q" * 20_000, "c": "r" * 18_000}
        calls = [_echo_call(text, f"call_{tag}") for tag, text in texts.items()]
        llm = FakeLLM(
            [
                Message(role="assistant", content="save three", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)
        received: list[str] = []

        async def _save(text: str) -> str:
            received.append(text)
            return f"saved {len(text)} chars"

        agent.executor.register("echo", _save)
        assert await agent.run("save three sections with echo") == "done"

        # The executor and history get the real arguments.
        assert received == list(texts.values())
        task = next(iter(agent.hippo.history._tasks.values()))
        steps = await agent.hippo.history.get_steps(task.task_id)
        assert [s.tool_call.arguments for s in steps if s.tool_call] == [
            {"text": text} for text in texts.values()
        ]

        # The next prompt keeps the whole turn under the same ids.
        second = _planning_calls(llm)[1]["messages"]
        assert_protocol_valid(second)
        turn = [m for m in second if m.tool_calls or m.role == "tool"]
        assert turn[0].content == "save three"
        assert [(tc.id, tc.arguments) for tc in turn[0].tool_calls] == [
            ("call_a", {"text": texts["a"]}),
            ("call_b", {"_elided": "20012 chars withheld"}),
            ("call_c", {"text": texts["c"]}),
        ]
        assert [(m.tool_call_id, m.content) for m in turn[1:]] == [
            ("call_a", "saved 15000 chars"),
            ("call_b", "saved 20000 chars"),
            ("call_c", "saved 18000 chars"),
        ]

    @pytest.mark.asyncio
    async def test_arguments_and_results_share_one_budget(self) -> None:
        # The calls and the results each fit on their own. Together, with
        # a long query, they would pass the pre_plan cap, fail the task,
        # and fail the session's next runs too until eviction.
        draft = "d" * 49_000
        calls = [
            ToolCall(tool_name="save", arguments={"text": draft}, id="call_save"),
            ToolCall(tool_name="read", arguments={"text": "ref.txt"}, id="call_read"),
        ]
        llm = FakeLLM(
            [
                Message(role="assistant", tool_calls=calls),
                Message(role="assistant", content="done"),
                Message(role="assistant", content="next"),
            ]
        )
        agent = Agent(llm=llm, embedding=None)
        for name in ("save", "read"):
            agent.hippo.register_tool(_echo_def(name))

        async def _save(text: str) -> str:
            return f"saved {len(text)} chars"

        async def _read(text: str) -> str:
            return "r" * 49_000

        agent.executor.register("save", _save)
        agent.executor.register("read", _read)
        query = "Update the draft with save and read the reference. " * 60
        assert await agent.run(query) == "done"

        second = _planning_calls(llm)[1]["messages"]
        assert_protocol_valid(second)
        turn = [m for m in second if m.tool_calls or m.role == "tool"]
        # The calls keep their share; the large result gives way.
        assert [(tc.id, tc.arguments) for tc in turn[0].tool_calls] == [
            ("call_save", {"text": draft}),
            ("call_read", {"text": "ref.txt"}),
        ]
        results = {m.tool_call_id: m.content for m in turn[1:]}
        assert results["call_save"] == "saved 49000 chars"
        # The large result is cut to the few hundred characters left.
        calls_chars = len(f'save({json.dumps({"text": draft})})') + len(
            '\nread({"text": "ref.txt"})'
        )
        room = MAX_MEMORY_TEXT_LEN - calls_chars - len(results["call_save"])
        assert WITHHELD_PREVIEW_CHARS <= len(results["call_read"]) <= room
        kept, left_out = _cut_parts(results["call_read"])
        assert kept == "r" * len(kept)
        assert len(kept) + left_out == 49_000
        # The session's next run is not affected.
        assert await agent.run("unrelated question") == "next"

    @pytest.mark.asyncio
    async def test_text_gives_way_before_results(self) -> None:
        # The text and the result each fit beside the call, but not both.
        # History keeps the text, so the text is what working memory drops.
        thought = "t" * 30_000
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    content=thought,
                    thinking="plan",
                    tool_call=_echo_call("x", "call_r"),
                ),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _agent(llm)

        async def _read(text: str) -> str:
            return "r" * 30_000

        agent.executor.register("echo", _read)
        assert await agent.run("read it with echo") == "done"

        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        assistant = next(m for m in window if m.tool_calls)
        assert (assistant.content, assistant.thinking) == ("", "")
        assert assistant.tool_calls[0].id == "call_r"
        assert [m.content for m in window if m.role == "tool"] == ["r" * 30_000]
        task = next(iter(agent.hippo.history._tasks.values()))
        steps = await agent.hippo.history.get_steps(task.task_id)
        assert steps[0].thought == thought

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [500, ASSISTANT_TEXT_SHARE])
    async def test_a_short_text_stays_beside_a_cut_result(self, size: int) -> None:
        # The text keeps its share of the turn, so the result is cut to the
        # room that the call and the text leave it.
        thought = "t" * size
        assistant, result = await _text_beside_a_long_result(thought)
        assert (assistant.content, assistant.thinking) == (thought, "plan")
        calls_text = 'echo({"text": "x"})'
        room = MAX_MEMORY_TEXT_LEN - len(calls_text) - len(f"{thought}\n")
        assert room - 10 <= len(result.content) <= room
        # The turn as the budget counts it: text, line break, calls, result.
        turn = f"{assistant.content}\n{calls_text}{result.content}"
        assert len(turn) <= MAX_MEMORY_TEXT_LEN

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [ASSISTANT_TEXT_SHARE + 1, 3_000])
    async def test_a_longer_text_gives_way_to_a_cut_result(self, size: int) -> None:
        # A longer text gets no share: it could not fit beside a result
        # that has to be cut. It is dropped, and the result gets all the
        # room the call leaves it.
        assistant, result = await _text_beside_a_long_result("t" * size)
        assert (assistant.content, assistant.thinking) == ("", "")
        room = MAX_MEMORY_TEXT_LEN - len('echo({"text": "x"})')
        assert room - 10 <= len(result.content) <= room

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
        await agent.run("echo long thought")
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
        await agent.run("echo summary")
        assert any("[called echo]" in e.text for e in agent.hippo.long._entries)
        window = agent.hippo.working.get_window()
        assert not any("[called echo]" in m.content for m in window)


# ── Tool results with images ───────────────────────────────────────────────

SHOT = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n screenshot bytes", "image/png")
# What the model reads in place of SHOT.
MARKER = f"[1 image: sha256:{SHOT.sha256[:12]} image/png]"


def _returning(output: ToolOutput) -> Callable[[], Coroutine[Any, Any, ToolOutput]]:
    async def _tool() -> ToolOutput:
        return output

    return _tool


def _image_agent(llm: BaseLLMProvider, **overrides: ToolOutput) -> Agent:
    """Agent whose tools return an image with text, alone, or with an error.

    *overrides* replace those outputs, or add tools, by name.
    """
    agent = Agent(llm=llm, embedding=None)
    outputs = {
        "shot": ToolOutput(text="captured", images=[SHOT]),
        "shot_only": ToolOutput(images=[SHOT]),
        "shot_error": ToolOutput(text="window not found", images=[SHOT], is_error=True),
        **overrides,
    }
    for name, output in outputs.items():
        agent.hippo.register_tool(
            ToolDef(
                name=name, description=f"{name} tool", parameters={"type": "object"}
            )
        )
        agent.executor.register(name, _returning(output))
    return agent


def _recording_transport(
    replies: list[dict[str, Any]], bodies: list[dict[str, Any]]
) -> httpx.MockTransport:
    """Answer with *replies* in order and record every request body."""
    queue = deque(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=queue.popleft())

    return httpx.MockTransport(handler)


def _assert_markers_without_images(body: dict[str, Any]) -> None:
    """*body* answers the shot and shot_only calls and carries no image."""
    sent = body["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "tool"]
    assert [m["content"] for m in sent[3:]] == [f"captured\n{MARKER}", MARKER]
    # No Ollama "images" field, no OpenAI content parts, no base64 anywhere.
    assert not any("images" in m or not isinstance(m["content"], str) for m in sent)
    assert SHOT.data not in json.dumps(body)


class TestToolResultImages:
    @pytest.mark.asyncio
    async def test_tool_message_names_the_images(self) -> None:
        calls = [
            ToolCall(tool_name="shot", id="call_text"),
            ToolCall(tool_name="shot_only", id="call_only"),
            ToolCall(tool_name="shot_error", id="call_error"),
        ]
        llm = FakeLLM(
            [
                Message(role="assistant", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        query = "take screenshots with shot, shot_only and shot_error"
        assert await _image_agent(llm).run(query) == "done"

        second = _planning_calls(llm)[1]["messages"]
        assert_protocol_valid(second)
        results = {m.tool_call_id: m.content for m in second if m.role == "tool"}
        # An image-only result reads the marker, not "(no output)".
        assert results == {
            "call_text": f"captured\n{MARKER}",
            "call_only": MARKER,
            "call_error": f"Error: window not found\n{MARKER}",
        }
        # The images themselves are not sent yet.
        assert not [img for m in second for img in m.images if img.data]
        assert not any(SHOT.data in m.content for m in second)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "size",
        [
            # With the marker, too large for the memory-write check.
            MAX_MEMORY_TEXT_LEN - 10,
            # With the marker, MAX_MEMORY_TEXT_LEN - 4 characters: fits the
            # check, but not the turn budget, of which the rendered call
            # "shot({})" takes 8 characters.
            MAX_MEMORY_TEXT_LEN - 1 - len(MARKER) - 4,
        ],
    )
    async def test_marker_is_not_counted_against_the_limits(self, size: int) -> None:
        # The output fits the check and the turn budget on its own, so the
        # marker after it must not get it withheld.
        output = "x" * size
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=ToolCall("shot", id="call_big")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(llm, shot=ToolOutput(text=output, images=[SHOT]))
        assert await agent.run("take a large screenshot") == "done"
        window = agent.hippo.working.get_window()
        [result] = [m for m in window if m.role == "tool"]
        assert result.content == f"{output}\n{MARKER}"
        assert result.images == [SHOT.as_reference()]

    @pytest.mark.asyncio
    async def test_cut_result_still_names_its_images(self) -> None:
        # Cut to MAX_OUTPUT_LEN and marked as truncated by the executor-
        # result check, the output is still too large for working memory,
        # which keeps its start.
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=ToolCall("shot", id="call_big")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(llm, shot=ToolOutput(text="x" * 60_000, images=[SHOT]))
        await agent.run("take a large screenshot")
        window = agent.hippo.working.get_window()
        [result] = [m for m in window if m.role == "tool"]
        text, marker = result.content.rsplit("\n", 1)
        assert marker == MARKER
        kept, left_out = _cut_parts(text)
        assert kept == "x" * len(kept)
        # The count covers the whole output, as the tool returned it.
        assert len(kept) + left_out == 60_000
        # The rendered call "shot({})" takes 8 of the turn's characters.
        assert len(text) <= MAX_MEMORY_TEXT_LEN - 8
        # Unlike a withheld result, a cut one keeps its image references.
        assert result.images == [SHOT.as_reference()]

    @pytest.mark.asyncio
    async def test_a_result_left_no_room_is_withheld(self) -> None:
        # The call's arguments leave its result less room than a preview,
        # so the result is withheld rather than cut to almost nothing. Like
        # any withheld result it keeps no images, but still names them.
        call = ToolCall("annotate", arguments={"text": "a" * 49_900}, id="call_a")
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=call),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(llm)

        async def _annotate(text: str) -> ToolOutput:
            return ToolOutput(text="y" * 1_000, images=[SHOT])

        agent.hippo.register_tool(
            ToolDef(name="annotate", description="annotate", parameters={})
        )
        agent.executor.register("annotate", _annotate)
        assert await agent.run("annotate the screenshot") == "done"
        window = agent.hippo.working.get_window()
        [result] = [m for m in window if m.role == "tool"]
        notice, preview, marker = result.content.split("\n")
        assert notice.startswith("[Tool result withheld")
        assert "Tool calls and results too large" in notice
        assert preview == "y" * WITHHELD_PREVIEW_CHARS
        assert marker == MARKER
        assert result.images == []

    @pytest.mark.asyncio
    async def test_history_and_working_memory_keep_references(self) -> None:
        calls = [
            ToolCall(tool_name="shot", id="call_text"),
            ToolCall(tool_name="shot_only", id="call_only"),
        ]
        llm = FakeLLM(
            [
                Message(role="assistant", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(llm)
        assert await agent.run("take screenshots with shot and shot_only") == "done"
        # The same digest, type and size as SHOT, without the base64 data.
        ref = SHOT.as_reference()

        task = next(iter(agent.hippo.history._tasks.values()))
        steps = await agent.hippo.history.get_steps(task.task_id)
        assert [s.tool_result.images for s in steps if s.tool_result] == [[ref]] * 2

        # The working-memory turn, as the next planning call received it.
        second = _planning_calls(llm)[1]["messages"]
        results = [m for m in second if m.role == "tool"]
        assert [m.images for m in results] == [[ref]] * 2
        assert [m.tool_result.images for m in results if m.tool_result] == [[ref]] * 2

    @pytest.mark.asyncio
    async def test_sqlite_history_keeps_references(self, tmp_path: Path) -> None:
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=ToolCall("shot", id="call_s")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(llm)
        history = SqliteHistory(tmp_path / "history.db")
        agent.hippo.history = history
        assert await agent.run("take a screenshot") == "done"

        conn = await history._ensure_ready()
        async with conn.execute("SELECT task_id FROM tasks") as cur:
            [(task_id,)] = await cur.fetchall()
        steps = await history.get_steps(task_id)
        assert [s.tool_result.images for s in steps if s.tool_result] == [
            [SHOT.as_reference()]
        ]
        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["shot", "shot_only"])
    async def test_memory_write_check_covers_the_images(
        self, monkeypatch: pytest.MonkeyPatch, tool: str
    ) -> None:
        import localmelo.melo.checker.validators as validators

        monkeypatch.setattr(validators, "MAX_IMAGES_PER_MEMORY_WRITE", 0)
        llm = FakeLLM(
            [
                Message(role="assistant", tool_call=ToolCall(tool, id="call_s")),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(llm)
        await agent.run(f"take a screenshot with {tool}")
        window = agent.hippo.working.get_window()
        [result] = [m for m in window if m.role == "tool"]
        # A rejected write keeps neither the text nor the images, but the
        # notice still names the images.
        reason = "Too many images in one memory write: 1 (max 0)"
        notice = f"[Tool result withheld from working memory: {reason}."
        expected = {
            "shot": f"{notice} First 8 characters follow.]\ncaptured\n{MARKER}",
            # An image-only result has no text to preview.
            "shot_only": f"{notice}]\n{MARKER}",
        }
        assert result.content == expected[tool]
        assert result.images == []

    @pytest.mark.asyncio
    async def test_turn_budget_cuts_a_result_with_images(self) -> None:
        # Each result fits the memory-write check on its own; together they
        # do not fit the turn budget, so the larger one gives way.
        calls = [
            ToolCall(tool_name="page", id="call_page"),
            ToolCall(tool_name="shot", id="call_shot"),
        ]
        llm = FakeLLM(
            [
                Message(role="assistant", tool_calls=calls),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _image_agent(
            llm,
            page=ToolOutput(text="p" * 20_000, images=[SHOT]),
            shot=ToolOutput(text="x" * 30_000, images=[SHOT]),
        )
        assert await agent.run("save the page and a screenshot") == "done"

        second = _planning_calls(llm)[1]["messages"]
        assert_protocol_valid(second)
        results = {m.tool_call_id: m for m in second if m.role == "tool"}
        page, shot = results["call_page"], results["call_shot"]
        assert page.content == f"{'p' * 20_000}\n{MARKER}"
        assert page.images == [SHOT.as_reference()]
        # Cut to the room the other result leaves it, it keeps its images,
        # and its message still ends with the line that names them.
        text, marker = shot.content.rsplit("\n", 1)
        assert marker == MARKER
        kept, left_out = _cut_parts(text)
        assert kept == "x" * len(kept)
        assert len(kept) + left_out == 30_000
        assert len(text) <= MAX_MEMORY_TEXT_LEN - len("page({})\nshot({})") - 20_000
        assert shot.images == [SHOT.as_reference()]

    @pytest.mark.asyncio
    async def test_openai_request_names_images_but_carries_none(self) -> None:
        calls = [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for name in ("shot", "shot_only")
        ]
        bodies: list[dict[str, Any]] = []
        transport = _recording_transport(
            [
                {"choices": [{"message": {"role": "assistant", "tool_calls": calls}}]},
                {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
            ],
            bodies,
        )
        llm = OpenAICompatLLM(
            base_url="http://omlx.test/v1", model="m", transport=transport
        )
        agent = _image_agent(llm)
        assert await agent.run("take screenshots with shot and shot_only") == "done"
        await agent.close()
        _assert_markers_without_images(bodies[1])

    @pytest.mark.asyncio
    async def test_ollama_request_names_images_but_carries_none(self) -> None:
        calls: list[dict[str, Any]] = [
            {"function": {"name": name, "arguments": {}}}
            for name in ("shot", "shot_only")
        ]
        bodies: list[dict[str, Any]] = []
        transport = _recording_transport(
            [
                {"message": {"role": "assistant", "content": "", "tool_calls": calls}},
                {"message": {"role": "assistant", "content": "done"}},
            ],
            bodies,
        )
        llm = OllamaNativeChat(
            base_url="http://ollama.test", model="m", transport=transport
        )
        agent = _image_agent(llm)
        assert await agent.run("take screenshots with shot and shot_only") == "done"
        await agent.close()
        _assert_markers_without_images(bodies[1])


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

    def test_drop_oldest_units_keeps_whole_units_and_the_newest(self) -> None:
        wm = WorkingMemory(max_len=20)
        wm.append(Message(role="user", content="q"))
        wm.append_turn(*_turn(2, "a"))
        wm.append_turn(*_turn(1, "b"))
        assert wm.drop_oldest_units(2) == 4  # "q", then the "a" turn
        window = wm.get_window()
        assert_protocol_valid(window)
        assert [m.tool_call_id for m in window if m.role == "tool"] == ["call_b0"]
        # The newest unit always stays.
        assert wm.drop_oldest_units(5) == 0
        assert len(wm.get_window()) == 2
        assert WorkingMemory(max_len=2).drop_oldest_units(1) == 0

    def test_tail_turns_never_splits(self) -> None:
        msgs: list[Message] = [Message(role="user", content="q")]
        a, a_results = _turn(2, "a")
        b, b_results = _turn(1, "b")
        msgs += [a, *a_results, b, *b_results]
        assert tail_turns(msgs, 3) == [b, *b_results]
        assert tail_turns(msgs, 5) == [a, *a_results, b, *b_results]
        assert tail_turns(msgs, 1) == [b, *b_results]  # newest turn is kept
        assert tail_turns(msgs, 100) == msgs

    def test_an_empty_reply_label_owns_its_nudge(self) -> None:
        query = Message(role="user", content="q")
        label = Message(role="assistant", content=EMPTY_REPLY_LABEL)
        nudge = Message(role="user", content=EMPTY_REPLY_NUDGE)
        a, a_results = _turn(1, "a")
        msgs = [query, label, nudge, a, *a_results]
        assert group_turns(msgs) == [[query], [label, nudge], [a, *a_results]]
        # Only that pair: either one next to another message stands alone.
        answer = Message(role="assistant", content="x")
        assert group_turns([label, query]) == [[label], [query]]
        assert group_turns([answer, nudge]) == [[answer], [nudge]]
        assert group_turns([label, nudge, nudge]) == [[label, nudge], [nudge]]
        # Trimming keeps both or neither.
        assert tail_turns(msgs, 3) == [a, *a_results]
        assert tail_turns(msgs, 4) == [label, nudge, a, *a_results]

    def test_eviction_and_dropping_keep_the_label_with_its_nudge(self) -> None:
        label = Message(role="assistant", content=EMPTY_REPLY_LABEL)
        nudge = Message(role="user", content=EMPTY_REPLY_NUDGE)
        a, a_results = _turn(1, "a")
        wm = WorkingMemory(max_len=3)
        wm.append(Message(role="user", content="q"))
        wm.append_turn(label, [nudge])  # 3
        wm.append_turn(a, a_results)  # 5 > 3: drop "q", then label and nudge
        assert wm.get_window() == [a, *a_results]

        wm = WorkingMemory(max_len=20)
        wm.append(Message(role="user", content="q"))
        wm.append_turn(label, [nudge])
        wm.append_turn(a, a_results)
        assert wm.drop_oldest_units(2) == 3  # "q", then label and nudge
        assert wm.get_window() == [a, *a_results]

        # The newest unit stays whole, even past max_len.
        wm = WorkingMemory(max_len=1)
        wm.append_turn(label, [nudge])
        assert wm.get_window() == [label, nudge]


class TestAgentWindowTruncation:
    @pytest.mark.asyncio
    async def test_every_planning_prompt_is_protocol_valid(self) -> None:
        options = AgentOptions(steps_per_attempt=8, max_attempts=1)
        responses = [
            Message(
                role="assistant",
                tool_calls=[_echo_call(f"s{i}c{j}") for j in range(i % 3 + 1)],
            )
            for i in range(6)
        ]
        responses.append(Message(role="assistant", content="done"))
        llm = FakeLLM(responses)
        agent = _agent(llm, options=options)
        agent.hippo.working = WorkingMemory(max_len=4)
        assert await agent.run("echo long task") == "done"

        planning = _planning_calls(llm)
        assert len(planning) == 7
        for call in planning:
            sent = call["messages"]
            assert_protocol_valid(sent)
            # The task query opens the conversation even after eviction.
            assert sent[1].role == "user"
            assert sent[1].content == "echo long task"

    @pytest.mark.asyncio
    async def test_reflection_window_is_protocol_valid(self) -> None:
        options = AgentOptions(steps_per_attempt=3, max_attempts=1)
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
        agent = _agent(llm, options=options)
        await agent.run("echo needs reflection")
        reflection = _reflection_calls(llm)
        assert len(reflection) == 1
        sent = reflection[0]["messages"]
        assert_protocol_valid(sent)
        assert sent[-1].role == "user"
        assert any(m.tool_calls for m in sent)


# ── Working memory that outgrows the prompt cap ────────────────────────────

# Checker.pre_plan fails a step whose prompt is larger than this.
PROMPT_CAP = 100_000


def _prompt_chars(messages: list[Message]) -> int:
    """What Checker.pre_plan counts of *messages*."""
    return sum(
        len(m.content)
        + sum(len(tc.tool_name) + len(json.dumps(tc.arguments)) for tc in m.tool_calls)
        for m in messages
    )


def _reader(llm: BaseLLMProvider, size: int = 40_000) -> Agent:
    """Agent whose read tool returns *size* characters a call."""
    agent = Agent(llm=llm, embedding=None)
    agent.hippo.register_tool(_echo_def("read"))

    async def _read(text: str) -> str:
        return text[-1] * size

    agent.executor.register("read", _read)
    return agent


def _read_call(part: str) -> Message:
    call = ToolCall(tool_name="read", arguments={"text": part}, id=f"call_{part}")
    return Message(role="assistant", tool_call=call)


class TestPromptCapRecovery:
    @pytest.mark.asyncio
    async def test_later_runs_drop_old_turns_instead_of_failing(self) -> None:
        # One large turn a run. From the third run on, the turns no longer
        # fit the cap together; the oldest whole units are dropped, so no
        # run fails with "Prompt too large".
        replies = []
        for i in range(6):
            replies += [_read_call(str(i)), Message(role="assistant", content=f"a{i}")]
        llm = FakeLLM(replies)
        agent = _reader(llm)
        for i in range(6):
            result = await agent.run_task(f"read part {i}")
            assert (result.status, result.result) == ("completed", f"a{i}")

        for call in _planning_calls(llm):
            sent = call["messages"]
            assert_protocol_valid(sent)
            assert _prompt_chars(sent[1:]) <= PROMPT_CAP
        window = agent.hippo.working.get_window()
        assert_protocol_valid(window)
        assert [m.tool_call_id for m in window if m.role == "tool"] == [
            "call_4",
            "call_5",
        ]

    @pytest.mark.asyncio
    async def test_large_turns_within_one_run(self) -> None:
        llm = FakeLLM(
            [
                _read_call("1"),
                _read_call("2"),
                _read_call("3"),
                _read_call("4"),
                Message(role="assistant", content="done"),
            ]
        )
        agent = _reader(llm)
        result = await agent.run_task("read parts 1 to 4")
        assert (result.status, result.result, result.steps) == ("completed", "done", 5)

        planning = _planning_calls(llm)
        for call in planning:
            sent = call["messages"]
            assert_protocol_valid(sent)
            assert _prompt_chars(sent[1:]) <= PROMPT_CAP
            # The task still opens the conversation after its message left.
            assert (sent[1].role, sent[1].content) == ("user", "read parts 1 to 4")
        last = planning[-1]["messages"]
        assert [m.tool_call_id for m in last if m.role == "tool"] == [
            "call_3",
            "call_4",
        ]

    @pytest.mark.asyncio
    async def test_a_leftover_oversized_message_is_dropped(self) -> None:
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = _agent(llm)
        agent.hippo.working.append(Message(role="system", content="x" * 110_000))
        result = await agent.run_task("hello")
        assert (result.status, result.result) == ("completed", "ok")
        assert [m.content for m in agent.hippo.working.get_window()] == ["hello"]

    @pytest.mark.asyncio
    async def test_nothing_is_dropped_when_the_newest_unit_cannot_fit(self) -> None:
        llm = FakeLLM([Message(role="assistant", content="never sent")])
        agent = _agent(llm)
        agent.hippo.working.append(Message(role="user", content="earlier"))
        query = "x" * (PROMPT_CAP + 1)
        result = await agent.run_task(query)
        assert result.status == "failed"
        assert result.error.startswith("Plan check failed: Prompt too large")
        assert llm.calls == []
        window = agent.hippo.working.get_window()
        assert [m.content for m in window] == ["earlier", query]

    @pytest.mark.asyncio
    async def test_an_empty_reply_label_goes_only_with_its_nudge(self) -> None:
        # The prompt, with the query that plan_step puts back in front,
        # fits with the nudge and the newest turn, but not with the label
        # as well. Dropping the label alone would leave the nudge to follow
        # that query, so the label and the nudge go together.
        agent = _agent(FakeLLM())
        query = Message(role="user", content="q")
        label = Message(role="assistant", content=EMPTY_REPLY_LABEL)
        nudge = Message(role="user", content=EMPTY_REPLY_NUDGE)
        a, a_results = _turn(1, "a")
        agent.hippo.working.append(query)
        agent.hippo.working.append_turn(label, [nudge])
        agent.hippo.working.append_turn(a, a_results)
        room = PROMPT_CAP - _prompt_chars([query, nudge, a, *a_results])
        await agent._fit_prompt_cap([Message(role="system", content="x" * room)], "q")
        assert agent.hippo.working.get_window() == [a, *a_results]

    @pytest.mark.asyncio
    async def test_the_query_counts_once_its_message_is_dropped(self) -> None:
        # A 60,000-character task and two 20,000-character results: the
        # second result does not fit beside the task and the first. The
        # request puts the task back in front once its message is dropped,
        # so dropping that message alone saves nothing: the first result
        # goes too.
        query = "read parts 1 and 2: " + "q" * 60_000
        llm = FakeLLM(
            [
                _read_call("1"),
                _read_call("2"),
                Message(role="assistant", content="done"),
            ]
        )
        result = await _reader(llm, size=20_000).run_task(query)
        assert (result.status, result.result) == ("completed", "done")

        planning = _planning_calls(llm)
        assert len(planning) == 3
        for call in planning:
            sent = call["messages"]
            assert_protocol_valid(sent)
            assert _prompt_chars(sent[1:]) <= PROMPT_CAP
            assert (sent[1].role, sent[1].content) == ("user", query)
        last = planning[-1]["messages"]
        assert [m.tool_call_id for m in last if m.role == "tool"] == ["call_2"]

    @pytest.mark.asyncio
    async def test_a_task_that_cannot_fit_beside_the_newest_turn_fails(self) -> None:
        # A 70,000-character task and one 45,000-character result: every
        # request would carry both, whatever is dropped, so the task fails
        # with "Prompt too large" instead of sending a request over the cap.
        query = "read this: " + "q" * 70_000
        llm = FakeLLM([_read_call("1"), Message(role="assistant", content="never")])
        result = await _reader(llm, size=45_000).run_task(query)
        assert result.status == "failed"
        assert result.error == "Plan check failed: Prompt too large: 115028 chars"

        [sent] = [call["messages"] for call in _planning_calls(llm)]
        assert _prompt_chars(sent[1:]) <= PROMPT_CAP

    @pytest.mark.asyncio
    async def test_the_query_counts_once_eviction_dropped_its_message(self) -> None:
        # A 60,000-character task, then results of 10, 10 and 45,000
        # characters. Eviction by message count drops the task's message
        # after the second turn, so the window alone fits the cap, but not
        # with the task the request puts back in front: the check before
        # the step must count the task too, and fail the run instead of
        # sending a request over the cap.
        query = "q" * 60_000
        llm = FakeLLM(
            [
                _read_call("1"),
                _read_call("2"),
                _read_call("3"),
                Message(role="assistant", content="never"),
            ]
        )
        agent = Agent(
            llm=llm,
            options=AgentOptions(
                builtins=frozenset(), tool_exposure="all", memory=False
            ),
        )
        agent.hippo.working = WorkingMemory(max_len=4)
        agent.hippo.register_tool(_echo_def("read"))

        async def _read(text: str) -> str:
            return "r" * (45_000 if text == "3" else 10)

        agent.executor.register("read", _read)
        result = await agent.run_task(query)
        assert (result.status, result.error, result.steps) == (
            "failed",
            "Plan check failed: Prompt too large: 105044 chars",
            3,
        )

        planning = _planning_calls(llm)
        assert len(planning) == 3
        for call in planning:
            assert _prompt_chars(call["messages"][1:]) <= PROMPT_CAP
        # Eviction took the task's message out of the window. Nothing was
        # dropped to fit the cap: not even the newest turn fits beside the
        # task.
        window = agent.hippo.working.get_window()
        assert query not in [m.content for m in window]
        assert [m.tool_call_id for m in window if m.role == "tool"] == [
            "call_2",
            "call_3",
        ]


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
    async def test_loop_stops_on_repeated_batches(self) -> None:
        options = AgentOptions(steps_per_attempt=6, max_attempts=1)
        llm = FakeLLM(
            [
                Message(
                    role="assistant",
                    tool_calls=[_echo_call("a"), _echo_call("b")][:: (-1) ** i],
                )
                for i in range(3)
            ]
        )
        llm.responses.append(
            Message(role="assistant", content='{"recommended_action": "stop"}')
        )
        agent = _agent(llm, options=options)
        await agent.run("echo repeat")
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

    @pytest.mark.asyncio
    async def test_unparsed_arguments_pass_a_strict_server(self) -> None:
        # Like SGLang, the server rejects a request whose history holds
        # tool-call arguments that do not parse to a JSON object.
        unparsed = ["print(6 * 7)", "[1, 2]", '"print(6 * 7)"']
        replies: deque[dict[str, Any]] = deque(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": "echo", "arguments": arguments},
                        }
                        for i, arguments in enumerate(unparsed)
                    ],
                },
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            for msg in json.loads(request.content)["messages"]:
                for tc in msg.get("tool_calls", []):
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = None
                    if not isinstance(arguments, dict):
                        return httpx.Response(400, json={"error": "not an object"})
            return httpx.Response(
                200, json={"choices": [{"message": replies.popleft()}]}
            )

        llm = OpenAICompatLLM(
            base_url="http://sglang.test/v1",
            model="Qwen3.8-27B-8bit",
            transport=httpx.MockTransport(handler),
        )
        agent = _agent(llm)
        assert await agent.run("run the code") == "first"
        # The turn stays in working memory, so the next run replays it too.
        assert await agent.run("and now?") == "second"
        await agent.close()
