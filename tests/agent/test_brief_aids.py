"""Brief runs only (``AgentOptions.brief_probe``): a push after steps of
reading, a note of the tool output the window dropped, and the pointers a
reflection hands to the next attempt."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest

from localmelo.melo.agent._brief_aids import (
    ARG_LIMIT,
    DROPPED_KEPT,
    REFLECTION_ADDENDUM,
    START_WITH_LIMIT,
    STEP_PUSH,
    dropped_notes,
    render_dropped,
    start_with,
)
from localmelo.melo.agent.agent import NO_CHANGE_PUSH
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.short import WorkingMemory
from localmelo.melo.schema import Message, ReflectionEntry, ToolCall
from localmelo.tests.agent.test_brief_protocol import _digest
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _agent,
    _answer,
    _echo_call,
    _is_reflection,
    _reflection,
)


async def _run(
    replies: list[Any], **options: Any
) -> tuple[Any, FakeLLM, list[AgentEvent]]:
    events: list[AgentEvent] = []
    llm = FakeLLM(replies)
    agent = _agent(llm, **options)
    result = await agent.run_task("echo the answer", on_event=events.append)
    return result, llm, events


def _step_notices(events: list[AgentEvent]) -> list[Mapping[str, Any]]:
    return [
        e.data
        for e in events
        if e.type == "notice"
        and e.data["kind"] == "no_change"
        and e.data["detail"].get("scope") == "step"
    ]


def _count(llm: FakeLLM, text: str) -> int:
    return sum(text in m.content for c in llm.calls for m in c["messages"])


def _reads(count: int) -> list[Message]:
    return [_echo_call(f"read {i}") for i in range(count)]


class TestStepPush:
    @pytest.mark.asyncio
    async def test_six_reads_without_a_change_get_one_push(self) -> None:
        result, llm, events = await _run(
            [*_reads(6), _answer("done")],
            max_steps=20,
            steps_per_attempt=20,
            brief_probe=_digest("same"),
        )
        assert result.status == "completed"
        [notice] = _step_notices(events)
        assert notice["detail"] == {"scope": "step"}
        # Planning request 7 is the first to carry it, at the end of its window.
        assert [_count(llm, STEP_PUSH)] == [1]
        assert llm.calls[6]["messages"][-1].content == STEP_PUSH
        assert not any(STEP_PUSH in m.content for m in llm.calls[5]["messages"])

    @pytest.mark.asyncio
    async def test_it_comes_at_most_once_every_six_steps(self) -> None:
        _result, _llm, events = await _run(
            [*_reads(13), _answer("done")],
            max_steps=30,
            steps_per_attempt=30,
            brief_probe=_digest("same"),
        )
        assert len(_step_notices(events)) == 2  # after steps 6 and 12

    @pytest.mark.asyncio
    async def test_a_change_restarts_the_count(self) -> None:
        # Start "a"; the probe at step 6 sees "b", so the baseline moves and
        # the next probe, at step 12, sees "b" again.
        _result, llm, events = await _run(
            [*_reads(12), _answer("done")],
            max_steps=30,
            steps_per_attempt=30,
            brief_probe=_digest("a", "b"),
        )
        assert len(_step_notices(events)) == 1
        assert _count(llm, STEP_PUSH) == 1
        assert STEP_PUSH in llm.calls[12]["messages"][-1].content

    @pytest.mark.asyncio
    async def test_no_push_before_six_steps_or_outside_a_git_tree(self) -> None:
        _result, llm, events = await _run(
            [*_reads(5), _answer("done")],
            max_steps=20,
            steps_per_attempt=20,
            brief_probe=_digest("same"),
        )
        assert _count(llm, STEP_PUSH) == 0
        _result, llm, events = await _run(
            [*_reads(7), _answer("done")],
            max_steps=20,
            steps_per_attempt=20,
            brief_probe=_digest(None),
        )
        assert _count(llm, STEP_PUSH) == 0 and not _step_notices(events)

    @pytest.mark.asyncio
    async def test_a_run_without_a_brief_gets_none(self) -> None:
        _result, llm, events = await _run(
            [*_reads(8), _answer("done")], max_steps=20, steps_per_attempt=20
        )
        assert _count(llm, STEP_PUSH) == 0 and not _step_notices(events)

    @pytest.mark.asyncio
    async def test_the_boundary_push_stays(self) -> None:
        _result, llm, events = await _run(
            [_echo_call("a"), _reflection("continue"), _answer("42")],
            max_attempts=3,
            steps_per_attempt=1,
            brief_probe=_digest("same"),
        )
        assert _count(llm, NO_CHANGE_PUSH) == 1
        assert not _step_notices(events)

    def test_the_line_is_the_one_in_the_brief(self) -> None:
        assert STEP_PUSH == (
            "You have read for 6 steps without changing anything. Write the part "
            "of the task you can already write now (the smallest item first), "
            "then continue."
        )


def _unit(i: int, call: ToolCall, output: str) -> list[Message]:
    return [
        Message(role="assistant", tool_calls=[call]),
        Message(role="tool", content=output, tool_call_id=call.id),
    ]


class TestDroppedNote:
    def test_a_note_names_the_tool_its_argument_and_the_length(self) -> None:
        read = ToolCall("file_read", {"path": "support/sessions/runner.py"})
        grep = ToolCall("shell_exec", {"command": "grep -n x\nfoo " + "y" * 200})
        notes = dropped_notes(
            [*_unit(0, read, "x" * 49999), *_unit(1, grep, "z" * 2310)]
        )
        assert notes[0] == "file_read support/sessions/runner.py (49,999 chars)"
        assert notes[1].startswith("shell_exec `grep -n x foo yyy")
        assert notes[1].endswith("... (2,310 chars)") is False
        assert notes[1].endswith("...` (2,310 chars)")
        assert len(notes[1]) < len("shell_exec ``  (2,310 chars)") + ARG_LIMIT + 1

    def test_the_line_has_the_wording_of_the_brief(self) -> None:
        assert render_dropped([]) == ""
        assert render_dropped(
            ["file_read a.py (10 chars)", "shell_exec `ls` (3 chars)"]
        ) == (
            "Earlier tool output no longer in your context: file_read a.py "
            "(10 chars); shell_exec `ls` (3 chars). Read again what you need "
            "before you edit."
        )

    def test_a_secret_in_an_argument_is_redacted(self) -> None:
        call = ToolCall(
            "shell_exec",
            {"command": "curl -H 'Authorization: Bearer sk-abcdefghijklmnop1234'"},
        )
        [note] = dropped_notes(_unit(0, call, "ok"))
        assert "sk-abcdefghijklmnop1234" not in note

    def test_working_memory_reports_what_eviction_drops(self) -> None:
        memory = WorkingMemory(4)
        seen: list[list[Message]] = []
        memory.on_drop = seen.append
        for i in range(3):
            call = ToolCall("file_read", {"path": f"f{i}.py"})
            memory.append_turn(*(lambda u: (u[0], u[1:]))(_unit(i, call, "o" * i)))
        [gone] = seen  # the oldest unit left when the window passed four messages
        assert [m.role for m in gone] == ["assistant", "tool"]
        assert dropped_notes(gone) == ["file_read f0.py (0 chars)"]
        memory.drop_oldest_units(1)
        assert dropped_notes(seen[-1]) == ["file_read f1.py (1 chars)"]

    @pytest.mark.asyncio
    async def test_a_brief_run_carries_the_newest_eight_at_the_tail(self) -> None:
        # Twenty messages are kept: after the 11th step the oldest units go.
        _result, llm, _events = await _run(
            [*_reads(24), _answer("done")],
            max_steps=40,
            steps_per_attempt=40,
            brief_probe=_digest(None),
        )
        last = llm.calls[-1]["messages"][-1]
        assert last.role == "user"
        assert last.content.startswith(
            "Earlier tool output no longer in your context: "
        )
        assert last.content.endswith("Read again what you need before you edit.")
        listed = last.content.count("(")
        assert listed == DROPPED_KEPT
        # The newest dropped read is listed last; the oldest one is gone.
        listed_reads = [int(w) for w in re.findall(r"echo read (\d+) ", last.content)]
        assert listed_reads == sorted(listed_reads) and len(listed_reads) == 8
        assert 0 not in listed_reads and listed_reads[-1] < 24 - 9
        # The first request has nothing to note.
        assert "no longer in your context" not in str(llm.calls[0]["messages"])

    @pytest.mark.asyncio
    async def test_a_run_without_a_brief_has_no_note(self) -> None:
        _result, llm, _events = await _run(
            [*_reads(16), _answer("done")], max_steps=30, steps_per_attempt=30
        )
        assert not any(
            "no longer in your context" in m.content
            for c in llm.calls
            for m in c["messages"]
        )


class TestReflectionPointers:
    @pytest.mark.asyncio
    async def test_the_request_carries_the_addendum_only_with_a_brief(self) -> None:
        for brief in (True, False):
            options: dict[str, Any] = {"brief_probe": _digest("x")} if brief else {}
            _result, llm, _events = await _run(
                [_echo_call("a"), _reflection("done", best_effort_result="ok")],
                steps_per_attempt=1,
                **options,
            )
            [call] = [c for c in llm.calls if _is_reflection(c)]
            last = call["messages"][-1].content
            assert (REFLECTION_ADDENDUM in last) is brief
            assert REFLECTION_ADDENDUM not in call["messages"][0].content

    @pytest.mark.asyncio
    async def test_start_with_carries_the_first_direction(self) -> None:
        replies = [
            _echo_call("a"),
            _reflection(
                "continue",
                next_promising_directions=["edit runner.py line 40", "then run tests"],
            ),
            _answer("42"),
        ]
        _result, llm, _events = await _run(
            replies, max_attempts=3, steps_per_attempt=1, brief_probe=_digest("same")
        )
        want = f"{NO_CHANGE_PUSH} Start with: edit runner.py line 40"
        assert llm.calls[2]["messages"][-1].content == want

    @pytest.mark.asyncio
    async def test_a_reflection_without_a_direction_leaves_the_line(self) -> None:
        _result, llm, _events = await _run(
            [_echo_call("a"), _reflection("continue"), _answer("42")],
            max_attempts=3,
            steps_per_attempt=1,
            brief_probe=_digest("same"),
        )
        assert llm.calls[2]["messages"][-1].content == NO_CHANGE_PUSH

    def test_the_direction_is_redacted_and_capped(self) -> None:
        entry = ReflectionEntry(
            next_promising_directions=["use sk-abcdefghijklmnop1234 " + "w " * 400]
        )
        line = start_with(entry)
        assert "sk-abcdefghijklmnop1234" not in line
        assert len(line) == len("Start with: ") + START_WITH_LIMIT
        assert line.endswith("...")
        assert start_with(None) == "" and start_with(ReflectionEntry()) == ""
