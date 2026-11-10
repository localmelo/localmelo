"""The work checkpoint: long work keeps what it learned when old results leave.

With a context budget, a planning request that would pass ``CHECKPOINT_LINE`` of
the input limit is preceded by one request for the model's notes; the oldest
units then leave the window for good and the checkpoint's block stands right
after the task's message. A drop by the request fit is named in the block too.
Without a budget nothing changes. All on a stub model.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._brief_aids import covered_lines, dropped_calls
from localmelo.melo.agent._checkpoint import (
    CHECKPOINT_CARRY_NOTE,
    CHECKPOINT_PROMPT,
    EARLY_NOTE,
    _CheckpointLoop,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import (
    CHECKPOINT_HEADER,
    CHECKPOINT_LINE,
    WINDOW_FIT_TARGET,
    Message,
    RunResult,
    ToolCall,
    ToolDef,
)
from localmelo.support.sessions.window import decode_window, encode_window

ECHO = ToolDef(
    name="echo",
    description="Echo text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)
TASK = "echo every part, then say done"
# About thirty units of PAD characters fit under the line.
BUDGET: dict[str, Any] = {
    "context_window": 40000,
    "output_reserve": 4000,
    "safety_margin": 1000,
}
LIMIT = 40000 - 4000 - 1000
PAD = 3000


def _notes(n: int) -> str:
    return json.dumps(
        {
            "goal": "something else entirely",
            "state": f"state {n}: parts read so far hold the numbers 7 and 9",
            "results": "nothing written yet",
            "failed": [],
            "works": ["echo one part per step"],
            "next": ["echo the next part"],
        }
    )


class Stub(BaseLLMProvider):
    """Calls echo for *steps* planning requests, then answers; writes notes when asked."""

    model = "stub"

    def __init__(
        self, steps: int, notes: str | None = None, overflow_at: int = 0
    ) -> None:
        self.steps = steps
        self.notes = notes
        self.overflow_at = overflow_at
        self.requests: list[list[Message]] = []
        self.plans = 0
        self.asked = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        if len(self.requests) == self.overflow_at:
            raise ProviderError(
                "context_length_exceeded", status_code=400, context_overflow=True
            )
        if CHECKPOINT_PROMPT in messages[-1].content:
            self.asked += 1
            text = self.notes if self.notes is not None else _notes(self.asked)
            return Message("assistant", text)
        self.plans += 1
        if self.plans > self.steps:
            return Message("assistant", "done")
        call = ToolCall("echo", {"text": f"part {self.plans}"}, id=f"call{self.plans}")
        return Message("assistant", tool_calls=[call])

    async def close(self) -> None:
        pass


async def _run(
    llm: Stub, pad: int = PAD, task: str = TASK, **options: Any
) -> tuple[RunResult, list[AgentEvent], Agent]:
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            tool_exposure="all",
            max_steps=80,
            steps_per_attempt=80,
            **{"memory": False, **BUDGET, **options},
        ),
    )
    agent.hippo.register_tool(ECHO)

    async def echo(text: str = "") -> str:
        return text + " " + "e" * pad

    agent.executor.register("echo", echo)
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task(task, on_event=events.append)
    finally:
        await agent.close()
    return result, events, agent


def _asks(request: list[Message]) -> bool:
    return CHECKPOINT_PROMPT in request[-1].content


def _wire(request: list[Message]) -> list[tuple[object, ...]]:
    """What a request sends of each message (its timestamps aside)."""
    return [
        (m.role, m.content, tuple(c.id for c in m.tool_calls), m.tool_call_id)
        for m in request
    ]


def _calls(request: list[Message]) -> list[str]:
    return [call.id for m in request for call in m.tool_calls]


def _checkpoints(events: list[AgentEvent]) -> list[dict[str, Any]]:
    return [
        dict(e.data["detail"])
        for e in events
        if e.type == "notice" and e.data["kind"] == "checkpoint"
    ]


def _block(request: list[Message]) -> str:
    """The checkpoint block a planning request carries (after the task's words)."""
    [text] = [m.content for m in request if CHECKPOINT_HEADER in m.content]
    return text[text.index(CHECKPOINT_HEADER) :]


async def test_the_checkpoint_comes_at_the_line_and_the_window_starts_over() -> None:
    llm = Stub(steps=40)
    result, events, _ = await _run(llm)
    assert result.status == "completed" and result.checkpoints >= 1
    first = next(i for i, r in enumerate(llm.requests) if _asks(r))
    before, asked, after = (
        llm.requests[first - 1],
        llm.requests[first],
        llm.requests[first + 1],
    )
    # Not before the line: no earlier request carries a block, and the last
    # planning request before it was under the line.
    assert not any(
        CHECKPOINT_HEADER in m.content for r in llm.requests[:first] for m in r
    )
    plans = [r for r in result.context_budget or () if r["kind"] == "plan"]
    assert plans[first - 1]["estimate"] <= CHECKPOINT_LINE * LIMIT
    [detail, *_] = _checkpoints(events)
    assert detail["before"] > CHECKPOINT_LINE * LIMIT
    assert detail["after"] <= WINDOW_FIT_TARGET * LIMIT and detail["units"] > 0
    # The checkpoint request is the next planning request plus the asking message.
    assert _wire(asked)[: len(before)] == _wire(before)
    # After it: [system][the task's words with the block][kept units], none of the
    # units that left, and the newest units kept whole.
    assert [m.role for m in after[:2]] == ["system", "user"]
    assert after[1].content.startswith(TASK)
    assert after[1].content.index(CHECKPOINT_HEADER) > len(TASK)
    assert "state 1:" in _block(after)
    kept = _calls(after)
    assert kept == _calls(asked)[-len(kept) :] and len(kept) < len(_calls(asked))
    # The requests after it extend it until the next checkpoint.
    nxt = next((i for i, r in enumerate(llm.requests) if i > first and _asks(r)), None)
    later = llm.requests[first + 1 : nxt]
    for earlier, request in zip(later, later[1:], strict=False):
        assert _wire(request)[: len(earlier)] == _wire(earlier)


async def test_a_second_checkpoint_replaces_the_first_and_saw_it() -> None:
    llm = Stub(steps=50)
    result, events, _ = await _run(llm)
    assert result.checkpoints >= 2 and len(_checkpoints(events)) == result.checkpoints
    second = [r for r in llm.requests if _asks(r)][1]
    assert "state 1:" in _block(second[:-1])
    assert CHECKPOINT_CARRY_NOTE in second[-1].content
    last = llm.requests[-1]
    assert sum(m.content.count(CHECKPOINT_HEADER) for m in last) == 1
    assert "state 1:" not in _block(last) and f"state {llm.asked}:" in _block(last)


def test_the_cue_list_is_one_line_per_source_and_bounded() -> None:
    messages: list[Message] = []
    calls = [
        *(
            ToolCall("web_fetch", {"url": "https://a.test/doc", "part": n})
            for n in (1, 2, 3, 5)
        ),
        ToolCall("shell_exec", {"command": "make test"}),
        ToolCall("file_read", {"path": "notes.md"}),
    ]
    for n, call in enumerate(calls):
        call.id = f"c{n}"
        out = "boom\n[exit code 2]" if call.tool_name == "shell_exec" else "x"
        messages += [
            Message("assistant", tool_calls=[call]),
            Message("tool", out, tool_call_id=call.id),
        ]
    lines = covered_lines(dropped_calls(messages))
    assert lines == [
        "web_fetch https://a.test/doc parts 1-3, 5",
        "shell_exec `make test` (exit 2)",
        "file_read notes.md",
    ]
    assert covered_lines(dropped_calls(messages), limit=2) == [
        "earlier: 1 web_fetch source, 1 shell_exec source",
        "file_read notes.md",
    ]


async def test_a_drop_by_the_fit_is_named_in_the_block() -> None:
    # The server refuses the sixth request as too long although its estimate fit:
    # the fit's retry (the seventh) drops the oldest unit, and the next planning
    # request says so.
    llm = Stub(steps=8, overflow_at=6)
    result, events, _ = await _run(llm, pad=50, context_window=20000)
    assert result.status == "completed" and result.checkpoints == 0
    assert not _checkpoints(events) and llm.asked == 0
    assert "call1" not in _calls(llm.requests[6])
    assert not any(CHECKPOINT_HEADER in m.content for m in llm.requests[6])
    block = _block(llm.requests[7])
    assert "State:" not in block and f"Goal: {TASK}" in block
    assert "- echo part 1" in block


@pytest.mark.parametrize("case", ["unreadable", "no calls left"])
async def test_without_the_models_notes_the_block_is_computed_and_the_run_goes_on(
    case: str,
) -> None:
    if case == "unreadable":
        llm = Stub(steps=40, notes="I would rather not.")
        result, _, _ = await _run(llm)
        assert llm.asked >= 1 and result.status == "completed"
    else:
        probe = Stub(steps=40)
        await _run(probe)
        first = next(i for i, r in enumerate(probe.requests) if _asks(r))
        # The plans before the checkpoint, one more plan, and the closing call.
        llm = Stub(steps=40)
        result, _, _ = await _run(llm, max_calls=first + 2)
        assert llm.asked == 0 and len(llm.requests) > first
    # The planning request after it carries the computed lines alone.
    carried = [
        r for r in llm.requests if any(CHECKPOINT_HEADER in m.content for m in r)
    ]
    block = _block(carried[0])
    assert f"Goal: {TASK}" in block and "State:" not in block
    assert "Already read or run" in block and "- echo part 1" in block


async def test_the_checkpoint_is_the_models_notes_and_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[str] = []
    for name in ("memorize", "propose", "flush"):
        real = getattr(Hippo, name)

        async def spy(self: Hippo, *args: Any, _real: Any = real, **kw: Any) -> Any:
            written.append(repr((args, kw)))
            return await _real(self, *args, **kw)

        def sync_spy(self: Hippo, *args: Any, _real: Any = real, **kw: Any) -> Any:
            written.append(repr((args, kw)))
            return _real(self, *args, **kw)

        is_async = inspect.iscoroutinefunction(real)
        monkeypatch.setattr(Hippo, name, spy if is_async else sync_spy)
    llm = Stub(steps=40)
    _, events, agent = await _run(llm, memory=True)
    [detail, *_] = _checkpoints(events)
    # The goal is the task's words, whatever the reply says it is.
    assert detail["checkpoint"]["goal"] == TASK
    assert "something else entirely" not in detail["text"]
    # Never the system message; nothing of it goes to memory or to notes.
    assert all(CHECKPOINT_HEADER not in r[0].content for r in llm.requests)
    assert written and not any("state 1:" in text for text in written)
    assert not any("state 1:" in entry.text for entry in agent.notes())


async def test_without_a_budget_the_requests_are_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_budget: dict[str, Any] = {
        "context_window": None,
        "output_reserve": None,
        "safety_margin": None,
    }
    llm = Stub(steps=30)
    result, _, _ = await _run(llm, pad=4000, **no_budget)
    assert result.checkpoints == 0 and llm.asked == 0

    async def unchanged(*args: Any, **kw: Any) -> bool:
        return False

    monkeypatch.setattr(_CheckpointLoop, "_checkpoint_step", unchanged)
    before = Stub(steps=30)
    await _run(before, pad=4000, **no_budget)
    assert [_wire(r) for r in llm.requests] == [_wire(r) for r in before.requests]


def test_a_stored_window_keeps_the_block() -> None:
    block = Message("user", f"{CHECKPOINT_HEADER}\nGoal: x\n[end of work checkpoint]")
    call = ToolCall("echo", {"text": "a"}, id="c1")
    window = [
        Message("user", "x"),
        block,
        Message("assistant", tool_calls=[call]),
        Message("tool", "a", tool_call_id="c1"),
    ]
    decoded = decode_window(encode_window(window))
    assert decoded is not None
    agent = Agent(llm=Stub(steps=0), options=AgentOptions(memory=False))
    assert agent.load_conversation(decoded) == 4
    assert agent.conversation()[1].content == block.content


@pytest.mark.parametrize(
    "task", [TASK, "echo every part and write the summary to summary.md"]
)
async def test_the_early_line_comes_once_in_each_fill_with_a_file_product(
    task: str,
) -> None:
    llm = Stub(steps=50)
    result, _, _ = await _run(llm, task=task)
    assert result.checkpoints >= 2
    early = EARLY_NOTE.split("{files}")[0]
    marks = [
        "early" if early in r[-1].content else "checkpoint" if _asks(r) else ""
        for r in llm.requests
    ]
    if "summary.md" not in task:
        assert "early" not in marks
        return
    assert EARLY_NOTE.format(files="summary.md") in str(llm.requests)
    # Once before the first checkpoint and at most once in each fill after it.
    fills = "".join("E" if m == "early" else "C" if m else "" for m in marks).split("C")
    assert fills[0] == "E" and all(len(fill) <= 1 for fill in fills)
    # Only the outgoing request carries it: the window does not keep it.
    after = [i + 1 for i, m in enumerate(marks) if m == "early"]
    assert all(early not in str(llm.requests[i]) for i in after if i < len(marks))
