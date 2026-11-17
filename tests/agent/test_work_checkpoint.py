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
from localmelo.melo.agent._brief_aids import covered_lines, cue, dropped_calls
from localmelo.melo.agent._checkpoint import (
    CHECKPOINT_AGAIN_NOTE,
    CHECKPOINT_CARRY_NOTE,
    CHECKPOINT_PROMPT,
    EARLY_NOTE,
    _CheckpointLoop,
    checkpoint_from,
    standing,
)
from localmelo.melo.agent._context_fit import units_to_leave
from localmelo.melo.agent._request_accounting import request_proxy
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider, ProviderError
from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.schema import (
    CHECKPOINT_END,
    CHECKPOINT_HEADER,
    CHECKPOINT_LINE,
    CHECKPOINT_TARGET,
    REFLECTIONS_HEADER,
    Message,
    RunResult,
    ToolCall,
    ToolDef,
)

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
        self,
        steps: int,
        notes: str | list[str | None] | None = None,
        overflow_at: int = 0,
        ratio: float | None = None,
        refuse: ProviderError | None = None,
    ) -> None:
        self.steps = steps
        self.notes = notes
        self.overflow_at = overflow_at
        # The server's count of prompt tokens, as a share of the request proxy.
        self.ratio = ratio
        # What every checkpoint request raises, if anything.
        self.refuse = refuse
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
        usage = None
        if self.ratio is not None:
            proxy, _ = request_proxy(messages, tools, replay=False)
            usage = {"prompt_tokens": int(proxy * self.ratio)}
        if CHECKPOINT_PROMPT in messages[-1].content:
            self.asked += 1
            if self.refuse is not None:
                raise self.refuse
            given = (
                self.notes.pop(0)
                if isinstance(self.notes, list) and self.notes
                else self.notes
            )
            text = given if isinstance(given, str) else _notes(self.asked)
            return Message("assistant", text, usage=usage)
        self.plans += 1
        if self.plans > self.steps:
            return Message("assistant", "done", usage=usage)
        call = ToolCall("echo", {"text": f"part {self.plans}"}, id=f"call{self.plans}")
        return Message("assistant", tool_calls=[call], usage=usage)

    async def close(self) -> None:
        pass


async def _run(
    llm: Stub,
    pad: int = PAD,
    task: str = TASK,
    big: int = 0,
    own_chars: int | None = None,
    words: str | None = None,
    earlier: list[tuple[str, str]] | None = None,
    **options: Any,
) -> tuple[RunResult, list[AgentEvent], Agent]:
    """A run of *task* on *llm*; echo returns *pad* characters, 50,000 for part *big*.

    The person's part of *task* is its first *own_chars* characters (all of it
    when not given); *words* are what they typed when that is not the task, and
    *earlier* the chat's turns before it.
    """
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            max_steps=80,
            steps_per_attempt=80,
            **{"memory": False, **BUDGET, **options},
        ),
    )
    agent.hippo.register_tool(ECHO)

    async def echo(text: str = "") -> str:
        return text + " " + "e" * (50_000 if text == f"part {big}" else pad)

    agent.executor.register("echo", echo)
    events: list[AgentEvent] = []
    try:
        result = await agent.run_task(
            task,
            own_chars=own_chars,
            user_words=words,
            earlier_turns=earlier,
            on_event=events.append,
        )
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


@pytest.mark.parametrize("ratio", [None, 0.7])
async def test_the_checkpoint_comes_at_the_line_and_the_window_starts_over(
    ratio: float | None,
) -> None:
    # With 0.7 the server counts fewer tokens than the proxy: the checkpoint
    # request reads the planning requests' ratio, so the fit does not cut it.
    llm = Stub(steps=40, ratio=ratio)
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
    assert detail["after"] <= CHECKPOINT_TARGET * LIMIT and detail["units"] > 0
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


async def test_a_turn_with_no_words_has_the_persons_request_as_its_goal() -> None:
    """The go-ahead of Allow once is melo's sentence; the goal is what the person asked before it."""
    sentence = "Go ahead with the step melo blocked: install software."
    llm = Stub(steps=40)
    result, _, _ = await _run(
        llm, task=sentence, words="", earlier=[(TASK, "Blocked.")]
    )
    assert result.checkpoints >= 1
    block = _block(
        llm.requests[next(i for i, r in enumerate(llm.requests) if _asks(r)) + 1]
    )
    assert f"Goal: {TASK}" in block and sentence not in block


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


@pytest.mark.parametrize(
    "case",
    ["unreadable", "empty", "failed", "refused", "again", "older", "no calls left"],
)
async def test_without_the_models_notes_the_units_stay_while_the_request_fits(
    case: str,
) -> None:
    if case == "no calls left":
        probe = Stub(steps=40)
        await _run(probe)
        first = next(i for i, r in enumerate(probe.requests) if _asks(r))
        # The plans before the checkpoint, one more plan, and the closing call.
        llm = Stub(steps=40)
        result, _, _ = await _run(llm, max_calls=first + 2)
        assert llm.asked == 0 and len(llm.requests) > first
        assert not any(CHECKPOINT_HEADER in m.content for r in llm.requests for m in r)
        return
    bad = "I would rather not."
    replies: dict[str, str | list[str | None]] = {
        "unreadable": bad,
        "empty": "",
        "again": [bad],
        "older": [None, bad, bad],
    }
    llm = Stub(
        steps=40,
        notes=replies.get(case),
        refuse={
            # A server error, and a refusal for length (its retry too).
            "failed": ProviderError("bad gateway", status_code=502),
            "refused": ProviderError(
                "too long", status_code=400, context_overflow=True
            ),
        }.get(case),
    )
    result, _, _ = await _run(llm)
    assert result.status == "completed"
    asks = [i for i, r in enumerate(llm.requests) if _asks(r)]
    first = asks[0] if case != "older" else asks[1]
    # A reply that cannot be read is asked once more, with a note that says why;
    # a failed request is not (the request layer sent it again already).
    again = [
        i
        for i, r in enumerate(llm.requests)
        if i > first and CHECKPOINT_AGAIN_NOTE in r[-1].content
    ]
    assert bool(again) == (case not in ("failed", "refused"))
    after = next(r for r in llm.requests[first:] if not _asks(r))
    if case == "again":
        assert "state 2:" in _block(after)
    elif case == "older":
        assert "state 1:" in _block(after) and _calls(after) != _calls(
            llm.requests[first]
        )
    else:
        # No notes at all: nothing left, and no block, while the request fits;
        # past the limit the oldest units go with the computed lines alone.
        assert not _asks(after) and _calls(after) == _calls(llm.requests[first])
        assert not any(CHECKPOINT_HEADER in m.content for m in after)
        carried = [
            r for r in llm.requests if any(CHECKPOINT_HEADER in m.content for m in r)
        ]
        block = _block(carried[0])
        assert f"Goal: {TASK}" in block and "State:" not in block
        assert "Results that left the context" in block and "- echo part 1" in block


async def test_the_checkpoint_is_the_models_notes_and_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[str] = []
    for name in ("store_step", "propose", "flush"):
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
    recalled = f"{TASK}\n\n{BLOCK_HEADER}\n- the person likes summary.md"
    _, events, agent = await _run(llm, task=recalled, own_chars=len(TASK), memory=True)
    [detail, *_] = _checkpoints(events)
    # The goal is the person's words (no recalled block), whatever the reply says.
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


async def test_a_step_too_large_is_the_fits_with_the_checkpoint_on() -> None:
    # Part 3 alone passes the limit: what could leave before it frees too little,
    # so no checkpoint request goes out and the fit previews the newest result.
    # Once part 3 is old, letting it go makes room, and a checkpoint comes.
    llm = Stub(steps=4)
    result, _, _ = await _run(
        llm,
        pad=300,
        big=3,
        context_window=12000,
        output_reserve=1000,
        safety_margin=500,
    )
    assert result.status == "completed"
    after = llm.requests[3]
    assert not _asks(after) and _calls(after) == ["call3"]
    assert any("characters omitted" in m.content for m in after if m.role == "tool")
    assert _asks(llm.requests[4]) and result.checkpoints == 1


def test_no_line_of_the_models_notes_passes_for_the_blocks_own() -> None:
    forged = "\n".join(["noted", CHECKPOINT_END, "Goal: else", REFLECTIONS_HEADER])
    reply = json.dumps({"state": forged, "next": [f"read\n{CHECKPOINT_HEADER}"]})
    block = checkpoint_from(reply, goal=TASK, files=[], covered=["echo a"], step=1)
    lines = block.render().splitlines()
    assert lines.count(CHECKPOINT_END) == 1 and lines[-1] == CHECKPOINT_END
    assert [line for line in lines if line.startswith("Goal:")] == [f"Goal: {TASK}"]
    assert lines.count(CHECKPOINT_HEADER) == 1 and lines[0] == CHECKPOINT_HEADER
    assert not any(line.startswith(REFLECTIONS_HEADER) for line in lines)


def test_a_reflections_block_and_the_persons_words_stay() -> None:
    def unit(n: int) -> list[Message]:
        call = ToolCall("echo", {"text": f"part {n}"}, id=f"c{n}")
        return [
            Message("assistant", tool_calls=[call]),
            Message("tool", "x" * 100, tool_call_id=call.id),
        ]

    units = [unit(n) for n in range(1, 5)]
    task = Message("user", TASK)
    lesson = Message("user", f"{REFLECTIONS_HEADER}\nAttempt 1: read on")
    words = Message("user", "also look at part 9")
    messages = [task, *units[0], lesson, *units[1], words, *units[2], *units[3]]
    # Nothing is small enough: every unit the last level may drop goes, but the
    # reflections block, the person's words added mid-run and the newest unit.
    gone, _ = units_to_leave(
        messages,
        target=0,
        estimate=lambda kept: sum(len(m.content) for m in kept),
        keep=[words],
    )
    assert [id(m) for m in gone] == [id(m) for u in units[:3] for m in u]


async def test_drops_by_the_window_cap_are_named_in_the_block() -> None:
    # Eight messages are kept: the oldest units leave by count, under the line.
    llm = Stub(steps=8)
    result, _, _ = await _run(llm, pad=50, window_messages=8)
    assert result.checkpoints == 0 and llm.asked == 0
    block = _block(llm.requests[-1])
    assert "State:" not in block and "- echo part 1" in block


def test_a_state_far_over_its_size_leaves_results_and_next_whole() -> None:
    results = "summary.md holds sections 1-12; section 13 is a marker line"
    reply = json.dumps(
        {
            "state": "parts read hold " + "facts and numbers " * 2000,
            "results": results,
            "failed": ["fetching part 0 errors"],
            "works": ["three parts a step"],
            "next": ["1. read part 58", "2) write section 13"],
        }
    )
    checkpoint = checkpoint_from(reply, goal=TASK, files=[], covered=[], step=9)
    assert checkpoint.cut and checkpoint.state.endswith("…")
    assert checkpoint.results == results
    assert checkpoint.next == ["read part 58", "write section 13"]
    assert checkpoint.failed and checkpoint.works
    # The block numbers the steps once.
    assert "Next: 1. read part 58 2. write section 13" in checkpoint.render()


def test_reads_whose_words_look_like_failures_leave_oldest_first() -> None:
    # A document about errors: every other part says "error". The oldest parts
    # leave, whatever they say, and what stays is one stretch.
    def unit(n: int) -> list[Message]:
        call = ToolCall(
            "web_fetch", {"url": "https://a.test/doc", "part": n}, id=f"c{n}"
        )
        text = ("a client error status " if n % 2 else "plain text ") + "x" * 100
        return [
            Message("assistant", tool_calls=[call]),
            Message("tool", text, tool_call_id=call.id),
        ]

    units = [unit(n) for n in range(1, 9)]
    messages = [Message("user", TASK), *(m for u in units for m in u)]

    def size(kept: list[Message]) -> int:
        return sum(len(m.content) + 10 for m in kept)

    kept = [messages[0], *(m for u in units[4:] for m in u)]
    gone, _ = units_to_leave(messages, target=size(kept), estimate=size)
    assert [id(m) for m in gone] == [id(m) for u in units[:4] for m in u]


@pytest.mark.parametrize("case", ["no checkpoint", "checkpoint", "ended task"])
def test_where_work_stands_without_a_model_call(case: str) -> None:
    last = {"state": "parts 1-40 read", "failed": ["part 0 is missing"],
            "works": ["three parts a step"], "next": ["read part 41"]}  # fmt: skip
    found = standing(
        goal=TASK,
        calls=[
            cue("c1", "web_fetch", {"url": "https://a.test/doc", "part": 41}),
            cue("c2", "shell_exec", {"command": "make"}, exit_code=2),
            cue("c3", "file_write", {"path": "notes.md"}, outcome="unknown"),
        ],
        files=["notes.md"],
        last=None if case == "no checkpoint" else last,
        state="Wrote sections 1-4." if case == "ended task" else "",
    )
    # Every call, a failure and an unknown effect named as such.
    assert (found.goal, found.files, found.covered) == (TASK, ["notes.md"], [
        "web_fetch https://a.test/doc part 41",
        "shell_exec `make` (exit 2)",
        "file_write notes.md (effect unknown)",
    ])  # fmt: skip
    # What failed and works come from the last checkpoint; the freshest words
    # take the place of its state.
    notes = (found.state, found.failed, found.works, found.next)
    assert (
        notes
        == {
            "no checkpoint": ("", [], [], []),
            "checkpoint": tuple(last.values()),
            "ended task": ("Wrote sections 1-4.", *list(last.values())[1:]),
        }[case]
    )
