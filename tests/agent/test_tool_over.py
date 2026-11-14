"""A tool whose result says it takes no more calls is over for the run.

The tool's own code says so with ``ToolOutput.no_more_calls``, never in words.
From the next step the agent does not offer the tool (the place ``hide_tools``
use), a call to it anyway does not run and reads one fixed reply, and the
request after that call offers no tool and asks for the answer. The state is
per run: the next run offers the tool again. No real model.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent._chat import TOOL_OVER_ANSWER_NOTE
from localmelo.melo.agent.agent import TOOL_OVER, TOOL_OVER_ANSWER
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import ToolOutput
from localmelo.melo.executor.models import ErrorCategory, ExecutionStatus
from localmelo.melo.schema import Message, ToolCall, ToolDef

ONCE_MORE = "Not kept. You may try once more."
LAST = "Not kept. Do not try this again in this turn."
ALL = ["chatty", "keep", "look"]


class FakeLLM(BaseLLMProvider):
    """Answers with the queued replies; keeps each request's messages and tool names."""

    def __init__(self, *replies: Message) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[tuple[list[Message], list[str] | None]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        offered = None if tools is None else [tool.name for tool in tools]
        self.requests.append((list(messages), offered))
        if not self.replies:
            raise AssertionError("the test queued no further reply")
        return self.replies.popleft()

    def offered(self) -> list[list[str] | None]:
        return [names for _, names in self.requests]

    async def close(self) -> None:
        pass


class Tools(BaseToolProvider):
    """``keep`` is over once it has run twice, ``look`` always runs, ``chatty`` only sounds over."""

    def __init__(self, last_text: str = LAST) -> None:
        self.ran: list[str] = []
        self.last_text = last_text

    async def start(self) -> list[ProvidedTool]:
        def definition(name: str) -> ToolDef:
            return ToolDef(
                name=name,
                description=name,
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )

        return [
            ProvidedTool(definition=definition(name), function=function)
            for name, function in (
                ("keep", self.keep),
                ("look", self.look),
                ("chatty", self.chatty),
            )
        ]

    async def close(self) -> None:
        return None

    async def keep(self, text: str = "") -> str | ToolOutput:
        self.ran.append(f"keep:{text}")
        if sum(ran.startswith("keep:") for ran in self.ran) >= 2:
            return ToolOutput(text=self.last_text, no_more_calls=True)
        return ONCE_MORE

    async def look(self, text: str = "") -> str:
        self.ran.append(f"look:{text}")
        return "seen"

    async def chatty(self, text: str = "") -> str:
        self.ran.append(f"chatty:{text}")
        # The words of a last refusal, but no flag: its code did not say it.
        return LAST


_ids = itertools.count()


def _step(*calls: tuple[str, str]) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(tool_name=name, arguments={"text": text}, id=f"call_{next(_ids)}")
            for name, text in calls
        ],
    )


def _say(text: str) -> Message:
    return Message(role="assistant", content=text)


async def _run(
    llm: FakeLLM,
    tools: Tools,
    *queries: str,
    events: list[AgentEvent] | None = None,
) -> list[Any]:
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            attempt_policy="none",
            answer_on_stop=False,
        ),
        tool_providers=[tools],
    )
    await agent.start()
    try:
        return [
            await agent.run_task(
                query, on_event=None if events is None else events.append
            )
            for query in queries
        ]
    finally:
        await agent.close()


async def test_a_tool_that_takes_no_more_calls_is_not_offered_from_the_next_step() -> (
    None
):
    llm = FakeLLM(_step(("keep", "a")), _step(("keep", "b")), _say("done"))
    tools = Tools()
    (outcome,) = await _run(llm, tools, "keep my notes")
    assert (outcome.status, outcome.result) == ("completed", "done")
    # The second call said it was the last: the step after it offers the others only.
    assert llm.offered() == [ALL, ALL, ["chatty", "look"]]
    assert tools.ran == ["keep:a", "keep:b"]


async def test_a_result_the_check_cuts_still_withdraws_its_tool() -> None:
    llm = FakeLLM(_step(("keep", "a")), _step(("keep", "b")), _say("done"))
    # Over the result check's limit: the agent works with the cut copy of the result.
    tools = Tools(last_text="x" * 60_000)
    await _run(llm, tools, "keep my notes")
    assert llm.offered() == [ALL, ALL, ["chatty", "look"]]


async def test_words_in_a_result_do_not_withdraw_a_tool() -> None:
    llm = FakeLLM(_step(("chatty", "a")), _step(("chatty", "b")), _say("done"))
    tools = Tools()
    (outcome,) = await _run(llm, tools, "talk")
    assert outcome.result == "done"
    # The result read like a last refusal, no flag came with it: still offered, still runs.
    assert llm.offered() == [ALL] * 3
    assert tools.ran == ["chatty:a", "chatty:b"]


async def test_the_next_run_offers_the_tool_again() -> None:
    llm = FakeLLM(_step(("keep", "a")), _step(("keep", "b")), _say("one"), _say("two"))
    first, second = await _run(llm, Tools(), "keep my notes", "and now?")
    assert (first.result, second.result) == ("one", "two")
    assert llm.offered()[-2:] == [["chatty", "look"], ALL]


async def test_a_call_to_an_over_tool_is_not_run_and_the_answer_is_next() -> None:
    events: list[AgentEvent] = []
    llm = FakeLLM(
        _step(("keep", "a")),
        _step(("keep", "b")),
        # The model calls it anyway, and another tool beside it.
        _step(("keep", "c"), ("look", "x")),
        _say("Here is your intro. I could not keep the note."),
    )
    tools = Tools()
    (outcome,) = await _run(
        llm, tools, "Introduce yourself, and keep my note.", events=events
    )
    # The two earlier calls ran, the third did not, its neighbour did.
    assert tools.ran == ["keep:a", "keep:b", "look:x"]
    # Three planning requests, then one more that offers no tool.
    assert llm.offered() == [ALL, ALL, ["chatty", "look"], None]
    request, _ = llm.requests[-1]
    assert request[-1].content.endswith(TOOL_OVER_ANSWER_NOTE)
    assert f"Result of keep: Error: {TOOL_OVER}" in request[-1].content
    assert outcome.status == "completed"
    assert outcome.result == "Here is your intro. I could not keep the note."
    # The call that did not run carries the executor's own pair for a tool not offered.
    skipped = [
        event.data
        for event in events
        if event.type == "tool.finished" and event.data["error"] == TOOL_OVER
    ]
    assert len(skipped) == 1
    assert skipped[0]["status"] == ExecutionStatus.ERROR.value
    assert skipped[0]["error_category"] == ErrorCategory.TOOL_NOT_OFFERED.value
    assert skipped[0]["output"] == ""


async def test_a_call_in_the_step_that_ends_the_tool_does_not_end_the_run() -> None:
    events: list[AgentEvent] = []
    llm = FakeLLM(
        _step(("keep", "a")),
        # The second call is the last one the tool takes; the third is in the same step.
        _step(("look", "x"), ("keep", "b"), ("keep", "c")),
        _step(("look", "y")),
        _say("done"),
    )
    tools = Tools()
    (outcome,) = await _run(llm, tools, "keep my notes", events=events)
    assert tools.ran == ["keep:a", "look:x", "keep:b", "look:y"]
    # The third call read the fixed reply, and no request asked for the answer:
    # the run went on with the other tools until the model answered.
    assert llm.offered() == [ALL, ALL, ["chatty", "look"], ["chatty", "look"]]
    assert (outcome.status, outcome.result) == ("completed", "done")
    (skipped,) = [
        event.data
        for event in events
        if event.type == "tool.finished" and event.data["error"] == TOOL_OVER
    ]
    assert skipped["name"] == "keep"


async def test_the_answer_has_a_fixed_text_when_its_request_gives_none() -> None:
    llm = FakeLLM(
        _step(("keep", "a")), _step(("keep", "b")), _step(("keep", "c")), _say("")
    )
    tools = Tools()
    (outcome,) = await _run(llm, tools, "keep my notes")
    assert tools.ran == ["keep:a", "keep:b"]
    assert (outcome.status, outcome.result) == ("completed", TOOL_OVER_ANSWER)
