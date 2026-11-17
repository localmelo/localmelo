"""A tool call that hands the rest of the run's work on ends the run (``hand_off``).

A provided tool names the boolean argument that makes a call of it a hand-off
(``ProvidedTool.hands_off``). Such a call must be the only call of its step:
a step that mixes it with another call runs none of them. Alone, the call may
mark itself with ``hand_off(receipt)``; the run then ends ``completed`` with
the receipt as its result and ``handed_off`` true, and takes no further step.
No real model.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.tools import (
    BaseToolProvider,
    ProvidedTool,
    can_hand_off,
    hand_off,
)
from localmelo.melo.schema import Message, ToolCall, ToolDef


class FakeLLM(BaseLLMProvider):
    def __init__(self, replies: list[Message]) -> None:
        self.replies: deque[Message] = deque(replies)
        self.calls = 0

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.calls += 1
        if not self.replies:
            raise AssertionError("the test queued no further reply")
        return self.replies.popleft()

    async def close(self) -> None:
        pass


class Tools(BaseToolProvider):
    """``move`` hands the work on when ``rest`` is true; ``note`` records its text."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def start(self) -> list[ProvidedTool]:
        def definition(name: str, **properties: Any) -> ToolDef:
            return ToolDef(
                name=name,
                description=name,
                parameters={"type": "object", "properties": properties},
            )

        return [
            ProvidedTool(
                definition=definition(
                    "move", what={"type": "string"}, rest={"type": "boolean"}
                ),
                function=self.move,
                hands_off="rest",
            ),
            ProvidedTool(
                definition=definition("note", text={"type": "string"}),
                function=self.note,
            ),
        ]

    async def close(self) -> None:
        return None

    async def move(self, what: str = "", rest: bool = False) -> str:
        self.ran.append(f"move:{what}:{can_hand_off()}")
        if rest is True and can_hand_off():
            return hand_off(f"Handed on: {what}.")
        return f"Started {what} beside you."

    async def note(self, text: str = "") -> str:
        self.ran.append(f"note:{text}")
        return "noted"


def _step(*calls: tuple[str, dict[str, Any]]) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(tool_name=name, arguments=args, id=f"c{at}")
            for at, (name, args) in enumerate(calls)
        ],
    )


async def _run(
    replies: list[Message], query: str = "do the work", **options: Any
) -> tuple[Any, FakeLLM, Tools]:
    llm, tools = FakeLLM(replies), Tools()
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            builtins=frozenset(),
            attempt_policy="none",
            **{"answer_on_stop": False, **options},
        ),
        tool_providers=[tools],
    )
    await agent.start()
    try:
        outcome = await agent.run_task(query)
    finally:
        await agent.close()
    return outcome, llm, tools


async def test_a_lone_hand_off_ends_the_run_with_its_receipt() -> None:
    outcome, llm, tools = await _run(
        [
            _step(("note", {"text": "done first"})),
            _step(("move", {"what": "the rest", "rest": True})),
            # Never asked for: the run ended at the hand-off.
            _step(("note", {"text": "after"})),
        ]
    )
    assert tools.ran == ["note:done first", "move:the rest:True"]
    assert (outcome.status, outcome.result) == ("completed", "Handed on: the rest.")
    assert outcome.handed_off is True and outcome.to_dict()["handed_off"] is True
    assert llm.calls == 2


async def test_a_receipt_gets_no_closing_line_the_request_asked_an_answer_for() -> None:
    outcome, llm, _ = await _run(
        [_step(("move", {"what": "the rest", "rest": True}))],
        "Do the work. End with a final line that starts with FINAL ANSWER:",
        answer_on_stop=True,
    )
    assert outcome.result == "Handed on: the rest." and not outcome.format_fixed
    assert llm.calls == 1, "no request after the hand-off"


async def test_a_step_that_mixes_a_hand_off_with_another_call_runs_neither() -> None:
    outcome, llm, tools = await _run(
        [
            _step(
                ("note", {"text": "beside"}),
                ("move", {"what": "the rest", "rest": True}),
            ),
            Message(role="assistant", content="I kept the work here."),
        ]
    )
    assert tools.ran == []
    assert (outcome.status, outcome.result) == ("completed", "I kept the work here.")
    assert outcome.handed_off is False and "handed_off" not in outcome.to_dict()
    assert llm.calls == 2


async def test_the_same_tool_without_its_flag_is_an_ordinary_call() -> None:
    outcome, _, tools = await _run(
        [
            _step(
                ("move", {"what": "a side task", "rest": False}),
                ("note", {"text": "beside"}),
            ),
            Message(role="assistant", content="Both done."),
        ]
    )
    # Not a hand-off: both run, and the tool may not hand off from there.
    assert tools.ran == ["move:a side task:False", "note:beside"]
    assert outcome.result == "Both done." and outcome.handed_off is False


def test_outside_a_run_nothing_can_hand_off() -> None:
    assert can_hand_off() is False
    assert hand_off("receipt") == "receipt"
