"""The run's hold: while the person has the run, no model call and no time limit counts."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.agent._pause import Hold
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.contracts.pause import PAUSE, Resumed, RunPause
from localmelo.melo.contracts.providers import ProviderError
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.tests.agent.test_reflection_gate import _scoped_chat
from localmelo.tests.agent.test_tool_providers import (
    FakeLLM,
    _answer,
    _call,
    _provided_only,
)


class Timed(FakeLLM):
    """Records when each model call was made."""

    def __init__(self, replies: list[Message]) -> None:
        super().__init__(replies)
        self.at: list[float] = []

    async def chat(self, messages: list[Message], tools: Any = None) -> Message:
        self.at.append(time.monotonic())
        return await super().chat(messages, tools)


class Holding(BaseToolProvider):
    """A tool ``hand`` that hands the run to the person, as melo's browser does with a page."""

    def __init__(self, *, wait_s: float = 0.0, timeout_ms: float = 100) -> None:
        self.wait_s = wait_s
        self.timeout_ms = timeout_ms
        self.pause: RunPause | None = None
        self.ended = 0
        self.written: list[str] = []

    async def start(self) -> list[ProvidedTool]:
        async def hand(text: str = "") -> str:
            pause = PAUSE.get()
            assert pause is not None
            self.pause = pause
            pause.hold("browser", {"tab": "t1"}, on_end=self._ended)
            if self.wait_s:
                async with pause.suspended():
                    await asyncio.sleep(self.wait_s)
                pause.release("done")
            return "handed"

        async def write(text: str = "") -> str:
            self.written.append(text)
            return "written"

        words = {"type": "object", "properties": {"text": {"type": "string"}}}
        definition = ToolDef("hand", "hand the page to the person", words)
        return [
            ProvidedTool(definition, hand, timeout_ms=self.timeout_ms),
            ProvidedTool(ToolDef("write", "write a note", words), write),
        ]

    def _ended(self, how: Resumed) -> None:
        self.ended += 1

    async def close(self) -> None:
        pass


def _types(events: list[AgentEvent]) -> list[str]:
    return [e.type for e in events if e.type.startswith(("run.", "tool."))]


async def test_a_call_that_waits_for_the_person_outlasts_its_time_limit() -> None:
    llm = Timed([_call("hand", "x"), _answer("done")])
    tools = Holding(wait_s=0.3, timeout_ms=100)
    agent = Agent(llm=llm, options=_provided_only(), tool_providers=[tools])
    events: list[AgentEvent] = []
    result = await agent.run_task("sign in", on_event=events.append)
    assert result.status == "completed"
    finished = next(e for e in events if e.type == "tool.finished")
    assert finished.data["status"] == "success" and finished.data["output"] == "handed"
    assert _types(events)[1:5] == [
        "tool.started",
        "run.paused",
        "run.resumed",
        "tool.finished",
    ]
    paused = next(e for e in events if e.type == "run.paused")
    resumed = next(e for e in events if e.type == "run.resumed")
    assert paused.data == {"what": "browser", "detail": {"tab": "t1"}}
    assert resumed.data["how"] == "done" and resumed.data["held_s"] >= 0.3
    await agent.close()


async def test_a_hold_left_on_stops_the_next_model_call_until_it_is_released() -> None:
    llm = Timed([_call("hand", "x"), _answer("done")])
    tools = Holding()
    agent = Agent(llm=llm, options=_provided_only(), tool_providers=[tools])
    run = asyncio.ensure_future(agent.run_task("sign in", on_event=lambda e: None))
    async with asyncio.timeout(2):
        while tools.pause is None:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(llm.at) == 1, "no model call while the person has the run"
    released = time.monotonic()
    tools.pause.release("closed")
    result = await run
    assert result.status == "completed" and llm.at[1] >= released
    await agent.close()


async def test_a_stop_while_held_ends_the_run_and_tells_the_holder() -> None:
    llm = Timed([_call("hand", "x"), _answer("done")])
    tools = Holding()
    agent = Agent(llm=llm, options=_provided_only(), tool_providers=[tools])
    events: list[AgentEvent] = []
    run = asyncio.ensure_future(agent.run_task("sign in", on_event=events.append))
    async with asyncio.timeout(2):
        while tools.pause is None or not tools.pause.held:
            await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    resumed = [e.data["how"] for e in events if e.type == "run.resumed"]
    assert resumed == ["stopped"] and tools.ended == 1
    assert events[-1].type == "run.finished"
    assert events[-1].data["result"]["status"] == "cancelled"
    await agent.close()


async def test_a_hold_that_lands_after_the_step_began_stops_its_planning_call() -> None:
    """The gate is at the request itself: a take can land while the step gets ready."""
    llm = Timed([_call("hand", "x"), _answer("done")])
    tools = Holding()
    agent = Agent(llm=llm, options=_provided_only(), tool_providers=[tools])
    plan = agent._do_plan
    calls = [0]

    async def holding_on_the_second(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == 2 and tools.pause is not None:
            tools.pause.release("done")  # the tool's own hold is over: the step goes on
            tools.pause.hold("browser", {"tab": "t1"})  # and the panel's take lands now
        return await plan(*args, **kwargs)

    agent._do_plan = holding_on_the_second  # type: ignore[method-assign]
    run = asyncio.ensure_future(agent.run_task("sign in", on_event=lambda e: None))
    async with asyncio.timeout(2):
        while calls[0] < 2:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(llm.at) == 1, "no model call while the person has the run"
    assert tools.pause is not None
    tools.pause.release("done")
    result = await run
    assert result.status == "completed" and len(llm.at) == 2
    await agent.close()


async def test_a_hold_stops_the_next_tool_the_model_chose_until_it_is_released() -> (
    None
):
    """Two calls in one reply: the person takes the page during the first."""
    both = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(tool_name="hand", arguments={"text": "x"}),
            ToolCall(tool_name="write", arguments={"text": "y"}),
        ],
    )
    llm = Timed([both, _answer("done")])
    tools = Holding()
    agent = Agent(llm=llm, options=_provided_only(), tool_providers=[tools])
    events: list[AgentEvent] = []
    run = asyncio.ensure_future(agent.run_task("sign in", on_event=events.append))
    async with asyncio.timeout(2):
        while tools.pause is None or not tools.pause.held:
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    started = [e.data["name"] for e in events if e.type == "tool.started"]
    assert tools.written == [] and started == ["hand"], "the second call waits"
    tools.pause.release("done")
    result = await run
    assert result.status == "completed" and tools.written == ["y"]
    await agent.close()


async def test_each_retry_of_a_request_waits_while_the_run_is_held() -> None:
    """A take that lands during a retry's delay stops the next send."""
    calls: list[bool] = []
    hold = Hold(lambda what, detail: None, lambda what, how, held_s: None)

    class Busy(FakeLLM):
        async def chat(self, messages: list[Message], tools: Any = None) -> Message:
            calls.append(hold.held)
            if len(calls) == 1:
                raise ProviderError("busy", status_code=503, retryable=True)
            return Message(role="assistant", content="ok")

    async def delay(seconds: float) -> None:
        hold.hold("browser", {"tab": "t1"})

    chat = _scoped_chat(Busy(), retry_delays=[0.0], sleep=delay)
    token = PAUSE.set(hold)
    try:
        request = asyncio.ensure_future(
            chat._request([Message(role="user", content="q")], tools=None)
        )
        await asyncio.sleep(0.2)
        assert calls == [False], "no send while held"
        hold.release("done")
        await asyncio.wait_for(request, 2)
    finally:
        PAUSE.reset(token)
    assert calls == [False, False]


async def test_a_waiter_woken_by_a_release_stays_for_the_hold_that_came_next() -> None:
    hold = Hold(lambda what, detail: None, lambda what, how, held_s: None)
    hold.hold("browser", {"tab": "t1"})
    waiter = asyncio.ensure_future(hold.wait())
    await asyncio.sleep(0.01)
    hold.release("done")
    hold.hold("browser", {"tab": "t2"})  # a new take before the waiter runs
    await asyncio.sleep(0.05)
    assert not waiter.done()
    hold.release("done")
    await asyncio.wait_for(waiter, 1)
