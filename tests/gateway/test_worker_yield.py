"""A full set of workers makes room for a higher-level queued task at a step boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway import workers as workers_module
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.scheduler.model import Machine
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore


async def eventually(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.001)


class Rig:
    """Workers with one slot and a model that takes its orders from the test."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.hold: dict[tuple[str, int], asyncio.Event] = {}
        self.tool_calls = 0
        self.read: Path | None = None
        self.write: Path | None = None
        self.workers: TaskWorkers | None = None
        self.workspace: Path | None = None
        self.gate: ChatGate | None = None

    def hold_call(self, name: str, step: int) -> asyncio.Event:
        self.hold[(name, step)] = asyncio.Event()
        return self.hold[(name, step)]

    def count(self, name: str) -> int:
        return self.calls.count(name)


@pytest.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rig]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(workers_module, "MAX_WORKERS", 1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    notes = workspace / "notes.txt"
    notes.write_text("notes")
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="class", n_srv=8, lock_gate="wait")
    hub = SessionHub(store)
    state = Rig()
    state.workspace, state.read, state.gate = workspace, notes, gate

    class Script(BaseLLMProvider):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            text = " ".join(str(m.content) for m in messages)
            name = "low" if "low-level-work" in text else "high"
            step = state.count(name)
            state.calls.append(name)
            wait = state.hold.get((name, step))
            if wait is not None:
                await wait.wait()
            if name == "low" and step == 0 and state.write is not None:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="file_write",
                            arguments={"path": str(state.write), "content": "x"},
                            id="write-1",
                        )
                    ],
                )
            if name == "low" and step == 0:
                state.tool_calls += 1
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="file_read",
                            arguments={"path": str(notes)},
                            id="read-1",
                        )
                    ],
                )
            return Message(role="assistant", content="done")

        async def close(self) -> None:
            pass

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](Script()),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    state.workers = workers
    await workers.open()
    try:
        yield state
    finally:
        for event in state.hold.values():
            event.set()
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def make(rig: Rig, text: str, level: str) -> str:
    assert rig.workers is not None and rig.workspace is not None
    row, _ = await rig.workers.create(
        {"prompt": text, "workspace": str(rig.workspace), "level": level}
    )
    return str(row["task_id"])


async def state_of(rig: Rig, task_id: str) -> str:
    assert rig.workers is not None
    return str((await rig.workers.get(task_id))["state"])


async def until(rig: Rig, task_id: str, state: str) -> None:
    async with asyncio.timeout(5):
        while await state_of(rig, task_id) != state:
            await asyncio.sleep(0.005)


async def test_a_lower_worker_parks_at_its_next_step_for_a_higher_queued_task(
    rig: Rig,
) -> None:
    assert rig.workers is not None
    first_call = rig.hold_call("low", 0)
    low = await make(rig, "low-level-work", "P3")
    await eventually(lambda: rig.count("low") == 1)
    high = await make(rig, "high-level-work", "P2")
    # The slot is taken: the higher task waits and the lower one is asked to park.
    await eventually(lambda: low in rig.workers._yielding)  # type: ignore[union-attr]
    assert await state_of(rig, high) == "queued"
    assert rig.count("high") == 0
    # The lower worker's call and its tool finish first; it never stops mid-tool.
    first_call.set()
    await until(rig, high, "done")
    # The lower worker made no second call before the higher task was done.
    assert rig.calls[:2] == ["low", "high"]
    await until(rig, low, "done")
    assert rig.calls == ["low", "high", "low"]
    assert rig.tool_calls == 1
    assert not rig.workers._yielding


async def test_workers_of_equal_level_never_displace_each_other(rig: Rig) -> None:
    assert rig.workers is not None
    first_call = rig.hold_call("low", 0)
    low = await make(rig, "low-level-work", "P2")
    await eventually(lambda: rig.count("low") == 1)
    high = await make(rig, "high-level-work", "P2")
    await asyncio.sleep(0.05)
    assert not rig.workers._yielding
    assert rig.count("high") == 0
    first_call.set()
    await until(rig, low, "done")
    await until(rig, high, "done")
    assert rig.calls == ["low", "low", "high"]


async def test_a_person_pausing_the_lower_worker_keeps_it_paused(rig: Rig) -> None:
    assert rig.workers is not None
    first_call = rig.hold_call("low", 0)
    low = await make(rig, "low-level-work", "P3")
    await eventually(lambda: rig.count("low") == 1)
    high = await make(rig, "high-level-work", "P2")
    await eventually(lambda: low in rig.workers._yielding)  # type: ignore[union-attr]
    await rig.workers.update(low, state="pause")
    assert low not in rig.workers._yielding
    first_call.set()
    await until(rig, high, "done")
    await until(rig, low, "parked")
    await asyncio.sleep(0.05)
    assert await state_of(rig, low) == "parked"
    assert rig.count("low") == 1


async def test_a_worker_waiting_for_an_answer_is_left_alone(rig: Rig) -> None:
    assert rig.workers is not None and rig.workspace is not None
    rig.write = rig.workspace / "out.txt"
    low = await make(rig, "low-level-work", "P3")
    await eventually(lambda: bool(rig.workers.broker.pending(low)))  # type: ignore[union-attr]
    high = await make(rig, "high-level-work", "P2")
    await asyncio.sleep(0.05)
    # The person is deciding: the lower worker keeps its slot and its question.
    assert not rig.workers._yielding
    assert rig.workers.broker.pending(low)
    assert rig.count("high") == 0 and await state_of(rig, high) == "queued"
    request = rig.workers.broker.pending(low)[0].request_id
    assert rig.workers.broker.answer(request, ApprovalDecision(True)) == "resolved"
    await until(rig, low, "done")
    await until(rig, high, "done")


async def test_a_held_lower_worker_parks_at_once_and_the_higher_one_goes_first(
    rig: Rig,
) -> None:
    assert rig.workers is not None and rig.gate is not None
    # Another job has the model: both workers' first calls are held.
    rig.gate.set_machine(Machine(lock="foreign"))
    low = await make(rig, "low-level-work", "P3")
    await eventually(lambda: rig.gate.held(low) == "Reservation")  # type: ignore[union-attr]
    high = await make(rig, "high-level-work", "P2")
    await eventually(lambda: rig.gate.held(high) == "Reservation")  # type: ignore[union-attr]
    assert low not in rig.workers._runs and high in rig.workers._runs
    await until(rig, low, "queued")
    rig.gate.set_machine(Machine(lock="none"))
    await until(rig, high, "done")
    await until(rig, low, "done")
    assert rig.calls == ["high", "low", "low"]
