"""A continued task starts its next run: the worker's tool-capture hook finds its execution.

The worker's Agent has a tool provider, so the Agent runs the capture hook
before the first tool starts. A first start has its capture row from the
recovery launch; a task continued by a message is an ordinary start whose
row is made only when the task goes to running. These tests build the Agent
with the real worker tool providers (the other follow-up tests leave them
out) on a scripted model. No model server or network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.finishing import OUT_OF_STEPS
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_store import TaskStoreError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore

Reply = Callable[[list[Message]], Awaitable[Message]]


class Scripted(BaseLLMProvider):
    """Answers each request with the next reply function; keeps every request."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests: list[list[Message]] = []
        # When set, the next agent the workers build fails to start with it.
        self.start_error: Exception | None = None
        # The task's state and its current capture's execution, as each agent
        # was being built (before any of its tools started).
        self.builds: list[tuple[str, str | None]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        return await self.replies.pop(0)(messages)

    async def close(self) -> None:
        pass


def says(text: str) -> Reply:
    async def reply(messages: list[Message]) -> Message:
        return Message(role="assistant", content=text)

    return reply


def lists_files() -> Reply:
    async def reply(messages: list[Message]) -> Message:
        call = ToolCall(tool_name="list_files", arguments={"path": "."}, id="c1")
        return Message(role="assistant", content="", tool_calls=[call])

    return reply


Setup = tuple[TaskWorkers, Scripted, Path]


@pytest.fixture
async def workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Setup]:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = Config()
    cfg.gateway.workspace = str(workspace)
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    llm = Scripted()

    async def build(**kwargs: Any) -> Agent:
        if llm.start_error is not None:
            raise llm.start_error
        (record,) = await opened.tasks.list_tasks()
        active = await RecoveryStore(store).active_capture(record.task_id)
        llm.builds.append((record.state, active))
        # Two steps and no reflection, so a model that keeps calling tools
        # ends the run out of steps; the worker's own tool providers stay.
        options = dataclasses.replace(
            session_options(kwargs["options"], kwargs["workspace"]),
            max_steps=2,
            attempt_policy="none",
            answer_on_stop=False,  # these runs end out of steps; no extra request
        )
        agent = Agent(
            llm=kwargs["wrap_llm"](llm),
            options=options,
            tool_providers=list(kwargs["tools"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    opened = TaskWorkers(store, gate, lambda: cfg, hub)
    await opened.open()
    try:
        yield opened, llm, workspace
    finally:
        await opened.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def _settled(pool: TaskWorkers, task_id: str, turns: int, state: str) -> None:
    async def check() -> bool:
        record = await pool.tasks.get_task(task_id)
        count = len(await pool.store.list_turns(task_id, limit=10))
        return record.state == state and count == turns and task_id not in pool._runs

    async with asyncio.timeout(5):
        while not await check():
            await asyncio.sleep(0.01)


async def test_a_done_task_continues_to_a_second_turn(workers: Setup) -> None:
    pool, llm, workspace = workers
    llm.replies = [lists_files(), says("Fixed; not committed."), says("Committed.")]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "done")
    _, how = await pool.message(task_id, "commit it")
    assert how == "continued"
    await _settled(pool, task_id, 2, "done")
    turns = await pool.store.list_turns(task_id, limit=10)
    assert [t.answer for t in turns][-1] == "Committed."


async def test_a_task_out_of_steps_continues_to_a_second_turn(workers: Setup) -> None:
    pool, llm, workspace = workers
    # The first run keeps calling a tool until its two steps are used up.
    llm.replies = [lists_files(), lists_files(), says("Committed.")]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "failed")
    _, how = await pool.message(task_id, "commit what you have")
    assert how == "continued"
    await _settled(pool, task_id, 2, "done")
    turns = await pool.store.list_turns(task_id, limit=10)
    assert [t.user_text for t in turns][-1] == "commit what you have"
    assert [t.answer for t in turns][-1] == "Committed."
    # Each run was running, with a capture of its own, before its tools set up.
    (first_state, first), (second_state, second) = llm.builds
    assert (first_state, second_state) == ("running", "running")
    assert first and second and first != second


async def test_a_run_out_of_steps_is_stopped_and_continues_from_its_page(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    llm.replies = [lists_files(), lists_files(), says("Committed.")]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "failed")
    # Still a failed task in the store, with the reason that makes it a stop.
    record = await pool.tasks.get_task(task_id)
    assert (record.state, record.wait_reason) == ("failed", OUT_OF_STEPS)
    body = await pool.get(task_id)
    assert body["out_of_steps"] is True
    _, how = await pool.continue_stopped(task_id)
    assert how == "continued"
    queued = await pool.tasks.get_task(task_id)
    assert queued.wait_reason == "" and queued.card
    await _settled(pool, task_id, 2, "done")
    assert (await pool.get(task_id))["out_of_steps"] is False
    # A task that is done has nothing to continue from here.
    with pytest.raises(TaskStoreError, match="ran out of steps"):
        await pool.continue_stopped(task_id)


async def test_a_run_that_could_not_start_is_not_a_stop(workers: Setup) -> None:
    pool, llm, workspace = workers
    llm.replies = [lists_files(), lists_files()]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "failed")
    llm.start_error = RuntimeError("the model server is not answering")
    await pool.message(task_id, "commit what you have")
    await _settled(pool, task_id, 1, "failed")
    body = await pool.get(task_id)
    assert body["out_of_steps"] is False
    with pytest.raises(TaskStoreError, match="ran out of steps"):
        await pool.continue_stopped(task_id)


async def test_a_failed_task_continues_after_a_second_message(workers: Setup) -> None:
    pool, llm, workspace = workers
    llm.replies = [
        lists_files(),
        lists_files(),
        lists_files(),
        lists_files(),
        says("Done at last."),
    ]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "failed")
    # Out of steps again on the continued run: it still starts a third time.
    await pool.message(task_id, "try once more")
    await _settled(pool, task_id, 2, "failed")
    await pool.message(task_id, "and again")
    await _settled(pool, task_id, 3, "done")


async def test_a_continued_run_that_cannot_start_says_so_and_the_next_one_can(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    llm.replies = [lists_files(), lists_files(), says("Committed.")]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "failed")
    llm.start_error = RuntimeError("the model server is not answering\nsecond line")
    await pool.message(task_id, "commit what you have")
    await _settled(pool, task_id, 1, "failed")
    record = await pool.tasks.get_task(task_id)
    assert record.wait_reason == (
        "Could not continue: the model server is not answering second line"
    )
    # The failure is not sticky: the person sends again and the run starts.
    llm.start_error = None
    await pool.message(task_id, "try again")
    await _settled(pool, task_id, 2, "done")
    assert (await pool.tasks.get_task(task_id)).wait_reason == ""


async def test_a_stop_before_a_continued_run_starts_ends_it_cancelled(
    workers: Setup, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pool, llm, workspace = workers
    llm.replies = [says("Fixed; not committed.")]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    task_id = row["task_id"]
    await _settled(pool, task_id, 1, "done")
    # Hold the continued run just before its start, as a Stop would find it.
    reached, go = asyncio.Event(), asyncio.Event()
    original = RecoveryStore.provenance

    async def held(self: RecoveryStore, task: str) -> Any:
        reached.set()
        await go.wait()
        return await original(self, task)

    monkeypatch.setattr(RecoveryStore, "provenance", held)
    await pool.message(task_id, "commit it")
    await asyncio.wait_for(reached.wait(), 3)
    assert (await pool.tasks.get_task(task_id)).state == "queued"
    await pool.update(task_id, state="stop")
    go.set()
    with caplog.at_level(logging.ERROR, logger="localmelo.support.gateway.workers"):
        await _settled(pool, task_id, 1, "cancelled")
    # Cancelled, not failed: no turn began, no model call, no failure to show.
    record = await pool.tasks.get_task(task_id)
    assert not record.wait_reason.startswith("Could not continue")
    events = await pool.store.list_events(task_id)
    ended = [
        e.event["data"]["state"]
        for e in events
        if e.event["type"].startswith("sched.f")
    ]
    assert ended == ["done"]
    assert len(llm.requests) == 1
    assert not caplog.records
