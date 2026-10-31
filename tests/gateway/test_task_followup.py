"""A message to a task: read at its next step while it runs, a new turn once it ended.

The store's reopen queues a finished task again with the message as its next
turn; real workers on a scripted model show that the new turn sees the
earlier one, in the same session, and that a message sent while the task
runs reaches its next step. No model server or network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, RunResult, ToolCall, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
from localmelo.support.gateway.workers import TaskInbox, TaskWorkers
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore

Reply = Callable[[list[Message]], Awaitable[Message]]


async def eventually(check: Callable[[], Awaitable[bool]]) -> None:
    async with asyncio.timeout(5):
        while not await check():
            await asyncio.sleep(0.01)


# ── The store ──


@pytest.fixture
async def tasks(tmp_path: Path) -> AsyncIterator[TaskStore]:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    opened = TaskStore(store)
    await opened.open()
    try:
        yield opened
    finally:
        await store.aclose()


async def _made(tasks: TaskStore, tmp_path: Path, **spec: Any) -> str:
    payload = {"objective": "Write the report", **spec}
    record, _ = await tasks.create_task(
        TaskSpec.from_payload(payload), workspace=str(tmp_path), origin="user"
    )
    return record.task_id


async def test_reopen_queues_a_finished_task_with_its_next_turn(
    tasks: TaskStore, tmp_path: Path
) -> None:
    task_id = await _made(tasks, tmp_path)
    store = tasks.store
    turn = await store.begin_turn(task_id, "Write the report", holder="h")
    await store.finish_turn(
        task_id,
        turn.turn,
        holder="h",
        status="completed",
        cause="run",
        result=RunResult(task_id="run-1", status="completed", result="ok").to_dict(),
        answer="ok",
    )
    await tasks.change(task_id, state="done", actor="scheduler")
    with pytest.raises(ValueError, match="empty"):
        await tasks.reopen(task_id, "  ")
    record = await tasks.reopen(task_id, "commit it")
    assert (record.state, record.card, record.finished_at) == (
        "queued",
        "commit it",
        None,
    )
    assert (record.level, record.reason) == (2, "")
    # A restart before the new turn starts re-admits it: turn 2 is reserved.
    assert await tasks.resume_turn(task_id) == 2
    assert await tasks.resume_waiting_turn(task_id)
    # Only a finished task continues.
    with pytest.raises(TaskStoreError, match="only a done, failed or stopped"):
        await tasks.reopen(task_id, "again")


async def test_reopen_takes_the_persons_level_or_keeps_its_own(
    tasks: TaskStore, tmp_path: Path
) -> None:
    task_id = await _made(tasks, tmp_path)
    await tasks.change(task_id, state="cancelled")
    record = await tasks.reopen(task_id, "now, urgently", level=1, reason="urgent")
    assert (record.level, record.reason) == (1, "urgent")
    with pytest.raises(ValueError):
        await tasks.reopen(task_id, "x", level=0)
    # A paused (Later) task that was stopped goes back to Normal.
    paused = await _made(tasks, tmp_path, objective="Tidy photos")
    await tasks.pause(paused)
    await tasks.change(paused, state="cancelled")
    assert (await tasks.reopen(paused, "go on")).level == 2


def test_the_inbox_hands_each_message_once() -> None:
    inbox = TaskInbox()
    inbox.validate("/anywhere")
    inbox.put("one")
    inbox.put("two")
    assert inbox.poll() == [(1, "one"), (2, "two")]
    inbox.accept(1)
    assert inbox.poll() == [(2, "two")]
    assert inbox.drain() == ["two"] and inbox.poll() == []


# ── Real workers on a scripted model ──


class Scripted(BaseLLMProvider):
    """Answers each request with the next reply function; keeps every request."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests: list[list[Message]] = []

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
        agent = Agent(
            llm=kwargs["wrap_llm"](llm),
            options=session_options(kwargs["options"], kwargs["workspace"]),
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


async def _ended(workers: TaskWorkers, task_id: str, turns: int) -> None:
    async def check() -> bool:
        record = await workers.tasks.get_task(task_id)
        done = len(await workers.store.list_turns(task_id, limit=10)) == turns
        return record.state == "done" and done and task_id not in workers._runs

    await eventually(check)


def _texts(messages: list[Message]) -> str:
    return "\n".join(message.content for message in messages)


async def test_a_follow_up_continues_a_done_task_in_its_own_session(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    llm.replies = [says("Changed registry.py; not committed yet."), says("Committed.")]
    row, _ = await pool.create(
        {"prompt": "Fix registry.py", "workspace": str(workspace)}
    )
    task_id = row["task_id"]
    await _ended(pool, task_id, 1)
    body, how = await pool.message(task_id, "finish it: commit registry.py")
    assert (how, body["task_id"], body["state"]) == ("continued", task_id, "queued")
    await _ended(pool, task_id, 2)
    # Turn 2 sees turn 1: its request and its answer come before the message.
    second = _texts(llm.requests[1])
    assert second.index("Fix registry.py") < second.index("not committed yet")
    assert second.index("not committed yet") < second.index("finish it: commit")
    turns = await pool.store.list_turns(task_id, limit=10)
    assert [t.user_text for t in turns] == [
        "Fix registry.py",
        "finish it: commit registry.py",
    ]
    assert [t.answer for t in turns][-1] == "Committed."
    assert len(await pool.tasks.list_tasks()) == 1


async def test_a_message_to_a_running_task_arrives_at_its_next_step(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    holder: dict[str, str] = {}

    async def first(messages: list[Message]) -> Message:
        body, how = await pool.message(holder["id"], "also count the files")
        assert (how, body["state"]) == ("queued", "running")
        return await lists_files()(messages)

    llm.replies = [first, says("Two files.")]
    row, _ = await pool.create(
        {"prompt": "List my folder", "workspace": str(workspace)}
    )
    holder["id"] = task_id = row["task_id"]
    await _ended(pool, task_id, 1)
    assert "also count the files" not in _texts(llm.requests[0])
    assert "also count the files" in _texts(llm.requests[1])
    events = await pool.store.list_events(task_id)
    accepted = [e.event["data"] for e in events if e.event["type"] == "inbox.accepted"]
    assert [data["text"] for data in accepted] == ["also count the files"]


async def test_a_message_after_the_last_step_goes_on_as_the_next_turn(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    holder: dict[str, str] = {}

    async def last(messages: list[Message]) -> Message:
        await pool.message(holder["id"], "and commit it")
        return Message(role="assistant", content="Fixed.")

    llm.replies = [last, says("Committed.")]
    row, _ = await pool.create({"prompt": "Fix it", "workspace": str(workspace)})
    holder["id"] = task_id = row["task_id"]
    await _ended(pool, task_id, 2)
    turns = await pool.store.list_turns(task_id, limit=10)
    assert [t.user_text for t in turns] == ["Fix it", "and commit it"]
    assert task_id not in pool._inboxes


async def test_a_task_stopped_before_it_ran_continues_with_its_request(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    row, _ = await pool.create(
        {"prompt": "Tidy photos", "workspace": str(workspace), "level": "P5"}
    )
    task_id = row["task_id"]
    await pool.update(task_id, state="stop")
    llm.replies = [says("Tidied.")]
    _, how = await pool.message(task_id, "do it now")
    assert how == "continued"
    await _ended(pool, task_id, 1)
    assert "Tidy photos\n\ndo it now" in _texts(llm.requests[0])


async def test_messages_a_task_cannot_take(workers: Setup) -> None:
    pool, _, workspace = workers
    row, _ = await pool.create(
        {"prompt": "Later", "workspace": str(workspace), "level": "P5"}
    )
    task_id = row["task_id"]
    with pytest.raises(ValueError, match="empty"):
        await pool.message(task_id, "   ")
    with pytest.raises(ValueError, match="at most"):
        await pool.message(task_id, "x" * 9000)
    # A paused task reads it when it resumes.
    _, how = await pool.message(task_id, "use the new folder")
    assert how == "queued"
    assert pool._inboxes[task_id].poll() == [(1, "use the new folder")]
    # A stop drops what it never read.
    await pool.update(task_id, state="stop")
    assert task_id not in pool._inboxes


async def test_a_restart_before_the_follow_up_ran_starts_it_again(
    workers: Setup,
) -> None:
    pool, llm, workspace = workers
    row, _ = await pool.create(
        {"prompt": "Tidy photos", "workspace": str(workspace), "level": "P5"}
    )
    task_id = row["task_id"]
    await pool.update(task_id, state="stop")
    # Queued in the store, then the gateway stops before the turn starts.
    await pool.tasks.reopen(task_id, "Tidy photos\n\ndo it now")
    await pool.aclose()
    gate = ChatGate(policy="off")
    again = TaskWorkers(pool.store, gate, pool.settings, pool.hub)
    llm.replies = [says("Tidied.")]
    try:
        await again.open()
        await _ended(again, task_id, 1)
        assert "do it now" in _texts(llm.requests[0])
    finally:
        await again.aclose()
        await gate.close()
