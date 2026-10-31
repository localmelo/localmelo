"""A task starts with what its chat knows: the pack is made once, at creation.

Real workers on a scripted model show that the pack reaches a task's first
request under its heading and stays out of the turn the transcript shows,
that a continue reuses the stored pack, that a repeated keyed request does
not make it again, and that a task the person did not make, or one whose
notes cannot be read, starts without one. The notes are a stand-in that
answers ``context_pack`` as ChatMemory does. No model server or network.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.task_policy import TaskSpec, is_persons_task
from localmelo.support.gateway.workers import TASK_PACK_HEADER, TaskWorkers
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.service import PACK_HEADER
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore

Reply = Callable[[list[Message]], Awaitable[Message]]
PACK_BODY = "- The person lives in Utrecht.\n- This folder uses uv."


class Notes:
    """Stands for ChatMemory.context_pack: records each call and answers *pack*."""

    def __init__(self, pack: str) -> None:
        self.pack = pack
        self.error: Exception | None = None
        self.calls: list[tuple[str, str, str | None]] = []

    async def context_pack(
        self,
        objective: str,
        chat_id: str,
        budget_chars: int = 1500,
        *,
        workspace: str | None = None,
    ) -> str:
        self.calls.append((objective, chat_id, workspace))
        if self.error is not None:
            raise self.error
        return self.pack


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


@dataclass
class Rig:
    workers: TaskWorkers
    llm: Scripted
    workspace: Path
    notes: Notes


@pytest.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rig]:
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
    notes = Notes(PACK_HEADER + "\n" + PACK_BODY)

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](llm),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    opened = TaskWorkers(store, gate, lambda: cfg, hub, memory=cast(ChatMemory, notes))
    await opened.open()
    try:
        yield Rig(opened, llm, workspace, notes)
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

    async with asyncio.timeout(5):
        while not await check():
            await asyncio.sleep(0.01)


def _texts(messages: list[Message]) -> str:
    return "\n".join(message.content for message in messages)


async def test_the_pack_reaches_the_first_request_under_its_heading(
    rig: Rig,
) -> None:
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "Fix registry.py", "workspace": str(rig.workspace)}
    )
    task_id = row["task_id"]
    await _ended(rig.workers, task_id, 1)
    first = _texts(rig.llm.requests[0])
    assert f"{TASK_PACK_HEADER}\n{PACK_BODY}" in first
    assert first.index("Fix registry.py") < first.index(TASK_PACK_HEADER)
    # The notes' own heading is replaced, not repeated under ours.
    assert PACK_HEADER not in first
    # The pack is made once, for the objective, in the task's folder.
    assert rig.notes.calls == [("Fix registry.py", "", os.path.realpath(rig.workspace))]
    # The transcript keeps the person's words only; the pack is stored with the spec.
    turns = await rig.workers.store.list_turns(task_id, limit=10)
    assert [turn.user_text for turn in turns] == ["Fix registry.py"]
    task = await rig.workers.get(task_id)
    assert task["spec"]["context_pack"] == PACK_BODY


async def test_no_notes_means_no_heading_and_no_stored_pack(rig: Rig) -> None:
    rig.notes.pack = ""
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    assert TASK_PACK_HEADER not in _texts(rig.llm.requests[0])
    task = await rig.workers.get(row["task_id"])
    assert "context_pack" not in task["spec"]


async def test_memory_off_means_no_heading_and_no_read(rig: Rig) -> None:
    rig.workers.memory = None
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    assert rig.notes.calls == []
    assert TASK_PACK_HEADER not in _texts(rig.llm.requests[0])


async def test_a_task_the_person_did_not_make_gets_no_pack(rig: Rig) -> None:
    job = rig.workspace / "job"
    job.mkdir()
    rig.llm.replies = [says("Done.")]
    spec = TaskSpec.from_payload(
        {"objective": "Sweep the cache", "workspace": str(job)}
    )
    task, _ = await rig.workers.create_spec(spec, origin="agent")
    await _ended(rig.workers, task["task_id"], 1)
    assert rig.notes.calls == []
    assert TASK_PACK_HEADER not in _texts(rig.llm.requests[0])


async def test_a_saved_chore_carries_the_pack_the_person_made(rig: Rig) -> None:
    assert is_persons_task("user") and is_persons_task("user_rule")
    assert not any(is_persons_task(o) for o in ("front", "harness", "agent", "sleep"))
    job = rig.workspace / "weekly"
    job.mkdir()
    rig.llm.replies = [says("Done.")]
    spec = TaskSpec.from_payload(
        {"objective": "Summarise the week", "workspace": str(job)}
    )
    task, _ = await rig.workers.create_spec(spec, origin="user_rule")
    await _ended(rig.workers, task["task_id"], 1)
    assert f"{TASK_PACK_HEADER}\n{PACK_BODY}" in _texts(rig.llm.requests[0])


async def test_a_failed_read_of_the_notes_still_starts_the_task(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    rig.notes.error = RuntimeError("the index is down")
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    task_id = row["task_id"]
    await _ended(rig.workers, task_id, 1)
    assert (await rig.workers.get(task_id))["state"] == "done"
    assert TASK_PACK_HEADER not in _texts(rig.llm.requests[0])
    assert "Could not read the chat's notes" in caplog.text


async def test_a_continue_reuses_the_stored_pack(rig: Rig) -> None:
    rig.llm.replies = [says("Done."), says("Finished.")]
    row, _ = await rig.workers.create(
        {"prompt": "Fix registry.py", "workspace": str(rig.workspace)}
    )
    task_id = row["task_id"]
    await _ended(rig.workers, task_id, 1)
    rig.notes.pack = PACK_HEADER + "\n- Changed after the task started."
    body, how = await rig.workers.message(task_id, "finish it")
    assert how == "continued"
    await _ended(rig.workers, task_id, 2)
    second = _texts(rig.llm.requests[1])
    assert f"{TASK_PACK_HEADER}\n{PACK_BODY}" in second
    assert "Changed after the task started" not in second
    assert len(rig.notes.calls) == 1


async def test_a_repeated_keyed_request_keeps_the_first_pack(rig: Rig) -> None:
    rig.llm.replies = [says("Done.")]
    request = {
        "prompt": "Fix registry.py",
        "workspace": str(rig.workspace),
        "request_key": "same-request",
    }
    first, repeated_first = await rig.workers.create(dict(request))
    await _ended(rig.workers, first["task_id"], 1)
    again, repeated = await rig.workers.create(dict(request))
    assert (repeated_first, repeated) == (False, True)
    assert again["task_id"] == first["task_id"]
    assert len(rig.notes.calls) == 1


async def test_a_restart_reuses_the_stored_pack(rig: Rig) -> None:
    rig.llm.replies = [says("Done."), says("Finished.")]
    row, _ = await rig.workers.create(
        {"prompt": "Fix registry.py", "workspace": str(rig.workspace)}
    )
    task_id = row["task_id"]
    await _ended(rig.workers, task_id, 1)
    old = rig.workers
    await old.aclose()
    # A gateway started again on the same store reads the pack from the task.
    restarted = TaskWorkers(
        old.store,
        old.gate,
        old.settings,
        old.hub,
        memory=cast(ChatMemory, rig.notes),
    )
    await restarted.open()
    try:
        rig.notes.pack = PACK_HEADER + "\n- Changed after the restart."
        _, how = await restarted.message(task_id, "finish it")
        assert how == "continued"
        await _ended(restarted, task_id, 2)
        second = _texts(rig.llm.requests[1])
        assert f"{TASK_PACK_HEADER}\n{PACK_BODY}" in second
        assert "Changed after the restart" not in second
        assert len(rig.notes.calls) == 1
    finally:
        await restarted.aclose()
