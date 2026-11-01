"""A task starts with what its chat knows: the pack is made once, at creation.

The project's own instruction files (``AGENTS.md``, ``CLAUDE.md``,
``GEMINI.md`` of the task's folder, read-only) are read at the same moment and
ride the same way, under their own heading after the pack.

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
from localmelo.support.gateway.workers import (
    PROJECT_HEADER,
    TASK_PACK_HEADER,
    TaskWorkers,
)
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


# ── The project's own instruction files ──


def _instructions_read(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, dict[str, Any]]]:
    """Every call the workers make to the finder, answered by the real one."""
    from localmelo.support.memory import project_files

    calls: list[tuple[str, dict[str, Any]]] = []
    real = project_files.instructions

    def spy(folder: str, **kwargs: Any) -> Any:
        calls.append((folder, kwargs))
        return real(folder, **kwargs)

    monkeypatch.setattr(project_files, "instructions", spy)
    return calls


async def test_the_projects_file_follows_the_pack_on_the_first_request(
    rig: Rig,
) -> None:
    (rig.workspace / "AGENTS.md").write_text("Test with `uv run pytest`.\n")
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "Fix registry.py", "workspace": str(rig.workspace)}
    )
    task_id = row["task_id"]
    await _ended(rig.workers, task_id, 1)
    first = _texts(rig.llm.requests[0])
    heading = PROJECT_HEADER.format(paths="AGENTS.md")
    block = f"{heading}\n--- AGENTS.md ---\nTest with `uv run pytest`."
    assert block in first
    assert "follow them for how this project is built, tested and laid out" in heading
    assert "the person's words, melo's rules and approvals come first" in heading
    # The person's words, then the pack, then the project's files.
    assert (
        first.index("Fix registry.py")
        < first.index(TASK_PACK_HEADER)
        < first.index(heading)
    )
    # Neither is in the transcript; both are stored with the spec.
    turns = await rig.workers.store.list_turns(task_id, limit=10)
    assert [turn.user_text for turn in turns] == ["Fix registry.py"]
    spec = (await rig.workers.get(task_id))["spec"]
    assert (
        spec["project_instructions"] == "--- AGENTS.md ---\nTest with `uv run pytest`."
    )
    assert spec["project_files"] == [{"path": "AGENTS.md", "chars": 26, "cut": False}]
    assert "project_skipped" not in spec


async def test_the_pack_alone_when_the_folder_has_no_file(rig: Rig) -> None:
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    first = _texts(rig.llm.requests[0])
    assert f"{TASK_PACK_HEADER}\n{PACK_BODY}" in first
    assert "this project's notes for assistants" not in first
    spec = (await rig.workers.get(row["task_id"]))["spec"]
    assert not [key for key in spec if key.startswith("project_")]


async def test_the_file_is_read_without_notes_and_the_pack_is_not_needed(
    rig: Rig,
) -> None:
    rig.workers.memory = None
    (rig.workspace / "CLAUDE.md").write_text("Lint with ruff.")
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    first = _texts(rig.llm.requests[0])
    assert TASK_PACK_HEADER not in first
    assert PROJECT_HEADER.format(paths="CLAUDE.md") + "\n--- CLAUDE.md ---" in first


async def test_the_nearest_file_comes_first_and_the_heading_names_each(
    rig: Rig,
) -> None:
    pkg = rig.workspace / "pkg"
    pkg.mkdir()
    (rig.workspace / "AGENTS.md").write_text("Root rule.")
    (pkg / "CLAUDE.md").write_text("Package rule.")
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create({"prompt": "Fix it", "workspace": str(pkg)})
    await _ended(rig.workers, row["task_id"], 1)
    first = _texts(rig.llm.requests[0])
    assert PROJECT_HEADER.format(paths="CLAUDE.md, ../AGENTS.md") in first
    assert first.index("Package rule.") < first.index("Root rule.")


async def test_the_setting_off_reads_nothing(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _instructions_read(rig, monkeypatch)
    (rig.workspace / "AGENTS.md").write_text("Test with pytest.")
    rig.workers.settings().memory.project_instructions = "off"
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "Fix registry.py", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    assert calls == []
    first = _texts(rig.llm.requests[0])
    assert "this project's notes for assistants" not in first
    assert f"{TASK_PACK_HEADER}\n{PACK_BODY}" in first, "the pack is not the setting's"
    spec = (await rig.workers.get(row["task_id"]))["spec"]
    assert not [key for key in spec if key.startswith("project_")]


async def test_the_homes_own_file_is_never_given_to_a_task(rig: Rig) -> None:
    # The rig's home is the folder that holds the workspace.
    (rig.workspace.parent / "AGENTS.md").write_text("The person's own file.")
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "Fix registry.py", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    assert "The person's own file." not in _texts(rig.llm.requests[0])


async def test_a_task_the_person_did_not_make_reads_no_project_file(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _instructions_read(rig, monkeypatch)
    job = rig.workspace / "job"
    job.mkdir()
    (job / "AGENTS.md").write_text("Test with pytest.")
    rig.llm.replies = [says("Done.")]
    spec = TaskSpec.from_payload(
        {"objective": "Sweep the cache", "workspace": str(job)}
    )
    task, _ = await rig.workers.create_spec(spec, origin="agent")
    await _ended(rig.workers, task["task_id"], 1)
    assert calls == []
    assert "this project's notes for assistants" not in _texts(rig.llm.requests[0])


async def test_a_saved_chore_reads_the_project_file(rig: Rig) -> None:
    job = rig.workspace / "weekly"
    job.mkdir()
    (job / "GEMINI.md").write_text("Weekly notes live in notes/.")
    rig.llm.replies = [says("Done.")]
    spec = TaskSpec.from_payload(
        {"objective": "Summarise the week", "workspace": str(job)}
    )
    task, _ = await rig.workers.create_spec(spec, origin="user_rule")
    await _ended(rig.workers, task["task_id"], 1)
    assert "--- GEMINI.md ---\nWeekly notes live in notes/." in _texts(
        rig.llm.requests[0]
    )


async def test_a_failed_read_of_the_file_still_starts_the_task(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from localmelo.support.memory import project_files

    def broken(folder: str, **kwargs: Any) -> Any:
        raise RuntimeError("the disk is gone")

    monkeypatch.setattr(project_files, "instructions", broken)
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    assert (await rig.workers.get(row["task_id"]))["state"] == "done"
    assert "this project's notes for assistants" not in _texts(rig.llm.requests[0])
    assert "Could not read the project's instructions" in caplog.text


async def test_a_file_too_big_to_read_is_named_and_gives_no_heading(rig: Rig) -> None:
    (rig.workspace / "AGENTS.md").write_text("x" * (256 * 1024 + 1))
    rig.llm.replies = [says("Done.")]
    row, _ = await rig.workers.create(
        {"prompt": "List the folder", "workspace": str(rig.workspace)}
    )
    await _ended(rig.workers, row["task_id"], 1)
    assert "this project's notes for assistants" not in _texts(rig.llm.requests[0])
    spec = (await rig.workers.get(row["task_id"]))["spec"]
    assert spec["project_skipped"] == ["AGENTS.md"]
    assert "project_instructions" not in spec and "project_files" not in spec


async def test_a_continue_and_a_repeated_request_keep_the_first_reading(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _instructions_read(rig, monkeypatch)
    agents = rig.workspace / "AGENTS.md"
    agents.write_text("First rule.")
    rig.llm.replies = [says("Done."), says("Finished.")]
    request = {
        "prompt": "Fix registry.py",
        "workspace": str(rig.workspace),
        "request_key": "same-request",
    }
    row, _ = await rig.workers.create(dict(request))
    task_id = row["task_id"]
    await _ended(rig.workers, task_id, 1)
    agents.write_text("Changed rule.")
    again, repeated = await rig.workers.create(dict(request))
    assert repeated and again["task_id"] == task_id
    _, how = await rig.workers.message(task_id, "finish it")
    assert how == "continued"
    await _ended(rig.workers, task_id, 2)
    assert "First rule." in _texts(rig.llm.requests[1])
    assert "Changed rule." not in _texts(rig.llm.requests[1])
    assert len(calls) == 1
