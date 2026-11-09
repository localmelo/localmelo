"""A task starts with what its chat knows: the pack is made once, at creation.

The project's own instruction files (``AGENTS.md``, ``CLAUDE.md``,
``GEMINI.md`` of the task's folder, read-only) are read at the same moment and
ride the same way, under their own heading after the pack. So do the person's
own words, when the request does not hold them, and what a chat hands a task
it starts (its earlier turns, what its turn recalled): ahead of the pack.

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
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.agent._earlier import EARLIER_END, EARLIER_HEAD
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.handover import (
    HANDOFF_HEADER,
    PICKED_HEADER,
    WORDS_HEADER,
    Handover,
    picked_block,
    turns_block,
    words_block,
)
from localmelo.support.gateway.task_policy import TaskSpec, is_persons_task
from localmelo.support.gateway.workers import (
    PROJECT_HEADER,
    TASK_PACK_CHARS,
    TASK_PACK_HEADER,
    TaskWorkers,
)
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import Item
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
        self.budgets: list[int] = []
        self.excluded: list[frozenset[str]] = []

    async def context_pack(
        self,
        objective: str,
        chat_id: str,
        budget_chars: int = 1500,
        *,
        workspace: str | None = None,
        exclude: Any = (),
    ) -> str:
        self.calls.append((objective, chat_id, workspace))
        self.budgets.append(budget_chars)
        self.excluded.append(frozenset(exclude))
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


async def test_a_saved_chores_run_reads_no_project_file(rig: Rig) -> None:
    """A chore's folder holds what earlier runs left: data, never the project's instructions (#570)."""
    job = rig.workspace / "weekly"
    job.mkdir()
    (job / "GEMINI.md").write_text("Weekly notes live in notes/.")
    rig.llm.replies = [says("Done.")]
    spec = TaskSpec.from_payload(
        {"objective": "Summarise the week", "workspace": str(job)}
    )
    task, _ = await rig.workers.create_spec(spec, origin="user_rule")
    await _ended(rig.workers, task["task_id"], 1)
    assert "Weekly notes live in notes/." not in _texts(rig.llm.requests[0])


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


# ── The person's own words, and what the chat hands over (#463) ──

WORDS = "read his résumé first, then tell me which roles fit"
ASKED = "Read Alex's résumé and list the roles that fit him."
TURNS = (
    ("c1", 1, "My friend Alex is looking for a new job.", "Noted."),
    ("c1", 2, "He wants a staff role at a large company.", "Got it."),
)
RECALLED = Item(
    key="d:3",
    text="User: Alex's résumé is two pages.\nmelo: Good.",
    where="chat “Résumé”, Oct 30",
    session_id="c9",
    turn=1,
    created_at=time.time(),
)


def _spec(rig: Rig, objective: str = ASKED, words: str = WORDS) -> TaskSpec:
    spec = TaskSpec.from_payload(
        {"objective": objective, "workspace": str(rig.workspace)}
    )
    return replace(spec, user_words=words)


async def _first(rig: Rig, spec: TaskSpec, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    """The task's first request and its stored spec."""
    rig.llm.replies = [says("Done.")]
    task, _ = await rig.workers.create_spec(spec, **kwargs)
    await _ended(rig.workers, task["task_id"], 1)
    turns = await rig.workers.store.list_turns(task["task_id"], limit=10)
    assert [turn.user_text for turn in turns] == [spec.objective], "never in the turn"
    return _texts(rig.llm.requests[0]), (await rig.workers.get(task["task_id"]))["spec"]


async def test_the_persons_words_ride_once_under_their_heading(rig: Rig) -> None:
    first, spec = await _first(rig, _spec(rig), chat_started=True)
    assert first.count(WORDS) == 1
    assert words_block(WORDS) in first
    assert (
        first.index(ASKED) < first.index(WORDS_HEADER) < first.index(TASK_PACK_HEADER)
    )
    assert spec["own_words"] == WORDS and spec["user_words"] == WORDS


async def test_no_words_when_the_request_holds_them_or_there_are_none(
    rig: Rig,
) -> None:
    first, spec = await _first(rig, _spec(rig, "Fix registry.py", "fix  registry.py"))
    assert WORDS_HEADER not in first and "own_words" not in spec
    # A saved chore, a task melo made: no words, nothing new.
    job = rig.workspace / "weekly"
    job.mkdir()
    chore = TaskSpec.from_payload(
        {"objective": "Summarise the week", "workspace": str(job)}
    )
    first, spec = await _first(rig, chore, origin="user_rule")
    assert WORDS_HEADER not in first and EARLIER_HEAD not in first
    assert not {"own_words", "chat_turns", "chat_picked"} & set(spec)


async def test_a_fresh_task_on_the_same_job_reuses_the_words_and_the_handover(
    rig: Rig,
) -> None:
    rig.llm.replies = [says("Done."), says("Finished.")]
    handover = Handover(turns=TURNS, used=(RECALLED,))
    task, _ = await rig.workers.create_spec(
        _spec(rig), chat_started=True, handover=handover
    )
    task_id = task["task_id"]
    await _ended(rig.workers, task_id, 1)
    record = await rig.workers.tasks.get_task(task_id)
    fresh, how = await rig.workers.successor(record, "finish it", by="person", key=None)
    assert how == "successor"
    await _ended(rig.workers, fresh["task_id"], 1)
    second = _texts(rig.llm.requests[1])
    for block in (WORDS_HEADER, EARLIER_HEAD, PICKED_HEADER, HANDOFF_HEADER):
        assert second.count(block) == 1, block


async def test_the_chats_turns_ride_ahead_of_the_pack(rig: Rig) -> None:
    first, spec = await _first(
        rig, _spec(rig), chat_started=True, handover=Handover(turns=TURNS)
    )
    lines = (
        "- the user: My friend Alex is looking for a new job.\n"
        "  melo: Noted.\n- the user: He wants a staff role at a large company.\n"
        "  melo: Got it."
    )
    assert turns_block(lines) in first
    assert turns_block(lines).startswith(f"{EARLIER_HEAD}\n<<turns nonce=")
    assert turns_block(lines).endswith(f">>\n{EARLIER_END}")
    assert first.index(WORDS_HEADER) < first.index(EARLIER_HEAD)
    assert first.index(EARLIER_END) < first.index(TASK_PACK_HEADER)
    assert spec["chat_turns"].startswith("- the user: My friend Alex")
    assert PICKED_HEADER not in first and "chat_picked" not in spec


async def test_what_the_chat_recalled_rides_ahead_of_the_notes_in_the_packs_room(
    rig: Rig,
) -> None:
    first, spec = await _first(
        rig,
        _spec(rig, words=""),
        chat_started=True,
        handover=Handover(turns=TURNS, used=(RECALLED,)),
    )
    line = (
        "- the user said “Alex's résumé is two pages.”; melo replied “Good.” "
        "(chat “Résumé”, Oct 30)"
    )
    assert picked_block(line) in first
    assert first.index(EARLIER_END) < first.index(PICKED_HEADER)
    assert first.index(PICKED_HEADER) < first.index(TASK_PACK_HEADER)
    assert spec["chat_picked"] == line
    # The notes get the pack's room the picked lines leave; none is left out by key.
    assert rig.notes.budgets == [TASK_PACK_CHARS - len(line) - 1]
    assert rig.notes.excluded == [frozenset()]


async def test_no_handover_no_budget_change(rig: Rig) -> None:
    first, spec = await _first(rig, _spec(rig, words=""))
    assert EARLIER_HEAD not in first and PICKED_HEADER not in first
    assert rig.notes.budgets == [TASK_PACK_CHARS] and TASK_PACK_CHARS == 1500


async def test_memory_off_keeps_the_words_and_the_chats_turns_only(rig: Rig) -> None:
    rig.workers.memory = None
    first, spec = await _first(
        rig,
        _spec(rig),
        chat_started=True,
        handover=Handover(turns=TURNS, used=(RECALLED,)),
    )
    # The chat's own conversation rides either way; nothing recalled, no notes.
    assert EARLIER_HEAD in first and "- the user: My friend Alex is looking" in first
    assert spec["chat_turns"].startswith("- the user: My friend Alex")
    assert PICKED_HEADER not in first and TASK_PACK_HEADER not in first
    assert not {"chat_picked", "context_pack"} & set(spec)
    assert words_block(WORDS) in first


async def test_memory_turned_off_in_the_settings_keeps_the_chats_turns(
    rig: Rig,
) -> None:
    rig.notes.enabled = False  # type: ignore[attr-defined]
    rig.notes.pack = ""
    first, spec = await _first(
        rig,
        _spec(rig),
        chat_started=True,
        handover=Handover(turns=TURNS, used=(RECALLED,)),
    )
    assert EARLIER_HEAD in first and PICKED_HEADER not in first
    assert "chat_picked" not in spec and spec["chat_turns"]


async def test_a_failed_read_of_what_the_chat_recalled_keeps_its_turns(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    # A note needs the notes' topics, which this stand-in cannot read.
    note = replace(RECALLED, key="r:5", text="Alex is a friend.")
    first, spec = await _first(
        rig, _spec(rig), chat_started=True, handover=Handover(turns=TURNS, used=(note,))
    )
    assert EARLIER_HEAD in first and PICKED_HEADER not in first
    assert "Could not read what the chat recalled" in caplog.text


async def test_a_request_near_the_query_limit_keeps_the_blocks_that_fit(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from localmelo.melo.checker.validators import MAX_QUERY_LEN

    queries: list[str] = []

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](rig.llm),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        run_task = agent.run_task

        async def recorded(query: str, **more: Any) -> Any:
            queries.append(query)
            return await run_task(query, **more)

        agent.run_task = recorded  # type: ignore[method-assign]
        return agent

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    # Room for the person's words, not for the chat's turns or the notes too.
    objective = "Read Alex's notes. " + "x" * (MAX_QUERY_LEN - 19 - 400)
    rig.llm.replies = [says("Done.")]
    await rig.workers.create_spec(
        _spec(rig, objective=objective),
        chat_started=True,
        handover=Handover(turns=TURNS * 2),
    )
    async with asyncio.timeout(5):
        while not queries:
            await asyncio.sleep(0.01)
    (query,) = queries
    assert len(query) <= MAX_QUERY_LEN
    assert WORDS_HEADER in query, "the highest block that fits stays"
    assert EARLIER_HEAD not in query and TASK_PACK_HEADER not in query
    assert "Left 2 of 3 context blocks out of the request" in caplog.text
