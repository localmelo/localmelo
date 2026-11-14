"""A chat's memory of its own past, end to end in the chat's runner.

The Agent's window keeps the newest messages of the chat (here 8, so four
turns, dropped one at a time). A turn that has left the window is recalled like a turn of another
chat, as ``this chat``; one the window still shows is not.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import BLOCK_HEADER
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.attachments import TextAttachment
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, StoredEvent
from localmelo.tests.fixtures.stub_llm import StubServer

DENTIST = "my dentist appointment is on Nov 3"
ASKS = "when is my dentist appointment?"
LATER = ("plan the garden", "buy tomato seeds", "paint the fence", "fix the gate")


def today() -> str:
    now = time.localtime()
    return f"{time.strftime('%b', now)} {now.tm_mday}"


@pytest.fixture()
def stub() -> Any:
    server = StubServer(
        scenarios={"answer": {"replies": [{"content": "Okay, noted."}]}}
    ).start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def talk(
    memory: ChatMemory,
    store: SessionStore,
    stub: Any,
    workspace: Path,
    said: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """One chat of the *said* turns in one runner; each turn's request text and
    its ``turn.memory`` event."""
    workspace.mkdir(exist_ok=True)
    events: list[StoredEvent] = []
    agent = await build_session_agent(
        cfg=None,
        base_url=stub.base_url,
        model="stub-model",
        workspace=str(workspace),
        options=AgentOptions(window_messages=8),
        identify=True,
        memory=memory,
    )
    runner = SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(kind="chat"),
        on_event=events.append,
        memory=memory,
    )
    out: list[tuple[str, dict[str, Any]]] = []
    try:
        for text in said:
            stub.requests.clear()
            events.clear()
            result = await runner.run_turn(text, title=said[0])
            assert result.status == "completed"
            await memory.drain()
            data = [e.event["data"] for e in events if e.event["type"] == "turn.memory"]
            out.append((stub.requests[0]["messages"][-1]["content"], data[0]))
    finally:
        await runner.close()
        await agent.close()
    return out


async def test_a_turn_that_left_the_window_is_recalled_as_this_chat(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    # Turn 1 is the dentist; turns 2-5 fill the window; turn 6 asks about it.
    turns = await talk(memory, store, stub, tmp_path / "w", [DENTIST, *LATER, ASKS])
    for request, data in turns[:-1]:
        assert request.startswith(("my dentist", "plan", "buy", "paint", "fix"))
        assert BLOCK_HEADER not in request and data["used"] == []
    request, data = turns[-1]
    (used,) = data["used"]
    assert used["kind"] == "chat" and used["turn"] == 1
    assert used["source"] == f"this chat, {today()}"
    assert request.startswith(ASKS + "\n\n" + BLOCK_HEADER)
    assert f"({'this chat'}, {today()})" in request and "Nov 3" in request


async def test_a_turn_the_window_still_shows_is_not_recalled(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    # The dentist is turn 2 of 3 earlier turns: the window holds all three.
    said = [LATER[0], DENTIST, LATER[1], ASKS]
    request, data = (await talk(memory, store, stub, tmp_path / "w", said))[-1]
    assert data["used"] == [] and BLOCK_HEADER not in request
    assert request == ASKS


async def test_a_turn_whose_question_left_the_window_is_recalled_though_its_answer_stays(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    # With chunk eviction the window drops down to 5 messages at once: turn 3's
    # answer is the first message left, its question is gone.
    said = [LATER[0], LATER[1], DENTIST, LATER[2], LATER[3], ASKS]
    request, data = (await talk(memory, store, stub, tmp_path / "w", said))[-1]
    (used,) = data["used"]
    assert used["turn"] == 3 and used["source"] == f"this chat, {today()}"
    assert "Nov 3" in request


# ── The record of repeated tasks: on disk, and none of the text ─────────────


def every_value(path: Path) -> str:
    """Every value of every table of the file, lowercased, in one text."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        values = [
            str(value)
            for table in tables
            for row in conn.execute(f"SELECT * FROM {table}")
            for value in row
        ]
    return "\n".join(values).lower()


async def test_chats_and_tasks_leave_a_record_a_new_agent_finds_and_none_of_their_text(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    workspace = tmp_path / "w"
    workspace.mkdir()
    record = Path(os.path.expanduser("~/.cache/localmelo/memory/procedures.db"))
    # One marker in each part of a turn's text: the person's words, the file
    # they attached, a note recalled from an earlier turn, a context block, and
    # a task's request.
    fact = "my quokka appointment is on Nov 3"
    ask = "when is my appointment? narwhal"
    attached = [TextAttachment("notes.txt", "okapi lines")]
    marks = ("narwhal", "okapi", "quokka", "platypus", "wombat")

    async def build() -> Agent:
        return await build_session_agent(
            cfg=None,
            base_url=stub.base_url,
            model="stub-model",
            workspace=str(workspace),
            options=AgentOptions(window_messages=8),
            identify=True,
            memory=memory,
        )

    agent = await build()
    chat = SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(kind="chat"),
        memory=memory,
        context="platypus news",
    )
    for said in (fact, *LATER):
        await chat.run_turn(said)
        await memory.drain()
    stub.requests.clear()
    await chat.run_turn(ask, attachments=attached)
    request = stub.requests[0]["messages"][-1]["content"]
    assert all(mark in request for mark in ("narwhal", "okapi", "quokka", "platypus"))
    await chat.close()
    await agent.close()
    assert record.exists()

    agent = await build()
    task = SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(kind="task"),
        context="platypus news",
    )
    await task.start_turn("check the wombat report", from_person=False)
    outcome = await task.wait()
    assert outcome is not None and outcome.status == "completed"
    await task.close()
    await agent.close()

    # A new agent, a new chat, other context: the same words find the record.
    agent = await build()
    events: list[StoredEvent] = []
    again = SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(kind="chat"),
        on_event=events.append,
        progress=True,
        context="other news",
    )
    await again.run_turn(ask, attachments=attached)
    await again.close()
    await agent.close()
    states = [e.event["data"] for e in events if e.event["type"] == "run.progress"]
    assert states[0]["past"]["runs"] == 1 and states[0]["basis"]["match"] == "exact"

    text = every_value(record)
    assert not [mark for mark in marks if mark in text]
    with contextlib.closing(sqlite3.connect(record)) as conn:
        texts = conn.execute("SELECT query_text FROM routines").fetchall()
    # Six turns and a task make seven routines; the repeated turn found its own.
    assert len(texts) == 7 and set(texts) == {("",)}
