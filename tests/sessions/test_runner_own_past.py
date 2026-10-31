"""A chat's memory of its own past, end to end in the chat's runner.

The Agent's window keeps the newest messages of the chat (here 8, so four
turns, dropped one at a time). A turn that has left the window is recalled like a turn of another
chat, as ``this chat``; one the window still shows is not.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.recall import BLOCK_HEADER
from localmelo.support.sessions.agents import build_session_agent
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
    eviction: Any = "slide",
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
        options=AgentOptions(window_messages=8, window_eviction=eviction),
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
    # The dentist is turn 3 of 5 earlier turns: the window holds turns 2-5.
    said = [LATER[0], LATER[1], DENTIST, LATER[2], LATER[3], ASKS]
    request, data = (await talk(memory, store, stub, tmp_path / "w", said))[-1]
    assert data["used"] == [] and BLOCK_HEADER not in request
    assert request == ASKS


async def test_a_turn_whose_question_left_the_window_is_recalled_though_its_answer_stays(
    memory: ChatMemory, store: SessionStore, stub: Any, tmp_path: Path
) -> None:
    # With chunk eviction the window drops down to 5 messages at once: turn 3's
    # answer is the first message left, its question is gone.
    said = [LATER[0], LATER[1], DENTIST, LATER[2], LATER[3], ASKS]
    chunked = await talk(memory, store, stub, tmp_path / "w", said, eviction="chunk")
    request, data = chunked[-1]
    (used,) = data["used"]
    assert used["turn"] == 3 and used["source"] == f"this chat, {today()}"
    assert "Nov 3" in request
