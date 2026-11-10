"""A gateway turn's reply as it is written: live frames beside the record.

End to end on the stub model, through the gate's wrapper and the hub: the
pieces reach a viewer before the stored reply, which equals their sum; no
live frame is stored; a viewer that joins in the middle gets the text so far
once, then the rest; a stop in the middle closes the model server's request;
a server that ignores ``stream`` gives one reply, as before.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
from localmelo.support.gateway.console.sse import frame
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.live import LIVE_EVENT, LiveReply
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, StoredEvent
from localmelo.tests.fixtures.stub_llm import StubServer

# The core releases complete lines (its redaction reads a line at a time).
PIECES = [
    "Hello there.\n",
    "This reply is",
    " written\n",
    "line by line.\n",
    "The end.",
]
REPLY = "".join(PIECES)


def slow(delay_s: float) -> dict[str, Any]:
    # Reasoning in a field of its own first, as a thinking model's server sends
    # it: the core then releases the content as it comes, with no declaration.
    return {
        "replies": [
            {
                "content": REPLY,
                "stream": {
                    "reasoning": ["Let me say hello."],
                    "chunks": PIECES,
                    "delay_s": delay_s,
                },
            }
        ]
    }


@pytest.fixture()
def stub() -> Iterator[StubServer]:
    server = StubServer(
        scenarios={
            "slow": slow(0.05),
            "stoppable": slow(0.3),
            "ignored": {"replies": [{"content": REPLY, "stream_ignored": True}]},
        },
        default="slow",
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


async def chat(
    store: SessionStore,
    hub: SessionHub,
    stub: StubServer,
    tmp_path: Path,
    scenario: str,
) -> SessionRunner:
    gate = ChatGate(policy="off", call_log_dir=None)
    agent = await build_session_agent(
        cfg=None,
        base_url=stub.base_url.replace("/v1", f"/s/{scenario}/v1"),
        model="stub-model",
        workspace=str(tmp_path),
        options=AgentOptions(),
        wrap_llm=lambda llm: GatedLLM(llm, gate.client("chat")),
    )
    return SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(kind="chat"),
        live=lambda info: LiveReply(info, hub.live),
    )


async def follow(hub: SessionHub, session_id: str, out: list[StoredEvent]) -> None:
    """What a page that shows the chat gets, until the turn ends."""
    async for event in hub.subscribe(session_id, viewer=True):
        out.append(event)
        if event.event.get("type") == "turn.finished":
            return


def live_text(events: list[StoredEvent]) -> str:
    text = ""
    for event in events:
        if event.event["type"] == LIVE_EVENT:
            data = event.event["data"]
            text = (text if data["at"] else "") + data["text"]
    return text


async def first_piece(events: list[StoredEvent]) -> None:
    while not any(e.event["type"] == LIVE_EVENT for e in events):
        await asyncio.sleep(0.01)


async def test_pieces_come_before_the_stored_reply_which_is_their_sum(
    store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    hub = SessionHub(store)
    runner = await chat(store, hub, stub, tmp_path, "slow")
    turn = await runner.start_turn("say hello")
    seen: list[StoredEvent] = []
    await asyncio.wait_for(follow(hub, turn.session_id, seen), 20)
    result = await runner.wait()
    assert result is not None and result.status == "completed"
    assert stub.requests[0]["stream"] is True

    kinds = [e.event["type"] for e in seen]
    live = [e for e in seen if e.event["type"] == LIVE_EVENT]
    assert len(live) >= 2 and kinds.index(LIVE_EVENT) < kinds.index("assistant.message")
    assert all(e.id is None and "id:" not in frame(e) for e in live)
    assert live_text(seen) == REPLY == result.result
    stored = await store.list_events(turn.session_id)
    assert LIVE_EVENT not in {e.event["type"] for e in stored}
    reply = next(e for e in stored if e.event["type"] == "assistant.message")
    assert reply.event["data"]["content"] == REPLY
    (row,) = await store.list_turns(turn.session_id)
    assert row.result is not None and 0 <= row.result["first_word_s"] < 20
    await runner.close()


async def test_a_viewer_that_joins_in_the_middle_gets_the_text_so_far_once(
    store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    hub = SessionHub(store)
    runner = await chat(store, hub, stub, tmp_path, "slow")
    turn = await runner.start_turn("say hello")
    early: list[StoredEvent] = []
    following = asyncio.create_task(follow(hub, turn.session_id, early))
    await asyncio.wait_for(first_piece(early), 20)
    late: list[StoredEvent] = []
    await asyncio.wait_for(follow(hub, turn.session_id, late), 20)
    await following
    first = next(e for e in late if e.event["type"] == LIVE_EVENT)
    assert first.event["data"]["at"] == 0 and REPLY.startswith(
        first.event["data"]["text"]
    )
    # Then the rest, each part once: the late viewer ends with the same text.
    assert live_text(late) == live_text(early) == REPLY
    joined = "".join(
        e.event["data"]["text"] for e in late if e.event["type"] == LIVE_EVENT
    )
    assert joined == REPLY
    await runner.close()


async def test_a_stop_in_the_middle_closes_the_model_server_request(
    store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    hub = SessionHub(store)
    runner = await chat(store, hub, stub, tmp_path, "stoppable")
    turn = await runner.start_turn("say hello")
    seen: list[StoredEvent] = []
    following = asyncio.create_task(follow(hub, turn.session_id, seen))
    await asyncio.wait_for(first_piece(seen), 20)
    assert runner.cancel()
    result = await asyncio.wait_for(runner.wait(), 20)
    assert result is not None and result.status == "cancelled"
    # The stub writes a piece every 0.3 s: it sees the closed connection at
    # its next write, long before its last piece.
    assert await asyncio.to_thread(stub.cut.wait, 3)
    await asyncio.wait_for(following, 20)
    shown = live_text(seen)
    assert shown and REPLY.startswith(shown) and shown != REPLY
    await runner.close()


async def test_a_server_that_ignores_stream_gives_one_reply_as_before(
    store: SessionStore, stub: StubServer, tmp_path: Path
) -> None:
    hub = SessionHub(store)
    runner = await chat(store, hub, stub, tmp_path, "ignored")
    turn = await runner.start_turn("say hello")
    seen: list[StoredEvent] = []
    await asyncio.wait_for(follow(hub, turn.session_id, seen), 20)
    result = await runner.wait()
    assert (
        result is not None and result.status == "completed" and result.result == REPLY
    )
    assert LIVE_EVENT not in {e.event["type"] for e in seen}
    (row,) = await store.list_turns(turn.session_id)
    assert row.result is not None and "first_word_s" not in row.result
    await runner.close()
