"""SessionHub: replay, then live, without gaps or repeats, also after an overflow."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    StoredEvent,
    TurnInfo,
)

from .helpers import core_event

HOLDER = "testhost:1:aaaaaaaa"


async def running(store: SessionStore) -> TurnInfo:
    return await store.begin_turn(None, "hi", holder=HOLDER, new_session=NewSession())


async def add(
    store: SessionStore, turn: TurnInfo, count: int, first_seq: int = 1
) -> None:
    for start in range(0, count, 200):
        batch = [
            core_event("step.started", seq=first_seq + i)
            for i in range(start, min(start + 200, count))
        ]
        await store.append_events(turn.session_id, turn.turn, batch, holder=HOLDER)


async def take(events: AsyncIterator[StoredEvent], count: int) -> list[StoredEvent]:
    taken = []
    async for event in events:
        taken.append(event)
        if len(taken) == count:
            break
    return taken


async def test_replay_then_live_with_events_stored_during_the_replay(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    hub = SessionHub(store)
    turn = await running(store)
    await add(store, turn, 2500)  # three replay pages
    stored = await store.list_events(turn.session_id, limit=1000)
    total = 2 + 2500  # turn.started and session.title, then the batches
    seen: list[StoredEvent] = []
    subscription = hub.subscribe(turn.session_id)
    async for event in subscription:
        seen.append(event)
        if len(seen) == 10:
            await add(store, turn, 300, first_seq=2501)  # during the replay
        if len(seen) == total + 300:
            break
    await subscription.aclose()
    ids = [e.id or 0 for e in seen]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) == total + 300
    assert seen[:5] == stored[:5]
    assert [e.event["seq"] for e in seen[2:]] == list(range(1, 2801))
    assert hub._subscribers == {}  # closing the iterator unregistered it
    await hub.aclose()
    await store.aclose()


async def test_after_an_id_and_two_subscribers_see_one_sequence(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    hub = SessionHub(store)
    turn = await running(store)
    await add(store, turn, 5)
    events = await store.list_events(turn.session_id)
    resumed = await take(hub.subscribe(turn.session_id, after_id=events[3].id or 0), 3)
    assert resumed == events[4:]
    first = asyncio.create_task(take(hub.subscribe(turn.session_id), 17))
    second = asyncio.create_task(take(hub.subscribe(turn.session_id), 17))
    await asyncio.sleep(0.05)
    await add(store, turn, 10, first_seq=6)
    one, two = await first, await second
    assert one == two and len(one) == 17
    await hub.aclose()
    await store.aclose()


async def test_an_overflow_re_reads_and_keeps_the_queue_bounded(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    hub = SessionHub(store, queue_size=5)
    turn = await running(store)
    subscription = hub.subscribe(turn.session_id)
    first = await subscription.__anext__()  # replayed: turn.started
    assert first.event["type"] == "turn.started"
    queued: list[int] = []
    for seq in range(1, 51):  # the consumer sleeps meanwhile
        await store.append_events(
            turn.session_id, 1, [core_event("step.started", seq=seq)], holder=HOLDER
        )
        queued.append(max(s.queue.qsize() for s in hub._subscribers[turn.session_id]))
    assert max(queued) <= 5
    rest = await take(subscription, 51)  # session.title, then the 50
    await subscription.aclose()
    assert [e.event["type"] for e in rest[:1]] == ["session.title"]
    assert [e.event["seq"] for e in rest[1:]] == list(range(1, 51))
    ids = [first.id or 0, *(e.id or 0 for e in rest)]
    assert ids == sorted(ids) and len(set(ids)) == 52
    await hub.aclose()
    await store.aclose()


async def test_out_of_turn_and_recovery_events_reach_subscribers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    store = SessionStore(path, writer="gateway")
    hub = SessionHub(store)
    turn = await running(store)
    await add(store, turn, 1)
    subscription = hub.subscribe(turn.session_id)
    replayed = await take(subscription, 3)
    stored = await store.append_session_event(
        turn.session_id, "task.progress", {"n": 1}
    )
    live = await subscription.__anext__()
    assert live == stored and live.turn is None
    await subscription.aclose()
    await hub.aclose()
    last_seen = stored.id or 0
    await store.aclose()  # the process stops with the turn running
    reopened = SessionStore(path, writer="gateway")
    again = SessionHub(reopened)
    await reopened.open()  # recovery
    events = await take(again.subscribe(turn.session_id, after_id=last_seen), 1)
    assert events[0].event["type"] == "turn.finished"
    assert events[0].event["data"]["cause"] == "crash"
    assert (events[0].id or 0) > max(e.id or 0 for e in replayed)
    await again.aclose()
    await reopened.aclose()


async def test_aclose_ends_every_subscription(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", writer="gateway")
    hub = SessionHub(store)
    turn = await running(store)
    reading = asyncio.create_task(take(hub.subscribe(turn.session_id), 100))
    await asyncio.sleep(0.05)
    await hub.aclose()
    assert len(await asyncio.wait_for(reading, timeout=5)) == 2
    assert await take(hub.subscribe(turn.session_id), 1) == []
    await store.aclose()


def test_a_hub_needs_the_gateways_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SessionHub needs the gateway's store"):
        SessionHub(SessionStore(tmp_path / "c.db", writer="cli"))
    with pytest.raises(ValueError):
        SessionHub(SessionStore(tmp_path / "g.db", writer="gateway"), queue_size=0)
