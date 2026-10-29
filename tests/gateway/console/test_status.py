"""``GET /api/sessions/{id}/status``: the terminal's status numbers, on the console."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localmelo.melo.contracts.events import EVENT_FIELDS
from localmelo.support.gateway.console import Console
from localmelo.support.sessions.store import NewSession
from localmelo.support.usage_snapshot import TaskCounts, UsageSnapshot

from .conftest import HOLDER, all_events, core_event, seed_turn


def message(prompt: int, cached: int | None, ts: float) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": 5}
    if cached is not None:
        usage["cached_tokens"] = cached
    return core_event("assistant.message", ts=ts, usage=usage)


def sized(type_: str, ts: float, **data: Any) -> dict[str, Any]:
    """An event whose payload is only the pinned keys, ``None`` unless given."""
    payload = {key: None for key in EVENT_FIELDS[type_]} | data
    return {
        "v": 1,
        "type": type_,
        "run_id": "run-1",
        "seq": 1,
        "ts": ts,
        "data": payload,
    }


def budget(ts: float, measured: int, limit: int) -> dict[str, Any]:
    return sized(
        "context_budget", ts, measured_prompt_tokens=measured, input_limit=limit
    )


def progress(ts: float, step: int) -> dict[str, Any]:
    return sized("run.progress", ts, level=0, state="running", step=step, attempt=1)


async def seed(console: Console, workspace: str, events: list[dict[str, Any]]) -> str:
    turn = await console.store.begin_turn(
        None,
        "q",
        holder=HOLDER,
        title="T",
        new_session=NewSession(model="stub-model", workspace=workspace),
    )
    await console.store.append_events(turn.session_id, turn.turn, events, holder=HOLDER)
    return turn.session_id


async def terminal_fold(console: Console, session_id: str) -> UsageSnapshot:
    """What the terminal's renderer holds after the same stored events."""
    snap = UsageSnapshot()
    for stored in await all_events(console.store, session_id):
        snap.observe(stored.event)
    return snap


def run_events(now: float, cached: int | None) -> list[dict[str, Any]]:
    return [
        core_event("run.started", ts=now - 400),
        message(1000, cached, now - 300),
        budget(now - 290, 1000, 8000),
        message(1500, None if cached is None else cached + 600, now - 200),
        progress(now - 100, 3),
    ]


@pytest.mark.parametrize("cached", [400, None])
async def test_the_status_equals_the_terminal_fold(
    client: TestClient, console: Console, workspace: str, cached: int | None
) -> None:
    now = time.time()
    session_id = await seed(console, workspace, run_events(now, cached))
    body = client.get(f"/api/sessions/{session_id}/status").json()
    snap = await terminal_fold(console, session_id)
    expected = snap.to_dict(local=True, now=now, tasks=TaskCounts(0, 0, 0, None))
    assert body.pop("session_id") == session_id
    # The clock is the one thing the two reads do not share; minutes of idle
    # time (200 s here) keep the segment text equal.
    assert abs(body["cache"].pop("idle_s") - expected["cache"].pop("idle_s")) < 5
    assert body == expected
    if cached is None:
        assert body["segments"][0].startswith("cache -- idle")
        assert body["cache"]["last_percent"] is None
        assert body["cache"]["session_percent"] is None
    else:
        assert body["cache"]["last_percent"] == 67
    assert body["segments"][1:] == [
        "ctx 1.0k/8.0k",
        "step 3",
        "tasks 0 running 0 queued",
    ]


async def test_a_session_with_no_events_has_no_segments_but_the_tasks(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    body = client.get(f"/api/sessions/{turn.session_id}/status").json()
    assert body["cache"]["requests"] == 0
    assert body["context"] is None and body["steps"] is None
    assert body["segments"] == ["tasks 0 running 0 queued"]


async def test_the_fold_is_cached_by_the_last_event_id(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    session_id = await seed(console, workspace, run_events(now, 400))
    last = (await all_events(console.store, session_id))[-1].id
    reads: list[int] = []
    original = console.store.list_events

    async def counting(session_id: str, **kwargs: Any) -> Any:
        reads.append(kwargs.get("after_id", 0))
        return await original(session_id, **kwargs)

    monkeypatch.setattr(console.store, "list_events", counting)
    first = client.get(f"/api/sessions/{session_id}/status").json()
    assert reads == [0]
    again = client.get(f"/api/sessions/{session_id}/status").json()
    assert reads == [0], "a poll with no new event must not read events"
    for body in (first, again):
        body["cache"].pop("idle_s")
    assert again == first
    # A new event reads only what follows the cached id.
    await console.store.append_events(
        session_id, 1, [message(2000, 1800, now)], holder=HOLDER
    )
    fresh = client.get(f"/api/sessions/{session_id}/status").json()
    assert reads == [0, last]
    assert fresh["cache"]["last_percent"] == 90


async def test_tasks_come_from_the_task_store(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Record:
        def __init__(self, state: str, level_eff: int) -> None:
            self.state, self.level_eff = state, level_eff

    async def listed() -> list[Record]:
        return [
            Record("running", 2),
            Record("queued", 3),
            Record("queued", 1),
            Record("held", 4),
            Record("done", 0),
        ]

    monkeypatch.setattr(console.workers.tasks, "list_tasks", listed)
    turn = await seed_turn(console.store, workspace)
    body = client.get(f"/api/sessions/{turn.session_id}/status").json()
    assert body["tasks"] == {"running": 1, "queued": 2, "held": 1, "top_level": 1}
    assert body["segments"][-1] == "tasks 1 running 2 queued"


def test_an_unknown_session_is_not_found(client: TestClient) -> None:
    for session_id in ("nope", "worker-session"):
        resp = client.get(f"/api/sessions/{session_id}/status")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "not_found"
