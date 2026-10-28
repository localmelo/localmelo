"""Transactional worker hold and park persistence."""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.park_store import ParkStore, ParkStoreError
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskNotFoundError, TaskStore
from localmelo.support.sessions.store import SessionStore, StoredEvent

Stores = tuple[SessionStore, TaskStore, ParkStore, str]


@pytest.fixture
async def stores(tmp_path: Path) -> AsyncIterator[Stores]:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    tasks = TaskStore(sessions)
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "Test"}),
        origin="user",
        workspace=str(tmp_path),
    )
    async with sessions.component_transaction("test_running") as conn:
        await conn.execute(
            "UPDATE tasks SET state='running' WHERE task_id=?", (task.task_id,)
        )
    try:
        yield sessions, tasks, ParkStore(sessions), task.task_id
    finally:
        await sessions.aclose()


def db_row(sessions: SessionStore, task_id: str) -> Any:
    with sqlite3.connect(sessions.path) as conn:
        return conn.execute(
            "SELECT state, hold_site, wait_reason, card, level, level_eff, peak_level, reason "
            "FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()


def hold_args() -> dict[str, Any]:
    return dict(site="reflect", step=3, agent="writer", attempt=2, reason="capacity")


def guard(sessions: SessionStore, task_id: str, operation_id: str) -> dict[str, Any]:
    with sqlite3.connect(sessions.path) as conn:
        row = conn.execute(
            "SELECT generation, operation_id FROM park_priority WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return {
        "expected_generation": row[0] if row else None,
        "expected_operation_id": row[1] if row else None,
        "operation_id": operation_id,
    }


async def test_transitions_and_duplicates(stores: Stores) -> None:
    sessions, _, parks, task_id = stores
    seen: list[StoredEvent] = []
    committed_states = []
    sessions.add_listener(seen.append)
    sessions.add_listener(
        lambda event: committed_states.append(db_row(sessions, task_id)[0])
    )
    initial = db_row(sessions, task_id)
    first_hold = guard(sessions, task_id, "hold-1")
    held = await parks.hold(task_id, **hold_args(), **first_hold)
    assert held.state == "held" and held.step == 3
    assert await parks.get(task_id) == held
    assert await parks.hold(task_id, **hold_args(), **first_hold) == held
    with pytest.raises(ParkStoreError):
        await parks.hold(task_id, **{**hold_args(), "step": 4}, **first_hold)
    first_resume = guard(sessions, task_id, "resume-1")
    resumed = await parks.resume(task_id, **first_resume)
    assert resumed.state == "running"
    assert await parks.resume(task_id, **first_resume) == resumed
    assert db_row(sessions, task_id)[0] == "running"
    assert db_row(sessions, task_id)[2] == ""
    await parks.hold(task_id, **hold_args(), **guard(sessions, task_id, "hold-2"))
    first_park = guard(sessions, task_id, "park-1")
    parked = await parks.park(task_id, card="Resume safely", **first_park)
    assert parked.state == "parked" and parked.card == "Resume safely"
    assert await parks.park(task_id, card="Resume safely", **first_park) == parked
    with pytest.raises(ParkStoreError):
        await parks.resume(task_id, **guard(sessions, task_id, "resume-2"))
    with pytest.raises(ParkStoreError):
        await parks.park(task_id, card="Different", **first_park)
    assert db_row(sessions, task_id)[4:] == initial[4:]
    assert committed_states == ["held", "running", "held", "parked"]
    assert [event.event["type"] for event in seen] == [
        "sched.held",
        "sched.resumed",
        "sched.held",
        "sched.parked",
    ]


async def test_pause_p5_remains_parked(stores: Stores) -> None:
    sessions, tasks, parks, task_id = stores
    await tasks.pause(task_id)
    before = db_row(sessions, task_id)
    assert before[0] == "parked" and before[4] == 5
    first_hold = guard(sessions, task_id, "hold")
    held = await parks.hold(task_id, **hold_args(), **first_hold)
    assert held.state == "parked"
    assert await parks.hold(task_id, **hold_args(), **first_hold) == held
    with pytest.raises(ParkStoreError):
        await parks.resume(task_id, **guard(sessions, task_id, "resume"))
    first_park = guard(sessions, task_id, "park")
    parked = await parks.park(task_id, card="Paused card", **first_park)
    assert parked.state == "parked"
    assert await parks.park(task_id, card="Paused card", **first_park) == parked
    assert db_row(sessions, task_id)[4:] == before[4:]


@pytest.mark.parametrize("audit_change", ["prune", "intervening"])
async def test_pause_after_hold_keeps_parkable_evidence(
    stores: Stores, audit_change: str
) -> None:
    sessions, tasks, parks, task_id = stores
    await parks.hold(task_id, **hold_args(), **guard(sessions, task_id, "hold"))
    await tasks.pause(task_id)
    async with sessions.component_transaction("test_audit") as conn:
        if audit_change == "prune":
            await conn.execute("DELETE FROM sched_log WHERE task_id = ?", (task_id,))
        else:
            await conn.execute(
                "INSERT INTO sched_log (ts, kind, task_id, data) VALUES (0, 'test', ?, '{}')",
                (task_id,),
            )
    parked = await parks.park(
        task_id, card="Paused after hold", **guard(sessions, task_id, "park")
    )
    assert parked.state == "parked"
    assert parked.site == "reflect" and parked.step == 3


async def test_failure_rolls_back_event_and_row(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, _, parks, task_id = stores
    seen: list[StoredEvent] = []
    sessions.add_listener(seen.append)
    before = db_row(sessions, task_id)

    def fail(label: str) -> None:
        if label == "park_held":
            raise RuntimeError("injected failure")

    monkeypatch.setattr(sessions, "_before_commit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        await parks.hold(task_id, **hold_args(), **guard(sessions, task_id, "hold"))
    assert db_row(sessions, task_id) == before
    assert not seen
    assert not await sessions.list_events(task_id)


async def test_terminal_and_missing_rejected(stores: Stores) -> None:
    sessions, _, parks, task_id = stores
    with pytest.raises(TaskNotFoundError):
        await parks.hold("missing", **hold_args(), **guard(sessions, "missing", "hold"))
    async with sessions.component_transaction("test_terminal") as conn:
        await conn.execute(
            "UPDATE tasks SET state='cancelled' WHERE task_id=?", (task_id,)
        )
    before = db_row(sessions, task_id)
    for call in (
        parks.hold(task_id, **hold_args(), **guard(sessions, task_id, "hold")),
        parks.resume(task_id, **guard(sessions, task_id, "resume")),
        parks.park(task_id, card="card", **guard(sessions, task_id, "park")),
    ):
        with pytest.raises(ParkStoreError):
            await call
    assert db_row(sessions, task_id) == before
