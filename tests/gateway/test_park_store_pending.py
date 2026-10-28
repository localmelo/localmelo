"""Durable pause marker and superseded park transition regressions."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.park_store import ParkStore, ParkStoreError
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore


@pytest.fixture
async def stores(
    tmp_path: Path,
) -> AsyncIterator[tuple[SessionStore, TaskStore, ParkStore, str]]:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    tasks = TaskStore(sessions)
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "test"}),
        origin="user",
        workspace=str(tmp_path),
    )
    async with sessions.component_transaction("running") as conn:
        await conn.execute(
            "UPDATE tasks SET state='running' WHERE task_id=?", (task.task_id,)
        )
    try:
        yield sessions, tasks, ParkStore(sessions), task.task_id
    finally:
        await sessions.aclose()


async def test_pending_pause_blocks_held_resume(
    stores: tuple[SessionStore, TaskStore, ParkStore, str],
) -> None:
    sessions, tasks, parks, task_id = stores
    held = await parks.hold(
        task_id, site="plan", step=1, attempt=1, reason="capacity", operation_id="hold"
    )
    assert not held.pause_pending
    await tasks.request_pending_pause(
        task_id,
        expected_generation=None,
        expected_operation_id=None,
        operation_id="pause",
    )
    current = await parks.get(task_id)
    assert current is not None and current.pause_pending and current.state == "held"
    before = await sessions.list_events(task_id)
    with pytest.raises(ParkStoreError, match="pause is pending"):
        await parks.resume(
            task_id,
            expected_generation=1,
            expected_operation_id="pause",
            operation_id="resume",
        )
    current = await parks.get(task_id)
    assert current is not None and current.pause_pending and current.state == "held"
    assert await tasks.park_identity(task_id) == (1, "pause")
    assert await sessions.list_events(task_id) == before


async def test_successful_park_clears_pending_row_and_event(
    stores: tuple[SessionStore, TaskStore, ParkStore, str],
) -> None:
    sessions, tasks, parks, task_id = stores
    await tasks.request_pending_pause(
        task_id,
        expected_generation=None,
        expected_operation_id=None,
        operation_id="pause",
    )
    held = await parks.hold(
        task_id,
        site="reflect",
        step=2,
        attempt=1,
        reason="capacity",
        expected_generation=1,
        expected_operation_id="pause",
        operation_id="hold",
    )
    assert held.pause_pending
    parked = await parks.park(
        task_id,
        card="card",
        expected_generation=2,
        expected_operation_id="pause",
        operation_id="park",
    )
    assert parked.state == "parked" and not parked.pause_pending
    assert await parks.get(task_id) == parked
    async with sessions.component_transaction("check_pending") as conn:
        row = await (
            await conn.execute(
                "SELECT pause_pending, snapshot_active FROM park_priority WHERE task_id = ?",
                (task_id,),
            )
        ).fetchone()
    assert row is not None and tuple(row) == (0, 1)
    event = [
        item.event
        for item in await sessions.list_events(task_id)
        if item.event["type"] == "sched.parked"
    ]
    assert len(event) == 1 and event[0]["data"]["pause_pending"] is False


async def test_superseded_hold_receipt_rejected_without_duplicate(
    stores: tuple[SessionStore, TaskStore, ParkStore, str],
) -> None:
    sessions, tasks, parks, task_id = stores
    await tasks.request_pending_pause(
        task_id,
        expected_generation=None,
        expected_operation_id=None,
        operation_id="pause",
    )
    await tasks.cancel_pending_pause(
        task_id,
        expected_generation=1,
        expected_operation_id="pause",
        operation_id="cancel",
    )
    args: dict[str, Any] = dict(
        site="reflect",
        step=1,
        attempt=1,
        reason="capacity",
        expected_generation=2,
        expected_operation_id=None,
        operation_id="first-hold",
    )
    await parks.hold(task_id, **args)
    await parks.resume(
        task_id,
        expected_generation=3,
        expected_operation_id=None,
        operation_id="resume",
    )
    await parks.hold(
        task_id,
        site="plan",
        step=7,
        attempt=1,
        reason="capacity",
        expected_generation=4,
        expected_operation_id=None,
        operation_id="new-hold",
    )
    with pytest.raises(ParkStoreError, match="superseded"):
        await parks.hold(task_id, **args)
    current = await parks.get(task_id)
    assert current is not None and (current.state, current.site, current.step) == (
        "held",
        "plan",
        7,
    )
    assert [item.event["type"] for item in await sessions.list_events(task_id)].count(
        "sched.held"
    ) == 2
