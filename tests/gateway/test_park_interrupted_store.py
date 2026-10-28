"""Guarded retry and interruption persistence."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.park_store import ParkStore, ParkStoreError
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
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
    parks = ParkStore(sessions)
    await tasks.request_pending_pause(
        task.task_id,
        expected_generation=None,
        expected_operation_id=None,
        operation_id="pause",
    )
    await parks.hold(
        task.task_id,
        site="reflect",
        step=3,
        attempt=2,
        reason="capacity",
        expected_generation=1,
        expected_operation_id="pause",
        operation_id="hold",
    )
    try:
        yield sessions, tasks, parks, task.task_id
    finally:
        await sessions.aclose()


async def test_retry_interrupt_and_resume(
    stores: tuple[SessionStore, TaskStore, ParkStore, str],
) -> None:
    sessions, tasks, parks, task_id = stores
    marked = await parks.mark_park_retry(
        task_id,
        expected_generation=2,
        expected_operation_id="pause",
        operation_id="retry",
    )
    assert marked.state == "held" and marked.reason == "park not saved; retrying"
    assert (await tasks.get_task(task_id)).wait_reason == "park not saved; retrying"
    assert (
        await parks.mark_park_retry(
            task_id,
            expected_generation=2,
            expected_operation_id="pause",
            operation_id="retry",
        )
        == marked
    )
    with pytest.raises(ParkStoreError):
        await parks.interrupt(
            task_id,
            card="card",
            expected_generation=2,
            expected_operation_id="pause",
            operation_id="stale",
        )
    interrupted = await parks.interrupt(
        task_id,
        card="Resume from card",
        expected_generation=3,
        expected_operation_id="pause",
        operation_id="interrupt",
    )
    assert interrupted.state == "interrupted" and interrupted.card == "Resume from card"
    assert await tasks.park_identity(task_id) == (4, None)
    async with sessions.component_transaction("verify_pending") as conn:
        row = await (
            await conn.execute(
                "SELECT pause_pending, snapshot_active FROM park_priority WHERE task_id=?",
                (task_id,),
            )
        ).fetchone()
    assert row is not None and tuple(row) == (0, 1)
    with pytest.raises(TaskStoreError):
        await tasks.resume_interrupted(
            task_id,
            expected_generation=3,
            expected_operation_id="pause",
            operation_id="stale-resume",
        )
    assert (await tasks.get_task(task_id)).state == "interrupted"
    result = await tasks.resume_interrupted(
        task_id,
        expected_generation=4,
        expected_operation_id=None,
        operation_id="resume",
    )
    assert (
        result.state == "queued"
        and result.level == 2
        and result.card == "Resume from card"
    )
    assert await tasks.park_identity(task_id) == (5, None)
    assert (
        await tasks.resume_interrupted(
            task_id,
            expected_generation=4,
            expected_operation_id=None,
            operation_id="resume",
        )
        == result
    )
    events = [e.event["type"] for e in await sessions.list_events(task_id)]
    assert events.count("sched.park_retrying") == events.count("sched.interrupted") == 1
    assert events.count("sched.resumed") == 1
    receipt = await parks.reconcile(task_id, operation_id="interrupt")
    assert receipt is not None and not receipt[1]


async def test_resume_uncertainty_is_idempotent(
    stores: tuple[SessionStore, TaskStore, ParkStore, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, tasks, parks, task_id = stores
    await parks.interrupt(
        task_id,
        card="card",
        expected_generation=2,
        expected_operation_id="pause",
        operation_id="interrupt",
    )
    original = sessions._commit

    async def uncertain(conn: Any, events: Any, label: str) -> None:
        await original(conn, events, label)
        if label == "task_resume":
            raise asyncio.CancelledError()

    monkeypatch.setattr(sessions, "_commit", uncertain)
    args: dict[str, Any] = dict(
        expected_generation=3, expected_operation_id=None, operation_id="resume"
    )
    with pytest.raises(asyncio.CancelledError):
        await tasks.resume_interrupted(task_id, **args)
    monkeypatch.setattr(sessions, "_commit", original)
    receipt = await tasks.reconcile_pause_operation(task_id, "resume")
    assert receipt is not None and receipt[1]
    assert (await tasks.resume_interrupted(task_id, **args)).state == "queued"
    assert [e.event["type"] for e in await sessions.list_events(task_id)].count(
        "sched.resumed"
    ) == 1


async def test_failure_uncertain_and_stop(
    stores: tuple[SessionStore, TaskStore, ParkStore, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, tasks, parks, task_id = stores

    def fail(label: str) -> None:
        if label == "park_interrupted":
            raise RuntimeError("injected")

    monkeypatch.setattr(sessions, "_before_commit", fail)
    args: dict[str, Any] = dict(
        card="resume",
        expected_generation=2,
        expected_operation_id="pause",
        operation_id="interrupt",
    )
    with pytest.raises(RuntimeError, match="injected"):
        await parks.interrupt(task_id, **args)
    assert (await tasks.get_task(task_id)).state == "held"
    assert await tasks.park_identity(task_id) == (2, "pause")
    assert await parks.reconcile(task_id, operation_id="interrupt") is None
    monkeypatch.setattr(sessions, "_before_commit", lambda label: None)
    original = sessions._commit

    async def uncertain(conn: Any, events: Any, label: str) -> None:
        await original(conn, events, label)
        if label == "park_interrupted":
            raise asyncio.CancelledError()

    monkeypatch.setattr(sessions, "_commit", uncertain)
    with pytest.raises(asyncio.CancelledError):
        await parks.interrupt(task_id, **args)
    monkeypatch.setattr(sessions, "_commit", original)
    receipt = await parks.reconcile(task_id, operation_id="interrupt")
    assert receipt is not None and receipt[1]
    assert (await parks.interrupt(task_id, **args)).state == "interrupted"
    assert [e.event["type"] for e in await sessions.list_events(task_id)].count(
        "sched.interrupted"
    ) == 1
    await tasks.change(task_id, state="cancelled")
    with pytest.raises(ParkStoreError, match="terminal"):
        await parks.mark_park_retry(
            task_id,
            expected_generation=3,
            expected_operation_id=None,
            operation_id="late",
        )
    with pytest.raises(TaskStoreError):
        await tasks.resume_interrupted(
            task_id,
            expected_generation=3,
            expected_operation_id=None,
            operation_id="resume",
        )
    assert (await tasks.get_task(task_id)).state == "cancelled"
