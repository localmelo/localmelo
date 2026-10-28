"""A cancelled pending pause cannot complete a stale park."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.park_store import ParkStore, ParkStoreError
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore


async def test_pending_pause_resume_then_stale_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
        tasks = TaskStore(sessions)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "test"}),
            origin="user",
            workspace=str(tmp_path),
        )
        task_id = task.task_id
        async with sessions.component_transaction("test_running") as conn:
            await conn.execute(
                "UPDATE tasks SET state='running' WHERE task_id=?", (task_id,)
            )
        parks = ParkStore(sessions)
        await tasks.request_pending_pause(
            task_id,
            expected_generation=None,
            expected_operation_id=None,
            operation_id="pause",
        )
        await parks.hold(
            task_id,
            site="plan",
            step=1,
            attempt=1,
            reason="capacity",
            expected_generation=1,
            expected_operation_id="pause",
            operation_id="hold",
        )
        event = [
            e.event
            for e in await sessions.list_events(task_id)
            if e.event["type"] == "sched.held"
        ][-1]
        assert (
            event["data"]["state"] == "held" and event["data"]["pause_pending"] is True
        )

        def fail(label: str) -> None:
            if label == "park_parked":
                raise RuntimeError("injected rollback")

        monkeypatch.setattr(sessions, "_before_commit", fail)
        with pytest.raises(RuntimeError, match="injected"):
            await parks.park(
                task_id,
                card="card",
                expected_generation=2,
                expected_operation_id="pause",
                operation_id="failed-park",
            )
        assert await tasks.park_identity(task_id) == (2, "pause")
        record = await parks.get(task_id)
        assert record is not None and record.state == "held"
        assert await parks.reconcile(task_id, operation_id="failed-park") is None
        monkeypatch.setattr(sessions, "_before_commit", lambda label: None)
        await tasks.cancel_pending_pause(
            task_id,
            expected_generation=2,
            expected_operation_id="pause",
            operation_id="resume",
        )
        with pytest.raises(ParkStoreError):
            await parks.park(
                task_id,
                card="stale card",
                expected_generation=2,
                expected_operation_id="pause",
                operation_id="stale-park",
            )
        record = await parks.get(task_id)
        assert record is not None and record.state == "held"
        assert await tasks.park_identity(task_id) == (3, None)
        assert not any(
            e.event["type"] == "sched.parked"
            for e in await sessions.list_events(task_id)
        )
        await tasks.change(task_id, state="cancelled")
        with pytest.raises(ParkStoreError, match="terminal"):
            await parks.park(
                task_id,
                card="stale",
                expected_generation=3,
                expected_operation_id=None,
                operation_id="late-park",
            )
        assert (await tasks.get_task(task_id)).state == "cancelled"
        receipt = await parks.reconcile(task_id, operation_id="hold")
        assert receipt is not None and receipt[1] is False
    finally:
        await sessions.aclose()


@pytest.mark.parametrize("pause_first", [False, True])
async def test_scheduler_park_absence_race(tmp_path: Path, pause_first: bool) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
        tasks = TaskStore(sessions)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "test"}),
            origin="user",
            workspace=str(tmp_path),
        )
        task_id = task.task_id
        async with sessions.component_transaction("test_running") as conn:
            await conn.execute(
                "UPDATE tasks SET state='running' WHERE task_id=?", (task_id,)
            )
        parks = ParkStore(sessions)
        await parks.hold(
            task_id,
            site="plan",
            step=1,
            attempt=1,
            reason="capacity",
            expected_generation=None,
            expected_operation_id=None,
            operation_id="hold",
        )
        if pause_first:
            await tasks.request_pending_pause(
                task_id,
                expected_generation=None,
                expected_operation_id=None,
                operation_id="pause",
            )
            with pytest.raises(ParkStoreError):
                await parks.park(
                    task_id,
                    card="card",
                    expected_generation=None,
                    expected_operation_id=None,
                    operation_id="park",
                )
            assert await tasks.park_identity(task_id) == (1, "pause")
            record = await parks.get(task_id)
            assert record is not None and record.state == "held"
        else:
            parked = await parks.park(
                task_id,
                card="card",
                expected_generation=None,
                expected_operation_id=None,
                operation_id="park",
            )
            assert parked.state == "parked"
            assert await tasks.park_identity(task_id) == (1, "park")
            result = await parks.reconcile(task_id, operation_id="park")
            assert result is not None
            receipt, current = result
            assert current and receipt["generation"] == 1
    finally:
        await sessions.aclose()


@pytest.mark.parametrize("paused", [False, True])
async def test_paused_hold_event_reports_durable_state(
    tmp_path: Path, paused: bool
) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
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
        if paused:
            await tasks.pause(task.task_id)
        generation, pending_id = await tasks.park_identity(task.task_id)
        record = await ParkStore(sessions).hold(
            task.task_id,
            site="reflect",
            step=1,
            attempt=1,
            reason="capacity",
            expected_generation=generation,
            expected_operation_id=pending_id,
            operation_id="hold",
        )
        event = [
            e.event
            for e in await sessions.list_events(task.task_id)
            if e.event["type"] == "sched.held"
        ][-1]
        assert (
            event["data"]["state"] == record.state == ("parked" if paused else "held")
        )
        assert event["data"]["pause_pending"] is False
    finally:
        await sessions.aclose()


async def test_old_park_and_resume_cannot_change_new_hold(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
        tasks = TaskStore(sessions)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "test"}),
            origin="user",
            workspace=str(tmp_path),
        )
        task_id = task.task_id
        async with sessions.component_transaction("running") as conn:
            await conn.execute(
                "UPDATE tasks SET state='running' WHERE task_id=?", (task_id,)
            )
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
        parks = ParkStore(sessions)
        await parks.hold(
            task_id,
            site="reflect",
            step=1,
            attempt=1,
            reason="capacity",
            expected_generation=2,
            expected_operation_id=None,
            operation_id="first-hold",
        )
        await parks.resume(
            task_id,
            expected_generation=3,
            expected_operation_id=None,
            operation_id="first-resume",
        )
        await parks.hold(
            task_id,
            site="plan",
            step=7,
            attempt=1,
            reason="capacity",
            expected_generation=4,
            expected_operation_id=None,
            operation_id="second-hold",
        )
        with pytest.raises(ParkStoreError):
            await parks.park(
                task_id,
                card="CARD FROM TURN 1",
                expected_generation=3,
                expected_operation_id=None,
                operation_id="old-park",
            )
        with pytest.raises(ParkStoreError):
            await parks.resume(
                task_id,
                expected_generation=3,
                expected_operation_id=None,
                operation_id="old-resume",
            )
        record = await parks.get(task_id)
        assert record is not None
        assert (record.state, record.site, record.step) == ("held", "plan", 7)
        assert not any(
            e.event["type"] == "sched.parked"
            for e in await sessions.list_events(task_id)
        )
    finally:
        await sessions.aclose()


async def test_uncertain_park_reconciles_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
        tasks = TaskStore(sessions)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "test"}),
            origin="user",
            workspace=str(tmp_path),
        )
        task_id = task.task_id
        async with sessions.component_transaction("test_running") as conn:
            await conn.execute(
                "UPDATE tasks SET state='running' WHERE task_id=?", (task_id,)
            )
        parks = ParkStore(sessions)
        await tasks.request_pending_pause(
            task_id,
            expected_generation=None,
            expected_operation_id=None,
            operation_id="pause",
        )
        await parks.hold(
            task_id,
            site="plan",
            step=1,
            attempt=1,
            reason="capacity",
            expected_generation=1,
            expected_operation_id="pause",
            operation_id="hold",
        )
        original = sessions._commit

        async def uncertain(conn: Any, events: Any, label: str) -> None:
            await original(conn, events, label)
            if label == "park_parked":
                raise asyncio.CancelledError()

        monkeypatch.setattr(sessions, "_commit", uncertain)
        args: dict[str, Any] = dict(
            card="card",
            expected_generation=2,
            expected_operation_id="pause",
            operation_id="park",
        )
        with pytest.raises(asyncio.CancelledError):
            await parks.park(task_id, **args)
        monkeypatch.setattr(sessions, "_commit", original)
        result = await parks.reconcile(task_id, operation_id="park")
        assert result is not None
        receipt, current = result
        assert current and receipt["generation"] == 3
        assert await tasks.park_identity(task_id) == (3, None)
        assert (await parks.park(task_id, **args)).state == "parked"
        assert [e.event["type"] for e in await sessions.list_events(task_id)].count(
            "sched.parked"
        ) == 1
        with pytest.raises(ParkStoreError):
            await parks.park(task_id, **{**args, "card": "different"})
        async with sessions.component_transaction("prune_receipt") as conn:
            await conn.execute(
                "DELETE FROM events WHERE session_id = ? AND type = 'sched.parked'",
                (task_id,),
            )
        assert await parks.reconcile(task_id, operation_id="park") is None
        with pytest.raises(ParkStoreError):
            await parks.park(task_id, **args)
    finally:
        await sessions.aclose()
