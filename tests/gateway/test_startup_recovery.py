"""Restart classification preserves uncertainty without launching workers."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.startup_recovery import StartupRecovery
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore, SessionStoreError
from localmelo.tests.sessions.helpers import core_event, run_result


async def test_unknown_tasks_interrupt_and_proven_new_stays_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        spec = TaskSpec.from_payload({"objective": "do work"})
        legacy, _ = await tasks.create_task(
            spec, origin="user", workspace=str(tmp_path)
        )
        new, _ = await tasks.create_task(
            spec, origin="user", workspace=str(tmp_path), _recovery_proof=True
        )
        recovery = StartupRecovery(store, tasks)
        await recovery.run()
        old = await tasks.get_task(legacy.task_id)
        assert old.state == "interrupted" and not old.card
        assert (await RecoveryStore(store).provenance(legacy.task_id)).unsafe_resume
        assert (await tasks.get_task(new.task_id)).state == "queued"
        await recovery.run()
        assert (await tasks.get_task(legacy.task_id)) == old
    finally:
        await store.aclose()


async def test_started_work_is_interrupted_once_with_recorded_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        spec = TaskSpec.from_payload({"objective": "do work"})
        task, _ = await tasks.create_task(
            spec, origin="user", workspace=str(tmp_path), _recovery_proof=True
        )
        recovery = RecoveryStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        await recovery.start(
            task.task_id,
            operation_id="launch",
            execution_id="exec",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        await store.begin_turn(task.task_id, "do work", holder="worker")
        startup = StartupRecovery(store, tasks)
        await startup.run()
        task = await tasks.get_task(task.task_id)
        assert task.state == "interrupted" and task.crashes == 1
        assert "completion unverified" in task.card
        assert (await recovery.provenance(task.task_id)).unsafe_resume
        await startup.run()
        assert (await tasks.get_task(task.task_id)).crashes == 1
    finally:
        await store.aclose()


@pytest.mark.parametrize(
    "proven,card,expected",
    [
        (True, "", "parked"),
        (False, "", "interrupted"),
        (False, "saved card", "interrupted"),
    ],
)
async def test_parked_matrix_preserves_proof_and_cards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proven: bool,
    card: str,
    expected: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_user_payload({"objective": "do work", "level": 5}),
            origin="user",
            workspace=str(tmp_path),
            _recovery_proof=proven,
        )
        await tasks.change(task.task_id, state="parked")
        if card:
            async with store.component_transaction("seed_card") as conn:
                await conn.execute(
                    "UPDATE tasks SET card = ? WHERE task_id = ?", (card, task.task_id)
                )
        await StartupRecovery(store, tasks).run()
        record = await tasks.get_task(task.task_id)
        assert record.state == expected and not record.card
        assert record.level == 5
    finally:
        await store.aclose()


async def test_new_evidence_replaces_old_card_on_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "work"}),
            origin="user",
            workspace=str(tmp_path),
            _recovery_proof=True,
        )
        generation, previous = await tasks.park_identity(task.task_id)
        await RecoveryStore(store).start(
            task.task_id,
            operation_id="start",
            execution_id="run-1",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        async with store.component_transaction("seed_old_card") as conn:
            await conn.execute(
                "UPDATE tasks SET card = 'old card' WHERE task_id = ?", (task.task_id,)
            )
        turn = await store.begin_turn(task.task_id, "work", holder="worker")
        await store.append_events(
            task.task_id,
            turn.turn,
            [
                core_event("run.started"),
                core_event("tool.started", seq=2, name="file_write"),
            ],
            holder="worker",
        )
        await StartupRecovery(store, tasks).run()
        card = (await tasks.get_task(task.task_id)).card
        assert "file_write started" in card and "without a recorded finish" in card
        assert card != "old card"
    finally:
        await store.aclose()


async def test_committed_park_card_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "work"}),
            origin="user",
            workspace=str(tmp_path),
        )
        await tasks.change(task.task_id, state="running")
        parks = ParkStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        await parks.hold(
            task.task_id,
            site="plan",
            step=1,
            attempt=1,
            reason="waiting",
            operation_id="hold",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        generation, previous = await tasks.park_identity(task.task_id)
        await parks.park(
            task.task_id,
            card="valid evidence card",
            operation_id="park",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        await StartupRecovery(store, tasks).run()
        task = await tasks.get_task(task.task_id)
        assert task.state == "parked" and task.card == "valid evidence card"
    finally:
        await store.aclose()


async def test_pending_pause_survives_startup_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "do work"}),
            origin="user",
            workspace=str(tmp_path),
            _recovery_proof=True,
        )
        generation, previous = await tasks.park_identity(task.task_id)
        await RecoveryStore(store).start(
            task.task_id,
            operation_id="launch",
            execution_id="execution",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        generation, previous = await tasks.park_identity(task.task_id)
        await tasks.request_pending_pause(
            task.task_id,
            operation_id="pause",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        await StartupRecovery(store, tasks).run()
        task = await tasks.get_task(task.task_id)
        assert task.state == "interrupted" and task.level == 5
        async with store.component_transaction("pending_check") as conn:
            row = await (
                await conn.execute(
                    "SELECT pause_pending, snapshot_active, operation_id FROM park_priority WHERE task_id = ?",
                    (task.task_id,),
                )
            ).fetchone()
        assert row is not None and tuple(row) == (1, 1, "pause")
    finally:
        await store.aclose()


async def test_worker_open_fences_recovered_tasks_but_allows_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    workers = TaskWorkers(store, gate, Config, hub)
    try:
        await workers.tasks.open()
        task, _ = await workers.tasks.create_task(
            TaskSpec.from_payload({"objective": "do work"}),
            origin="user",
            workspace=str(tmp_path),
        )
        await workers.open()
        assert (await workers.get(task.task_id))["state"] == "interrupted"
        with pytest.raises(TaskStoreError):
            await workers.update(task.task_id, state="resume")
        with pytest.raises(TaskStoreError):
            await workers.update(task.task_id, level=1)
        assert not workers.is_active(task.task_id)
        assert (await workers.update(task.task_id, state="stop"))[
            "state"
        ] == "cancelled"
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


@pytest.mark.parametrize("same_execution", [True, False])
async def test_completed_outcome_requires_matching_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_execution: bool
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "sessions.db"
    store = SessionStore(path, writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "do work"}),
        origin="user",
        workspace=str(tmp_path),
        _recovery_proof=True,
    )
    generation, previous = await tasks.park_identity(task.task_id)
    await RecoveryStore(store).start(
        task.task_id,
        operation_id="launch",
        execution_id="run-1" if same_execution else "other",
        expected_generation=generation,
        expected_operation_id=previous,
    )
    turn = await store.begin_turn(task.task_id, "do work", holder="worker")
    await store.append_events(
        task.task_id,
        turn.turn,
        [
            core_event("run.started"),
            core_event("run.finished", seq=2, result=run_result()),
        ],
        holder="worker",
    )
    await store.aclose()
    reopened = SessionStore(path, writer="gateway")
    await reopened.open()
    try:
        tasks = TaskStore(reopened)
        await tasks.open()
        startup = StartupRecovery(reopened, tasks)
        await startup.run()
        assert (await tasks.get_task(task.task_id)).state == (
            "done" if same_execution else "interrupted"
        )
        await startup.run()
        assert (await tasks.get_task(task.task_id)).crashes == (
            0 if same_execution else 1
        )
    finally:
        await reopened.aclose()


async def test_startup_supersedes_stale_settled_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "work"}),
            origin="user",
            workspace=str(tmp_path),
        )
        recovery = RecoveryStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        await recovery.prepare(
            task.task_id,
            operation_id="stale",
            execution_id="old",
            expected_generation=generation,
            expected_operation_id=previous,
            outcome="interrupted",
            card="",
            reason="restart",
            crash=False,
            unsafe_resume=True,
        )
        await tasks.change(task.task_id, state="parked")
        await StartupRecovery(store, tasks).run()
        receipt = await recovery.receipt(task.task_id, "stale")
        assert receipt is not None and receipt[0]["phase"] == "superseded"
        assert (await tasks.get_task(task.task_id)).state == "interrupted"
    finally:
        await store.aclose()


async def test_commit_then_cancel_does_not_duplicate_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "work"}),
            origin="user",
            workspace=str(tmp_path),
            _recovery_proof=True,
        )
        generation, previous = await tasks.park_identity(task.task_id)
        await RecoveryStore(store).start(
            task.task_id,
            operation_id="start",
            execution_id="execution",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        original = store._commit
        fired = False

        async def commit_then_cancel(conn: Any, events: Any, label: str) -> None:
            nonlocal fired
            await original(conn, events, label)
            if label == "recovery_commit" and not fired:
                fired = True
                raise asyncio.CancelledError()

        monkeypatch.setattr(store, "_commit", commit_then_cancel)
        startup = StartupRecovery(store, tasks)
        with pytest.raises(asyncio.CancelledError):
            await startup.run()
        await startup.run()
        assert (await tasks.get_task(task.task_id)).crashes == 1
        assert [
            event.event["type"] for event in await store.list_events(task.task_id)
        ].count("sched.recovered") == 1
    finally:
        await store.aclose()


async def test_startup_failure_stays_unavailable_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    gate, hub = ChatGate(policy="off"), SessionHub(store)
    workers = TaskWorkers(store, gate, Config, hub)
    try:
        await workers.tasks.open()
        task, _ = await workers.tasks.create_task(
            TaskSpec.from_payload({"objective": "do work"}),
            origin="user",
            workspace=str(tmp_path),
        )
        original = store._before_commit

        def fail(label: str) -> None:
            if label == "recovery_commit":
                raise RuntimeError("injected failure")
            original(label)

        monkeypatch.setattr(store, "_before_commit", fail)
        with pytest.raises(SessionStoreError):
            await workers.open()
        assert not workers._opened and not workers.is_active(task.task_id)
        assert await RecoveryStore(store).active_intent(task.task_id)
        monkeypatch.setattr(store, "_before_commit", original)
        await workers.open()
        assert (await workers.get(task.task_id))["state"] == "interrupted"
        assert await RecoveryStore(store).active_intent(task.task_id) is None
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()
