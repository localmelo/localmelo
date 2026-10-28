"""Regression coverage for startup recovery classification."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.gateway.park_store import ParkStore
from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.startup_recovery import StartupRecovery
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.sessions.helpers import core_event, run_result


@pytest.fixture
async def stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[SessionStore, TaskStore]]:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    try:
        yield store, tasks
    finally:
        await store.aclose()


async def test_priority_change_preserves_committed_park_card(
    stores: tuple[SessionStore, TaskStore], tmp_path: Path
) -> None:
    store, tasks = stores
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
    await tasks.set_level(task.task_id, 3)
    await StartupRecovery(store, tasks).run()
    task = await tasks.get_task(task.task_id)
    assert task.state == "parked" and task.card == "valid evidence card"
    assert task.level == 3


@pytest.mark.parametrize("binding", [None, "launch-id", "different-id"])
async def test_terminal_result_requires_submission_binding(
    stores: tuple[SessionStore, TaskStore], tmp_path: Path, binding: str | None
) -> None:
    store, tasks = stores
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
        execution_id="launch-id" if binding == "launch-id" else "run-1",
        expected_generation=generation,
        expected_operation_id=previous,
    )
    turn = await store.begin_turn(
        task.task_id, "work", holder="worker", client_request_id=binding
    )
    await store.append_events(
        task.task_id,
        turn.turn,
        [
            core_event("run.started"),
            core_event("run.finished", seq=2, result=run_result()),
        ],
        holder="worker",
    )
    await store.finish_turn(
        task.task_id,
        turn.turn,
        holder="worker",
        status="completed",
        cause="run",
        answer="done",
        result=run_result(),
    )
    # The matching case binds the launch id; the default run ID alone is insufficient.
    startup = StartupRecovery(store, tasks)
    await startup.run()
    await startup.run()
    task = await tasks.get_task(task.task_id)
    assert task.state == ("done" if binding == "launch-id" else "interrupted")
    assert task.crashes == (0 if binding == "launch-id" else 1)


async def test_legacy_running_turn_counts_one_crash(
    stores: tuple[SessionStore, TaskStore], tmp_path: Path
) -> None:
    store, tasks = stores
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(tmp_path),
    )
    await tasks.change(task.task_id, state="running")
    turn = await store.begin_turn(task.task_id, "work", holder="worker")
    startup = StartupRecovery(store, tasks)
    await startup.run()
    await startup.run()
    task = await tasks.get_task(task.task_id)
    assert task.state == "interrupted" and task.crashes == 1
    provenance = await RecoveryStore(store).provenance(task.task_id)
    assert provenance.execution_id == f"turn:{turn.id}"
    assert provenance.unsafe_resume


@pytest.mark.parametrize("change_identity", [False, True])
async def test_terminal_task_supersedes_prepared_intent(
    stores: tuple[SessionStore, TaskStore], tmp_path: Path, change_identity: bool
) -> None:
    store, tasks = stores
    task, _ = await tasks.create_task(
        TaskSpec.from_payload({"objective": "work"}),
        origin="user",
        workspace=str(tmp_path),
    )
    recovery = RecoveryStore(store)
    generation, previous = await tasks.park_identity(task.task_id)
    await recovery.prepare(
        task.task_id,
        operation_id="pending",
        execution_id="old",
        expected_generation=generation,
        expected_operation_id=previous,
        outcome="interrupted",
        card="",
        reason="restart",
        crash=True,
        unsafe_resume=True,
    )
    async with store.component_transaction("terminalize") as conn:
        await conn.execute(
            "UPDATE tasks SET state = 'done' WHERE task_id = ?", (task.task_id,)
        )
        if change_identity:
            await conn.execute(
                "UPDATE park_priority SET generation = generation + 1 WHERE task_id = ?",
                (task.task_id,),
            )
    startup = StartupRecovery(store, tasks)
    await startup.run()
    await startup.run()
    receipt = await recovery.receipt(task.task_id, "pending")
    assert receipt is not None and receipt[0]["phase"] == "superseded"
    task = await tasks.get_task(task.task_id)
    assert task.state == "done" and task.crashes == 0
