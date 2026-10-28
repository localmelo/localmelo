"""Contract tests for durable first-start and recovery transactions."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
from localmelo.support.sessions.store import SessionStore, StoredEvent


@pytest.fixture
async def stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple]:
    monkeypatch.setenv("HOME", str(tmp_path))
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    tasks = TaskStore(sessions)
    await tasks.open()
    task, duplicate = await tasks.create_task(
        TaskSpec.from_payload({"objective": "Recover a task"}),
        origin="user",
        workspace=str(tmp_path),
    )
    assert not duplicate
    recovery = RecoveryStore(sessions)
    try:
        yield sessions, tasks, recovery, task.task_id
    finally:
        await sessions.aclose()


async def mark_created(sessions: SessionStore, tasks: TaskStore) -> str:
    task, repeated = await tasks.create_task(
        TaskSpec.from_payload({"objective": "Proven new task"}),
        origin="user",
        workspace=str(Path(sessions.path).parent),
        _recovery_proof=True,
    )
    assert not repeated
    return task.task_id


async def identity(tasks: TaskStore, task_id: str) -> dict:
    generation, operation_id = await tasks.park_identity(task_id)
    return {"expected_generation": generation, "expected_operation_id": operation_id}


async def start(tasks: TaskStore, recovery: RecoveryStore, task_id: str) -> dict:
    args = dict(
        operation_id="start-1",
        execution_id="execution-1",
        **await identity(tasks, task_id),
    )
    launched = await recovery.start(task_id, **args)
    assert launched.acquired
    return args


async def prepare(
    tasks: TaskStore,
    recovery: RecoveryStore,
    task_id: str,
    operation_id: str,
    **overrides: object,
) -> dict:
    args = dict(
        operation_id=operation_id,
        execution_id="execution-1",
        outcome="interrupted",
        card="Resume safely",
        reason="worker crashed",
        crash=True,
        unsafe_resume=True,
        **await identity(tasks, task_id),
    )
    args.update(overrides)
    await recovery.prepare(task_id, **args)
    return args


async def test_recovery_terminal_clears_pending_snapshot_and_finishes(
    stores: tuple,
) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    await tasks.request_pending_pause(
        task_id, operation_id="pause", **await identity(tasks, task_id)
    )
    await prepare(tasks, recovery, task_id, "terminal-recovery", outcome="failed")
    await recovery.commit(task_id, operation_id="terminal-recovery")
    task = await tasks.get_task(task_id)
    assert task.state == "failed" and task.finished_at is not None
    async with sessions.component_transaction("check_terminal_snapshot") as conn:
        row = await (
            await conn.execute(
                "SELECT pause_pending, snapshot_active, operation_id FROM park_priority WHERE task_id = ?",
                (task_id,),
            )
        ).fetchone()
    assert row is not None and tuple(row) == (0, 0, None)


async def test_repeated_create_cannot_upgrade_unknown_provenance(stores: tuple) -> None:
    sessions, tasks, recovery, _ = stores
    spec = TaskSpec.from_payload({"objective": "Old task"})
    kwargs = dict(
        origin="user", workspace=str(Path(sessions.path).parent), request_key="old"
    )
    old, _ = await tasks.create_task(spec, **kwargs)
    repeated, was_repeated = await tasks.create_task(
        spec, **kwargs, _recovery_proof=True
    )
    assert was_repeated and repeated.task_id == old.task_id
    assert (await recovery.provenance(old.task_id)).first_start == "unknown"


async def test_unknown_and_new_task_provenance(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    old = await recovery.provenance(task_id)
    assert old.first_start == "unknown" and old.execution_id is None
    assert await recovery.receipt(task_id, "absent") is None
    with pytest.raises(TaskStoreError):
        await recovery.start(
            task_id,
            operation_id="unsafe",
            execution_id="e",
            **await identity(tasks, task_id),
        )
    task_id = await mark_created(sessions, tasks)
    new = await recovery.provenance(task_id)
    assert new.first_start == "never_started" and new.execution_id is None


async def test_same_launch_operation_grants_only_one_permit(stores: tuple) -> None:
    sessions, tasks, recovery, _ = stores
    task_id = await mark_created(sessions, tasks)
    args = dict(
        operation_id="launch",
        execution_id="execution-1",
        **await identity(tasks, task_id),
    )
    launches = await asyncio.gather(
        *(recovery.start(task_id, **args) for _ in range(2))
    )
    assert sum(launch.acquired for launch in launches) == 1


async def test_first_start_rejects_unparked_p5(stores: tuple) -> None:
    sessions, tasks, recovery, _ = stores
    task, _ = await tasks.create_task(
        TaskSpec.from_user_payload({"objective": "paused", "level": 5}),
        origin="user",
        workspace=str(Path(sessions.path).parent),
        _recovery_proof=True,
    )
    with pytest.raises(TaskStoreError):
        await recovery.start(
            task.task_id,
            operation_id="start",
            execution_id="exec",
            **await identity(tasks, task.task_id),
        )
    assert (await tasks.get_task(task.task_id)).state == "queued"


async def test_concurrent_start_has_one_winner(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    guard = await identity(tasks, task_id)
    events: list[StoredEvent] = []
    sessions.add_listener(events.append)
    results = await asyncio.gather(
        *(
            recovery.start(
                task_id, operation_id=f"start-{i}", execution_id=f"exec-{i}", **guard
            )
            for i in range(2)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, TaskStoreError) for result in results) == 1
    assert (await tasks.get_task(task_id)).state == "running"
    provenance = await recovery.provenance(task_id)
    assert provenance.first_start == "started" and provenance.execution_id in {
        "exec-0",
        "exec-1",
    }
    assert len(events) == 1


async def test_start_retry_and_changed_payload(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    events: list[StoredEvent] = []
    sessions.add_listener(events.append)
    args = await start(tasks, recovery, task_id)
    first = await recovery.start(task_id, **args)
    assert not first.acquired
    assert await recovery.start(task_id, **args) == first
    with pytest.raises(TaskStoreError):
        await recovery.start(task_id, **{**args, "execution_id": "different"})
    receipt = await recovery.receipt(task_id, "start-1")
    assert receipt is not None and receipt[1] is True
    assert len(events) == 1


async def test_resumed_execution_requires_review_and_changes_identity(
    stores: tuple,
) -> None:
    sessions, tasks, recovery, _ = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    await prepare(tasks, recovery, task_id, "recover-1")
    await recovery.commit(task_id, operation_id="recover-1")
    await tasks.change(task_id, state="queued")
    args = dict(
        operation_id="resume-start",
        execution_id="execution-2",
        expected_execution_id="execution-1",
        **await identity(tasks, task_id),
    )
    with pytest.raises(TaskStoreError):
        await recovery.start(task_id, **args)
    # PR2/3 must provide the verified-gone or explicit-confirmation transition.
    async with sessions.component_transaction("simulate_review") as conn:
        await conn.execute(
            "UPDATE recovery_provenance SET unsafe_resume = 0 WHERE task_id = ?",
            (task_id,),
        )
    launched = await recovery.start(task_id, **args)
    assert launched.acquired and launched.provenance.execution_id == "execution-2"
    assert launched.provenance.unsafe_resume
    with pytest.raises(TaskStoreError):
        await recovery.record_process(
            task_id,
            execution_id="execution-1",
            pgid=11,
            leader_pid=11,
            boot_id="boot",
            start_token="old",
        )
    await recovery.record_process(
        task_id,
        execution_id="execution-2",
        pgid=12,
        leader_pid=12,
        boot_id="boot",
        start_token="new",
    )
    assert await recovery.processes(task_id, execution_id="execution-2") == [
        (12, 12, "boot", "new")
    ]


async def test_process_identity_is_execution_bound_and_retained(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    assert await recovery.processes(task_id, execution_id="execution-1") == []
    process = dict(
        execution_id="execution-1",
        pgid=123,
        leader_pid=123,
        boot_id="host-boot",
        start_token="birth-token",
    )
    await recovery.record_process(task_id, **process)
    await recovery.record_process(task_id, **process)
    assert await recovery.processes(task_id, execution_id="execution-1") == [
        (123, 123, "host-boot", "birth-token")
    ]
    with pytest.raises(TaskStoreError):
        await recovery.record_process(task_id, **{**process, "execution_id": "old"})
    with pytest.raises(ValueError):
        await recovery.record_process(task_id, **{**process, "pgid": True})
    await sessions.apply_retention()
    assert await recovery.processes(task_id, execution_id="execution-1") == [
        (123, 123, "host-boot", "birth-token")
    ]


async def test_prepared_recovery_cannot_overwrite_later_park(stores: tuple) -> None:
    sessions, tasks, recovery, _ = stores
    task_id = await mark_created(sessions, tasks)
    await recovery.prepare(
        task_id,
        operation_id="recovery",
        execution_id="unstarted",
        outcome="queued",
        card="",
        reason="",
        crash=False,
        unsafe_resume=False,
        **await identity(tasks, task_id),
    )
    await tasks.change(task_id, state="parked")
    with pytest.raises(TaskStoreError):
        await recovery.commit(task_id, operation_id="recovery")
    assert (await tasks.get_task(task_id)).state == "parked"


async def test_prepared_recovery_fences_first_launch(stores: tuple) -> None:
    sessions, tasks, recovery, _ = stores
    task_id = await mark_created(sessions, tasks)
    guard = await identity(tasks, task_id)
    assert await recovery.prepare(
        task_id,
        operation_id="recover",
        execution_id="unstarted",
        outcome="queued",
        card="",
        reason="",
        crash=False,
        unsafe_resume=False,
        **guard,
    )
    with pytest.raises(TaskStoreError):
        await recovery.start(
            task_id, operation_id="launch", execution_id="launch-id", **guard
        )
    assert (await tasks.get_task(task_id)).state == "queued"


async def test_started_recovery_cannot_claim_unverified_safety(stores: tuple) -> None:
    sessions, tasks, recovery, _ = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    with pytest.raises(TaskStoreError):
        await recovery.prepare(
            task_id,
            operation_id="recover",
            execution_id="execution-1",
            outcome="interrupted",
            card="card",
            reason="unknown",
            crash=False,
            unsafe_resume=False,
            **await identity(tasks, task_id),
        )


async def test_prepared_intent_is_discoverable_and_exclusive(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    args = await prepare(tasks, recovery, task_id, "intent-1")
    active = await recovery.active_intent(task_id)
    assert active is not None and active[0] == "intent-1"
    assert active[1]["execution_id"] == "execution-1"
    with pytest.raises(TaskStoreError):
        await recovery.prepare(task_id, **{**args, "operation_id": "intent-2"})
    await recovery.commit(task_id, operation_id="intent-1")
    assert await recovery.active_intent(task_id) is None


async def test_prepare_is_inert_and_commit_retry_after_cancel(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    before = await tasks.get_task(task_id)
    before_identity = await identity(tasks, task_id)
    args = await prepare(tasks, recovery, task_id, "recover-1")
    assert await tasks.get_task(task_id) == before
    assert await identity(tasks, task_id) == before_identity
    assert await recovery.prepare(task_id, **args) is False
    with pytest.raises(TaskStoreError):
        await recovery.prepare(task_id, **{**args, "card": "different"})
    events: list[StoredEvent] = []
    sessions.add_listener(events.append)

    async def lose_response() -> None:
        await recovery.commit(task_id, operation_id="recover-1")
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(lose_response())
    receipt = await recovery.receipt(task_id, "recover-1")
    assert receipt is not None and receipt[1] is True
    committed = await recovery.commit(task_id, operation_id="recover-1")
    assert await recovery.commit(task_id, operation_id="recover-1") == committed
    task = await tasks.get_task(task_id)
    assert (
        task.state == "interrupted"
        and task.card == "Resume safely"
        and task.crashes == 1
    )
    assert (await identity(tasks, task_id))["expected_generation"] > before_identity[
        "expected_generation"
    ]
    assert len(events) == 1


async def test_stale_cas_and_stop_supersede_prepared_recovery(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    stale = await identity(tasks, task_id)
    await start(tasks, recovery, task_id)
    with pytest.raises(TaskStoreError):
        await recovery.prepare(
            task_id,
            operation_id="stale",
            execution_id="execution-1",
            outcome="interrupted",
            card="card",
            reason="crash",
            crash=True,
            unsafe_resume=True,
            **stale,
        )
    await prepare(tasks, recovery, task_id, "recover-1")
    guard = await identity(tasks, task_id)
    with pytest.raises(TaskStoreError):
        await recovery.supersede(task_id, "recover-1", **guard)
    await tasks.change(task_id, state="cancelled", operation_id="stop-1", **guard)
    await recovery.supersede(task_id, "recover-1", **await identity(tasks, task_id))
    assert await recovery.active_intent(task_id) is None
    with pytest.raises(TaskStoreError):
        await recovery.commit(task_id, operation_id="recover-1")
    with pytest.raises(TaskStoreError):
        await recovery.start(
            task_id, operation_id="start-1", execution_id="execution-1", **stale
        )
    task = await tasks.get_task(task_id)
    assert task.state == "cancelled" and task.crashes == 0
    receipt = await recovery.receipt(task_id, "start-1")
    assert receipt is not None and receipt[1] is False


async def test_distinct_recoveries_count_one_crash_per_execution(stores: tuple) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    await prepare(tasks, recovery, task_id, "recover-1")
    await recovery.commit(task_id, operation_id="recover-1")
    # A second recovery of the same execution must not double-count the crash.
    await tasks.change(task_id, state="running")
    await prepare(tasks, recovery, task_id, "recover-2")
    await recovery.commit(task_id, operation_id="recover-2")
    assert (await tasks.get_task(task_id)).crashes == 1
    assert (await recovery.receipt(task_id, "recover-1"))[1] is False
    assert (await recovery.receipt(task_id, "recover-2"))[1] is True


async def test_receipt_and_provenance_survive_retention_and_reopen(
    stores: tuple,
) -> None:
    sessions, tasks, recovery, task_id = stores
    task_id = await mark_created(sessions, tasks)
    await start(tasks, recovery, task_id)
    await prepare(tasks, recovery, task_id, "recover-1", outcome="failed")
    await recovery.commit(task_id, operation_id="recover-1")
    before = await recovery.receipt(task_id, "recover-1")
    assert before is not None and before[1] is True
    await sessions.apply_retention()
    await tasks.apply_retention()
    await sessions.aclose()
    reopened = SessionStore(sessions.path, writer="gateway")
    await reopened.open()
    try:
        reopened_tasks = TaskStore(reopened)
        await reopened_tasks.open()
        recovered = RecoveryStore(reopened)
        assert await recovered.receipt(task_id, "recover-1") == before
        assert (await recovered.provenance(task_id)).execution_id == "execution-1"
        assert (await reopened_tasks.get_task(task_id)).crashes == 1
    finally:
        await reopened.aclose()
