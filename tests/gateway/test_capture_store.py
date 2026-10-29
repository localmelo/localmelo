"""Per-execution process capture remains conservative across restarts."""

from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import (
    SCHEDULER_VERSION,
    TaskStore,
    TaskStoreError,
)
from localmelo.support.sessions.store import SessionStore


async def test_v3_capture_migration_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "preserve"}),
            origin="user",
            workspace=str(tmp_path),
        )
        async with store.component_transaction("old_schema") as conn:
            await conn.execute("DROP TABLE recovery_capture")
            await conn.execute(
                "UPDATE schema_components SET version = 3 WHERE name = 'scheduler'"
            )
        original = store._before_commit

        def fail(label: str) -> None:
            original(label)
            if label == "component_scheduler":
                raise RuntimeError("migration failed")

        monkeypatch.setattr(store, "_before_commit", fail)
        with pytest.raises(RuntimeError):
            await TaskStore(store).open()
        async with store.component_transaction("check_schema") as conn:
            row = await (
                await conn.execute(
                    "SELECT version FROM schema_components WHERE name = 'scheduler'"
                )
            ).fetchone()
            table = await (
                await conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'recovery_capture'"
                )
            ).fetchone()
        assert row is not None and row[0] == 3 and table is None
        assert await tasks.get_task(task.task_id) == task
        monkeypatch.setattr(store, "_before_commit", original)
        migrated = TaskStore(store)
        assert await migrated.open() == SCHEDULER_VERSION
        assert await migrated.get_task(task.task_id) == task
        assert await migrated.open() == SCHEDULER_VERSION
        async with store.component_transaction("check_schema_tables") as conn:
            names = await (
                await conn.execute(
                    "SELECT name FROM sqlite_master WHERE name IN "
                    "('recovery_capture', 'recovery_one_capture')"
                )
            ).fetchall()
        assert {row[0] for row in names} == {
            "recovery_capture",
            "recovery_one_capture",
        }
    finally:
        await store.aclose()


async def test_v4_migration_adds_the_host_column(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        tasks = TaskStore(store)
        await tasks.open()
        async with store.component_transaction("old_schema") as conn:
            await conn.execute("ALTER TABLE recovery_processes DROP COLUMN host_id")
            await conn.execute(
                "UPDATE schema_components SET version = 4 WHERE name = 'scheduler'"
            )
        migrated = TaskStore(store)
        assert await migrated.open() == SCHEDULER_VERSION
        async with store.component_transaction("check_host") as conn:
            columns = await (
                await conn.execute("PRAGMA table_info(recovery_processes)")
            ).fetchall()
        assert "host_id" in {row[1] for row in columns}
    finally:
        await store.aclose()


async def _host_columns(store: SessionStore) -> set[str]:
    async with store.component_transaction("check_host") as conn:
        rows = await (
            await conn.execute("PRAGMA table_info(recovery_processes)")
        ).fetchall()
    return {row[1] for row in rows}


async def test_v3_store_gains_the_host_column_and_records_a_host(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        await TaskStore(store).open()
        async with store.component_transaction("old_schema") as conn:
            # v3 had recovery_processes (without host_id) but no capture table.
            await conn.execute("ALTER TABLE recovery_processes DROP COLUMN host_id")
            await conn.execute("DROP TABLE recovery_capture")
            await conn.execute(
                "UPDATE schema_components SET version = 3 WHERE name = 'scheduler'"
            )
        tasks = TaskStore(store)
        assert await tasks.open() == SCHEDULER_VERSION
        assert "host_id" in await _host_columns(store)
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "work"}),
            origin="user",
            workspace=str(tmp_path),
            _recovery_proof=True,
        )
        recovery = RecoveryStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        await recovery.start(
            task.task_id,
            operation_id="start",
            execution_id="run",
            expected_generation=generation,
            expected_operation_id=previous,
            _capture=True,
        )
        await recovery.record_process(
            task.task_id,
            execution_id="run",
            pgid=11,
            leader_pid=11,
            boot_id="boot",
            start_token="start",
            host_id="host-a",
        )
        assert await recovery.process_host(task.task_id, execution_id="run") == "host-a"
    finally:
        await store.aclose()


async def test_v5_store_without_the_host_column_is_repaired_on_open(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    try:
        await TaskStore(store).open()
        async with store.component_transaction("old_schema") as conn:
            await conn.execute("ALTER TABLE recovery_processes DROP COLUMN host_id")
        assert "host_id" not in await _host_columns(store)
        repaired = TaskStore(store)
        assert await repaired.open() == SCHEDULER_VERSION
        assert "host_id" in await _host_columns(store)
        assert await TaskStore(store).open() == SCHEDULER_VERSION
    finally:
        await store.aclose()


async def test_process_host_is_unknown_unless_every_group_shares_it(
    tmp_path: Path,
) -> None:
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
        recovery = RecoveryStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        await recovery.start(
            task.task_id,
            operation_id="start",
            execution_id="run",
            expected_generation=generation,
            expected_operation_id=previous,
            _capture=True,
        )
        assert await recovery.process_host(task.task_id, execution_id="run") == ""

        async def record(pgid: int, host: str) -> None:
            await recovery.record_process(
                task.task_id,
                execution_id="run",
                pgid=pgid,
                leader_pid=pgid,
                boot_id="boot",
                start_token="start",
                host_id=host,
            )

        await record(11, "host-a")
        await record(12, "host-a")
        assert await recovery.process_host(task.task_id, execution_id="run") == "host-a"
        # A group recorded on another host makes the whole set unknown.
        await record(13, "host-b")
        assert await recovery.process_host(task.task_id, execution_id="run") == ""
    finally:
        await store.aclose()


async def test_capture_is_execution_bound_and_incomplete_is_permanent(
    tmp_path: Path,
) -> None:
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
        recovery = RecoveryStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        launch_args: dict[str, Any] = dict(
            operation_id="start",
            execution_id="run",
            expected_generation=generation,
            expected_operation_id=previous,
            _capture=True,
        )
        assert not await recovery.capture_complete(task.task_id, execution_id="run")
        await recovery.start(task.task_id, **launch_args)
        assert await recovery.capture_complete(task.task_id, execution_id="run")
        assert not (await recovery.start(task.task_id, **launch_args)).acquired
        generation, previous = await tasks.park_identity(task.task_id)
        process: dict[str, Any] = dict(
            execution_id="run",
            pgid=123,
            leader_pid=123,
            boot_id="boot",
            start_token="birth",
            expected_generation=generation,
            expected_operation_id=previous,
        )
        await recovery.record_captured_process(task.task_id, **process)
        assert await recovery.processes(task.task_id, execution_id="run") == [
            (123, 123, "boot", "birth")
        ]
        assert not (await recovery.start(task.task_id, **launch_args)).acquired
        with pytest.raises(TaskStoreError):
            await recovery.record_captured_process(task.task_id, **process)
        stale = await tasks.park_identity(task.task_id)
        await recovery.mark_capture_incomplete(
            task.task_id,
            execution_id="run",
            expected_generation=stale[0],
            expected_operation_id=stale[1],
        )
        assert not await recovery.capture_complete(task.task_id, execution_id="run")
        with pytest.raises(TaskStoreError):
            await recovery.start(
                task.task_id, **{**launch_args, "operation_id": "second"}
            )
        with pytest.raises(TaskStoreError):
            await recovery.mark_capture_incomplete(
                task.task_id,
                execution_id="other",
                expected_generation=stale[0],
                expected_operation_id=stale[1],
            )
    finally:
        await store.aclose()


@pytest.mark.parametrize("action", ["mark", "record"])
@pytest.mark.parametrize("scenario", ["new_launch", "ordinary_launch", "prepared"])
async def test_capture_writes_cannot_cross_execution_or_recovery_fence(
    tmp_path: Path, action: str, scenario: str
) -> None:
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
        recovery = RecoveryStore(store)
        generation, previous = await tasks.park_identity(task.task_id)
        await recovery.start(
            task.task_id,
            operation_id="first",
            execution_id="old",
            expected_generation=generation,
            expected_operation_id=previous,
            _capture=True,
        )
        if scenario == "new_launch":
            # Simulate a separately authorized future Resume, which PR3 owns.
            async with store.component_transaction("authorize_resume") as conn:
                await conn.execute(
                    "UPDATE tasks SET state='queued', card='card' WHERE task_id=?",
                    (task.task_id,),
                )
                await conn.execute(
                    "UPDATE recovery_provenance SET unsafe_resume=0 WHERE task_id=?",
                    (task.task_id,),
                )
            generation, previous = await tasks.park_identity(task.task_id)
            await recovery.start(
                task.task_id,
                operation_id="second",
                execution_id="new",
                expected_execution_id="old",
                expected_generation=generation,
                expected_operation_id=previous,
            )
            assert not await recovery.capture_complete(task.task_id, execution_id="old")
            async with store.component_transaction("stale_capture") as conn:
                row = await (
                    await conn.execute(
                        "SELECT complete, trusted, active FROM recovery_capture "
                        "WHERE task_id = ? AND execution_id = 'old'",
                        (task.task_id,),
                    )
                ).fetchone()
                assert row is not None and tuple(row) == (0, 1, 0)
                await conn.execute(
                    "UPDATE recovery_capture SET complete = 1, trusted = 1 "
                    "WHERE task_id = ? AND execution_id = 'old'",
                    (task.task_id,),
                )
            assert not await recovery.capture_complete(task.task_id, execution_id="old")
        elif scenario == "ordinary_launch":
            await tasks.change(task.task_id, state="queued")
            await tasks.change(task.task_id, state="running")
            assert not await recovery.capture_complete(task.task_id, execution_id="old")
        else:
            generation, previous = await tasks.park_identity(task.task_id)
            await recovery.prepare(
                task.task_id,
                operation_id="recovery",
                execution_id="old",
                expected_generation=generation,
                expected_operation_id=previous,
                outcome="interrupted",
                card="card",
                reason="restart",
                crash=True,
                unsafe_resume=True,
            )
        generation, previous = await tasks.park_identity(task.task_id)
        with pytest.raises(TaskStoreError):
            if action == "mark":
                await recovery.mark_capture_incomplete(
                    task.task_id,
                    execution_id="old",
                    expected_generation=generation,
                    expected_operation_id=previous,
                )
            else:
                await recovery.record_captured_process(
                    task.task_id,
                    execution_id="old",
                    pgid=123,
                    leader_pid=123,
                    boot_id="boot",
                    start_token="birth",
                    expected_generation=generation,
                    expected_operation_id=previous,
                )
        assert await tasks.park_identity(task.task_id) == (generation, previous)
        if scenario == "prepared":
            await recovery.commit(task.task_id, operation_id="recovery")
            assert (await tasks.get_task(task.task_id)).state == "interrupted"
    finally:
        await store.aclose()
