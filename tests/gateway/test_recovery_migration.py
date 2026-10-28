"""The scheduler's recovery schema migration is atomic and conservative."""

from pathlib import Path

import pytest

from localmelo.support.gateway.recovery_store import RecoveryStore
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore


async def test_v2_recovery_migration_rolls_back_and_preserves_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await store.open()
    tasks = TaskStore(store)
    await tasks.open()
    try:
        task, _ = await tasks.create_task(
            TaskSpec.from_payload({"objective": "test", "level": 1}),
            origin="user",
            workspace=str(tmp_path),
        )
        await tasks.pause(task.task_id)
        before = await tasks.get_task(task.task_id)
        identity = await tasks.park_identity(task.task_id)
        async with store.component_transaction("make_v2") as conn:
            for table in (
                "recovery_capture",
                "recovery_processes",
                "recovery_crashes",
                "recovery_operations",
                "recovery_provenance",
            ):
                await conn.execute(f"DROP TABLE {table}")
            await conn.execute(
                "UPDATE schema_components SET version = 2 WHERE name = 'scheduler'"
            )
        original = store._before_commit

        def fail(label: str) -> None:
            original(label)
            if label == "component_scheduler":
                raise RuntimeError("migration failed")

        monkeypatch.setattr(store, "_before_commit", fail)
        with pytest.raises(RuntimeError, match="migration failed"):
            await TaskStore(store).open()
        async with store.component_transaction("inspect_failed_migration") as conn:
            version = await (
                await conn.execute(
                    "SELECT version FROM schema_components WHERE name = 'scheduler'"
                )
            ).fetchone()
            existing_table = await (
                await conn.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'recovery_operations'"
                )
            ).fetchone()
        assert version is not None and version[0] == 2
        assert existing_table is None
        monkeypatch.setattr(store, "_before_commit", original)
        await TaskStore(store).open()
        assert await tasks.get_task(task.task_id) == before
        assert await tasks.park_identity(task.task_id) == identity
        provenance = await RecoveryStore(store).provenance(task.task_id)
        assert provenance.first_start == "unknown" and provenance.unsafe_resume
    finally:
        await store.aclose()
