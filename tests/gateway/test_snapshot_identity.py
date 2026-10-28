"""Regression for keeping identity after a pause snapshot is consumed."""

from pathlib import Path

from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.task_store import TaskStore
from localmelo.support.sessions.store import SessionStore


async def test_retained_identity_does_not_restore_consumed_priority(
    tmp_path: Path,
) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
        tasks = TaskStore(sessions)
        await tasks.open()
        record, _ = await tasks.create_task(
            TaskSpec.from_user_payload({"objective": "Test", "level": 1}),
            origin="user",
            workspace=str(tmp_path),
        )
        await tasks.pause(record.task_id)
        assert (await tasks.resume(record.task_id)).level == 1
        async with sessions.component_transaction("identity_check") as conn:
            row = await (
                await conn.execute(
                    "SELECT generation FROM park_priority WHERE task_id = ?",
                    (record.task_id,),
                )
            ).fetchone()
            assert row is not None
        await tasks.change(record.task_id, priority=5, state="parked")
        resumed = await tasks.resume(record.task_id)
        assert (resumed.level, resumed.level_source) == (2, "computed")
    finally:
        await sessions.aclose()
