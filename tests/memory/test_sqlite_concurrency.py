"""Multi-session SQLite stress coverage for SqliteHistory.

Existing test in ``test_persistence.py`` exercises 20 concurrent ``add_step``
calls on a single :class:`SqliteHistory` instance, which only validates the
per-instance ``_write_lock``. The realistic gateway pattern is that each
session holds its own ``SqliteHistory`` object (its own aiosqlite connection)
pointing at the same ``history.db``. Between instances, the only serialization
is SQLite's ``BEGIN IMMEDIATE`` plus ``busy_timeout=5000`` plus WAL — this
test asserts that combination still produces dense, unique ``seq`` values
under contention.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.schema import StepRecord, TaskRecord


@pytest.mark.asyncio
async def test_multi_session_concurrent_writers(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    task_id = "stress-1"
    writers = 20
    steps_per_writer = 2
    expected_total = writers * steps_per_writer

    # Seed the shared TaskRecord with a dedicated, then-closed instance so
    # the parallel writers each open their own connection — mirrors the
    # one-connection-per-session shape used by the gateway.
    seeder = SqliteHistory(db_path)
    await seeder.save_task(TaskRecord(query="stress", task_id=task_id))
    await seeder.aclose()

    async def writer(wid: int) -> None:
        h = SqliteHistory(db_path)
        try:
            for sid in range(steps_per_writer):
                await h.add_step(task_id, StepRecord(thought=f"w{wid}-s{sid}"))
        finally:
            await h.aclose()

    await asyncio.gather(*(writer(wid) for wid in range(writers)))

    reader = SqliteHistory(db_path)
    try:
        steps = await reader.get_steps(task_id)
    finally:
        await reader.aclose()

    expected_thoughts = [
        f"w{wid}-s{sid}" for wid in range(writers) for sid in range(steps_per_writer)
    ]
    assert len(steps) == expected_total
    assert sorted(s.thought for s in steps) == sorted(expected_thoughts)

    # seq is not surfaced via get_steps; read it directly to confirm the
    # cross-instance BEGIN IMMEDIATE actually serializes writes.
    conn = await aiosqlite.connect(str(db_path))
    try:
        async with conn.execute(
            "SELECT seq FROM steps WHERE task_id = ? ORDER BY seq",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
    finally:
        await conn.close()
    seqs = [r[0] for r in rows]
    assert seqs == list(range(expected_total))
