"""Multi-session SQLite stress coverage for SqliteHistory.

Existing test in ``test_persistence.py`` exercises 20 concurrent ``add_step``
calls on a single :class:`SqliteHistory` instance, which only validates the
per-instance ``_write_lock``. The realistic gateway pattern is that each
session holds its own ``SqliteHistory`` object (its own aiosqlite connection)
pointing at the same ``history.db``. Between instances, the only serialization
is SQLite's ``BEGIN IMMEDIATE`` plus ``busy_timeout=5000`` plus WAL — this
test asserts that combination still produces dense, unique ``seq`` values
under contention. A cancellation while another connection holds the write
lock must not leave a connection inside its transaction either.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from localmelo.melo.memory import _sqlite as sqlite_helpers
from localmelo.melo.memory._sqlite import prepare_connection
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
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


@pytest.mark.asyncio
async def test_a_cancelled_add_step_leaves_no_transaction_open(tmp_path: Path) -> None:
    # A cancellation can reach add_step while its BEGIN IMMEDIATE waits for
    # another connection's write lock (up to busy_timeout). aiosqlite still
    # runs the BEGIN once the lock is free, so without a rollback queued
    # behind it the connection would stay inside a transaction: holding the
    # write lock, and refusing the next add_step's BEGIN.
    db_path = tmp_path / "history.db"
    history = SqliteHistory(db_path)
    try:
        await history.save_task(TaskRecord(query="q", task_id="t1"))
        with contextlib.closing(
            sqlite3.connect(db_path, isolation_level=None)
        ) as other:
            other.execute("BEGIN IMMEDIATE")
            writing = asyncio.create_task(
                history.add_step("t1", StepRecord(thought="cancelled"))
            )
            await asyncio.sleep(0.2)  # add_step waits at its BEGIN IMMEDIATE
            writing.cancel()
            await asyncio.sleep(0.05)
            other.execute("ROLLBACK")
            with pytest.raises(asyncio.CancelledError):
                await writing

        await history.add_step("t1", StepRecord(thought="next"))
        assert history._conn is not None
        assert not history._conn.in_transaction
        # No write lock is kept from other connections.
        with contextlib.closing(sqlite3.connect(db_path, timeout=0.1)) as other:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
        assert [step.thought for step in await history.get_steps("t1")] == ["next"]
    finally:
        await history.aclose()


class _LockedConn:
    """A connection whose schema step fails *locked* times, then works."""

    def __init__(self, locked: int, error: str = "database is locked") -> None:
        self.locked = locked
        self.error = error
        self.scripts = 0
        self.rollbacks = 0

    async def execute(self, sql: str) -> None:
        return None

    async def executescript(self, sql: str) -> None:
        self.scripts += 1
        if self.locked:
            self.locked -= 1
            raise sqlite3.OperationalError(self.error)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        self.rollbacks += 1


async def test_prepare_connection_runs_a_locked_open_again() -> None:
    conn = _LockedConn(locked=2)
    await prepare_connection(conn, "CREATE TABLE t (x)")  # type: ignore[arg-type]
    assert (conn.scripts, conn.rollbacks) == (3, 2)


async def test_prepare_connection_raises_any_other_error_at_once() -> None:
    conn = _LockedConn(locked=1, error="no such table: t")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        await prepare_connection(conn, "CREATE TABLE t (x)")  # type: ignore[arg-type]
    assert conn.scripts == 1


async def test_prepare_connection_gives_up_after_its_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sqlite_helpers, "LOCKED_RETRY_S", 0.05)
    conn = _LockedConn(locked=10_000)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        await prepare_connection(conn, "CREATE TABLE t (x)")  # type: ignore[arg-type]
    assert 1 < conn.scripts < 100


@pytest.mark.parametrize("store", ["history", "long_term"])
async def test_eight_connections_open_one_new_file_without_an_error(
    tmp_path: Path, store: str
) -> None:
    # Connections that open one new file at once used to get "database is
    # locked" without the busy timeout, about one run in eight.
    for attempt in range(10):
        path = tmp_path / f"{store}-{attempt}.db"
        if store == "history":
            stores: list[Any] = [SqliteHistory(path) for _ in range(8)]
            await asyncio.gather(*(s.count_tasks() for s in stores))
        else:
            stores = [SqliteLongTerm(path) for _ in range(8)]
            await asyncio.gather(*(s.count_entries() for s in stores))
        await asyncio.gather(*(s.aclose() for s in stores))
