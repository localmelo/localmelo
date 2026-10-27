"""A locked procedures.db holds a run up for at most its busy timeout.

The store waits 500 ms for another writer (BUSY_TIMEOUT_MS), also while it
opens the file and creates the schema: a run's end that is cut short
must not leave SQLite's thread waiting seconds longer, which closing the
store, and so the process's exit, would wait for.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from localmelo.melo.memory.procedures.normalize import key_for_query
from localmelo.melo.memory.procedures.store import (
    BUSY_TIMEOUT_MS,
    RunFacts,
    SqliteProcedures,
)


def facts() -> RunFacts:
    return RunFacts(
        task_id=uuid.uuid4().hex,
        key=key_for_query("run the tests", "/work"),
        model="m",
        options={},
        status="completed",
        steps=2,
    )


@pytest.fixture
def locked(tmp_path: Path) -> Iterator[Path]:
    """A new procedures.db, without schema, whose write lock another holds."""
    path = tmp_path / "procedures.db"
    holder = sqlite3.connect(path, timeout=0, isolation_level=None)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("CREATE TABLE elsewhere (a)")
    try:
        yield path
    finally:
        holder.execute("ROLLBACK")
        holder.close()


async def test_a_locked_new_file_fails_within_about_the_busy_timeout(
    locked: Path,
) -> None:
    assert BUSY_TIMEOUT_MS == 500
    store = SqliteProcedures(locked)
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            await store.learn(facts())
    finally:
        await store.aclose()
    assert time.monotonic() - started < 2.0


async def test_closing_after_a_cut_short_write_does_not_wait_seconds(
    locked: Path,
) -> None:
    store = SqliteProcedures(locked)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await store.learn(facts())
    started = time.monotonic()
    await store.aclose()
    assert time.monotonic() - started < 1.5


async def test_a_lookup_reads_a_current_file_while_another_writes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "procedures.db"
    writer = SqliteProcedures(path)
    for _ in range(2):
        await writer.learn(facts())
    await writer.aclose()
    holder = sqlite3.connect(path, timeout=0, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        reader = SqliteProcedures(path)
        try:
            # A new connection, at the current schema version: opening it
            # takes no write lock, so the read is not refused.
            view = await reader.lookup(facts().key, model="m", options={})
        finally:
            await reader.aclose()
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert view is not None and view.version == 2


async def test_an_old_file_is_still_brought_to_the_current_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "procedures.db"
    sqlite3.connect(path).close()  # an empty file: user_version 0
    store = SqliteProcedures(path)
    try:
        await store.export_rows()
    finally:
        await store.aclose()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (1,)
