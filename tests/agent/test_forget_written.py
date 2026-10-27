"""Forgetting a routine leaves none of it in a procedures.db that runs wrote.

Runs append template versions and supersede older ones, which leaves
stale copies of rows in the file's free pages. ``melo-procedures forget``
must leave no label or id of the forgotten routine in the file or its
-wal, while the routines it keeps stay readable.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.procedures import store as store_module
from localmelo.melo.memory.procedures.store import SqliteProcedures
from localmelo.tests.agent.procedural_helpers import (
    ScriptedLLM,
    answer,
    make_agent,
    read,
)

SECRET_FILE = "/p/zebra_quokka_payroll.py"


@pytest.fixture(autouse=True)
def secure_delete_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Act as an SQLite built with secure_delete off, as macOS's is.

    Some builds turn it on by default; the store must not rely on that.
    """
    real = store_module.prepare_connection

    async def prepare(conn: Any, schema: str, **kwargs: Any) -> None:
        await conn.execute("PRAGMA secure_delete = OFF")
        await real(conn, schema, **kwargs)

    monkeypatch.setattr(store_module, "prepare_connection", prepare)


def _bytes(memory: Path) -> bytes:
    return b"".join(
        path.read_bytes()
        for path in sorted(memory.iterdir())
        if path.name.startswith("procedures.db")
    )


async def test_forget_leaves_no_trace_of_a_routine_runs_wrote(tmp_path: Path) -> None:
    script = []
    for i in range(4):
        script += [read(SECRET_FILE, f"call_{i}"), answer("done")]
    script += [read("/p/kept.py"), answer("done")]
    agent = make_agent(ScriptedLLM(script), memory_dir=str(tmp_path))
    try:
        for _ in range(4):
            await agent.run_task("summarize the payroll file")
        await agent.run_task("summarize the kept file")
    finally:
        await agent.close()
    with closing(sqlite3.connect(tmp_path / "procedures.db")) as db:
        (forgotten,) = db.execute(
            "SELECT routine_id FROM routines WHERE query_text LIKE '%payroll%'"
        ).fetchone()
    assert b"zebra_quokka" in _bytes(tmp_path)

    store = SqliteProcedures(tmp_path / "procedures.db", create=False)
    try:
        assert await store.forget(forgotten) > 0
    finally:
        await store.aclose()

    data = _bytes(tmp_path)
    assert b"zebra_quokka" not in data
    assert b"payroll" not in data
    assert forgotten.encode() not in data
    with closing(sqlite3.connect(tmp_path / "procedures.db")) as db:
        kept = db.execute("SELECT query_text FROM routines").fetchall()
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert kept == [("summarize the kept file",)]


async def test_every_connection_overwrites_what_it_deletes(tmp_path: Path) -> None:
    store = SqliteProcedures(tmp_path / "procedures.db")
    try:
        conn = await store._open()
        async with conn.execute("PRAGMA secure_delete") as cursor:
            row = await cursor.fetchone()
    finally:
        await store.aclose()
    assert row is not None and row[0] == 1
