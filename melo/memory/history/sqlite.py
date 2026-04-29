from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiosqlite

from localmelo.melo.memory._sqlite import apply_pragmas
from localmelo.melo.memory.history import History
from localmelo.melo.schema import StepRecord, TaskRecord, ToolCall, ToolResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    query   TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'running',
    result  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS steps (
    step_id          TEXT PRIMARY KEY,
    task_id          TEXT NOT NULL,
    thought          TEXT NOT NULL DEFAULT '',
    tool_call_json   TEXT,
    tool_result_json TEXT,
    timestamp        REAL NOT NULL,
    seq              INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
"""


class SqliteHistory(History):
    """Persistent history backed by SQLite via aiosqlite.

    The connection is opened lazily on first awaited call so the
    constructor stays synchronous (compatible with ``Agent.__init__``).
    All write paths run under ``_write_lock`` so the
    ``SELECT MAX(seq) → INSERT`` sequence in :meth:`add_step` is atomic
    across concurrent callers on the same event loop.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def _ensure_ready(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            if self._conn is not None:
                return self._conn
            conn = await aiosqlite.connect(self._db_path)
            await apply_pragmas(conn)
            await conn.executescript(_SCHEMA)
            await conn.commit()
            self._conn = conn
            return conn

    # ── public interface (mirrors History) ──

    async def save_task(self, task: TaskRecord) -> None:
        conn = await self._ensure_ready()
        async with self._write_lock:
            await conn.execute(
                """INSERT OR REPLACE INTO tasks (task_id, query, status, result)
                   VALUES (?, ?, ?, ?)""",
                (task.task_id, task.query, task.status, task.result),
            )
            await conn.commit()

    async def get_task(self, task_id: str) -> TaskRecord | None:
        conn = await self._ensure_ready()
        async with conn.execute(
            "SELECT task_id, query, status, result FROM tasks WHERE task_id = ?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        task = TaskRecord(query=row[1], task_id=row[0], status=row[2], result=row[3])
        task.steps = await self.get_steps(task_id)
        return task

    async def add_step(self, task_id: str, step: StepRecord) -> None:
        conn = await self._ensure_ready()

        tc_json = None
        if step.tool_call:
            tc_json = json.dumps(
                {
                    "tool_name": step.tool_call.tool_name,
                    "arguments": step.tool_call.arguments,
                }
            )

        tr_json = None
        if step.tool_result:
            tr_json = json.dumps(
                {
                    "tool_name": step.tool_result.tool_name,
                    "output": step.tool_result.output,
                    "error": step.tool_result.error,
                    "duration_ms": step.tool_result.duration_ms,
                }
            )

        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                async with conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) FROM steps WHERE task_id = ?",
                    (task_id,),
                ) as cur:
                    row = await cur.fetchone()
                seq = (row[0] if row else -1) + 1
                await conn.execute(
                    """INSERT INTO steps
                       (step_id, task_id, thought, tool_call_json, tool_result_json,
                        timestamp, seq)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        step.step_id,
                        task_id,
                        step.thought,
                        tc_json,
                        tr_json,
                        step.timestamp,
                        seq,
                    ),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def get_steps(self, task_id: str) -> list[StepRecord]:
        conn = await self._ensure_ready()
        async with conn.execute(
            """SELECT step_id, thought, tool_call_json, tool_result_json, timestamp
               FROM steps WHERE task_id = ? ORDER BY seq""",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()

        steps: list[StepRecord] = []
        for row in rows:
            tc = None
            if row[2]:
                d = json.loads(row[2])
                tc = ToolCall(
                    tool_name=d["tool_name"],
                    arguments=d.get("arguments", {}),
                )

            tr = None
            if row[3]:
                d = json.loads(row[3])
                tr = ToolResult(
                    tool_name=d["tool_name"],
                    output=d.get("output", ""),
                    error=d.get("error", ""),
                    duration_ms=d.get("duration_ms", 0.0),
                )

            steps.append(
                StepRecord(
                    step_id=row[0],
                    thought=row[1],
                    tool_call=tc,
                    tool_result=tr,
                    timestamp=row[4],
                )
            )
        return steps

    async def count_tasks(self) -> int:
        """Diagnostic helper: total tasks persisted."""
        conn = await self._ensure_ready()
        async with conn.execute("SELECT COUNT(*) FROM tasks") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def count_steps(self) -> int:
        """Diagnostic helper: total steps across all tasks."""
        conn = await self._ensure_ready()
        async with conn.execute("SELECT COUNT(*) FROM steps") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def aclose(self) -> None:
        """Close the underlying aiosqlite connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def close(self) -> None:
        """Sync compatibility shim. Prefer :meth:`aclose` in new code.

        Drops the connection reference so GC eventually releases the FD.
        Awaitable cleanup must go through :meth:`aclose`.
        """
        self._conn = None
