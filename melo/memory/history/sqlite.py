from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import aiosqlite

from localmelo.melo.memory._sqlite import (
    THINKING_RETENTION_DAYS,
    apply_pragmas,
    create_private_file,
)
from localmelo.melo.memory.history import History
from localmelo.melo.schema import (
    ImagePart,
    StepRecord,
    TaskRecord,
    ToolCall,
    ToolResult,
)

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

logger = logging.getLogger(__name__)

# The history schema this code reads and writes, kept in the file's
# ``PRAGMA user_version``. Version 0 is a file written before llm_calls.
HISTORY_SCHEMA_VERSION: Final = 1

# One row per model response (see :meth:`SqliteHistory.add_llm_call`). No
# foreign key, so no row is ever refused for its task, and no index: the
# primary key serves per-task reads.
_LLM_CALLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
  task_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  step INTEGER, attempt INTEGER,
  outcome TEXT NOT NULL,
  content TEXT NOT NULL DEFAULT '', thinking TEXT NOT NULL DEFAULT '',
  reasoning_key TEXT NOT NULL DEFAULT '',
  step_ids_json TEXT,
  redactions INTEGER NOT NULL DEFAULT 0,
  tool_calls_json TEXT, usage_json TEXT, duration_s REAL, timestamp REAL NOT NULL,
  PRIMARY KEY (task_id, seq));
"""


class HistorySchemaError(RuntimeError):
    """``history.db`` has a newer schema version than this code reads."""


# The tags of an inline reasoning block, in any letter case.
_OPEN_TAG = "<[Tt][Hh][Ii][Nn][Kk]>"
_CLOSE_TAG = "</[Tt][Hh][Ii][Nn][Kk]>"
# After leading whitespace, a block up to its first closing tag and the
# whitespace after it.
_LEADING_BLOCK = re.compile(rf"\s*{_OPEN_TAG}.*?{_CLOSE_TAG}\s*", re.DOTALL)
# After leading whitespace, an opening tag.
_LEADING_OPENING = re.compile(rf"\s*(?={_OPEN_TAG})")
_OPENING = re.compile(_OPEN_TAG)
# A closing tag and the whitespace after it.
_CLOSING = re.compile(rf"{_CLOSE_TAG}\s*")


def _ollama_prefix(thinking: str) -> str:
    """The copy of *thinking* that Ollama's provider puts before the content."""
    return f"<think>\n{thinking}\n</think>\n\n"


def _without_inline_blocks(text: str) -> str:
    """*text* without the reasoning a reply without a reasoning parser left in it.

    The first of these forms that applies is removed, again until none
    applies, so the result is a fixed point:

    1. after leading whitespace, a ``<think>`` up to and including the
       first ``</think>`` and the whitespace after it;
    2. after leading whitespace, a ``<think>`` with no ``</think>`` after
       it: everything from the ``<think>`` on (output cut off while the
       model thought);
    3. a ``</think>`` with no ``<think>`` before it: everything up to and
       including it and the whitespace after it (a chat template that
       opens the block in the prompt).

    Any other text is left as it is, a tag in its middle too.
    """
    start = 0
    while True:
        block = _LEADING_BLOCK.match(text, start)
        if block is not None:
            start = block.end()
            continue
        opening = _LEADING_OPENING.match(text, start)
        if opening is not None:
            return text[start : opening.end()]
        closing = _CLOSING.search(text, start)
        if closing is not None and not _OPENING.search(text, start, closing.start()):
            start = closing.end()
            continue
        return text[start:]


def _content_without_reasoning(reasoning_key: str, thinking: str, content: str) -> str:
    """*content* without the inline reasoning its row's *reasoning_key* implies.

    - ``thinking`` (Ollama's provider): the copy of *thinking* that the
      provider put before the content, when it is there;
    - ``""`` (no separate reasoning): see :func:`_without_inline_blocks`;
    - any other key, such as ``reasoning_content``: none, since the server
      returned the reasoning apart from the content.
    """
    if reasoning_key == "thinking":
        prefix = _ollama_prefix(thinking)
        if thinking and content.startswith(prefix):
            return content[len(prefix) :]
        return content
    if reasoning_key == "":
        return _without_inline_blocks(content)
    return content


class SqliteHistory(History):
    """Persistent history backed by SQLite via aiosqlite.

    The connection is opened lazily on first awaited call so the
    constructor stays synchronous (compatible with ``Agent.__init__``).
    All write paths run under ``_write_lock`` so the
    ``SELECT MAX(seq) → INSERT`` sequence in :meth:`add_step` is atomic
    across concurrent callers on the same event loop. That sequence runs
    in a ``BEGIN IMMEDIATE`` transaction, which a failure or a
    cancellation at any of its statements, the ``BEGIN`` included, rolls
    back, so the shared connection is never left inside it.

    Tool-result images are stored as references (MIME type, SHA-256
    digest, size), never their base64 payload, and load back as
    reference :class:`ImagePart` objects.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        thinking_retention_days: float = THINKING_RETENTION_DAYS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not (math.isfinite(thinking_retention_days) and thinking_retention_days > 0):
            raise ValueError(
                "thinking_retention_days must be a finite number of days "
                f"greater than 0, not {thinking_retention_days!r}"
            )
        self._db_path = str(db_path)
        self._thinking_retention_days = thinking_retention_days
        # Read only for the retention's cutoff.
        self._clock = clock
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        # The retention pass runs once per instance, at the first call that
        # opens the connection; a reconnection after aclose() skips it.
        self._retention_done = False

    async def _ensure_ready(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            if self._conn is not None:
                return self._conn
            if self._db_path != ":memory:":
                create_private_file(self._db_path)
            conn = await aiosqlite.connect(self._db_path)
            try:
                await apply_pragmas(conn)
                await conn.executescript(_SCHEMA)
                await conn.commit()
                await self._migrate(conn)
            except BaseException:
                await conn.close()
                raise
            self._conn = conn
            return conn

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        """Bring the file to :data:`HISTORY_SCHEMA_VERSION`, in one transaction.

        The version is read inside the ``BEGIN IMMEDIATE`` transaction, so
        two openers cannot both migrate. A newer file raises
        :class:`HistorySchemaError`. Every statement runs with
        ``execute``: ``executescript`` would commit the transaction first.
        """
        migrated = False
        try:
            await conn.execute("BEGIN IMMEDIATE")
            async with conn.execute("PRAGMA user_version") as cur:
                row = await cur.fetchone()
            version = int(row[0]) if row else 0
            if version > HISTORY_SCHEMA_VERSION:
                raise HistorySchemaError(
                    f"{self._db_path} has schema version {version}; this "
                    f"localmelo reads version {HISTORY_SCHEMA_VERSION}"
                )
            await conn.execute(_LLM_CALLS_SCHEMA)
            if version < HISTORY_SCHEMA_VERSION:
                await conn.execute(f"PRAGMA user_version = {HISTORY_SCHEMA_VERSION}")
                migrated = True
            if not self._retention_done:
                await self._clear_old_reasoning(conn)
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
        self._retention_done = True
        if migrated:
            logger.info(
                "Migrated %s to history schema version %d",
                self._db_path,
                HISTORY_SCHEMA_VERSION,
            )

    async def _clear_old_reasoning(self, conn: aiosqlite.Connection) -> None:
        """Clear the reasoning of the rows older than the retention period.

        Inside :meth:`_migrate`'s transaction. Each ``llm_calls`` row whose
        ``timestamp`` is older than ``thinking_retention_days`` loses its
        ``thinking`` and the inline reasoning its ``content`` holds (see
        :func:`_content_without_reasoning`); every other column, and the
        row, stays. A row that would not change is not written, so a pass
        over rows cleared before writes nothing.
        """
        cutoff = self._clock() - self._thinking_retention_days * 86400
        updates: list[tuple[str, str, int]] = []
        # Only rows that can change: with thinking, or with no reasoning key
        # and a tag in their content (LIKE ignores ASCII letter case).
        async with conn.execute(
            """SELECT task_id, seq, reasoning_key, thinking, content
               FROM llm_calls
               WHERE timestamp < ?
                 AND (thinking != ''
                      OR (reasoning_key = '' AND content LIKE '%think>%'))""",
            (cutoff,),
        ) as cur:
            async for task_id, seq, reasoning_key, thinking, content in cur:
                kept = _content_without_reasoning(reasoning_key, thinking, content)
                if thinking or kept != content:
                    updates.append((kept, task_id, seq))
        if updates:
            await conn.executemany(
                """UPDATE llm_calls SET thinking = '', content = ?
                   WHERE task_id = ? AND seq = ?""",
                updates,
            )

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
                    "id": step.tool_call.id,
                }
            )

        tr_json = None
        if step.tool_result:
            tr_fields: dict[str, Any] = {
                "tool_name": step.tool_result.tool_name,
                "output": step.tool_result.output,
                "error": step.tool_result.error,
                "duration_ms": step.tool_result.duration_ms,
            }
            if step.tool_result.images:
                # References only: the base64 payload is never stored.
                tr_fields["images"] = [
                    {
                        "mime_type": image.mime_type,
                        "sha256": image.sha256,
                        "size_bytes": image.size_bytes,
                    }
                    for image in step.tool_result.images
                ]
            tr_json = json.dumps(tr_fields)

        async with self._write_lock:
            # BEGIN IMMEDIATE is inside the try, so that a BEGIN that fails or
            # is cancelled is rolled back too. While another connection holds
            # the write lock, it waits up to busy_timeout, and aiosqlite still
            # runs it after its caller is cancelled: the rollback, queued
            # behind it, ends the transaction it then opens, which would
            # otherwise keep the write lock and refuse the next BEGIN.
            try:
                await conn.execute("BEGIN IMMEDIATE")
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
                # Rows written before tool call ids were stored have no
                # "id"; ToolCall then generates a fresh one.
                tc = ToolCall(
                    tool_name=d["tool_name"],
                    arguments=d.get("arguments", {}),
                    id=d.get("id", ""),
                )

            tr = None
            if row[3]:
                d = json.loads(row[3])
                # Rows written before image references were stored have
                # no "images".
                tr = ToolResult(
                    tool_name=d["tool_name"],
                    output=d.get("output", ""),
                    error=d.get("error", ""),
                    duration_ms=d.get("duration_ms", 0.0),
                    images=[
                        ImagePart(
                            mime_type=image["mime_type"],
                            sha256=image["sha256"],
                            size_bytes=image["size_bytes"],
                        )
                        for image in d.get("images", [])
                    ],
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
