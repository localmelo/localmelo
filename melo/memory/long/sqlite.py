from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from localmelo.melo.memory._sqlite import (
    checkpoint_wal,
    create_private_file,
    prepare_connection,
    secure_deletes,
)
from localmelo.melo.memory.long import LongTerm

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS long_term (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    text     TEXT NOT NULL,
    vector   BLOB NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


class SqliteLongTerm(LongTerm):
    """Persistent long-term memory backed by SQLite via aiosqlite.

    Vectors are stored as raw float32 ``BLOB``s. The connection is
    opened lazily on first awaited call; a missing file is created with
    mode 0600, an existing one keeps its mode. Writes serialize through
    ``_write_lock``; the cosine-similarity scan inside :meth:`search`
    runs in a worker thread so a long fetch does not stall the loop.
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
            if self._db_path != ":memory:":
                create_private_file(self._db_path)
            conn = await aiosqlite.connect(self._db_path)
            await prepare_connection(conn, _SCHEMA)
            self._conn = conn
            return conn

    async def add(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = await self._ensure_ready()
        vec = np.asarray(embedding, dtype=np.float32)
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO long_term (text, vector, metadata) VALUES (?, ?, ?)",
                (text, vec.tobytes(), json.dumps(metadata or {})),
            )
            await conn.commit()

    async def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_unit = q / q_norm

        conn = await self._ensure_ready()
        async with conn.execute("SELECT text, vector, metadata FROM long_term") as cur:
            raw_rows = await cur.fetchall()

        if not raw_rows:
            return []

        rows: list[tuple[str, bytes, str]] = [
            (str(r[0]), bytes(r[1]), str(r[2])) for r in raw_rows
        ]
        return await asyncio.to_thread(_rank_rows, rows, q_unit, top_k)

    async def count_entries(self) -> int:
        """Diagnostic helper: total long-term entries persisted."""
        conn = await self._ensure_ready()
        async with conn.execute("SELECT COUNT(*) FROM long_term") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def forget(self, task_id: str) -> None:
        """Delete the entries whose metadata names *task_id* as their task.

        An entry's provenance is its ``metadata`` JSON object: it names a
        task when its ``task_id`` key equals *task_id*. The reflections
        that ``Hippo.promote_reflections`` writes carry that key; a step
        summary carries only ``step_id``, so a per-task forget leaves it
        and :meth:`forget_all` is the only call that removes it.

        The read that picks the ids and the delete run in one
        ``BEGIN IMMEDIATE`` transaction under the write lock. As in
        ``SqliteHistory.forget``, the commit overwrites what it frees and
        is followed by a WAL checkpoint with truncation; one that another
        connection makes impossible is skipped with a WARNING.
        """
        await self._forget(task_id)

    async def forget_all(self) -> None:
        """Delete every long-term entry, whatever its provenance."""
        await self._forget(None)

    async def _forget(self, task_id: str | None) -> None:
        conn = await self._ensure_ready()
        async with self._write_lock:
            async with secure_deletes(conn):
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    if task_id is None:
                        await conn.execute("DELETE FROM long_term")
                    else:
                        ids: list[int] = []
                        async with conn.execute(
                            "SELECT id, metadata FROM long_term"
                        ) as cur:
                            async for row_id, metadata in cur:
                                try:
                                    parsed = json.loads(metadata)
                                except (TypeError, ValueError):
                                    continue
                                if (
                                    isinstance(parsed, dict)
                                    and parsed.get("task_id") == task_id
                                ):
                                    ids.append(int(row_id))
                        if ids:
                            await conn.executemany(
                                "DELETE FROM long_term WHERE id = ?",
                                [(row_id,) for row_id in ids],
                            )
                    await conn.commit()
                except BaseException:
                    with contextlib.suppress(Exception):
                        await conn.rollback()
                    raise
            if not await checkpoint_wal(conn):
                logger.warning(
                    "Forgot %s but could not truncate the WAL of %s: "
                    "another connection holds it",
                    "every entry" if task_id is None else f"the entries of {task_id}",
                    self._db_path,
                )

    async def aclose(self) -> None:
        """Close the underlying aiosqlite connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def close(self) -> None:
        """Sync compatibility shim. Prefer :meth:`aclose` in new code."""
        self._conn = None


def _rank_rows(
    rows: list[tuple[str, bytes, str]],
    q_unit: np.ndarray,
    top_k: int,
) -> list[tuple[str, float, dict[str, Any]]]:
    scores: list[tuple[str, float, dict[str, Any]]] = []
    for text, vec_bytes, meta_json in rows:
        vec = np.frombuffer(vec_bytes, dtype=np.float32)
        e_norm = np.linalg.norm(vec)
        if e_norm == 0:
            continue
        sim = float(np.dot(q_unit, vec / e_norm))
        scores.append((text, sim, json.loads(meta_json)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
