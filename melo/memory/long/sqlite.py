from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from localmelo.melo.memory._sqlite import apply_pragmas
from localmelo.melo.memory.long import LongTerm

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
    opened lazily on first awaited call. Writes serialize through
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
            conn = await aiosqlite.connect(self._db_path)
            await apply_pragmas(conn)
            await conn.executescript(_SCHEMA)
            await conn.commit()
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
